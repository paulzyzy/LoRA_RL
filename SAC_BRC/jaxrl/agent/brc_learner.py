import functools
from typing import Optional

import jax
import jax.numpy as jnp
import numpy as np
import optax

from jaxrl.agent.update import build_actor_input, update_actor, update_critic, update_target_critic, update_temperature

import flax.core
from flax.traverse_util import flatten_dict

from jaxrl.networks import NormalTanhPolicy, Critic, Temperature
from jaxrl.lora import CriticLoRA, update_target_critic_lora, count_lora_params, scale_lora_init_norms
from jaxrl.utils import Model, PRNGKey, Batch


def _create_lora_optimizer(critic_lr, lora_weight_decay=6e-4):
    """Create a LoRA-aware optimizer with 3 param groups (matching SimbaV2).

    - lora:   adamw for LoRA A/B params (with lora_weight_decay)
    - base:   adamw for other trainable params (BRC default wd=1e-4)
    - frozen: set_to_zero for frozen base kernels in LoRA layers
    """
    def param_labels(params):
        flat = flatten_dict(flax.core.unfreeze(params), sep='/')
        frozen_kernels = set()
        for name in flat:
            if name.endswith('/kernel'):
                prefix = name.rsplit('/kernel', 1)[0]
                if f'{prefix}/lora_A' in flat:
                    frozen_kernels.add(name)

        def _label(path, _):
            parts = [str(p.key) if hasattr(p, 'key') else str(p) for p in path]
            path_str = '/'.join(parts)
            if path_str.endswith('/lora_A') or path_str.endswith('/lora_B'):
                return 'lora'
            if path_str in frozen_kernels:
                return 'frozen'
            return 'base'

        return jax.tree_util.tree_map_with_path(_label, params)

    transforms = {
        'lora': optax.adamw(learning_rate=critic_lr, weight_decay=lora_weight_decay),
        'base': optax.adamw(learning_rate=critic_lr),
        'frozen': optax.set_to_zero(),
    }
    return optax.multi_transform(transforms, param_labels)


@jax.jit
def _get_temperature(temp):
    temp_val = temp()
    return temp_val

@jax.jit
def _sample_actions(
    rng: PRNGKey,
    actor: Model,
    inputs: np.ndarray,
    temperature: float = 1.0,
):
    dist = actor(inputs, temperature)
    rng, key = jax.random.split(rng)
    actions = dist.sample(seed=key)
    return rng, actions


# ============================================================
# Method-specific update functions
# ============================================================

def _make_update_fn(
    method,
    lora_alpha=None,
    lora_input_alpha=None,
    lora_output_alpha=None,
):
    """Factory: create a method-specific single-step update function.

    The returned function has the same signature for all methods so that
    _do_multiple_updates can wrap it uniformly.
    """
    def _update(rng, actor, critic, target_critic, temp, batch,
                discount, tau, target_entropy, num_bins, v_max, multitask):
        rng, actor_key, critic_key = jax.random.split(rng, 3)

        # --- Critic update ---
        new_critic, critic_info = update_critic(
            critic_key, actor, critic, target_critic, temp,
            batch, discount, num_bins, v_max, multitask)

        # --- Target critic update ---
        if method == 'lora':
            new_target_critic = update_target_critic_lora(
                new_critic, target_critic, tau,
                lora_alpha, lora_input_alpha, lora_output_alpha)
        else:
            new_target_critic = update_target_critic(new_critic, target_critic, tau)

        # --- Actor update ---
        new_actor, actor_info = update_actor(
            actor_key, actor, new_critic, temp, batch,
            num_bins, v_max, multitask)

        # --- Temperature update ---
        new_temp, alpha_info = update_temperature(
            temp, actor_info['entropy'], target_entropy)

        return rng, new_actor, new_critic, new_target_critic, new_temp, {
            **critic_info, **actor_info, **alpha_info,
        }
    return _update


def _make_do_multiple_updates(update_fn):
    """Wrap a single-step update_fn into a jitted fori_loop."""
    @functools.partial(jax.jit, static_argnames=(
        'discount', 'tau', 'target_entropy', 'num_bins', 'v_max',
        'multitask', 'num_updates'))
    def _do_multiple_updates(
        rng, actor, critic, target_critic, temp, batches,
        discount, tau, target_entropy, num_bins, v_max, multitask,
        step, num_updates,
    ):
        def one_step(i, state):
            step, rng, actor, critic, target_critic, temp, info = state
            step = step + 1
            new_rng, new_actor, new_critic, new_target_critic, new_temp, info = update_fn(
                rng, actor, critic, target_critic, temp,
                jax.tree.map(lambda x: jnp.take(x, i, axis=0), batches),
                discount, tau, target_entropy, num_bins, v_max, multitask,
            )
            return step, new_rng, new_actor, new_critic, new_target_critic, new_temp, info

        step, rng, actor, critic, target_critic, temp, info = one_step(
            0, (step, rng, actor, critic, target_critic, temp, {}))
        return jax.lax.fori_loop(
            1, num_updates, one_step,
            (step, rng, actor, critic, target_critic, temp, info))

    return _do_multiple_updates


class BRC(object):
    def __init__(
        self,
        seed: int,
        observations: jnp.ndarray,
        actions: jnp.ndarray,
        num_tasks: int,
        embedding_size: int = 32,
        ensemble_size: int = 2,
        actor_lr: float = 3e-4,
        critic_lr: float = 3e-4,
        temp_lr: float = 3e-4,
        discount: float = 0.99,
        tau: float = 0.005,
        target_entropy: Optional[float] = None,
        init_temperature: float = 0.1,
        updates_per_step: int = 10,
        width_critic: int = 512,
        width_actor: int = 256,
        depth_critic: int = 2,
        num_bins: int = 101,
        v_max: float = 10.0,
        # Method selection: 'base' or 'lora'
        method: str = 'lora',
        # LoRA params (only used when method='lora')
        lora_rank: int = 128,
        lora_alpha: float = 128.0,
        lora_input_rank: int = 0,
        lora_input_alpha: float = -1.0,
        lora_output_rank: int = 0,
        lora_output_alpha: float = -1.0,
        lora_a_init: str = 'normal',
        lora_b_init: str = 'normal',
        lora_kernel_init_scale: float = 0.5,
        lora_weight_decay: float = 6e-4,
    ) -> None:

        action_dim = actions.shape[-1]
        self.action_dim = float(action_dim)
        self.seed = seed
        self.target_entropy = -self.action_dim / 2 if target_entropy is None else target_entropy
        self.tau = tau
        self.discount = discount
        self.num_bins = num_bins
        self.v_max = v_max
        self.method = method
        self.lora_alpha = lora_alpha
        lora_input_alpha = lora_alpha if lora_input_alpha < 0 else lora_input_alpha
        lora_output_alpha = lora_alpha if lora_output_alpha < 0 else lora_output_alpha

        if method not in ('base', 'lora'):
            raise ValueError(f"Unsupported method '{method}'. Expected 'base' or 'lora'.")

        self.num_tasks = num_tasks
        self.embedding_size = embedding_size
        self.task_ids = jnp.arange(num_tasks, dtype=jnp.int32)

        task_embedding_init = jnp.zeros((1, embedding_size))
        task_ids_init = self.task_ids[:1]
        self.multitask = True if num_tasks > 1 else False

        actor_init = jnp.concatenate((observations, task_embedding_init), axis=-1) if self.multitask else observations

        def _init_models(seed):
            rng = jax.random.PRNGKey(seed)
            rng, actor_key, critic_key, temp_key = jax.random.split(rng, 4)

            # Actor is always the same regardless of method
            actor_def = NormalTanhPolicy(action_dim=action_dim, hidden_dims=width_actor)

            # Critic depends on method
            if method == 'lora':
                critic_def = CriticLoRA(
                    num_tasks=num_tasks, embedding_size=embedding_size,
                    ensemble_size=ensemble_size, hidden_dims=width_critic,
                    depth=depth_critic, output_nodes=num_bins,
                    multitask=self.multitask,
                    rank=lora_rank, alpha=lora_alpha,
                    input_rank=lora_input_rank, input_alpha=lora_input_alpha,
                    output_rank=lora_output_rank, output_alpha=lora_output_alpha,
                    a_init_name=lora_a_init, b_init_name=lora_b_init,
                    kernel_init_scale=lora_kernel_init_scale,
                )
            else:
                critic_def = Critic(
                    num_tasks=num_tasks, embedding_size=embedding_size,
                    ensemble_size=ensemble_size, hidden_dims=width_critic,
                    depth=depth_critic, output_nodes=num_bins,
                    multitask=self.multitask,
                )

            actor = Model.create(actor_def,
                inputs=[actor_key, actor_init],
                tx=optax.adamw(learning_rate=actor_lr))
            if method == 'lora':
                critic_tx = _create_lora_optimizer(critic_lr, lora_weight_decay)
            else:
                critic_tx = optax.adamw(learning_rate=critic_lr)
            critic = Model.create(critic_def,
                inputs=[critic_key, observations, actions, task_ids_init],
                tx=critic_tx)
            target_critic = Model.create(critic_def,
                inputs=[critic_key, observations, actions, task_ids_init])

            # For LoRA, post-init rescale so ||W||_col = ||(alpha/r)*A@B||_col
            # = sqrt(kernel_init_scale). With scale=0.5 each is ~1/sqrt(2),
            # yielding ~unit effective weight at step 0 (as in SimbaV2/FastTD3).
            if method == 'lora' and lora_kernel_init_scale > 0:
                new_params = scale_lora_init_norms(
                    critic.params,
                    lora_kernel_init_scale,
                    lora_alpha,
                    lora_input_alpha,
                    lora_output_alpha,
                )
                new_opt_state = critic.tx.init(new_params)
                critic = critic.replace(params=new_params, opt_state=new_opt_state)
                target_critic = target_critic.replace(params=new_params)

            temp = Model.create(Temperature(init_temperature),
                inputs=[temp_key],
                tx=optax.adam(learning_rate=temp_lr, b1=0.5))
            return actor, critic, target_critic, temp, rng

        self.init_models = jax.jit(_init_models)
        self.actor, self.critic, self.target_critic, self.temp, self.rng = self.init_models(self.seed)
        self.step = 1

        # --- Parameter counting ---
        self._log_param_counts()

        # --- Build method-specific update functions ---
        update_fn = _make_update_fn(
            method,
            lora_alpha,
            lora_input_alpha,
            lora_output_alpha,
        )
        self._do_multiple_updates = _make_do_multiple_updates(update_fn)

    def _log_param_counts(self):
        """Log parameter counts and store as a dict for external use."""
        actor_total = sum(p.size for p in jax.tree_util.tree_leaves(self.actor.params))
        critic_total = sum(p.size for p in jax.tree_util.tree_leaves(self.critic.params))

        info = {'actor_total': actor_total, 'critic_total': critic_total}

        if self.method == 'lora':
            total, trainable, frozen = count_lora_params(self.critic.params)
            info.update(critic_trainable=trainable, critic_frozen=frozen)
            print(f"[Method: LoRA]")
            print(f"  [Actor]  total params = {actor_total:,}")
            print(f"  [Critic] total params = {total:,}, trainable = {trainable:,}, frozen = {frozen:,}")
        else:
            print(f"[Method: Baseline]")
            print(f"  [Actor]  total params = {actor_total:,}")
            print(f"  [Critic] total params = {critic_total:,}")

        self.param_info = info

    def sample_actions(self, observations: np.ndarray, temperature: float = 1.0):
        inputs = build_actor_input(self.critic, observations, self.task_ids, self.multitask)
        rng, actions = _sample_actions(self.rng, self.actor, inputs, temperature)
        self.rng = rng
        actions = np.asarray(actions)
        return np.clip(actions, -1, 1)

    def update(self, batch: Batch, num_updates: int, env_step: int):
        step, rng, actor, critic, target_critic, temp, info = self._do_multiple_updates(
            self.rng,
            self.actor,
            self.critic,
            self.target_critic,
            self.temp,
            batch,
            self.discount,
            self.tau,
            self.target_entropy,
            self.num_bins,
            self.v_max,
            self.multitask,
            self.step,
            num_updates
        )
        self.step = step
        self.rng = rng
        self.actor = actor
        self.critic = critic
        self.target_critic = target_critic
        self.temp = temp
        return info

    def get_temperature(self):
        return _get_temperature(self.temp)

    def reset(self):
        self.step = 1
        self.actor, self.critic, self.target_critic, self.temp, self.rng = self.init_models(self.seed)

    def save(self, path):
        self.actor.save(f'{path}/actor.txt')
        self.critic.save(f'{path}/critic.txt')
        self.target_critic.save(f'{path}/target_critic.txt')
        self.temp.save(f'{path}/temp.txt')

    def load(self, path):
        self.actor = self.actor.load(f'{path}/actor.txt')
        self.critic = self.critic.load(f'{path}/critic.txt')
        self.target_critic = self.target_critic.load(f'{path}/target_critic.txt')
        self.temp = self.temp.load(f'{path}/temp.txt')
