from typing import Any, Dict, Tuple

import flax
import jax
import jax.numpy as jnp

from scale_rl.agents.jax_utils.network import Network, PRNGKey
from scale_rl.buffers import Batch


def update_actor(
    key: PRNGKey,
    actor: Network,
    critic: Network,
    temperature: Network,
    batch: Batch,
    use_cdq: bool,
    bc_alpha: float,
) -> Tuple[Network, Dict[str, float]]:
    def actor_loss_fn(
        actor_params: flax.core.FrozenDict[str, Any],
    ) -> Tuple[jnp.ndarray, Dict[str, float]]:
        dist, _ = actor.apply(
            variables={"params": actor_params},
            observations=batch["observation"],
        )
        actions = dist.sample(seed=key)
        log_probs = dist.log_prob(actions)

        if use_cdq:
            # qs: (2, n)
            qs, q_infos = critic(observations=batch["observation"], actions=actions)
            q = jnp.minimum(qs[0], qs[1])
        else:
            q, _ = critic(observations=batch["observation"], actions=actions)

        actor_loss = (log_probs * temperature() - q).mean()

        if bc_alpha > 0:
            # https://arxiv.org/abs/2306.02451
            q_abs = jax.lax.stop_gradient(jnp.abs(q).mean())
            bc_loss = ((actions - batch["action"]) ** 2).mean()
            actor_loss = actor_loss + bc_alpha * q_abs * bc_loss

        actor_info = {
            "actor/loss": actor_loss,
            "actor/entropy": -log_probs.mean(),
            "actor/mean_action": jnp.mean(actions),
        }
        return actor_loss, actor_info

    actor, info = actor.apply_gradient(actor_loss_fn)

    return actor, info


def update_critic(
    key: PRNGKey,
    actor: Network,
    critic: Network,
    target_critic: Network,
    temperature: Network,
    batch: Batch,
    use_cdq: bool,
    gamma: float,
    n_step: int,
) -> Tuple[Network, Dict[str, float]]:
    # compute the target q-value
    next_dist, _ = actor(observations=batch["next_observation"])
    next_actions = next_dist.sample(seed=key)
    next_actor_log_probs = next_dist.log_prob(next_actions)
    next_actor_entropy = temperature() * next_actor_log_probs

    if use_cdq:
        # next_qs: (2, n)
        next_qs, next_q_infos = target_critic(
            observations=batch["next_observation"], actions=next_actions
        )
        next_q = jnp.minimum(next_qs[0], next_qs[1])
    else:
        next_q, next_q_info = target_critic(
            observations=batch["next_observation"],
            actions=next_actions,
        )

    # compute the td-target, incorporating the n-step accumulated reward
    # https://gymnasium.farama.org/tutorials/gymnasium_basics/handling_time_limits/
    target_q = batch["reward"] + (gamma**n_step) * (1 - batch["terminated"]) * (
        next_q - next_actor_entropy
    )

    def critic_loss_fn(
        critic_params: flax.core.FrozenDict[str, Any],
    ) -> Tuple[jnp.ndarray, Dict[str, float]]:
        # compute predicted q-value
        if use_cdq:
            pred_qs, pred_q_infos = critic.apply(
                variables={"params": critic_params},
                observations=batch["observation"],
                actions=batch["action"],
            )
            loss_1 = (pred_qs[0] - target_q) ** 2
            loss_2 = (pred_qs[1] - target_q) ** 2
            critic_loss = (loss_1 + loss_2).mean()
        else:
            pred_q, _ = critic.apply(
                variables={"params": critic_params},
                observations=batch["observation"],
                actions=batch["action"],
            )
            critic_loss = ((pred_q - target_q) ** 2).mean()

        critic_info = {
            "critic/loss": critic_loss,
            "critic/batch_rew_min": batch["reward"].min(),
            "critic/batch_rew_mean": batch["reward"].mean(),
            "critic/batch_rew_max": batch["reward"].max(),
        }

        return critic_loss, critic_info

    critic, info = critic.apply_gradient(critic_loss_fn)

    return critic, info


def update_target_network(
    network: Network,
    target_network: Network,
    target_tau: bool,
) -> Tuple[Network, Dict[str, float]]:
    new_target_params = jax.tree_map(
        lambda p, tp: p * target_tau + tp * (1 - target_tau),
        network.params,
        target_network.params,
    )
    target_network = target_network.replace(params=new_target_params)

    info = {}

    return target_network, info


def update_temperature(
    temperature: Network, entropy: float, target_entropy: float
) -> Tuple[Network, Dict[str, float]]:
    def temperature_loss_fn(
        temperature_params: flax.core.FrozenDict[str, Any],
    ) -> Tuple[jnp.ndarray, Dict[str, float]]:
        temperature_value = temperature.apply({"params": temperature_params})
        temperature_loss = temperature_value * (entropy - target_entropy).mean()
        temperature_info = {
            "temperature/value": temperature_value,
            "temperature/loss": temperature_loss,
        }

        return temperature_loss, temperature_info

    temperature, info = temperature.apply_gradient(temperature_loss_fn)

    return temperature, info
