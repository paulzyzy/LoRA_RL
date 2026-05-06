"""LoRA (Low-Rank Adaptation) for BRC networks.

Matches the FastTD3 LoRA pattern:
- LoRA can be applied to the BroNet input Dense, residual-block Dense layers,
  and final output Dense independently.
- Base weights frozen via jax.lax.stop_gradient.
- Optional kernel_init_scale at init.
- NO L2 normalization during training (BRC doesn't use it).
- Target update merges effective weights and zeros target B.
"""

from typing import Callable, Optional

import flax.core
import flax.linen as nn
from flax.traverse_util import flatten_dict, unflatten_dict
import jax
import jax.numpy as jnp

from jaxrl.networks import TaskEmbedding


def default_init(scale: float = jnp.sqrt(2)):
    return nn.initializers.orthogonal(scale)


def get_lora_initializer(name: str) -> Callable:
    """Return a Flax-compatible parameter initializer by name."""
    name = (name or "zero").lower()
    if name == "zero":
        return nn.initializers.zeros
    if name == "normal":
        def _normal_init(key, shape, dtype=jnp.float32):
            return jax.random.normal(key, shape, dtype=dtype) / jnp.sqrt(
                jnp.asarray(min(shape[0], shape[1]), dtype=dtype)
            )
        return _normal_init
    if name == "orthogonal":
        return nn.initializers.orthogonal()
    if name == "kaiming_uniform":
        return nn.initializers.variance_scaling(
            scale=2.0, mode="fan_in", distribution="uniform"
        )
    raise ValueError(f"Unsupported LoRA initializer: {name}")


class LoRADense(nn.Module):
    """Dense layer with frozen base weights and trainable LoRA adapters.

    Forward: y = stop_grad(x @ W) + bias + (alpha/r) * (x @ A) @ B

    Matches FastTD3 LoRALinear. kernel_init_scale optionally scales base weight
    column norms to sqrt(scale) at init.
    """
    features: int
    rank: int
    alpha: float = 1.0
    a_init_name: str = "normal"
    b_init_name: str = "normal"
    kernel_init_scale: float = 0.0  # 0 = no scaling (keep BRC default init)

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        in_features = x.shape[-1]

        # Custom kernel initializer with optional scaling
        ki_scale = self.kernel_init_scale

        def _kernel_init(key, shape, dtype=jnp.float32):
            # Orthogonal init matching BRC default_init(sqrt(2))
            kernel = nn.initializers.orthogonal(scale=jnp.sqrt(2))(key, shape, dtype)

            # Scale column norms (matches FastTD3 scale_lora_base_weights)
            if ki_scale > 0:
                target_norm = jnp.sqrt(ki_scale)
                col_norms = jnp.linalg.norm(kernel, ord=2, axis=0, keepdims=True)
                denom = jnp.maximum(col_norms, 1e-8) / target_norm
                kernel = kernel / denom

            return kernel

        # Base weight (frozen via stop_gradient)
        kernel = self.param("kernel", _kernel_init, (in_features, self.features))
        bias = self.param("bias", nn.initializers.zeros, (self.features,))
        kernel = jax.lax.stop_gradient(kernel)
        y = jnp.dot(x, kernel) + bias

        # LoRA adapter
        r = max(1, min(self.rank, in_features, self.features))
        a_init = get_lora_initializer(self.a_init_name)
        b_init = get_lora_initializer(self.b_init_name)
        lora_A = self.param("lora_A", a_init, (in_features, r))
        lora_B = self.param("lora_B", b_init, (r, self.features))
        scaling = self.alpha / r
        y = y + scaling * jnp.dot(jnp.dot(x, lora_A), lora_B)

        return y


def _lora_alpha_for_prefix(
    prefix: str,
    block_alpha: float,
    input_alpha: Optional[float] = None,
    output_alpha: Optional[float] = None,
) -> float:
    """Return the LoRA alpha that belongs to a flattened parameter prefix."""
    if "/input_lora" in prefix or prefix.endswith("input_lora"):
        return block_alpha if input_alpha is None else input_alpha
    if "/output_lora" in prefix or prefix.endswith("output_lora"):
        return block_alpha if output_alpha is None else output_alpha
    return block_alpha


def _dense_or_lora(
    features: int,
    rank: int,
    alpha: float,
    a_init_name: str,
    b_init_name: str,
    kernel_init_scale: float,
    name: Optional[str] = None,
) -> nn.Module:
    if rank > 0:
        return LoRADense(
            features=features,
            rank=rank,
            alpha=alpha,
            a_init_name=a_init_name,
            b_init_name=b_init_name,
            kernel_init_scale=kernel_init_scale,
            name=name,
        )
    return nn.Dense(features, kernel_init=default_init(), name=name)


class BronetBlockLoRA(nn.Module):
    """Residual block with optional LoRA on both Dense layers."""
    hidden_dims: int
    rank: int
    alpha: float = 1.0
    a_init_name: str = "normal"
    b_init_name: str = "normal"
    kernel_init_scale: float = 0.0
    activations: Callable[[jnp.ndarray], jnp.ndarray] = nn.relu

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        res = _dense_or_lora(
            self.hidden_dims, self.rank, self.alpha,
            self.a_init_name, self.b_init_name,
            self.kernel_init_scale,
        )(x)
        res = nn.LayerNorm()(res)
        res = self.activations(res)
        res = _dense_or_lora(
            self.hidden_dims, self.rank, self.alpha,
            self.a_init_name, self.b_init_name,
            self.kernel_init_scale,
        )(res)
        res = nn.LayerNorm()(res)
        return res + x


class BroNetLoRA(nn.Module):
    """BroNet backbone with independently controlled LoRA zones."""
    hidden_dims: int
    depth: int
    rank: int
    alpha: float = 1.0
    input_rank: int = 0
    input_alpha: float = 1.0
    output_rank: int = 0
    output_alpha: float = 1.0
    a_init_name: str = "normal"
    b_init_name: str = "normal"
    kernel_init_scale: float = 0.0
    add_final_layer: bool = False
    output_nodes: int = 101
    activations: Callable[[jnp.ndarray], jnp.ndarray] = nn.relu

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        # Input projection before residual blocks.
        x = _dense_or_lora(
            self.hidden_dims, self.input_rank, self.input_alpha,
            self.a_init_name, self.b_init_name,
            self.kernel_init_scale,
            name="input_lora" if self.input_rank > 0 else None,
        )(x)
        x = nn.LayerNorm()(x)
        x = self.activations(x)
        # Residual blocks.
        for i in range(self.depth):
            x = BronetBlockLoRA(
                hidden_dims=self.hidden_dims,
                rank=self.rank,
                alpha=self.alpha,
                a_init_name=self.a_init_name,
                b_init_name=self.b_init_name,
                kernel_init_scale=self.kernel_init_scale,
                activations=self.activations,
            )(x)
        # Final projection after residual blocks.
        if self.add_final_layer:
            x = _dense_or_lora(
                self.output_nodes, self.output_rank, self.output_alpha,
                self.a_init_name, self.b_init_name,
                self.kernel_init_scale,
                name="output_lora" if self.output_rank > 0 else None,
            )(x)
        return x


class QValueLoRA(nn.Module):
    hidden_dims: int = 512
    depth: int = 2
    rank: int = 128
    alpha: float = 128.0
    input_rank: int = 0
    input_alpha: float = 1.0
    output_rank: int = 0
    output_alpha: float = 1.0
    a_init_name: str = "normal"
    b_init_name: str = "normal"
    kernel_init_scale: float = 0.0
    activations: Callable[[jnp.ndarray], jnp.ndarray] = nn.relu
    output_nodes: int = 101

    def setup(self):
        self.critic = BroNetLoRA(
            hidden_dims=self.hidden_dims, depth=self.depth,
            rank=self.rank, alpha=self.alpha,
            input_rank=self.input_rank, input_alpha=self.input_alpha,
            output_rank=self.output_rank, output_alpha=self.output_alpha,
            a_init_name=self.a_init_name, b_init_name=self.b_init_name,
            kernel_init_scale=self.kernel_init_scale,
            activations=self.activations,
            add_final_layer=True, output_nodes=self.output_nodes,
        )

    def __call__(self, inputs: jnp.ndarray) -> jnp.ndarray:
        return self.critic(inputs)


class QValueEnsembleLoRA(nn.Module):
    ensemble_size: int = 2
    hidden_dims: int = 512
    depth: int = 2
    rank: int = 128
    alpha: float = 128.0
    input_rank: int = 0
    input_alpha: float = 1.0
    output_rank: int = 0
    output_alpha: float = 1.0
    a_init_name: str = "normal"
    b_init_name: str = "normal"
    kernel_init_scale: float = 0.0
    activations: Callable[[jnp.ndarray], jnp.ndarray] = nn.relu
    output_nodes: int = 101

    def setup(self):
        VmapCritic = nn.vmap(
            QValueLoRA,
            variable_axes={'params': 0},
            split_rngs={'params': True},
            in_axes=None,
            out_axes=0,
            axis_size=self.ensemble_size,
        )
        self.q_value_ensemble = VmapCritic(
            hidden_dims=self.hidden_dims, depth=self.depth,
            rank=self.rank, alpha=self.alpha,
            input_rank=self.input_rank, input_alpha=self.input_alpha,
            output_rank=self.output_rank, output_alpha=self.output_alpha,
            a_init_name=self.a_init_name, b_init_name=self.b_init_name,
            kernel_init_scale=self.kernel_init_scale,
            activations=self.activations, output_nodes=self.output_nodes,
        )

    def __call__(self, inputs: jnp.ndarray) -> jnp.ndarray:
        return self.q_value_ensemble(inputs)


class CriticLoRA(nn.Module):
    """Critic with LoRA on residual blocks of the Q-value network."""
    num_tasks: int
    embedding_size: int
    ensemble_size: int = 2
    hidden_dims: int = 512
    depth: int = 2
    rank: int = 128
    alpha: float = 128.0
    input_rank: int = 0
    input_alpha: float = 1.0
    output_rank: int = 0
    output_alpha: float = 1.0
    a_init_name: str = "normal"
    b_init_name: str = "normal"
    kernel_init_scale: float = 0.0
    activations: Callable[[jnp.ndarray], jnp.ndarray] = nn.relu
    output_nodes: int = 101
    multitask: bool = False

    def setup(self):
        if self.multitask:
            self.task_embedding = TaskEmbedding(self.num_tasks, self.embedding_size)
        self.q_value_ensemble = QValueEnsembleLoRA(
            ensemble_size=self.ensemble_size,
            hidden_dims=self.hidden_dims, depth=self.depth,
            rank=self.rank, alpha=self.alpha,
            input_rank=self.input_rank, input_alpha=self.input_alpha,
            output_rank=self.output_rank, output_alpha=self.output_alpha,
            a_init_name=self.a_init_name, b_init_name=self.b_init_name,
            kernel_init_scale=self.kernel_init_scale,
            activations=self.activations, output_nodes=self.output_nodes,
        )

    def __call__(self, observations: jnp.ndarray, actions: jnp.ndarray,
                 task_ids: jnp.ndarray, return_embeddings: bool = False):
        if self.multitask is False:
            inputs = jnp.concatenate((observations, actions), axis=-1)
        else:
            task_embedding = self.task_embedding(task_ids)
            if return_embeddings:
                return task_embedding
            inputs = jnp.concatenate((observations, actions, task_embedding), axis=-1)
        q_values = self.q_value_ensemble(inputs)
        return q_values


def update_target_critic_lora(
    critic,
    target_critic,
    tau,
    lora_alpha,
    lora_input_alpha=None,
    lora_output_alpha=None,
):
    """LoRA-aware Polyak averaging for target network update.

    Matches FastTD3 soft_update_with_lora:
    - For LoRA layers: merge effective weight, Polyak average, zero target B
    - For other params: standard Polyak averaging
    """
    online_flat = flatten_dict(flax.core.unfreeze(critic.params), sep='/')
    target_flat = flatten_dict(flax.core.unfreeze(target_critic.params), sep='/')

    # Standard Polyak for all leaves first.
    new_flat = {
        name: tau * online_flat[name] + (1 - tau) * target_flat[name]
        for name in online_flat
    }

    # LoRA leaves get merged into the frozen kernel with the path-specific alpha.
    for name in list(online_flat.keys()):
        if not name.endswith('/kernel'):
            continue
        prefix = name.rsplit('/kernel', 1)[0]
        a_key = f'{prefix}/lora_A'
        b_key = f'{prefix}/lora_B'
        if a_key not in online_flat or b_key not in online_flat:
            continue

        alpha = _lora_alpha_for_prefix(
            prefix, lora_alpha, lora_input_alpha, lora_output_alpha)
        A_on = online_flat[a_key]
        B_on = online_flat[b_key]
        A_tgt = target_flat[a_key]
        B_tgt = target_flat[b_key]
        r = A_on.shape[-1]

        W_eff_on = online_flat[name] + (alpha / r) * jnp.matmul(A_on, B_on)
        W_eff_tgt = target_flat[name] + (alpha / r) * jnp.matmul(A_tgt, B_tgt)

        new_flat[name] = tau * W_eff_on + (1 - tau) * W_eff_tgt
        new_flat[a_key] = target_flat[a_key]
        new_flat[b_key] = jnp.zeros_like(B_tgt)

    new_target_params = unflatten_dict(new_flat, sep='/')
    return target_critic.replace(params=flax.core.freeze(new_target_params))


def scale_lora_init_norms(
    params,
    kernel_init_scale: float,
    lora_alpha: float,
    lora_input_alpha=None,
    lora_output_alpha=None,
):
    """Post-init rescaling for LoRA layers so that, at initialization,
    both the base kernel and the effective LoRA contribution have column
    L2 norm ~= sqrt(kernel_init_scale).

    For a 2-D kernel (in, out) the "column" axis is 0 (input dim).
    For a 3-D vmapped kernel (ens, in, out) the axis is 1.

    Matches FastTD3's scale_lora_base_weights + scale_lora_AB and the
    SimbaV2 _DenseKernel kernel_init_scale normalization.
    No-op on zero lora_B columns because their direction is undefined.
    """
    eps = 1e-8
    target_norm = jnp.asarray(jnp.sqrt(kernel_init_scale))

    flat = flatten_dict(flax.core.unfreeze(params), sep='/')

    for name in list(flat.keys()):
        if not name.endswith('/kernel'):
            continue
        prefix = name.rsplit('/kernel', 1)[0]
        a_key = f'{prefix}/lora_A'
        b_key = f'{prefix}/lora_B'
        if a_key not in flat or b_key not in flat:
            continue

        kernel = flat[name]
        A = flat[a_key]
        B = flat[b_key]

        if kernel.ndim == 2:
            axis = 0
        elif kernel.ndim == 3:
            axis = 1
        else:
            raise ValueError(f'Unexpected kernel ndim: {kernel.ndim}')

        # Rescale base kernel columns to target_norm.
        col_norms = jnp.linalg.norm(kernel, ord=2, axis=axis, keepdims=True)
        flat[name] = kernel * (target_norm / jnp.maximum(col_norms, eps))

        # Rescale lora_B per output column so that (alpha/r)*A@B has
        # column norm = target_norm, matching the frozen base kernel.
        r = A.shape[-1]
        alpha = _lora_alpha_for_prefix(
            prefix, lora_alpha, lora_input_alpha, lora_output_alpha)
        scaling = alpha / r
        lora_w = scaling * jnp.matmul(A, B)
        col_norms_lora = jnp.linalg.norm(lora_w, ord=2, axis=axis, keepdims=True)
        scale = jnp.where(
            col_norms_lora > eps,
            target_norm / jnp.maximum(col_norms_lora, eps),
            jnp.ones_like(col_norms_lora),
        ).astype(B.dtype)
        if B.ndim == 2:
            flat[b_key] = B * scale
        elif B.ndim == 3:
            flat[b_key] = B * scale
        else:
            raise ValueError(f'Unexpected lora_B ndim: {B.ndim}')

    return flax.core.freeze(unflatten_dict(flat, sep='/'))


def count_lora_params(params):
    """Count total, trainable, and frozen params for a LoRA model.

    Returns (total, trainable, frozen).
    Frozen = base kernels in LoRA layers (identified by having lora_A sibling).
    """
    from flax.traverse_util import flatten_dict
    flat = flatten_dict(flax.core.unfreeze(params), sep='/')

    total = 0
    frozen = 0
    for name, param in flat.items():
        total += param.size
        if name.endswith('/kernel'):
            prefix = name.rsplit('/kernel', 1)[0]
            if f'{prefix}/lora_A' in flat:
                frozen += param.size
    return total, total - frozen, frozen
