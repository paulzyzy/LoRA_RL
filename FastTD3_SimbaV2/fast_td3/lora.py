"""
LoRA (Low-Rank Adaptation) implementation for FastTD3.

Mirrors the JAX/Flax LoRA implementation in scale_rl/agents/simbaV2/simbaV2_layer.py.
When LoRA is disabled, the network behaves exactly as the baseline.
When LoRA is enabled:
  - Base weights are frozen (requires_grad=False)
  - LoRA A and B matrices are trainable
  - Forward: y = Linear(x) + (alpha/r) * x @ A @ B
  - Target network update merges LoRA into base weights, zeros B in target
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def get_initializer(name: str):
    """Return an initialization function matching the JAX implementation."""
    if name == "zero":
        return nn.init.zeros_
    elif name == "normal":
        def _normal_init(tensor):
            std = 1.0 / math.sqrt(min(tensor.shape))
            return nn.init.normal_(tensor, mean=0.0, std=std)
        return _normal_init
    elif name == "kaiming_uniform":
        def _kaiming_init(tensor):
            return nn.init.kaiming_uniform_(tensor, a=math.sqrt(5))
        return _kaiming_init
    elif name == "orthogonal":
        def _ortho_init(tensor):
            if tensor.ndim < 2:
                return nn.init.normal_(tensor)
            return nn.init.orthogonal_(tensor)
        return _ortho_init
    elif name == "identity":
        def _identity_init(tensor):
            if tensor.ndim < 2:
                return nn.init.ones_(tensor)
            return nn.init.eye_(tensor)
        return _identity_init
    else:
        raise ValueError(f"Unknown initializer: {name}")


class LoRALinear(nn.Module):
    """Linear layer with LoRA adapter.

    When active, the base weight is frozen and the output is:
        y = x @ W^T + bias + (alpha / rank) * x @ A @ B

    Where A has shape (in_features, rank) and B has shape (rank, out_features).
    This matches the JAX LoRAAdapter convention:
        LoRA(x) = (alpha/r) * x @ A @ B
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        rank: int,
        alpha: float = 1.0,
        a_init: str = "kaiming_uniform",
        b_init: str = "zero",
        bias: bool = True,
        device: torch.device = None,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank

        # Base linear layer (weight frozen, bias trainable)
        self.linear = nn.Linear(in_features, out_features, bias=bias, device=device)
        self.linear.weight.requires_grad = False

        # LoRA matrices: A (in_features, rank), B (rank, out_features)
        self.lora_A = nn.Parameter(torch.empty(in_features, rank, device=device))
        self.lora_B = nn.Parameter(torch.empty(rank, out_features, device=device))

        # Initialize LoRA matrices
        get_initializer(a_init)(self.lora_A)
        get_initializer(b_init)(self.lora_B)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Base forward (weight is frozen via requires_grad=False)
        y = self.linear(x)
        # LoRA contribution: (alpha/r) * x @ A @ B
        lora_out = (x @ self.lora_A) @ (self.lora_B * self.scaling)
        return y + lora_out

    def get_effective_weight(self) -> torch.Tensor:
        """Return the effective weight W + (alpha/r) * (A @ B)^T.

        Returns weight in nn.Linear convention: (out_features, in_features).
        """
        # linear.weight is (out, in), LoRA produces (in, out) via A @ B
        return self.linear.weight + (self.lora_A @ self.lora_B).T * self.scaling


@torch.no_grad()
def scale_lora_base_weights(module: nn.Module, kernel_init_scale: float):
    """Scale base weights of LoRA layers so column norms equal sqrt(kernel_init_scale).

    Matches the JAX SimbaV2 _DenseKernel initialization:
        norm = ||kernel||_col
        kernel = kernel / (norm / sqrt(kernel_init_scale))

    This rescales each column to have L2 norm = sqrt(kernel_init_scale).
    """
    eps = 1e-8
    target_norm = math.sqrt(kernel_init_scale)
    for mod in module.modules():
        if isinstance(mod, LoRALinear):
            w = mod.linear.weight.data  # (out_features, in_features)
            # JAX kernel shape is (in, out), norm along axis=0 (input dim)
            # → normalizes each output neuron's weight vector.
            # nn.Linear weight is (out, in), so the equivalent is dim=1 (input dim).
            col_norms = torch.linalg.norm(w, ord=2, dim=1, keepdim=True)  # (out_features, 1)
            denom = torch.clamp(col_norms, min=eps) / target_norm
            mod.linear.weight.data.copy_(w / denom)


@torch.no_grad()
def scale_lora_AB(module: nn.Module, kernel_init_scale: float):
    """Scale LoRA B so that (alpha/r)*(A@B)^T matches the base row norms.

    The LoRA contribution in W-space is: (alpha/r) * (A @ B)^T, shape (out, in).
    We scale each column of B independently so each output row has L2 norm
    sqrt(kernel_init_scale), matching the frozen base weight row norm.
    """
    eps = 1e-8
    target_norm = math.sqrt(kernel_init_scale)
    for mod in module.modules():
        if isinstance(mod, LoRALinear):
            # LoRA contribution in W-space: (alpha/r) * (A @ B)^T, shape (out, in)
            # A: (in, rank), B: (rank, out)
            # A @ B: (in, out), transposed: (out, in)
            lora_w = (mod.lora_A.data @ mod.lora_B.data).T * mod.scaling  # (out, in)
            # Row norms (one per output neuron), same convention as scale_lora_base_weights.
            row_norms = torch.linalg.norm(lora_w, ord=2, dim=1)  # (out,)
            scale = torch.where(
                row_norms > eps,
                torch.as_tensor(target_norm, dtype=row_norms.dtype, device=row_norms.device)
                / torch.clamp(row_norms, min=eps),
                torch.ones_like(row_norms),
            )
            # B has shape (rank, out); scaling column j scales row j of lora_w.
            mod.lora_B.data.mul_(scale.unsqueeze(0))


def replace_linear_with_lora(
    module: nn.Module,
    rank: int,
    alpha: float,
    a_init: str = "kaiming_uniform",
    b_init: str = "zero",
    device: torch.device = None,
) -> nn.Module:
    """Replace all nn.Linear layers in a module with LoRALinear layers.

    The base weights are copied from the original Linear layers.
    This function modifies the module in-place and returns it.
    """
    for name, child in list(module.named_children()):
        if isinstance(child, nn.Linear):
            lora_layer = LoRALinear(
                in_features=child.in_features,
                out_features=child.out_features,
                rank=rank,
                alpha=alpha,
                a_init=a_init,
                b_init=b_init,
                bias=child.bias is not None,
                device=device,
            )
            # Copy base weights
            lora_layer.linear.weight.data.copy_(child.weight.data)
            if child.bias is not None:
                lora_layer.linear.bias.data.copy_(child.bias.data)
            setattr(module, name, lora_layer)
        elif isinstance(child, nn.Sequential):
            # For Sequential, replace Linear layers inside
            new_children = []
            for subchild in child:
                if isinstance(subchild, nn.Linear):
                    lora_layer = LoRALinear(
                        in_features=subchild.in_features,
                        out_features=subchild.out_features,
                        rank=rank,
                        alpha=alpha,
                        a_init=a_init,
                        b_init=b_init,
                        bias=subchild.bias is not None,
                        device=device,
                    )
                    lora_layer.linear.weight.data.copy_(subchild.weight.data)
                    if subchild.bias is not None:
                        lora_layer.linear.bias.data.copy_(subchild.bias.data)
                    new_children.append(lora_layer)
                else:
                    new_children.append(subchild)
            setattr(module, name, nn.Sequential(*new_children))
        else:
            replace_linear_with_lora(child, rank, alpha, a_init, b_init, device)
    return module


def get_lora_params(model: nn.Module):
    """Get LoRA parameters (A and B matrices)."""
    params = []
    for name, param in model.named_parameters():
        if "lora_A" in name or "lora_B" in name:
            params.append(param)
    return params


def get_non_lora_trainable_params(model: nn.Module):
    """Get trainable non-LoRA parameters (biases, etc.)."""
    params = []
    for name, param in model.named_parameters():
        if param.requires_grad and "lora_A" not in name and "lora_B" not in name:
            params.append(param)
    return params


@torch.no_grad()
def soft_update_with_lora(
    src: nn.Module,
    tgt: nn.Module,
    tau: float,
):
    """Soft update for networks with LoRA.

    For LoRA layers:
      - Compute effective weight of source: W_eff = W + (alpha/r) * (A @ B)^T
      - Compute effective weight of target too, because target B can be nonzero
        right after target.load_state_dict(source.state_dict()) when B_init is nonzero
      - new_target_W = tau * W_eff_src + (1 - tau) * W_eff_tgt
      - Zero target's B matrix

    For non-LoRA parameters: standard Polyak averaging.

    This matches the JAX update_target_network() in simbaV2_update.py.
    """
    src_modules = dict(src.named_modules())
    tgt_modules = dict(tgt.named_modules())

    # Collect LoRA module names for special handling
    lora_module_names = set()
    for name, mod in src.named_modules():
        if isinstance(mod, LoRALinear):
            lora_module_names.add(name)

    # Handle LoRA layers
    for name in lora_module_names:
        src_mod = src_modules[name]
        tgt_mod = tgt_modules[name]

        # Effective weights: W + (alpha/r) * (A @ B)^T
        src_eff_weight = src_mod.get_effective_weight()
        tgt_eff_weight = tgt_mod.get_effective_weight()

        # Polyak average
        new_tgt_weight = tau * src_eff_weight + (1 - tau) * tgt_eff_weight
        tgt_mod.linear.weight.data.copy_(new_tgt_weight)

        # Zero target B
        tgt_mod.lora_B.data.zero_()

        # Bias: standard Polyak
        if src_mod.linear.bias is not None:
            tgt_mod.linear.bias.data.mul_(1 - tau).add_(src_mod.linear.bias.data, alpha=tau)

    # Handle non-LoRA parameters
    handled_params = set()
    for name in lora_module_names:
        for pname, _ in src_modules[name].named_parameters():
            handled_params.add(f"{name}.{pname}")

    src_params = []
    tgt_params = []
    for (name, sp), (_, tp) in zip(src.named_parameters(), tgt.named_parameters()):
        if name not in handled_params:
            src_params.append(sp.data)
            tgt_params.append(tp.data)

    if src_params:
        torch._foreach_mul_(tgt_params, 1.0 - tau)
        torch._foreach_add_(tgt_params, src_params, alpha=tau)
