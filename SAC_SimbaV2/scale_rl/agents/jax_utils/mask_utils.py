from __future__ import annotations

from typing import Optional, Sequence

import flax
import jax
import jax.numpy as jnp
from flax.core import FrozenDict, freeze, unfreeze
from flax.traverse_util import flatten_dict


def tree_ones_like(tree: FrozenDict) -> FrozenDict:
    return jax.tree_util.tree_map(lambda x: jnp.ones_like(x), tree)


def combine_masks(a: Optional[FrozenDict], b: Optional[FrozenDict]) -> Optional[FrozenDict]:
    if a is None:
        return b
    if b is None:
        return a
    combined = jax.tree_util.tree_map(lambda x, y: x * y, a, b)
    return combined


def apply_value_mask(params: FrozenDict, mask: Optional[FrozenDict]) -> FrozenDict:
    if mask is None:
        return params
    return freeze(jax.tree_util.tree_map(lambda p, m: p * m, params, mask))


def build_freeze_mask(
    params: FrozenDict,
    specs: Sequence[Sequence[str]],
) -> Optional[FrozenDict]:
    if not specs:
        return None

    flat_params = flatten_dict(unfreeze(params))
    mask_flat = {}
    for path, value in flat_params.items():
        path_str = "/".join(path)
        freeze_param = any(all(token in path_str for token in spec) for spec in specs)
        if freeze_param:
            mask_flat[path] = jnp.zeros_like(value)
        else:
            mask_flat[path] = jnp.ones_like(value)

    return freeze(unflatten_dict(mask_flat))
