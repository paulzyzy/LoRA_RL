import copy
from functools import partial
from typing import Any, Optional, Tuple

import flax
import flax.core
import flax.linen as nn
import jax
import jax.numpy as jnp
import optax
from flax.training import checkpoints

PRNGKey = jnp.ndarray


@flax.struct.dataclass
class Network:
    network_def: nn.Module = flax.struct.field(pytree_node=False)
    params: flax.core.FrozenDict[str, Any]
    tx: Optional[optax.GradientTransformation] = flax.struct.field(pytree_node=False)
    opt_state: Optional[optax.OptState] = None
    update_step: int = 0
    mask: Optional[flax.core.FrozenDict[str, Any]] = flax.struct.field(
        default=None, pytree_node=False
    )
    value_mask: Optional[flax.core.FrozenDict[str, Any]] = flax.struct.field(
        default=None, pytree_node=False
    )
    """
    dataclass decorator makes custom class to be passed safely to Jax.
    https://flax.readthedocs.io/en/latest/api_reference/flax.struct.html

    Network class wraps network & optimizer to easily optimize the network under the hood.

    args:
        network_def: flax.nn style of network definition.
        params: network parameters.
        tx: optimizer (e.g., optax.Adam).
        opt_state: current state of the optimizer (e.g., beta_1 in Adam).
        update_step: number of update step so far.
    """

    @classmethod
    def create(
        cls,
        network_def: nn.Module,
        network_inputs: flax.core.FrozenDict[str, jnp.ndarray],
        tx: Optional[optax.GradientTransformation] = None,
        mask: Optional[flax.core.FrozenDict[str, Any]] = None,
        value_mask: Optional[flax.core.FrozenDict[str, Any]] = None,
    ) -> "Network":
        variables = network_def.init(**network_inputs)
        params = variables.pop("params")
        
        # Ensure params is a FrozenDict to maintain consistency with opt_state
        if not isinstance(params, flax.core.FrozenDict):
            params = flax.core.freeze(params)

        if value_mask is not None:
            params = jax.tree_util.tree_map(lambda p, m: p * m, params, value_mask)

        if tx is not None:
            opt_state = tx.init(params)
        else:
            opt_state = None

        network = cls(
            network_def=network_def,
            params=params,
            tx=tx,
            opt_state=opt_state,
            mask=mask,
            value_mask=value_mask,
        )

        return network

    def __call__(self, *args, **kwargs):
        return self.network_def.apply({"params": self.params}, *args, **kwargs)

    def save(self, path: str, step: int = 0, keep: int = 1) -> None:
        """
        Save parameters, optimizer state, and other metadata to the given path.
        """
        ckpt = {
            "params": self.params,
            "opt_state": self.opt_state,
            "update_step": self.update_step,
        }
        checkpoints.save_checkpoint(
            ckpt_dir=path,
            target=ckpt,
            step=step,
            overwrite=True,
            keep=keep,
        )

    def load(self, path: str, param_key: str = None, only_param: bool = False) -> None:
        """
        Load parameters, optimizer state, and other metadata from the given path.
        args:
            path (str): The path to the checkpoint directory.
            param_key (str): If specified, only the subset of parameters is loaded.
            only_param (bool): If True, only the parameters are loaded.
        """
        ckpt = checkpoints.restore_checkpoint(ckpt_dir=path, target=None)

        def _key_exists(d, key):
            """
            Recursively check if key exists in dictionary d.
            """
            if key in d:
                return True
            return any(_key_exists(v, key) for v in d.values() if isinstance(v, dict))

        def _recursive_replace(source, target, key):
            """
            Recursively replace the value of key in source from target.
            """
            for k, v in source.items():
                if k == key:
                    source[k] = target[k]
                elif (
                    isinstance(v, dict) and k in source and isinstance(target[k], dict)
                ):
                    _recursive_replace(source[k], target[k], key)
            return source

        if param_key:
            if not _key_exists(self.params, param_key):
                raise ValueError(f"The key '{param_key}' is missing")
            new_params = copy.deepcopy(self.params)
            new_params = _recursive_replace(new_params, ckpt["params"], param_key)
        else:
            new_params = ckpt["params"]

        if only_param:
            network = self.replace(params=new_params)
        else:
            # self.opt_state: named_tuple
            # ckpt['opt_state]: dictionary
            new_opt_state = jax.tree_util.tree_unflatten(
                jax.tree_util.tree_structure(self.opt_state),
                jax.tree_util.tree_leaves(ckpt["opt_state"]),
            )

            network = self.replace(
                params=new_params,
                opt_state=new_opt_state,
                update_step=ckpt["update_step"],
            )

        return network

    def apply(self, *args, **kwargs):
        return self.network_def.apply(*args, **kwargs)
    
    def apply_gradient(self, loss_fn, get_info=True) -> Tuple[Any, "Network"]:
        grad_fn = jax.grad(loss_fn, has_aux=True)
        grads, info = grad_fn(self.params)
        info["_grads"] = grads
        is_fin = True

        if self.mask is not None:
            grads = jax.tree_util.tree_map(lambda g, m: g * m, grads, self.mask)
            info["_grads"] = grads

        # Check if optimizer is SAM (requires grad_fn parameter)
        # SAM optimizers from optax.contrib.sam have a specific state structure
        # We detect SAM by checking if opt_state has the expected SAM structure
        # or by trying to import contrib and check the state type
        is_sam = False
        try:
            from optax import contrib
            # Check if state is a SAMState (works for both opaque and non-opaque modes)
            # For opaque mode, the state might be wrapped, so we check recursively
            def _is_sam_state(state):
                if isinstance(state, contrib.SAMState):
                    return True
                # For opaque mode or multi_transform, check nested states
                if isinstance(state, (tuple, list)):
                    return any(_is_sam_state(s) for s in state)
                if isinstance(state, dict):
                    return any(_is_sam_state(v) for v in state.values())
                return False
            
            is_sam = _is_sam_state(self.opt_state)
        except (ImportError, AttributeError):
            pass
        
        # Create grad_fn for SAM (takes params and step, but step is ignored)
        if is_sam:
            # SAM requires grad_fn that takes (params, step) where step is typically ignored
            # When using multi_transform with masking, SAM may pass params with MaskedNode objects
            # We need to unwrap them by replacing MaskedNode with corresponding values from self.params
            try:
                from optax import MaskedNode
                
                # Capture original params for reference (will be accessed inside traced function)
                original_params_ref = self.params
                
                # Create a wrapper that unwraps MaskedNodes before calling loss_fn
                def sam_loss_wrapper(perturbed_params, step):
                    # Flatten both trees, treating MaskedNode as a leaf
                    # This ensures we can properly match and replace MaskedNode values
                    def _is_leaf(x):
                        # Treat MaskedNode as a leaf so it's included in the leaves list
                        return isinstance(x, MaskedNode)
                    
                    # Flatten both trees with MaskedNode as leaf
                    perturbed_leaves, perturbed_treedef = jax.tree_util.tree_flatten(
                        perturbed_params, is_leaf=_is_leaf
                    )
                    original_leaves, original_treedef = jax.tree_util.tree_flatten(
                        original_params_ref, is_leaf=_is_leaf
                    )
                    
                    # Replace MaskedNode leaves with corresponding original leaves
                    unwrapped_leaves = []
                    for i, p_leaf in enumerate(perturbed_leaves):
                        if isinstance(p_leaf, MaskedNode):
                            # Use corresponding original leaf (should be at same index)
                            if i < len(original_leaves):
                                unwrapped_leaves.append(original_leaves[i])
                            else:
                                # Shouldn't happen, but keep MaskedNode as fallback
                                unwrapped_leaves.append(p_leaf)
                        else:
                            unwrapped_leaves.append(p_leaf)
                    
                    # Reconstruct using original structure (which doesn't have MaskedNode)
                    unwrapped_params = jax.tree_util.tree_unflatten(
                        original_treedef, unwrapped_leaves
                    )
                    
                    return loss_fn(unwrapped_params)[0]
                
                sam_grad_fn = jax.grad(sam_loss_wrapper)
            except (ImportError, AttributeError):
                # Fallback: if we can't import MaskedNode, use simple approach
                sam_grad_fn = jax.grad(lambda params, _: loss_fn(params)[0])
            
            updates, new_opt_state = self.tx.update(
                grads, self.opt_state, self.params, grad_fn=sam_grad_fn
            )
        else:
            updates, new_opt_state = self.tx.update(grads, self.opt_state, self.params)

        if self.mask is not None:
            updates = jax.tree_util.tree_map(lambda u, m: u * m, updates, self.mask)
        
        new_params = optax.apply_updates(self.params, updates)

        network = self.replace(
            params=jax.tree_util.tree_map(
                partial(jnp.where, is_fin), new_params, self.params
            ),
            opt_state=jax.tree_util.tree_map(
                partial(jnp.where, is_fin), new_opt_state, self.opt_state
            ),
            update_step=self.update_step + 1,
        )

        return network, info

    # def apply_gradient(self, loss_fn, get_info=True) -> Tuple[Any, "Network"]:
    #     # Ensure params is a plain dict to match optax state (which seems to be using dicts)
    #     params = self.params
    #     if isinstance(params, flax.core.FrozenDict):
    #         params = flax.core.unfreeze(params)

    #     grad_fn = jax.grad(loss_fn, has_aux=True)
    #     grads, info = grad_fn(params)
    #     info["_grads"] = grads
    #     is_fin = True

    #     if self.mask is not None:
    #         grads = jax.tree_util.tree_map(lambda g, m: g * m, grads, self.mask)
    #         info["_grads"] = grads

    #     updates, new_opt_state = self.tx.update(grads, self.opt_state, params)
    #     if self.mask is not None:
    #         updates = jax.tree_util.tree_map(lambda u, m: u * m, updates, self.mask)
    #     new_params = optax.apply_updates(params, updates)

    #     network = self.replace(
    #         params=jax.tree_util.tree_map(
    #             partial(jnp.where, is_fin), new_params, params
    #         ),
    #         opt_state=new_opt_state,
    #         update_step=self.update_step + 1,
    #     )

    #     return network, info
