"""Utils for JAX."""

import jax
import torch
from jaxtyping import PyTree


def tensor_to_jax(tensor: PyTree[torch.Tensor]) -> PyTree[jax.Array]:
    """Convert PyTorch tensor to JAX array using DLpack."""
    return jax.tree.map(
        lambda t: jax.dlpack.from_dlpack(t),
        tensor,
        is_leaf=lambda t: isinstance(t, torch.Tensor),
    )


def jax_to_tensor(array: PyTree[jax.Array]) -> PyTree[torch.Tensor]:
    """Convert JAX array to PyTorch tensor using DLpack."""
    return jax.tree.map(
        lambda a: torch.from_dlpack(a),
        array,
        is_leaf=lambda a: isinstance(a, jax.Array),
    )
