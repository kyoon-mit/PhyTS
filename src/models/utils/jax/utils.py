"""Utils for JAX."""

import logging

import jax
import jax.numpy as jnp
import numpy as np
import torch
from jaxtyping import PyTree

logger = logging.getLogger(__name__)


def _jax_has_gpu() -> bool:
    """True if this JAX build exposes at least one GPU device."""
    try:
        return any(getattr(d, "platform", None) == "gpu" for d in jax.local_devices())
    except Exception:
        return False


def _torch_tensor_for_dlpack_to_jax(t: torch.Tensor) -> torch.Tensor:
    """Return a view/copy on a device JAX can import from.

    DLPack round-trips tensors on matching logical devices. If PyTorch uses CUDA
    but JAX is CPU-only (typical on clusters with ``jax`` pypi, ``jaxlib`` CPU),
    ``from_dlpack`` fails with *Unknown backend cuda*.  Moving to CPU first
    fixes that; JAX + LinOSS then run on CPU.
    """
    t = t.detach()
    if (t.is_cuda or getattr(t, "is_mps", False)) and not _jax_has_gpu():
        return t.cpu().contiguous()
    if t.is_cuda or getattr(t, "is_mps", False):
        return t.contiguous()
    return t


def _maybe_put_jax_cpu_to_first_gpu(arr: jax.Array) -> jax.Array:
    """If ``arr`` lives on CPU and a JAX GPU exists, copy to that GPU for compute."""
    if not _jax_has_gpu():
        return arr
    devs = arr.devices()
    if not devs:
        return arr
    dev = next(iter(devs))
    if getattr(dev, "platform", None) != "cpu":
        return arr
    gpu = next((d for d in jax.local_devices() if d.platform == "gpu"), None)
    if gpu is None:
        return arr
    return jax.device_put(arr, gpu)


def _tensor_to_jax_numpy_bridge(t: torch.Tensor) -> jax.Array:
    """CPU tensor → JAX array without DLPack (host copy).

    Some PyTorch/JAX/jaxlib combinations reject ``from_dlpack`` even for CPU
    tensors (``DLDeviceType`` mismatch). Numpy is the slow but reliable path.
    """
    t = t.detach().cpu()
    if t.dtype == torch.bfloat16:
        t = t.to(torch.float32)
    if not t.is_contiguous():
        t = t.contiguous()
    return jnp.asarray(np.asarray(t))


def _tensor_to_jax_leaf(t: torch.Tensor) -> jax.Array:
    """DLPack PyTorch → JAX; fall back to CPU DLPack, then host numpy.

    Even when both frameworks see a GPU, ``from_dlpack`` can still fail with
    *unsupported device type* (e.g. stack-specific DLPack/cuda compatibility).
    A further failure mode is JAX rejecting DLPack from CPU tensors; we then
    use :func:`_tensor_to_jax_numpy_bridge`.

    The first ``from_dlpack`` attempt catches any :exc:`TypeError` (not only
    a specific message) so minor JAX/PyTorch version differences still route
    to the same fallbacks.
    """
    t = _torch_tensor_for_dlpack_to_jax(t)
    try:
        arr = jax.dlpack.from_dlpack(t)
    except TypeError as err:
        logger.debug("DLPack from device failed (%s); trying CPU DLPack.", err)
        t_cpu = t.detach().cpu().contiguous()
        try:
            arr = jax.dlpack.from_dlpack(t_cpu)
        except TypeError as err2:
            logger.debug(
                "DLPack from CPU also failed (%s); using numpy bridge (slower).",
                err2,
            )
            arr = _tensor_to_jax_numpy_bridge(t_cpu)
    return _maybe_put_jax_cpu_to_first_gpu(arr)


def tensor_to_jax(tensor: PyTree[torch.Tensor]) -> PyTree[jax.Array]:
    """Convert PyTorch tensor to JAX array using DLpack.

    When PyTorch is on GPU and JAX is CPU-only, tensors are moved to CPU first
    so DLPack does not require a missing CUDA JAX backend.

    If GPU DLPack is still unusable, tensors are copied to CPU and imported
    again (JAX may still run the model on GPU after the array is created).
    """
    return jax.tree.map(
        _tensor_to_jax_leaf,
        tensor,
        is_leaf=lambda t: isinstance(t, torch.Tensor),
    )


def _jax_to_tensor_leaf(a: jax.Array) -> torch.Tensor:
    """JAX array → PyTorch; fall back to host numpy if DLPack is rejected.

    Symmetric to :func:`_tensor_to_jax_leaf`: some builds fail
    ``torch.from_dlpack`` for JAX outputs (GPU or odd dtypes).  Moving through
    host numpy is slower but portable.

    Parameters
    ----------
    a
        A concrete ``jax.Array`` (outputs from :func:`jax_inference` and similar).

    Returns
    -------
    torch.Tensor
        CPU tensor in the default branch; the DLPack success path may follow the
        JAX array's device when PyTorch supports it.
    """
    try:
        return torch.from_dlpack(a)
    except (TypeError, RuntimeError) as err:
        logger.debug("torch.from_dlpack failed (%s); using numpy bridge.", err)
    host = np.asarray(jax.device_get(a))
    return torch.from_numpy(host)


def jax_to_tensor(array: PyTree[jax.Array]) -> PyTree[torch.Tensor]:
    """Convert JAX array(s) to PyTorch tensor(s), preferring DLPack.

    Falls back to ``jax.device_get`` + :func:`torch.from_numpy` when DLPack
    round-trip fails (same class of environment issues as :func:`tensor_to_jax`).
    """
    return jax.tree.map(
        _jax_to_tensor_leaf,
        array,
        is_leaf=lambda a: isinstance(a, jax.Array),
    )
