"""JAX/equinox denoising task for the TIDMAD dark matter dataset.

Mirrors ``tasks.toy.toy_denoising.DenoisingMSE`` but backed by a JAX model
wrapped via ``models.utils.jax.wrapper.JAXLightningModule``.

Batch layout (from ``TIDMADDataset.__getitem__``):
    noisy   (B, L)  float32  channel0001 (noisy input)
    clean   (B, L)  float32  channel0002 (clean target)
    params  (B, 3)  float32  [frequency_hz, amplitude_mV, snr]

Usage (LightningCLI YAML): see ``configs/TIDMAD/train_tidmad_linoss_denoising.yaml``.
"""

from pathlib import Path

import equinox as eqx
import jax.numpy as jnp
import optax
from jaxtyping import Array, PyTree

from models.utils.jax.wrapper import JAXLightningModule


def mse_loss(y_hat: Array, y: Array) -> Array:
    return jnp.mean((y_hat - y) ** 2)


class LinOSSTIDMADDenoising(JAXLightningModule):
    """Seq2seq denoising from noisy (B, L) TIDMAD channel to clean (B, L) channel.

    Expects a LinOSS model with ``task='forecasting'`` and ``output_step=1``
    so it produces a full-resolution (L, output_dim=1) output per sample.
    """

    def __init__(
        self,
        model: eqx.Module,
        lr: float = 1e-3,
        clip_grad_norm: float | None = None,
        load_from_checkpoint: str | Path | None = None,
        seed: int = 0,
    ):
        super().__init__(
            model=model,
            loss_fn=mse_loss,
            optimizer=optax.adamw(lr),
            clip_grad_norm=clip_grad_norm,
            load_from_checkpoint=load_from_checkpoint,
            seed=seed,
        )

    def _prepare_batch(self, batch: PyTree[Array]) -> tuple[Array, Array]:
        noisy, clean, _params = batch
        x = noisy[..., None]  # (B, L, 1)
        y = clean[..., None]  # (B, L, 1) — matches model output shape (forecasting, output_dim=1)
        return x, y
