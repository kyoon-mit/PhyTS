"""JAX/equinox regression task for the toy sinusoidal + white noise dataset.

Mirrors ``tasks.toy.toy_regression.RegressionMSE`` but is backed by a JAX model
wrapped via ``models.utils.jax.wrapper.JAXLightningModule``.

Batch layout (from ``ToyDataset.__getitem__``):
    sig_bkg  (B, L)   noisy input  -> model sees (B, L, 1)
    sig      (B, L)   clean signal (unused)
    params   (B, 5)   [amplitude, frequency_hz, phase_rad, noise_amplitude, snr]

Usage (LightningCLI YAML): see ``configs/toy/train_toy_linoss_regression_raw.yaml``.
"""

from pathlib import Path

import equinox as eqx
import jax.numpy as jnp
import optax
from jaxtyping import Array, PyTree

from dataloader.toy_dataloader import Param
from models.utils.jax.wrapper import JAXLightningModule


def mse_loss(y_hat: Array, y: Array) -> Array:
    """Mean-squared error over all elements."""
    return jnp.mean((y_hat - y) ** 2)


class LinOSSToyRegression(JAXLightningModule):
    """Pooled regression from a (B, L) noisy signal to len(target_params) targets."""

    target_idx: tuple[int, ...]

    def __init__(
        self,
        model: eqx.Module,
        target_params: list[str] = ["amplitude", "frequency_hz", "phase_rad"],
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
        self.target_idx = tuple(int(Param[p]) for p in target_params)

    def _prepare_batch(self, batch: PyTree[Array]) -> tuple[Array, Array]:
        """Select noisy signal (+ feature dim) as x and target params as y."""
        sig_bkg, _sig, params = batch
        x = sig_bkg[..., None]  # (B, L, 1)
        y = params[:, jnp.asarray(self.target_idx)]
        return x, y
