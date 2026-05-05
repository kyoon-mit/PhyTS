"""JAX/equinox denoising task for LIGO gravitational wave strain data.

Seq2seq: noisy injected strain (H1 + L1) -> clean waveform (H1 + L1).

Batch layout (from LIGODataset with clean_data_key='whitened_signal'):
    X  (B, n_ifos, L)  whitened_injected (signal + noise) -> model sees (B, L, n_ifos)
    y  (B, n_ifos, L)  whitened_signal (signal only)      -> target (B, L, n_ifos)
    z  (B, n_obs)      observed variables (e.g. snr)      -- unused

Usage (LightningCLI YAML): see configs/LIGO/train_ligo_linoss_denoising.yaml.
"""

from pathlib import Path

import equinox as eqx
import jax.numpy as jnp
import optax
from jaxtyping import Array, PyTree

from models.utils.jax.wrapper import JAXLightningModule


def mse_loss(y_hat: Array, y: Array) -> Array:
    return jnp.mean((y_hat - y) ** 2)


class LinOSSLIGODenoising(JAXLightningModule):
    """Seq2seq denoising from noisy (B, n_ifos, L) LIGO strain to clean (B, n_ifos, L).

    Expects LinOSS with task='forecasting' and output_step=1 so it produces a
    full-resolution (L, output_dim) output per sample. Set N=n_ifos and output_dim=n_ifos.
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
        X, y, _z = batch
        x = jnp.transpose(X, (0, 2, 1))  # (B, n_ifos, L) -> (B, L, n_ifos)
        y = jnp.transpose(y, (0, 2, 1))  # (B, n_ifos, L) -> (B, L, n_ifos)
        return x, y
