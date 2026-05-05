"""JAX/equinox regression task for LIGO gravitational wave strain data.

Batch layout (from LIGODataset.__getitem__):
    X  (B, n_ifos, L)  injected strain (H1 + L1)  -> model sees (B, L, n_ifos)
    y  (B, n_targets)  target variables (e.g. chirp_mass)
    z  (B, n_obs)      observed variables (e.g. snr) -- unused

Usage (LightningCLI YAML): see configs/LIGO/train_ligo_linoss_regression.yaml.
"""

from pathlib import Path

import equinox as eqx
import jax.numpy as jnp
import optax
from jaxtyping import Array, PyTree

from models.utils.jax.wrapper import JAXLightningModule


def mse_loss(y_hat: Array, y: Array) -> Array:
    return jnp.mean((y_hat - y) ** 2)


class LinOSSLIGORegression(JAXLightningModule):
    """Pooled regression from (B, n_ifos, L) LIGO strain to (B, n_targets) predictions.

    Expects LinOSS with task='regression' so it mean-pools the sequence and outputs
    (B, output_dim). Set output_dim = len(target_variables) in the config.
    """

    def __init__(
        self,
        model: eqx.Module,
        lr: float = 1e-3,
        clip_grad_norm: float | None = None,
        load_from_checkpoint: str | Path | None = None,
        seed: int = 0,
    ):
        learning_rate_schedule = optax.warmup_cosine_decay_schedule(
            init_value=0.0,
            peak_value=float(lr),
            warmup_steps=1000,
            decay_steps=100000,
            end_value=1e-6,
        )
        optimizer = optax.adamw(learning_rate_schedule)
        super().__init__(
            model=model,
            loss_fn=mse_loss,
            optimizer=optimizer,
            clip_grad_norm=clip_grad_norm,
            load_from_checkpoint=load_from_checkpoint,
            seed=seed,
        )

    def _prepare_batch(self, batch: PyTree[Array]) -> tuple[Array, Array]:
        X, y, _z = batch
        x = jnp.transpose(X, (0, 2, 1))  # (B, n_ifos, L) -> (B, L, n_ifos)
        return x, y
