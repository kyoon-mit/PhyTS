"""JAX/equinox heteroscedastic regression task for Project 8 with GaussianNLL.

Mirrors :class:`tasks.Project8.project8_regression.Project8Regression` but is
backed by a JAX/equinox model wrapped via
:class:`models.utils.jax.wrapper.JAXLightningModule`.

The encoder must produce a 2-channel output ``(B, 2)`` interpreted as
``[mean, raw_var]`` for one z-scored regression target.  ``softplus`` enforces
positivity on the variance and the loss is heteroscedastic Gaussian NLL on the
z-scored target.  Per-target RMSE is logged in both z-score space and original
units (using the DataModule's ``mu`` / ``stds``).

Usage (LightningCLI YAML): see
``configs/Project8/train_project8_linoss_regression_energy_gaussiannll.yaml``.
"""

from pathlib import Path
from typing import Optional

import equinox as eqx
import jax
import jax.numpy as jnp
import optax
from jaxtyping import Array, PyTree

from models.utils.jax.training import jax_inference
from models.utils.jax.utils import tensor_to_jax
from models.utils.jax.wrapper import JAXLightningModule


def gaussian_nll_loss(y_hat: Array, y: Array, eps: float = 1e-6) -> Array:
    """Heteroscedastic Gaussian NLL on a single z-scored target.

    ``y_hat``: ``(B, 2)`` ``[mean, raw_var]``; ``y``: ``(B,)`` or ``(B, 1)``.
    """
    if y.ndim > 1 and y.shape[-1] == 1:
        y = y.squeeze(-1)
    mu, raw_var = jnp.split(y_hat, 2, axis=-1)
    mu = mu.squeeze(-1)
    v = jax.nn.softplus(raw_var.squeeze(-1)) + eps
    return 0.5 * jnp.mean(jnp.log(v) + (mu - y) ** 2 / v)


class Project8RegressionJAX(JAXLightningModule):
    """Heteroscedastic Gaussian-NLL regression of a single Project 8 variable.

    Designed for JAX/equinox encoders (e.g. ``models.linoss.LinOSS``) configured
    with ``output_dim=2``.
    """

    _target_name: Optional[str]
    _target_mu: Optional[float]
    _target_std: Optional[float]
    eps: float

    def __init__(
        self,
        model: eqx.Module,
        lr: float = 1e-3,
        clip_grad_norm: float | None = None,
        load_from_checkpoint: str | Path | None = None,
        seed: int = 0,
        eps: float = 1e-6,
    ):
        super().__init__(
            model=model,
            loss_fn=lambda y_hat, y: gaussian_nll_loss(y_hat, y, eps=eps),
            optimizer=optax.adamw(lr),
            clip_grad_norm=clip_grad_norm,
            load_from_checkpoint=load_from_checkpoint,
            seed=seed,
        )
        self.eps = eps
        self._target_name = None
        self._target_mu = None
        self._target_std = None

    def setup(self, stage: Optional[str] = None):
        dm = self.trainer.datamodule
        variables = list(dm.hparams.variables)
        if len(variables) != 1:
            raise ValueError(
                "Project8RegressionJAX expects exactly one variable in the "
                f"DataModule, got {variables}"
            )
        self._target_name = variables[0]
        if dm.mu is not None and dm.stds is not None:
            self._target_mu = float(dm.mu[0])
            self._target_std = float(dm.stds[0])
        else:
            self._target_mu, self._target_std = 0.0, 1.0

    def _prepare_batch(self, batch: PyTree[Array]) -> tuple[Array, Array]:
        x, var = batch
        if var.ndim > 1 and var.shape[-1] == 1:
            var = var.squeeze(-1)
        return x, var

    def _log_rmse(self, mu: Array, y: Array, prefix: str) -> None:
        rmse_z = jnp.sqrt(jnp.mean((mu - y) ** 2))
        self.log(
            f"{prefix}/rmse_z/{self._target_name}",
            float(rmse_z),
            on_step=False, on_epoch=True, sync_dist=True,
        )
        rmse_real = float(rmse_z) * (self._target_std + 1e-8)
        self.log(
            f"{prefix}/rmse/{self._target_name}",
            rmse_real,
            on_step=False, on_epoch=True, sync_dist=True,
        )

    def _eval_step(self, batch, prefix: str) -> None:
        batch = tensor_to_jax(batch)
        x, y = self._prepare_batch(batch)
        batch_size = jax.tree.leaves(x)[0].shape[0]
        keys = self._batched_keys(self.key, batch_size)
        outputs = jax_inference(
            model=self.jax_model,
            x=x,
            state=self.jax_model_state,
            key=keys,
        )
        loss = jax.jit(self.loss_fn)(outputs, y)
        self.log(
            f"{prefix}/loss",
            loss.item(),
            on_step=False, on_epoch=True, prog_bar=True, sync_dist=True,
        )
        mu, _ = jnp.split(outputs, 2, axis=-1)
        mu = mu.squeeze(-1)
        if y.ndim > 1 and y.shape[-1] == 1:
            y = y.squeeze(-1)
        self._log_rmse(mu, y, prefix)

    def validation_step(self, batch, batch_idx):
        self._eval_step(batch, "val")

    def test_step(self, batch, batch_idx):
        self._eval_step(batch, "test")
