"""LinOSS (JAX) version of Project8Regression: same batches, loss, metrics and test CSV.

Configure LinOSS with N=2 (I/Q), output_dim=2 and task="regression" so it returns the raw
[mean, variance] pair; the Gaussian NLL is applied here.
"""

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import optax

from models.utils.jax.training import jax_inference
from models.utils.jax.utils import tensor_to_jax
from models.utils.jax.wrapper import JAXLightningModule
from tasks.Project8.regression import EnergyEval


def gaussian_nll(out, y, eps: float = 1e-6):
    """(B, 2) [mean, raw variance] and (B,) z-scored targets -> mean Gaussian NLL."""
    var = jax.nn.softplus(out[:, 1]) + eps
    return 0.5 * jnp.mean(jnp.log(var) + (out[:, 0] - y) ** 2 / var)


class LinOSSProject8Regression(EnergyEval, JAXLightningModule):
    """LinOSS Gaussian-NLL regressor: (B, L, 2) -> (B, 2) [mean, raw variance]."""

    def __init__(
        self,
        model: eqx.Module,
        lr: float = 1e-3,
        clip_grad_norm: float | None = 1.0,
        seed: int = 0,
        csv_out: str = "results/Project8/linoss_gaussnll_regression/test_predictions.csv",
    ):
        super().__init__(model=model, loss_fn=gaussian_nll, optimizer=optax.adamw(lr),
                         clip_grad_norm=clip_grad_norm, seed=seed)
        self.csv_out = csv_out

    def _prepare_batch(self, batch):
        x, y = batch
        return x, y[:, 0]

    def _eval_step(self, batch, stage: str):
        x, y = self._prepare_batch(tensor_to_jax(batch))
        out = jax_inference(model=self.jax_model, x=x, state=self.jax_model_state,
                            key=self._batched_keys(self.key, x.shape[0]))
        self.log(f"{stage}/loss", float(gaussian_nll(out, y)), on_step=False, on_epoch=True,
                 prog_bar=True, batch_size=x.shape[0])
        self._collect(np.asarray(y), np.asarray(out[:, 0]),
                      np.asarray(jax.nn.softplus(out[:, 1]) + 1e-6))

    def validation_step(self, batch, batch_idx):
        self._eval_step(batch, "val")

    def test_step(self, batch, batch_idx):
        self._eval_step(batch, "test")
