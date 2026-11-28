"""LinOSS (JAX) versions of the TESS tasks: same batches, losses, metrics and test CSVs.

LinOSS gets (B, L, 2): the flux and, as a second channel, the padding mask, which it uses for
masked mean pooling (see models/linoss.py). Configure it with N=1 and task="regression" so it
returns raw outputs (logits, for classification); the loss is applied here.
"""

import equinox as eqx
import jax.numpy as jnp
import numpy as np
import optax

from models.utils.jax.training import jax_inference
from models.utils.jax.utils import tensor_to_jax
from models.utils.jax.wrapper import JAXLightningModule
from tasks.TESS.classification import ClassificationEval
from tasks.TESS.regression import RegressionEval


def cross_entropy(logits, label):
    """(B, C) logits and (B,) class indices -> mean cross-entropy."""
    return optax.softmax_cross_entropy_with_integer_labels(logits, label).mean()


def mse(pred, frot):
    """(B, 1) predictions and (B,) targets -> mean squared error."""
    return jnp.mean((pred[:, 0] - frot) ** 2)


class _LinOSSTESS(JAXLightningModule):
    """AdamW LinOSS training on (flux, mask) batches; subclasses map outputs to predictions."""

    def __init__(self, model, loss_fn, lr, weight_decay, clip_grad_norm, seed, csv_out):
        super().__init__(
            model=model,
            loss_fn=loss_fn,
            optimizer=optax.adamw(lr, weight_decay=weight_decay),
            clip_grad_norm=clip_grad_norm,
            seed=seed,
        )
        self.csv_out = csv_out

    def _predict(self, out) -> np.ndarray:
        raise NotImplementedError

    def _prepare_batch(self, batch):
        flux, mask, target = batch
        return jnp.stack([flux, mask.astype(flux.dtype)], axis=-1), target

    def _eval_step(self, batch, stage: str):
        x, y = self._prepare_batch(tensor_to_jax(batch))
        out = jax_inference(model=self.jax_model, x=x, state=self.jax_model_state,
                            key=self._batched_keys(self.key, x.shape[0]))
        self.log(f"{stage}/loss", float(self.loss_fn(out, y)), on_step=False, on_epoch=True,
                 prog_bar=True, batch_size=x.shape[0])
        self._collect(np.asarray(y), self._predict(out))

    def validation_step(self, batch, batch_idx):
        self._eval_step(batch, "val")

    def test_step(self, batch, batch_idx):
        self._eval_step(batch, "test")


class LinOSSTESSClassification(ClassificationEval, _LinOSSTESS):
    """LinOSS classifier: (B, L, 2) -> (B, n_classes) logits, cross-entropy loss."""

    def __init__(
        self,
        model: eqx.Module,
        lr: float = 1e-3,
        weight_decay: float = 1e-4,
        clip_grad_norm: float | None = 1.0,
        seed: int = 0,
        csv_out: str = "results/TESS/linoss_classification/test_predictions.csv",
    ):
        super().__init__(model, cross_entropy, lr, weight_decay, clip_grad_norm, seed, csv_out)

    def _predict(self, out) -> np.ndarray:
        return np.asarray(out.argmax(-1))


class LinOSSTESSRegression(RegressionEval, _LinOSSTESS):
    """LinOSS regressor: (B, L, 2) -> (B, 1) frot, MSE loss."""

    def __init__(
        self,
        model: eqx.Module,
        lr: float = 1e-3,
        weight_decay: float = 1e-4,
        clip_grad_norm: float | None = 1.0,
        seed: int = 0,
        csv_out: str = "results/TESS/linoss_regression/test_predictions.csv",
    ):
        super().__init__(model, mse, lr, weight_decay, clip_grad_norm, seed, csv_out)

    def _predict(self, out) -> np.ndarray:
        return np.asarray(out[:, 0])
