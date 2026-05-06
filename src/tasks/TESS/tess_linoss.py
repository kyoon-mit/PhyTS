"""LinOSS TESS tasks: regression (frot) and classification (label).

These tasks wrap the JAX/Equinox LinOSS model via JAXLightningModule for
training with PyTorch Lightning. The bridge works as follows:

    PyTorch tensors  ──tensor_to_jax──►  JAX arrays
    LinOSS forward (equinox/jax, JIT-compiled)
    JAX arrays  ──jax_to_tensor──►  PyTorch tensors (for eval/logging)

Batch convention (from TESSRegressionDataset / TESSClassificationDataset):
    flux   (B, L)   — z-score normalized, padded to seq_len
    mask   (B, L)   — True where cadence is valid (unused directly by LinOSS)
    target (B,)     — frot (float32) or label (int64)

LinOSS expects (B, L, N) input; _prepare_batch adds the feature dim and
appends the validity mask as an extra channel so LinOSS can do masked mean
pooling (see LinOSS.__call__ in models/linoss.py).

Usage (LightningCLI YAML):
    python main.py fit --config configs/TESS/other/train_tess_linoss_regression.yaml
"""

import os
from pathlib import Path

import equinox as eqx
import jax
import jax.numpy as jnp
import lightning as L
import optax
from jaxtyping import Array, PyTree

from models.utils.jax.training import jax_inference
from models.utils.jax.utils import jax_to_tensor, tensor_to_jax
from models.utils.jax.wrapper import JAXLightningModule
from tasks.param_count import attach_scalar_hyperparams, jax_equinox_model_hyper_dict
from tasks.TESS.eval_plots import log_validation_plots_to_wandb
from tasks.TESS.classification_metrics import log_extended_metrics


# ── Checkpoint callback ───────────────────────────────────────────────────────

class JAXModelCheckpoint(L.Callback):
    """Save the best JAX model as ``best.eqx`` whenever the monitored metric improves.

    Analogous to ``lightning.pytorch.callbacks.ModelCheckpoint`` but for
    Equinox models, which cannot be saved via PyTorch's ``torch.save``.

    Uses ``on_validation_end`` (not ``on_validation_epoch_end``): in Lightning
    2.x, epoch-reduced validation metrics are committed after
    ``on_validation_epoch_end``, so the built-in checkpoint and early-stop
    callbacks read ``trainer.callback_metrics`` from ``on_validation_end``.

    If the environment variable ``TESS_LINOSS_CKPT_DIR`` is set, it overrides
    ``dirpath``. LightningCLI (jsonargparse) does not accept nested list
    callbacks on the command line (e.g. ``--trainer.callbacks[1]...``), so
    cluster submit scripts can set that variable instead of a CLI override.

    Parameters
    ----------
    dirpath : str
        Directory for ``best.eqx``. Used only when ``TESS_LINOSS_CKPT_DIR`` is
        unset.
    monitor : str, optional
        Logged metric name to track (default ``val/loss``).
    mode : str, optional
        ``"min"`` or ``"max"`` (default ``"min"``).
    """

    def __init__(self, dirpath: str, monitor: str = "val/loss", mode: str = "min"):
        # TESS_LINOSS_CKPT_DIR: see class docstring (Slurm / LightningCLI list override).
        _dir = os.environ.get("TESS_LINOSS_CKPT_DIR", dirpath)
        self.dirpath = Path(_dir)
        self.monitor = monitor
        self.mode = mode
        self.best = float("inf") if mode == "min" else float("-inf")

    def on_validation_end(self, trainer: L.Trainer, pl_module: L.LightningModule):
        if trainer.sanity_checking:
            return
        current = trainer.callback_metrics.get(self.monitor)
        if current is None:
            current = trainer.logged_metrics.get(self.monitor)
        if current is None:
            return
        current = float(current.squeeze().item() if hasattr(current, "squeeze") else current)
        improved = current < self.best if self.mode == "min" else current > self.best
        if improved:
            self.best = current
            self.dirpath.mkdir(parents=True, exist_ok=True)
            path = self.dirpath / "best.eqx"
            eqx.tree_serialise_leaves(str(path), (pl_module.jax_model, pl_module.jax_model_state))
            print(f"Saved best JAX checkpoint → {path}  ({self.monitor}={current:.6f})")


# ── Loss functions ────────────────────────────────────────────────────────────

def _mse_loss(y_hat: Array, y: Array) -> Array:
    """MSE for (B, 1) predictions vs (B,) targets."""
    return jnp.mean((y_hat.squeeze(-1) - y) ** 2)


def _ce_loss(y_hat: Array, y: Array) -> Array:
    """Cross-entropy for (B, C) logits vs (B,) integer labels."""
    # optax handles the log-softmax numerically stably
    per_sample = optax.softmax_cross_entropy_with_integer_labels(y_hat, y)
    return jnp.mean(per_sample)


# ── Regression task ───────────────────────────────────────────────────────────

class TESSLinOSSRegressionMSE(JAXLightningModule):
    """End-to-end LinOSS regression: flux → frot via MSE.

    Configure the LinOSS model with ``task="regression"`` and ``output_dim=1``.
    The training loop (JAX) and metric logging are handled here; the
    ``forward`` method converts outputs back to PyTorch for eval compatibility.

    Example YAML config snippet::

        model:
          class_path: tasks.TESS.tess_linoss.TESSLinOSSRegressionMSE
          init_args:
            lr: 1.0e-3
            clip_grad_norm: 1.0
            model:
              class_path: models.linoss.LinOSS
              init_args:
                num_blocks: 4
                N: 1
                ssm_size: 64
                H: 128
                output_dim: 1
                task: regression
                output_step: 1
                discretization: IM
    """

    def __init__(
        self,
        model: eqx.Module,
        lr: float = 1e-3,
        weight_decay: float = 0.0,
        clip_grad_norm: float | None = 1.0,
        seed: int = 0,
    ):
        super().__init__(
            model=model,
            loss_fn=_mse_loss,
            optimizer=optax.adamw(lr, weight_decay=weight_decay),
            clip_grad_norm=clip_grad_norm,
            seed=seed,
        )
        self.save_hyperparameters(ignore=["model"])
        attach_scalar_hyperparams(self, jax_equinox_model_hyper_dict(self.jax_model, self.jax_model_state))

    def _prepare_batch(self, batch: PyTree[Array]) -> tuple[Array, Array]:
        """Extract (flux, frot) from the TESS batch; add feature dim and append mask channel."""
        flux, mask, frot = batch
        mask_f = mask.astype(jnp.float32)                                     # (B, L)
        x = jnp.concatenate([flux[..., None], mask_f[..., None]], axis=-1)   # (B, L, 2)
        return x, frot  # (B, L, 2), (B,)

    def forward(self, x):
        """Run inference and return a PyTorch tensor (B,) for the eval pipeline.

        x may be (B, L, 1) for mask-unaware callers or (B, L, 2) when the
        mask has been appended by the evaluation loop.
        """
        y_jax = super().forward(x)              # JAX array (B, 1)
        return jax_to_tensor(y_jax).squeeze(-1)  # PyTorch (B,)

    def on_train_epoch_start(self):
        self._train_preds: list = []
        self._train_labels: list = []

    def _training_step_extra_logs(self, model_output: Array, y: Array) -> None:
        import numpy as np
        self._train_preds.append(np.asarray(model_output.squeeze(-1)))
        self._train_labels.append(np.asarray(y))

    def on_train_epoch_end(self):
        import numpy as np
        y_hat = np.concatenate(self._train_preds)
        y = np.concatenate(self._train_labels)
        self.log("train/rmse", float(np.sqrt(np.mean((y_hat - y) ** 2))))

    def on_validation_epoch_start(self):
        self._val_preds: list = []
        self._val_labels: list = []

    def validation_step(self, batch, batch_idx):
        import numpy as np
        batch = tensor_to_jax(batch)
        x, y = self._prepare_batch(batch)
        keys = self._batched_keys(self.key, jax.tree.leaves(x)[0].shape[0])
        outputs = jax_inference(self.jax_model, x, self.jax_model_state, keys)
        loss = jax.jit(self.loss_fn)(outputs, y)
        self.log("val/loss", loss.item(), on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        self._val_preds.append(np.asarray(outputs.squeeze(-1)))
        self._val_labels.append(np.asarray(y))

    def on_validation_epoch_end(self):
        import numpy as np
        y_hat = np.concatenate(self._val_preds)
        y = np.concatenate(self._val_labels)
        ss_res = float(np.sum((y_hat - y) ** 2))
        ss_tot = float(max(np.sum((y - y.mean()) ** 2), 1e-8))
        rmse = float(np.sqrt(np.mean((y_hat - y) ** 2)))
        self.log("val/rmse", rmse)
        self.log("val/r2", 1.0 - ss_res / ss_tot)
        log_validation_plots_to_wandb(
            self,
            kind="regression",
            y_true=y,
            y_hat=y_hat,
        )

    def on_test_epoch_start(self):
        self._test_preds: list = []
        self._test_labels: list = []

    def test_step(self, batch, batch_idx):
        batch_jax = tensor_to_jax(batch)
        x, y = self._prepare_batch(batch_jax)
        keys = self._batched_keys(self.key, jax.tree.leaves(x)[0].shape[0])
        outputs = jax_inference(self.jax_model, x, self.jax_model_state, keys)
        loss = jax.jit(self.loss_fn)(outputs, y)
        self.log("test/loss", loss.item(), on_step=False, on_epoch=True, sync_dist=True)
        self._test_preds.append(jax_to_tensor(outputs.squeeze(-1)).cpu().numpy())
        self._test_labels.append(jax_to_tensor(y).cpu().numpy())

    def on_test_epoch_end(self):
        import numpy as np
        y_hat = np.concatenate(self._test_preds)
        y = np.concatenate(self._test_labels)
        rmse = float(np.sqrt(np.mean((y_hat - y) ** 2)))
        mae = float(np.mean(np.abs(y_hat - y)))
        ss_res = float(np.sum((y_hat - y) ** 2))
        ss_tot = float(max(np.sum((y - y.mean()) ** 2), 1e-8))
        self.log("test/rmse", rmse)
        self.log("test/mae", mae)
        self.log("test/r2", 1.0 - ss_res / ss_tot)


# ── Classification task ───────────────────────────────────────────────────────

class TESSLinOSSClassificationCE(JAXLightningModule):
    """End-to-end LinOSS classification: flux → variability label via cross-entropy.

    Configure the LinOSS model with ``task="regression"`` and
    ``output_dim=num_classes`` to get raw logits; the cross-entropy loss is
    applied inside ``_ce_loss`` (no softmax in the model head).

    Example YAML config snippet::

        model:
          class_path: tasks.TESS.tess_linoss.TESSLinOSSClassificationCE
          init_args:
            num_classes: 8
            lr: 1.0e-3
            clip_grad_norm: 1.0
            model:
              class_path: models.linoss.LinOSS
              init_args:
                num_blocks: 4
                N: 1
                ssm_size: 64
                H: 128
                output_dim: 8       # must equal num_classes
                task: regression    # raw logits; CE applied in the loss fn
                output_step: 1
                discretization: IM
    """

    def __init__(
        self,
        model: eqx.Module,
        num_classes: int,
        lr: float = 1e-3,
        weight_decay: float = 0.0,
        clip_grad_norm: float | None = 1.0,
        seed: int = 0,
    ):
        super().__init__(
            model=model,
            loss_fn=_ce_loss,
            optimizer=optax.adamw(lr, weight_decay=weight_decay),
            clip_grad_norm=clip_grad_norm,
            seed=seed,
        )
        self.num_classes = num_classes
        self.save_hyperparameters(ignore=["model"])
        attach_scalar_hyperparams(self, jax_equinox_model_hyper_dict(self.jax_model, self.jax_model_state))

    def _prepare_batch(self, batch: PyTree[Array]) -> tuple[Array, Array]:
        """Extract (flux, label) from the TESS batch; add feature dim and append mask channel."""
        flux, mask, label = batch
        mask_f = mask.astype(jnp.float32)                                     # (B, L)
        x = jnp.concatenate([flux[..., None], mask_f[..., None]], axis=-1)   # (B, L, 2)
        return x, label  # (B, L, 2), (B,) int64

    def forward(self, x):
        """Run inference and return PyTorch logits (B, num_classes) for eval pipeline.

        x may be (B, L, 1) for mask-unaware callers or (B, L, 2) when the
        mask has been appended by the evaluation loop.
        """
        y_jax = super().forward(x)   # JAX (B, num_classes)
        return jax_to_tensor(y_jax)  # PyTorch (B, num_classes)

    def on_train_epoch_start(self):
        self._train_preds: list = []
        self._train_labels: list = []

    def _training_step_extra_logs(self, model_output: Array, y: Array) -> None:
        import numpy as np
        self._train_preds.append(np.asarray(jnp.argmax(model_output, axis=-1)))
        self._train_labels.append(np.asarray(y))

    def on_train_epoch_end(self):
        import numpy as np
        preds = np.concatenate(self._train_preds)
        labels = np.concatenate(self._train_labels)
        self.log("train/acc", float(np.mean(preds == labels)))

    def on_validation_epoch_start(self):
        self._val_preds: list = []
        self._val_labels: list = []

    def validation_step(self, batch, batch_idx):
        import numpy as np
        batch = tensor_to_jax(batch)
        x, y = self._prepare_batch(batch)
        keys = self._batched_keys(self.key, jax.tree.leaves(x)[0].shape[0])
        outputs = jax_inference(self.jax_model, x, self.jax_model_state, keys)
        loss = jax.jit(self.loss_fn)(outputs, y)
        preds = jnp.argmax(outputs, axis=-1)
        self.log("val/loss", loss.item(), on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        self._val_preds.append(np.asarray(preds))
        self._val_labels.append(np.asarray(y))

    def on_validation_epoch_end(self):
        import numpy as np
        preds  = np.concatenate(self._val_preds)
        labels = np.concatenate(self._val_labels)
        self.log("val/acc", float(np.mean(preds == labels)))
        per_class = [
            float(np.mean(preds[labels == c] == c))
            for c in range(self.num_classes) if np.any(labels == c)
        ]
        if per_class:
            self.log("val/balanced_acc", sum(per_class) / len(per_class))
        log_extended_metrics(self, preds, labels, num_classes=self.num_classes, prefix="val")
        log_validation_plots_to_wandb(
            self,
            kind="classification",
            y_true=labels,
            y_hat=preds,
        )

    def on_test_epoch_start(self):
        self._test_preds: list = []
        self._test_labels: list = []

    def test_step(self, batch, batch_idx):
        batch_jax = tensor_to_jax(batch)
        x, y = self._prepare_batch(batch_jax)
        keys = self._batched_keys(self.key, jax.tree.leaves(x)[0].shape[0])
        outputs = jax_inference(self.jax_model, x, self.jax_model_state, keys)
        loss = jax.jit(self.loss_fn)(outputs, y)
        preds = jnp.argmax(outputs, axis=-1)
        self.log("test/loss", loss.item(), on_step=False, on_epoch=True, sync_dist=True)
        self._test_preds.append(jax_to_tensor(preds).cpu().numpy())
        self._test_labels.append(jax_to_tensor(y).cpu().numpy())

    def on_test_epoch_end(self):
        import numpy as np
        preds = np.concatenate(self._test_preds)
        labels = np.concatenate(self._test_labels)
        self.log("test/acc", float(np.mean(preds == labels)))
        per_class = [
            float(np.mean(preds[labels == c] == c))
            for c in range(self.num_classes) if np.any(labels == c)
        ]
        if per_class:
            self.log("test/balanced_acc", sum(per_class) / len(per_class))
        log_extended_metrics(self, preds, labels, num_classes=self.num_classes, prefix="test")
        for c in range(self.num_classes):
            mask_c = labels == c
            if mask_c.sum() > 0:
                self.log(f"test/acc_class_{c}", float(np.mean(preds[mask_c] == c)))
