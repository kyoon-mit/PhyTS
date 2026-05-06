"""JAX/LinOSS regression tasks for LIGO gravitational-wave strain data.

Mirrors ``tasks.toy.toy_regression_jax.LinOSSToyRegression`` but targets LIGO
physical parameters (e.g. ``chirp_mass``) and uses a Gaussian NLL loss.

Batch layout (from ``LIGODataModule.__getitem__``)::

    X_injected  (B, n_ifos, L)   noisy strain      shape e.g. (B, 2, 1024)
    y_targets   (B, n_targets)   physical params   e.g. chirp_mass
    z_observed  (B, n_obs)       auxiliary scalars  e.g. snr

The task transposes X to ``(B, L, n_ifos)`` before passing to the LinOSS model
(which is vmapped over the batch, so each sample is ``(L, n_ifos)``).

The model output is ``(2 * n_targets,)``: the first half are predicted means,
the second half are predicted log-variances.  Gaussian NLL loss is minimised.

Test step writes a CSV with columns:
    <observed_name>, true_<target>, pred_<target>_mean, pred_<target>_logvar

Usage (LightningCLI YAML): see ``configs/LIGO/train_ligo_linoss_gaussnll_regression.yaml``.
"""

from pathlib import Path
from typing import Sequence

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import optax
from jaxtyping import Array, PyTree

from models.utils.jax.training import jax_inference
from models.utils.jax.utils import tensor_to_jax
from models.utils.jax.wrapper import JAXLightningModule


def _soft_clamp_lv(lv_raw: Array) -> Array:
    """Soft clamp log-variance to (-5, 5) via tanh; gradient flows at all values."""
    return 5.0 * jnp.tanh(lv_raw / 5.0)


def gaussian_nll_loss(y_hat: Array, y: Array) -> Array:
    """Gaussian NLL averaged over batch and targets (used for validation/test)."""
    n = y.shape[-1]
    mu = y_hat[:, :n]
    lv = _soft_clamp_lv(y_hat[:, n:])
    return jnp.mean(0.5 * (lv + (y - mu) ** 2 / jnp.exp(lv)))


def beta_nll_loss(y_hat: Array, y: Array, beta: float = 0.5) -> Array:
    """Beta-NLL loss (Seitzer et al. 2022) used for training.

    Stops gradient through the variance weight to prevent variance collapse.
    """
    n = y.shape[-1]
    mu = y_hat[:, :n]
    lv = _soft_clamp_lv(y_hat[:, n:])
    var = jnp.exp(lv)
    mse = (y - mu) ** 2
    weight = jax.lax.stop_gradient(var ** beta)
    return jnp.mean(0.5 * (weight * (mse / var + lv)))


class LinOSSLIGOGaussNLL(JAXLightningModule):
    """LinOSS regression from 2-channel LIGO strain to physical parameters.

    Minimises Gaussian NLL; predicts per-parameter mean + log-variance.
    Test step writes a CSV: observed vars, true targets, predicted mean + logvar.
    """

    n_targets: int

    def __init__(
        self,
        model: eqx.Module,
        n_targets: int = 1,
        lr: float = 1e-3,
        clip_grad_norm: float | None = 1.0,
        load_from_checkpoint: str | Path | None = None,
        seed: int = 0,
        target_names: Sequence[str] = ("chirp_mass",),
        observed_names: Sequence[str] = ("snr",),
        csv_out: str = "results/LIGO/linoss_gaussnll_regression/test_predictions.csv",
    ):
        super().__init__(
            model=model,
            loss_fn=gaussian_nll_loss,
            optimizer=optax.adamw(lr),
            clip_grad_norm=clip_grad_norm,
            load_from_checkpoint=load_from_checkpoint,
            seed=seed,
        )
        self.n_targets = n_targets
        self.target_names = list(target_names)
        self.observed_names = list(observed_names)
        self.csv_out = csv_out

    def _prepare_batch(self, batch: PyTree[Array]) -> tuple[Array, Array]:
        """Transpose X to (B, L, n_ifos) and select target parameters."""
        X, y_targets, _z = batch
        # X: (B, n_ifos, L) -> (B, L, n_ifos)  [jax arrays after tensor_to_jax]
        x = jnp.swapaxes(X, 1, 2)
        return x, y_targets

    # ── test step: collect predictions and write CSV ───────────────────────────

    def on_test_epoch_start(self):
        self._mus: list[np.ndarray] = []
        self._lvs: list[np.ndarray] = []
        self._targets: list[np.ndarray] = []
        self._observed: list[np.ndarray] = []

    def test_step(self, batch, batch_idx: int):
        # Keep z_observed from the original torch batch before jax conversion
        z_np = batch[2].cpu().numpy()  # (B, n_obs)

        batch_jax = tensor_to_jax(batch)
        x, y_targets = self._prepare_batch(batch_jax)
        batch_size = jax.tree.leaves(x)[0].shape[0]
        keys = self._batched_keys(self.key, batch_size)

        y_hat = jax_inference(
            model=self.jax_model,
            x=x,
            state=self.jax_model_state,
            key=keys,
        )
        loss = jax.jit(self.loss_fn)(y_hat, y_targets)
        self.log("test/loss", loss.item(), on_step=False, on_epoch=True, prog_bar=True)

        n = self.n_targets
        mu = np.array(y_hat[:, :n])
        lv = np.tanh(np.array(y_hat[:, n:]) / 5.0) * 5.0
        self._mus.append(mu)
        self._lvs.append(lv)
        self._targets.append(np.array(y_targets))
        self._observed.append(z_np)

    def on_test_epoch_end(self):
        import pandas as pd

        mus = np.concatenate(self._mus, axis=0)
        lvs = np.concatenate(self._lvs, axis=0)
        targets = np.concatenate(self._targets, axis=0)
        obs = np.concatenate(self._observed, axis=0)

        rows: dict = {}
        for i, name in enumerate(self.observed_names):
            rows[name] = obs[:, i]
        for i, name in enumerate(self.target_names):
            rows[f"true_{name}"] = targets[:, i]
            rows[f"pred_{name}_mean"] = mus[:, i]
            rows[f"pred_{name}_logvar"] = lvs[:, i]

        out = Path(self.csv_out)
        out.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(rows).to_csv(out, index=False)
        print(f"Test predictions -> {out}  ({len(rows[next(iter(rows))]):,} rows)")
