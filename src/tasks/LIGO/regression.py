"""PyTorch regression tasks for LIGO gravitational-wave strain data.

Batch layout (from LIGODataModule):
    X  (B, n_ifos, L)   noisy strain — H1 index 0, L1 index 1
    y  (B, n_targets)   physical parameters (e.g. chirp_mass)
    z  (B, n_obs)       auxiliary scalars   (e.g. snr)

All tasks transpose X → (B, L, n_ifos) before the model, so every model
receives time as the leading sequence dimension.

Test step saves a CSV with columns:
    <observed_name>, true_<target>, pred_<target>              (MSE)
    <observed_name>, true_<target>, pred_<target>_mean,
                     pred_<target>_logvar                       (GaussNLL)
"""

from pathlib import Path
from typing import Sequence

import lightning as L
import torch
import torch.nn as nn
import torch.nn.functional as F
from jaxtyping import Float
from torch import Tensor, optim

Batch = tuple[
    Float[Tensor, "B n_ifos L"],
    Float[Tensor, "B n_targets"],
    Float[Tensor, "B n_obs"],
]

_DATA_ROOT = (
    "/n/holystore01/LABS/iaifi_lab/Lab/kyoon/DATA"
    "/ai4gw@cern/bns_snr_5_50_powerlaw_256Hz_80K_10K_100K"
)


class LIGORegressionMSE(L.LightningModule):
    """MSE regression from 2-channel LIGO strain to physical parameters.

    Compatible with any model that accepts (B, L, n_ifos) -> (B, n_targets).
    """

    def __init__(
        self,
        model: nn.Module,
        target_names: Sequence[str] = ("chirp_mass",),
        observed_names: Sequence[str] = ("snr",),
        lr: float = 1e-3,
        lr_decay: float = 0.99,
        csv_out: str = "results/LIGO/s4d_mse_regression/test_predictions.csv",
    ):
        super().__init__()
        self.save_hyperparameters(ignore=["model"])
        self.model = model
        self.target_names = list(target_names)
        self.observed_names = list(observed_names)
        self.lr = lr
        self.lr_decay = lr_decay
        self.csv_out = csv_out
        self.criterion = nn.MSELoss()

    def forward(self, x: Float[Tensor, "B n_ifos L"]) -> Float[Tensor, "B n_targets"]:
        # (B, n_ifos, L) -> (B, L, n_ifos) -> model -> (B, n_targets)
        return self.model(x.transpose(1, 2))

    def _step(self, batch: Batch):
        X, y, z = batch
        y_hat = self(X)
        loss = self.criterion(y_hat, y)
        return loss, y_hat, y, z

    def training_step(self, batch: Batch, batch_idx: int):
        loss, _, _, _ = self._step(batch)
        self.log("train/loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch: Batch, batch_idx: int):
        loss, y_hat, y, _ = self._step(batch)
        self.log("val/loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        for i, name in enumerate(self.target_names):
            rmse = torch.sqrt(F.mse_loss(y_hat[:, i], y[:, i]))
            self.log(f"val/rmse_{name}", rmse, on_step=False, on_epoch=True)

    def on_test_epoch_start(self):
        self._preds: list[Tensor] = []
        self._targets: list[Tensor] = []
        self._observed: list[Tensor] = []

    def test_step(self, batch: Batch, batch_idx: int):
        X, y, z = batch
        y_hat = self(X)
        self._preds.append(y_hat.cpu())
        self._targets.append(y.cpu())
        self._observed.append(z.cpu())

    def on_test_epoch_end(self):
        import pandas as pd

        preds = torch.cat(self._preds, 0).numpy()
        targets = torch.cat(self._targets, 0).numpy()
        obs = torch.cat(self._observed, 0).numpy()

        rows: dict = {}
        for i, name in enumerate(self.observed_names):
            rows[name] = obs[:, i]
        for i, name in enumerate(self.target_names):
            rows[f"true_{name}"] = targets[:, i]
            rows[f"pred_{name}"] = preds[:, i]

        out = Path(self.csv_out)
        out.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(rows).to_csv(out, index=False)
        print(f"Test predictions -> {out}  ({len(rows[next(iter(rows))]):,} rows)")

    def configure_optimizers(self):
        optimizer = optim.AdamW(self.parameters(), lr=self.lr)
        scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=self.lr_decay)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "interval": "epoch"},
        }


class LIGORegressionGaussNLL(L.LightningModule):
    """Gaussian NLL regression from 2-channel LIGO strain to physical parameters.

    Model must output (B, 2*n_targets): first half = means, second = log-variances.
    Compatible with S4Model(d_output=2*n_targets) or similar.
    """

    def __init__(
        self,
        model: nn.Module,
        target_names: Sequence[str] = ("chirp_mass",),
        observed_names: Sequence[str] = ("snr",),
        lr: float = 1e-3,
        lr_decay: float = 0.99,
        csv_out: str = "results/LIGO/s4d_gaussnll_regression/test_predictions.csv",
    ):
        super().__init__()
        self.save_hyperparameters(ignore=["model"])
        self.model = model
        self.target_names = list(target_names)
        self.observed_names = list(observed_names)
        self.lr = lr
        self.lr_decay = lr_decay
        self.csv_out = csv_out

    def forward(self, x: Float[Tensor, "B n_ifos L"]) -> Float[Tensor, "B two_n_targets"]:
        return self.model(x.transpose(1, 2))

    @staticmethod
    def _split_output(y_hat: Tensor, n: int) -> tuple[Tensor, Tensor]:
        mu = y_hat[:, :n]
        # Soft clamp via tanh: gradient flows at all values, no dead-gradient zone.
        # Range (-5, 5) → sigma ∈ (0.007, 12), appropriate for chirp_mass ∈ [0.87, 2.17].
        lv = 5.0 * torch.tanh(y_hat[:, n:] / 5.0)
        return mu, lv

    @staticmethod
    def _gaussnll(mu: Tensor, lv: Tensor, y: Tensor) -> Tensor:
        return 0.5 * (lv + (y - mu) ** 2 / torch.exp(lv)).mean()

    @staticmethod
    def _beta_nll(mu: Tensor, lv: Tensor, y: Tensor, beta: float = 0.5) -> Tensor:
        """Beta-NLL loss (Seitzer et al. 2022).

        Stops gradient through the variance weight to prevent variance collapse:
        overconfident predictions (small var) get down-weighted, so the optimizer
        can't drive lv to -inf by memorising training targets.
        """
        var = torch.exp(lv)
        mse = (y - mu) ** 2
        return 0.5 * (var.detach() ** beta * (mse / var + lv)).mean()

    def training_step(self, batch: Batch, batch_idx: int):
        X, y, _ = batch
        y_hat = self(X)
        mu, lv = self._split_output(y_hat, y.shape[-1])
        loss = self._gaussnll(mu, lv, y)
        self.log("train/loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch: Batch, batch_idx: int):
        X, y, _ = batch
        y_hat = self(X)
        mu, lv = self._split_output(y_hat, y.shape[-1])
        loss = self._gaussnll(mu, lv, y)
        self.log("val/loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        for i, name in enumerate(self.target_names):
            rmse = torch.sqrt(F.mse_loss(mu[:, i], y[:, i]))
            self.log(f"val/rmse_{name}", rmse, on_step=False, on_epoch=True)

    def on_test_epoch_start(self):
        self._mus: list[Tensor] = []
        self._lvs: list[Tensor] = []
        self._targets: list[Tensor] = []
        self._observed: list[Tensor] = []

    def test_step(self, batch: Batch, batch_idx: int):
        X, y, z = batch
        y_hat = self(X)
        mu, lv = self._split_output(y_hat, y.shape[-1])
        self._mus.append(mu.cpu())
        self._lvs.append(lv.cpu())
        self._targets.append(y.cpu())
        self._observed.append(z.cpu())

    def on_test_epoch_end(self):
        import pandas as pd

        mus = torch.cat(self._mus, 0).numpy()
        lvs = torch.cat(self._lvs, 0).numpy()
        targets = torch.cat(self._targets, 0).numpy()
        obs = torch.cat(self._observed, 0).numpy()

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

    def configure_optimizers(self):
        optimizer = optim.AdamW(self.parameters(), lr=self.lr)
        scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=self.lr_decay)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "interval": "epoch"},
        }
