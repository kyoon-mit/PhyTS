"""TESS rotation regression: light curve -> near-core rotation frequency frot (1/day), MSE loss.

Batches are (flux, mask, frot) from TESSRegressionDataModule. The model gets the flux as
(B, L, 1) and the padding mask, and returns (B, 1). Validation and test report R^2, RMSE and MAE
over the whole split (R^2 is the paper's Table 2 metric). The test step also writes the true and
predicted frot of every test light curve to csv_out, which benchmarks/TESS/score.py reads.
"""

from pathlib import Path

import lightning as L
import numpy as np
import pandas as pd
import torch.nn.functional as F
from torch import nn, optim


def r2_score(true: np.ndarray, pred: np.ndarray) -> float:
    """Coefficient of determination on this split (predicting its mean scores 0)."""
    return float(1 - np.sum((pred - true) ** 2) / np.sum((true - true.mean()) ** 2))


class RegressionEval:
    """R^2, RMSE and MAE over the full val / test split, and the test-prediction CSV.

    Mixed into the PyTorch and the LinOSS task; their eval steps call _collect.
    """

    csv_out: str

    def _collect(self, true: np.ndarray, pred: np.ndarray) -> None:
        self._true.append(true)
        self._pred.append(pred)

    def _reset(self) -> None:
        self._true, self._pred = [], []

    def _log_metrics(self, stage: str) -> tuple[np.ndarray, np.ndarray]:
        true, pred = np.concatenate(self._true), np.concatenate(self._pred)
        err = pred - true
        self.log(f"{stage}/r2", r2_score(true, pred))
        self.log(f"{stage}/rmse", float(np.sqrt(np.mean(err**2))))
        self.log(f"{stage}/mae", float(np.mean(np.abs(err))))
        return true, pred

    def on_validation_epoch_start(self):
        self._reset()

    def on_test_epoch_start(self):
        self._reset()

    def on_validation_epoch_end(self):
        self._log_metrics("val")

    def on_test_epoch_end(self):
        true, pred = self._log_metrics("test")
        out = Path(self.csv_out)
        out.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame({"true_frot": true, "pred_frot": pred}).to_csv(out, index=False)
        print(f"Test predictions -> {out} ({len(true):,} rows)")


class TESSRegression(RegressionEval, L.LightningModule):
    """MSE regressor around model(x (B, L, 1), mask (B, L)) -> (B, 1)."""

    def __init__(
        self,
        model: nn.Module,
        lr: float = 1e-3,
        lr_decay: float = 0.99,
        weight_decay: float = 1e-4,
        csv_out: str = "results/TESS/regression/test_predictions.csv",
    ):
        super().__init__()
        self.save_hyperparameters(ignore=["model"])
        self.model = model
        self.csv_out = csv_out

    def forward(self, flux, mask):
        return self.model(flux.unsqueeze(-1), mask=mask).squeeze(-1)

    def _step(self, batch, stage: str):
        flux, mask, frot = batch
        pred = self(flux, mask)
        loss = F.mse_loss(pred, frot)
        self.log(f"{stage}/loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        if stage != "train":
            self._collect(frot.cpu().numpy(), pred.float().cpu().numpy())
        return loss

    def training_step(self, batch, batch_idx):
        return self._step(batch, "train")

    def validation_step(self, batch, batch_idx):
        self._step(batch, "val")

    def test_step(self, batch, batch_idx):
        self._step(batch, "test")

    def configure_optimizers(self):
        opt = optim.AdamW(self.parameters(), lr=self.hparams.lr,
                          weight_decay=self.hparams.weight_decay)
        sched = optim.lr_scheduler.ExponentialLR(opt, gamma=self.hparams.lr_decay)
        return {"optimizer": opt, "lr_scheduler": {"scheduler": sched, "interval": "epoch"}}
