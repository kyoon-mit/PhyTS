"""Project 8 energy regression with a Gaussian negative log-likelihood (paper Appendix E).

Batches are (x, y) from Project8DataModule: x (B, L, 2) noisy I/Q, y (B, 1) z-scored energy.
The model returns (B, 2): the mean and a raw variance of the z-scored energy; softplus makes the
variance positive. Validation and test report RMSE in eV over the whole split, and test also
R^2 (Table 2). The test step writes the true energy and the predicted mean and standard
deviation of every test event, in eV, to csv_out, which benchmarks/Project8/score.py reads.
"""

from pathlib import Path

import lightning as L
import numpy as np
import pandas as pd
import torch.nn.functional as F
from torch import nn, optim


class EnergyEval:
    """RMSE (and test R^2) in eV over the full val / test split, and the test-prediction CSV.

    Mixed into the PyTorch and the LinOSS task; their eval steps call _collect with z-scored
    values, which are converted with the DataModule's training-split mean and std.
    """

    csv_out: str

    def _collect(self, true: np.ndarray, mean: np.ndarray, var: np.ndarray) -> None:
        self._rows.append(np.stack([true, mean, var], axis=1))

    def on_validation_epoch_start(self):
        self._rows = []

    def on_test_epoch_start(self):
        self._rows = []

    def _in_ev(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        dm = self.trainer.datamodule
        true, mean, var = np.concatenate(self._rows).T
        return (true * dm.target_std + dm.target_mean, mean * dm.target_std + dm.target_mean,
                np.sqrt(var) * dm.target_std)

    def on_validation_epoch_end(self):
        true, mean, _ = self._in_ev()
        self.log("val/rmse", float(np.sqrt(np.mean((mean - true) ** 2))))

    def on_test_epoch_end(self):
        true, mean, sigma = self._in_ev()
        sq_err = np.sum((mean - true) ** 2)
        self.log("test/rmse", float(np.sqrt(sq_err / len(true))))
        self.log("test/r2", float(1 - sq_err / np.sum((true - true.mean()) ** 2)))
        out = Path(self.csv_out)
        out.parent.mkdir(parents=True, exist_ok=True)
        table = pd.DataFrame({"true_energy_eV": true, "pred_energy_eV_mean": mean,
                              "pred_energy_eV_sigma": sigma})
        table.to_csv(out, index=False)
        print(f"Test predictions -> {out} ({len(true):,} rows)")


class Project8Regression(EnergyEval, L.LightningModule):
    """Gaussian-NLL regressor around model(x (B, L, 2)) -> (B, 2) [mean, raw variance].

    AdamW; the learning rate drops by lr_factor when the validation loss stalls for
    lr_patience epochs.
    """

    def __init__(
        self,
        model: nn.Module,
        lr: float = 1e-3,
        lr_factor: float = 0.25,
        lr_patience: int = 5,
        lr_threshold: float = 1e-3,
        lr_min: float = 1e-6,
        eps: float = 1e-6,
        csv_out: str = "results/Project8/gaussnll_regression/test_predictions.csv",
    ):
        super().__init__()
        self.save_hyperparameters(ignore=["model"])
        self.model = model
        self.csv_out = csv_out

    def forward(self, x):
        return self.model(x)

    def _step(self, batch, stage: str):
        x, y = batch
        out = self(x)
        mean, var = out[:, 0], F.softplus(out[:, 1])
        loss = F.gaussian_nll_loss(mean, y[:, 0], var, eps=self.hparams.eps)
        self.log(f"{stage}/loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        if stage != "train":
            self._collect(y[:, 0].cpu().numpy(), mean.detach().float().cpu().numpy(),
                          var.detach().float().cpu().numpy())
        return loss

    def training_step(self, batch, batch_idx):
        return self._step(batch, "train")

    def validation_step(self, batch, batch_idx):
        self._step(batch, "val")

    def test_step(self, batch, batch_idx):
        self._step(batch, "test")

    def configure_optimizers(self):
        hp = self.hparams
        opt = optim.AdamW(self.parameters(), lr=hp.lr)
        sched = optim.lr_scheduler.ReduceLROnPlateau(
            opt, mode="min", factor=hp.lr_factor, patience=hp.lr_patience,
            threshold=hp.lr_threshold, min_lr=hp.lr_min)
        return {"optimizer": opt,
                "lr_scheduler": {"scheduler": sched, "monitor": "val/loss", "interval": "epoch"}}
