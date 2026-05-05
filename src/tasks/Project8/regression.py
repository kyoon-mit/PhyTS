"""
Regression task for the Project 8 CRES dataset.

Targets are scalar physics labels (energy_eV by default).  The model receives a
multivariate (B, L, C=2) trace [I, Q] (already z-scored per channel by the
dataloader) and predicts the z-scored target(s); we additionally log RMSE in
physical units so the headline number is interpretable (eV for energy).

Any model with signature  forward(x: (B, L, C)) -> (B, n_targets)  plugs in.

Usage (LightningCLI YAML):
    model:
      class_path: tasks.Project8.regression.RegressionMSE
      init_args:
        lr: 1.0e-3
        model:
          class_path: models.s4d.S4Model
          init_args:
            d_input: 2
            d_output: 1
            d_model: 128
            d_state: 64
            n_layers: 4

Batch convention (Project8DataModule):
    X    (B, L, C)         z-scored I/Q (or FFT) trace
    var  (B, n_variables)  z-scored physics target(s)
DataModule attaches `mu`, `stds` (numpy arrays of length n_variables) used here
to back out RMSE in physical units.
"""

import numpy as np
import torch
import torch.nn as nn
from torch import optim
import lightning as L


class RegressionMSE(L.LightningModule):
    """Multi-channel time-series regression with MSE loss.

    Logs train/val/test RMSE per target in BOTH normalised (z-score) units and
    physical units.  Physical units require `mu` / `stds` to be set on the
    LightningModule before training (the DataModule's `setup()` populates these
    on `dm.mu, dm.stds`; we capture them in `on_fit_start` / `on_test_start`).
    """

    def __init__(
        self,
        model: nn.Module,
        target_params: list[str] = ['energy_eV'],
        lr: float = 1e-3,
        lr_decay: float = 0.99,
    ):
        super().__init__()
        self.save_hyperparameters(ignore=['model'])
        self.model         = model
        self.lr            = lr
        self.lr_decay      = lr_decay
        self.target_params = target_params
        self.criterion     = nn.MSELoss()
        # Physical-unit norm stats — populated from datamodule at fit/test time
        self.register_buffer('y_mu',  torch.zeros(len(target_params)),  persistent=False)
        self.register_buffer('y_std', torch.ones(len(target_params)),   persistent=False)

    # ── pull norm stats off the attached datamodule ───────────────────────
    def _attach_norm_stats(self):
        dm = self.trainer.datamodule
        mu  = np.asarray(getattr(dm, 'mu',   np.zeros(len(self.target_params))), dtype=np.float32)
        std = np.asarray(getattr(dm, 'stds', np.ones(len(self.target_params))),  dtype=np.float32)
        self.y_mu  = torch.from_numpy(mu).to(self.device)
        self.y_std = torch.from_numpy(std).to(self.device)

    def on_fit_start(self):    self._attach_norm_stats()
    def on_test_start(self):   self._attach_norm_stats()

    def forward(self, x):
        # x: (B, L, C) → model → (B, n_targets)
        return self.model(x)

    def _step(self, batch):
        X, y = batch                                            # (B, L, C), (B, n_targets)
        y_hat = self(X)
        loss  = self.criterion(y_hat, y)
        return loss, y_hat, y

    def _log_rmse(self, y_hat, y, prefix):
        diff_norm = y_hat - y                                   # z-score residual
        diff_phys = diff_norm * self.y_std                      # un-normalise
        for i, name in enumerate(self.target_params):
            self.log(f'{prefix}/rmse_norm/{name}',
                     diff_norm[:, i].pow(2).mean().sqrt(),
                     on_step=False, on_epoch=True)
            self.log(f'{prefix}/rmse_phys/{name}',
                     diff_phys[:, i].pow(2).mean().sqrt(),
                     on_step=False, on_epoch=True)

    def training_step(self, batch, _):
        loss, y_hat, y = self._step(batch)
        self.log('train/loss', loss, on_step=False, on_epoch=True, prog_bar=True)
        self._log_rmse(y_hat, y, 'train')
        return loss

    def validation_step(self, batch, _):
        loss, y_hat, y = self._step(batch)
        self.log('val/loss', loss, on_step=False, on_epoch=True, prog_bar=True)
        self._log_rmse(y_hat, y, 'val')

    def test_step(self, batch, _):
        loss, y_hat, y = self._step(batch)
        self.log('test/loss', loss, on_step=False, on_epoch=True, prog_bar=True)
        self._log_rmse(y_hat, y, 'test')

    def configure_optimizers(self):
        opt   = optim.AdamW(self.parameters(), lr=self.lr)
        sched = optim.lr_scheduler.ExponentialLR(opt, gamma=self.lr_decay)
        return {'optimizer': opt, 'lr_scheduler': {'scheduler': sched, 'interval': 'epoch'}}
