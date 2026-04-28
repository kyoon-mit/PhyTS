"""
Regression task for the Project 8 simulation dataset with GaussianNLL loss.

Predicts a single normalised regression target (default: ``energy_eV``) from
whatever signal the DataModule supplies.  Set ``freq_transform`` on the
DataModule to choose the input domain:

    freq_transform=None   -> time-series I/Q
    freq_transform='fft'  -> FFT real/imag (z-scored)

The encoder outputs ``[mu, log_var]`` and the loss is
``torch.nn.functional.gaussian_nll_loss`` on the z-scored target.  RMSE is
logged in both z-score space and original units (using the DataModule's
``mu``/``stds``).

Usage (LightningCLI YAML):
    model:
      class_path: tasks.Project8.project8_regression.Project8Regression
      init_args:
        encoder:
          class_path: models.s4d.S4Model
          init_args:
            d_input: 2
            d_output: 2     # [mu, log_var] for one target
            d_model: 128
            n_layers: 6
            dropout: 0.0
            prenorm: false

Batch convention (Project8DataModule):
    X    (B, cutoff, C)        — ts or fft, depending on freq_transform
    var  (B, 1)                — z-scored regression target (single variable)
"""

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
import lightning as L


class Project8Regression(L.LightningModule):
    """Heteroscedastic regression of a single Project 8 variable via GaussianNLL.

    The encoder predicts (mu, log_var); loss is GaussianNLL on z-scored target.
    Per-target RMSE is logged in both z-score space and original units.
    """

    def __init__(
        self,
        encoder: nn.Module,
        learning_rate: float = 1e-3,
        gamma: float = 0.25,
        lr_patience: int = 5,
        threshold: float = 1e-3,
        lr_min: float = 1e-6,
        var_min: float = 1e-6,
    ):
        super().__init__()
        self.save_hyperparameters(ignore=['encoder'])
        self.encoder = encoder
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.lr_patience = lr_patience
        self.threshold = threshold
        self.lr_min = lr_min
        self.var_min = var_min

        # Filled in `setup` from the DataModule.
        self._target_name: Optional[str] = None
        self._target_mu: Optional[float] = None
        self._target_std: Optional[float] = None

    # ─── DataModule wiring ──────────────────────────────────────────────────
    def setup(self, stage: Optional[str] = None):
        dm = self.trainer.datamodule
        variables = list(dm.hparams.variables)
        if len(variables) != 1:
            raise ValueError(
                f"Project8Regression expects exactly one variable in the "
                f"DataModule, got {variables}"
            )
        self._target_name = variables[0]
        if dm.mu is not None and dm.stds is not None:
            self._target_mu = float(dm.mu[0])
            self._target_std = float(dm.stds[0])
        else:
            self._target_mu, self._target_std = 0.0, 1.0

    # ─── forward / step ─────────────────────────────────────────────────────
    def forward(self, x):
        # x: (B, L, C) -> encoder -> (B, 2)
        return self.encoder(x)

    def _step(self, batch):
        x, var = batch
        y = var.squeeze(-1)                                # (B,)
        out = self(x)                                      # (B, 2)
        mu, log_var = out[:, 0], out[:, 1]
        v = log_var.exp().clamp(min=self.var_min)
        loss = F.gaussian_nll_loss(mu, y, v)
        return loss, mu, v, y

    def _log_rmse(self, mu, y, prefix):
        rmse_z = (mu - y).pow(2).mean().sqrt()
        self.log(f'{prefix}/rmse_z/{self._target_name}', rmse_z,
                 on_step=False, on_epoch=True)
        rmse_real = rmse_z * (self._target_std + 1e-8)
        self.log(f'{prefix}/rmse/{self._target_name}', rmse_real,
                 on_step=False, on_epoch=True)

    def training_step(self, batch, batch_idx):
        loss, mu, _, y = self._step(batch)
        self.log('train/loss', loss, on_step=False, on_epoch=True, prog_bar=True)
        self._log_rmse(mu, y, 'train')
        return loss

    def validation_step(self, batch, batch_idx):
        loss, mu, _, y = self._step(batch)
        self.log('val/loss', loss, on_step=False, on_epoch=True, prog_bar=True)
        self._log_rmse(mu, y, 'val')

    def test_step(self, batch, batch_idx):
        loss, mu, v, y = self._step(batch)
        self.log('test/loss', loss, on_step=False, on_epoch=True, prog_bar=True)
        self._log_rmse(mu, y, 'test')
        sigma_real = v.sqrt().mean() * (self._target_std + 1e-8)
        self.log(f'test/sigma_pred/{self._target_name}', sigma_real,
                 on_step=False, on_epoch=True)

    def configure_optimizers(self):
        optimizer = optim.AdamW(self.parameters(), lr=self.learning_rate)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=self.gamma,
            patience=self.lr_patience,
            threshold=self.threshold,
            min_lr=self.lr_min,
        )
        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'monitor': 'val/loss',
                'interval': 'epoch',
            },
        }
