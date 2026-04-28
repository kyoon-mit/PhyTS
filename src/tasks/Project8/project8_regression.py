"""
Regression task for the Project 8 simulation dataset with GaussianNLL loss.

Predicts a single normalised regression target (default: ``energy_eV``) from
the FFT-transformed I/Q time series.  The encoder outputs ``[mu, log_var]``
and the loss is ``torch.nn.functional.gaussian_nll_loss`` on the z-scored
target.  RMSE is logged in both z-score space and original units (using the
DataModule's ``mu``/``stds``).

Usage (LightningCLI YAML):
    model:
      class_path: tasks.Project8.project8_regression.Project8RegressionGaussianNLL
      init_args:
        target: energy_eV
        encoder:
          class_path: models.s4d.S4Model
          init_args:
            d_input: 2
            d_output: 2     # [mu, log_var] for one target
            d_model: 128
            n_layers: 6
            dropout: 0.0
            prenorm: false

Batch convention (Project8DataModule with freq_transform='fft'):
    ts   (B, cutoff, C)        — std-normed time series with cav/gauss noise
    fft  (B, cutoff, C)        — z-scored FFT real/imag of (I + jQ)
    var  (B, n_variables)      — z-scored regression targets
"""

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
import lightning as L


class Project8RegressionGaussianNLL(L.LightningModule):
    """Heteroscedastic regression of a single Project 8 variable via GaussianNLL.

    The encoder predicts (mu, log_var); loss is GaussianNLL on z-scored target.
    Per-target RMSE is logged in both z-score space and original units.
    """

    def __init__(
        self,
        encoder: nn.Module,
        target: str = 'energy_eV',
        apply_fft: bool = True,
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
        self.target = target
        self.apply_fft = apply_fft
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.lr_patience = lr_patience
        self.threshold = threshold
        self.lr_min = lr_min
        self.var_min = var_min

        # Filled in `setup` from the DataModule.
        self._target_idx: Optional[int] = None
        self._target_mu: Optional[float] = None
        self._target_std: Optional[float] = None

    # ─── DataModule wiring ──────────────────────────────────────────────────
    def setup(self, stage: Optional[str] = None):
        dm = self.trainer.datamodule
        variables = list(dm.hparams.variables)
        if self.target not in variables:
            raise ValueError(
                f"target {self.target!r} not in datamodule variables {variables}"
            )
        self._target_idx = variables.index(self.target)
        if dm.mu is not None and dm.stds is not None:
            self._target_mu = float(dm.mu[self._target_idx])
            self._target_std = float(dm.stds[self._target_idx])
        else:
            self._target_mu, self._target_std = 0.0, 1.0

    # ─── forward / step ─────────────────────────────────────────────────────
    def _select_input(self, batch):
        # freq_transform='fft' -> (ts, fft, var); else (ts, var)
        if self.apply_fft:
            if len(batch) != 3:
                raise RuntimeError(
                    "apply_fft=True requires a DataModule with freq_transform='fft' "
                    "(batch should be (ts, fft, var))."
                )
            _, x, var = batch
        else:
            ts, var = batch[0], batch[-1]
            x = ts
        return x, var

    def forward(self, x):
        # x: (B, L, C) -> encoder -> (B, 2)
        return self.encoder(x)

    def _step(self, batch):
        x, var = self._select_input(batch)
        y = var[:, self._target_idx]                       # (B,)
        out = self(x)                                      # (B, 2)
        mu, log_var = out[:, 0], out[:, 1]
        v = log_var.exp().clamp(min=self.var_min)
        loss = F.gaussian_nll_loss(mu, y, v)
        return loss, mu, v, y

    def _log_rmse(self, mu, y, prefix):
        rmse_z = (mu - y).pow(2).mean().sqrt()
        self.log(f'{prefix}/rmse_z/{self.target}', rmse_z,
                 on_step=False, on_epoch=True)
        rmse_real = rmse_z * (self._target_std + 1e-8)
        self.log(f'{prefix}/rmse/{self.target}', rmse_real,
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
        # mean predictive sigma in original units, useful as a calibration sanity check
        sigma_real = v.sqrt().mean() * (self._target_std + 1e-8)
        self.log(f'test/sigma_pred/{self.target}', sigma_real,
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
