"""
Regression task for the sinusoidal + white noise toy dataset.

The model receives a signal (raw noisy OR pre-denoised) and predicts
physical signal parameters: amplitude, frequency_hz, phase_rad.

Any model with signature  forward(x: (B,L,1)) -> (B,n_targets)
can be plugged in via the `model` argument (e.g. S4Model with mean pooling).

Usage (LightningCLI YAML):
    model:
      class_path: tasks.toy_regression.RegressionMSE
      init_args:
        target_params:
          - amplitude
          - frequency_hz
          - phase_rad
        lr: 1.0e-3
        lr_decay: 0.99
        model:
          class_path: models.s4d.S4Model
          init_args:
            d_input: 1
            d_output: 3    # must match len(target_params)
            d_model: 64
            d_state: 16
            n_layers: 2
            dropout: 0.1

Batch convention (from ToyDataset.__getitem__):
    X  = sig_bkg  (B, L)   noisy input  (or pre-denoised signal)
    y  = sig      (B, L)   clean signal  (unused here)
    z  = params   (B, 5)   [amplitude, frequency_hz, phase_rad, noise_amplitude, snr]
"""

import importlib

import yaml
import torch
import torch.nn as nn
from torch import optim
import lightning as L

from dataloader.toy_dataloader import Param


def _load_denoiser(ckpt_path: str, cfg_path: str) -> nn.Module:
    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)
    mc = cfg['model']['init_args']['model']
    mod, cls_name = mc['class_path'].rsplit('.', 1)
    cls = getattr(importlib.import_module(mod), cls_name)
    model = cls(**mc.get('init_args', {}))
    ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=True)
    sd = {k.replace('model.', ''): v
          for k, v in ckpt['state_dict'].items() if k.startswith('model.')}
    model.load_state_dict(sd)
    model.eval()
    return model

# SNR bins for stratified test metrics
SNR_BINS = {'low': (0.0, 0.1), 'mid': (0.1, 0.3), 'high': (0.3, float('inf'))}


class RegressionMSE(L.LightningModule):
    """Parameter regression from a (possibly denoised) signal via MSE loss.

    Logs per-param RMSE at train/val and SNR-stratified RMSE at test.
    """

    def __init__(
        self,
        model: nn.Module,
        target_params: list[str] = ['amplitude', 'frequency_hz', 'phase_rad'],
        use_clean_input: bool = False,
        denoiser_ckpt: str | None = None,
        denoiser_cfg:  str | None = None,
        lr: float = 1e-3,
        lr_decay: float = 0.99,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.model           = model
        self.lr              = lr
        self.lr_decay        = lr_decay
        self.target_params   = target_params
        self.target_idx      = [int(Param[p]) for p in target_params]
        self.use_clean_input = use_clean_input
        self.criterion       = nn.MSELoss()

        # Optional frozen denoiser — inputs are denoised on-the-fly during training
        if denoiser_ckpt and denoiser_cfg:
            self.denoiser = _load_denoiser(denoiser_ckpt, denoiser_cfg)
            for p in self.denoiser.parameters():
                p.requires_grad_(False)
        else:
            self.denoiser = None

    def forward(self, x):
        # x: (B, L) -> (B, L, 1) -> model -> (B, n_targets)
        return self.model(x.unsqueeze(-1))

    def _get_targets(self, params):
        return params[:, self.target_idx]                      # (B, n_targets)

    def _log_per_param_rmse(self, y_hat, y, prefix):
        for i, name in enumerate(self.target_params):
            rmse = (y_hat[:, i] - y[:, i]).pow(2).mean().sqrt()
            self.log(f'{prefix}/rmse/{name}', rmse, on_step=False, on_epoch=True)

    def _step(self, batch):
        sig_bkg, sig, params = batch
        if self.use_clean_input:
            X = sig                                            # (B, L) clean signal
        elif self.denoiser is not None:
            with torch.no_grad():
                X = self.denoiser(sig_bkg.unsqueeze(-1)).squeeze(-1)  # denoised
        else:
            X = sig_bkg                                        # (B, L) noisy
        y     = self._get_targets(params)                      # (B, n_targets)
        y_hat = self(X)
        loss  = self.criterion(y_hat, y)
        return loss, y_hat, y

    def training_step(self, batch, batch_idx):
        loss, y_hat, y = self._step(batch)
        self.log('train/loss', loss, on_step=False, on_epoch=True, prog_bar=True)
        self._log_per_param_rmse(y_hat, y, 'train')
        return loss

    def validation_step(self, batch, batch_idx):
        loss, y_hat, y = self._step(batch)
        self.log('val/loss', loss, on_step=False, on_epoch=True, prog_bar=True)
        self._log_per_param_rmse(y_hat, y, 'val')

    def test_step(self, batch, batch_idx):
        sig_bkg, sig, params = batch
        if self.use_clean_input:
            X = sig
        elif self.denoiser is not None:
            with torch.no_grad():
                X = self.denoiser(sig_bkg.unsqueeze(-1)).squeeze(-1)
        else:
            X = sig_bkg
        y     = self._get_targets(params)
        y_hat = self(X)
        loss  = self.criterion(y_hat, y)
        self.log('test/loss', loss, on_step=False, on_epoch=True, prog_bar=True)
        self._log_per_param_rmse(y_hat, y, 'test')

        # SNR-stratified RMSE
        snr = params[:, int(Param.snr)]
        for bin_name, (lo, hi) in SNR_BINS.items():
            mask = (snr >= lo) & (snr < hi)
            if mask.sum() == 0:
                continue
            for i, name in enumerate(self.target_params):
                rmse = (y_hat[mask, i] - y[mask, i]).pow(2).mean().sqrt()
                self.log(f'test/rmse_snr_{bin_name}/{name}', rmse,
                         on_step=False, on_epoch=True)

    def configure_optimizers(self):
        optimizer = optim.AdamW(self.parameters(), lr=self.lr)
        scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=self.lr_decay)
        return {
            'optimizer': optimizer,
            'lr_scheduler': {'scheduler': scheduler, 'interval': 'epoch'},
        }
