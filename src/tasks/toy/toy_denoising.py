"""
Denoising task for the sinusoidal + white noise toy dataset.

The model receives the noisy signal and predicts the clean signal.
Any seq2seq nn.Module with signature  forward(x: (B,L,1)) -> (B,L,1)
can be plugged in via the `model` argument.

Usage (LightningCLI YAML):
    model:
      class_path: tasks.toy_denoising.DenoisingMSE
      init_args:
        lr: 1.0e-3
        lr_decay: 0.99
        model:
          class_path: models.s4d.S4ModelSeq2Seq
          init_args:
            d_input: 1
            d_output: 1
            d_model: 256
            d_state: 64
            n_layers: 4
            dropout: 0.2
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
import lightning as L


class PSDLoss(nn.Module):
    """MSE loss in the power spectral density domain."""

    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        psd_pred = torch.fft.rfft(y_pred).abs().pow(2)
        psd_true = torch.fft.rfft(y_true).abs().pow(2)
        return F.mse_loss(psd_pred, psd_true)


class DenoisingMSE(L.LightningModule):
    """Seq2seq denoising via MSE loss.

    Batch convention (from ToyDataset.__getitem__):
        X  = sig_bkg  (B, L)   noisy input
        y  = sig      (B, L)   clean target
        z  = params   (B, 5)   [amplitude, frequency_hz, phase_rad, noise_amplitude, snr]
    """

    def __init__(
        self,
        model: nn.Module,
        lr: float = 1e-3,
        lr_decay: float = 0.99,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.model    = model
        self.lr       = lr
        self.lr_decay = lr_decay
        self.criterion = nn.MSELoss()

    def forward(self, x):
        # x: (B, L) -> unsqueeze channel -> (B, L, 1) -> model -> (B, L, 1) -> squeeze
        return self.model(x.unsqueeze(-1)).squeeze(-1)  # (B, L)

    def _step(self, batch):
        X, y, _ = batch             # (B, L), (B, L), (B, 5)
        y_hat = self(X)             # (B, L)
        loss  = self.criterion(y_hat, y)
        return loss

    def training_step(self, batch, batch_idx):
        loss = self._step(batch)
        self.log('train/loss', loss, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        loss = self._step(batch)
        self.log('val/loss', loss, on_step=False, on_epoch=True, prog_bar=True)

    def test_step(self, batch, batch_idx):
        loss = self._step(batch)
        self.log('test/loss', loss, on_step=False, on_epoch=True, prog_bar=True)

    def configure_optimizers(self):
        optimizer = optim.AdamW(self.parameters(), lr=self.lr)
        scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=self.lr_decay)
        return {
            'optimizer': optimizer,
            'lr_scheduler': {'scheduler': scheduler, 'interval': 'epoch'},
        }


class DenoisingPSD(DenoisingMSE):
    """Seq2seq denoising with PSD loss."""

    def __init__(self, model: nn.Module, lr: float = 1e-3, lr_decay: float = 0.99):
        super().__init__(model, lr, lr_decay)
        self.criterion = PSDLoss()
