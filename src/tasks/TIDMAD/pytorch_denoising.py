"""PyTorch Lightning tasks for TIDMAD seq2seq denoising.

Batch convention (from TIDMADDataModule):
    noisy   (B, L)   float32  channel0001 (SQUID)
    clean   (B, L)   float32  channel0002 (reference injection)
    params  (B, 3)   float32  [frequency_hz, amplitude_mV, snr]  (unused during training)

Both tasks add a channel dimension before the model and squeeze it back after:
    (B, L) → (B, L, 1) → model → (B, L, 1) → (B, L)

Compatible with ConvAE(n_layers=..., latent_channels=..., kernel_size=...).
"""

import lightning as L
import torch
import torch.nn as nn
from torch import Tensor, optim

from functions.losses import MixtureMSESpectralLoss


class TIDMADDenoisingMSE(L.LightningModule):
    """Time-domain MSE denoising task for TIDMAD.

    Loss: MSE between model output and clean reference (time domain).
    """

    def __init__(
        self,
        model: nn.Module,
        lr: float = 1e-3,
        lr_decay: float = 0.99,
    ):
        super().__init__()
        self.save_hyperparameters(ignore=["model"])
        self.model = model
        self.lr = lr
        self.lr_decay = lr_decay
        self.criterion = nn.MSELoss()

    def forward(self, x: Tensor) -> Tensor:
        return self.model(x.unsqueeze(-1)).squeeze(-1)

    def _step(self, batch: tuple) -> Tensor:
        noisy, clean, _ = batch
        pred = self(noisy)
        return self.criterion(pred, clean)

    def training_step(self, batch, batch_idx: int):
        loss = self._step(batch)
        self.log("train/loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx: int):
        loss = self._step(batch)
        self.log("val/loss", loss, on_step=False, on_epoch=True, prog_bar=True)

    def configure_optimizers(self):
        optimizer = optim.AdamW(self.parameters(), lr=self.lr)
        scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=self.lr_decay)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "interval": "epoch"},
        }


class TIDMADDenoisingPSD(L.LightningModule):
    """Frequency-domain (PSD) denoising task for TIDMAD.

    Loss: MSE between the magnitude spectra of predicted and clean signals.
    Aligned with the PSD-based SNR benchmark metric (TIDMAD Benchmark 1).
    """

    def __init__(
        self,
        model: nn.Module,
        lr: float = 1e-3,
        lr_decay: float = 0.99,
    ):
        super().__init__()
        self.save_hyperparameters(ignore=["model"])
        self.model = model
        self.lr = lr
        self.lr_decay = lr_decay
        self.criterion = MixtureMSESpectralLoss(alpha=0.0)

    def forward(self, x: Tensor) -> Tensor:
        return self.model(x.unsqueeze(-1)).squeeze(-1)

    def _step(self, batch: tuple) -> Tensor:
        noisy, clean, _ = batch
        pred = self(noisy)
        return self.criterion(pred, clean)

    def training_step(self, batch, batch_idx: int):
        loss = self._step(batch)
        self.log("train/loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx: int):
        loss = self._step(batch)
        self.log("val/loss", loss, on_step=False, on_epoch=True, prog_bar=True)

    def configure_optimizers(self):
        optimizer = optim.AdamW(self.parameters(), lr=self.lr)
        scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=self.lr_decay)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "interval": "epoch"},
        }
