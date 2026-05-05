"""
Denoising task for Project 8 (CRES I/Q traces).

Model:  forward(x: (B, L, C)) -> (B, L, C)   — typical seq2seq.
Loss:   MSE in the time domain (z-scored coordinate space).

Batch convention (Project8DenoisingDataModule):
    noisy  (B, L, C=2)   I/Q with noise added (per-channel std-normalised)
    clean  (B, L, C=2)   I/Q without noise    (same scale as noisy)

Usage (LightningCLI YAML):
    model:
      class_path: tasks.Project8.denoising.DenoisingMSE
      init_args:
        lr: 1.0e-3
        model:
          class_path: models.s4d_seq2seq.S4ModelSeq2Seq
          init_args:
            d_input:  2
            d_output: 2
            d_model:  128
            d_state:  64
            n_layers: 4
"""

import torch
import torch.nn as nn
from torch import optim
import lightning as L


class DenoisingMSE(L.LightningModule):
    """Multi-channel seq2seq denoising with MSE loss."""

    def __init__(
        self,
        model: nn.Module,
        lr: float = 1e-3,
        lr_decay: float = 0.99,
    ):
        super().__init__()
        self.save_hyperparameters(ignore=['model'])
        self.model     = model
        self.lr        = lr
        self.lr_decay  = lr_decay
        self.criterion = nn.MSELoss()

    def forward(self, x):
        # x: (B, L, C) -> (B, L, C)
        return self.model(x)

    def _step(self, batch):
        noisy, clean = batch
        pred = self(noisy)
        return self.criterion(pred, clean), pred

    def training_step(self, batch, _):
        loss, _ = self._step(batch)
        self.log('train/loss', loss, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, _):
        loss, _ = self._step(batch)
        self.log('val/loss', loss, on_step=False, on_epoch=True, prog_bar=True)

    def test_step(self, batch, _):
        noisy, clean = batch
        pred = self(noisy)
        loss = self.criterion(pred, clean)
        self.log('test/loss', loss, on_step=False, on_epoch=True, prog_bar=True)
        # Per-channel test RMSE for diagnostic granularity (I vs Q).
        for c in range(clean.shape[-1]):
            rmse_c = (pred[..., c] - clean[..., c]).pow(2).mean().sqrt()
            self.log(f'test/rmse_ch{c}', rmse_c, on_step=False, on_epoch=True)

    def configure_optimizers(self):
        opt   = optim.AdamW(self.parameters(), lr=self.lr)
        sched = optim.lr_scheduler.ExponentialLR(opt, gamma=self.lr_decay)
        return {'optimizer': opt, 'lr_scheduler': {'scheduler': sched, 'interval': 'epoch'}}
