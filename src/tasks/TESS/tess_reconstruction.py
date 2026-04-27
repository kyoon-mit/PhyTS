"""Reconstruction (denoising pretraining) task for TESS lightcurves.

TESSReconstructionMSE — trains a seq2seq backbone to reconstruct clean flux
from noise-corrupted input. Serves as self-supervised pretraining for the
frozen-backbone downstream tasks (TESSFrozenBackboneRegressionMSE,
TESSFrozenBackboneClassificationCE).

Batch convention (from TESSReconstructionDataset.__getitem__):
    noisy_flux  (B, L)   — flux + Gaussian noise (noise_std≈0.3)
    clean_flux  (B, L)   — original z-score normalized flux
    mask        (B, L)   — True where cadence is valid (not padding)

Loss is MSE computed only on valid (non-padded) positions.

Usage (LightningCLI YAML):
    model:
      class_path: tasks.TESS.tess_reconstruction.TESSReconstructionMSE
      init_args:
        lr: 1.0e-3
        lr_decay: 0.99
        model:
          class_path: models.s4d_seq2seq.S4ModelSeq2Seq
          init_args:
            d_input: 1
            d_output: 1
            d_model: 256
            d_state: 64
            n_layers: 4
            dropout: 0.1
"""

import torch
import torch.nn as nn
from torch import Tensor, optim
import lightning as L


class TESSReconstructionMSE(L.LightningModule):
    """Seq2seq reconstruction pretraining: noisy_flux → clean_flux via masked MSE."""

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

    def forward(self, x: Tensor) -> Tensor:
        # x: (B, L) → (B, L, 1) → model → (B, L, 1) → (B, L)
        return self.model(x.unsqueeze(-1)).squeeze(-1)

    def _masked_mse(self, pred: Tensor, target: Tensor, mask: Tensor) -> Tensor:
        loss_elem = (pred - target).pow(2)
        return (loss_elem * mask.float()).sum() / mask.float().sum().clamp(min=1.0)

    def _step(self, batch: tuple) -> tuple[Tensor, Tensor]:
        noisy, clean, mask = batch
        reconstructed = self(noisy)
        loss = self._masked_mse(reconstructed, clean, mask)
        # SNR-like metric: signal power / residual power (on valid positions only)
        signal_power = (clean * mask.float()).pow(2).sum() / mask.float().sum().clamp(min=1.0)
        residual_power = loss
        snr = 10.0 * torch.log10((signal_power / residual_power.clamp(min=1e-10)))
        return loss, snr

    def training_step(self, batch, batch_idx):
        loss, snr = self._step(batch)
        self.log("train/loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("train/snr_db", snr, on_step=False, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        loss, snr = self._step(batch)
        self.log("val/loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val/snr_db", snr, on_step=False, on_epoch=True)

    def test_step(self, batch, batch_idx):
        loss, snr = self._step(batch)
        self.log("test/loss", loss, on_step=False, on_epoch=True)
        self.log("test/snr_db", snr, on_step=False, on_epoch=True)

    def configure_optimizers(self):
        opt = optim.AdamW(self.parameters(), lr=self.lr)
        sched = optim.lr_scheduler.ExponentialLR(opt, gamma=self.lr_decay)
        return {"optimizer": opt, "lr_scheduler": {"scheduler": sched, "interval": "epoch"}}
