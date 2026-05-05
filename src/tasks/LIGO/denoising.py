"""Denoising task for LIGO gravitational wave strain data.

The model receives the noisy strain (H1 + L1) and predicts the clean waveform.
Any seq2seq nn.Module with signature  forward(x: (B,L,d_input)) -> (B,L,d_output)
can be plugged in via the `model` argument.

Usage (LightningCLI YAML):
    model:
      class_path: tasks.LIGO.denoising.DenoisingMSE
      init_args:
        lr: 1.0e-3
        lr_decay: 0.99
        model:
          class_path: models.s4d.S4ModelSeq2Seq
          init_args:
            d_input: 2       # H1 + L1
            d_output: 2      # denoised H1 + L1
            d_model: 256
            d_state: 64
            n_layers: 4
            dropout: 0.2

Note:
    The LIGODataset must expose a clean waveform key (e.g. 'clean_data') alongside
    'injected_data' for this task to work. Update LIGODataModule accordingly.

Batch convention (from LIGODataset.__getitem__):
    X  = injected_data  (B, n_ifos, L)   noisy strain
    y  = clean target   (B, n_ifos, L)   clean waveform  <- requires dataloader update
    z  = observed vars  (B, n_obs)       e.g. snr
"""

import lightning as L
import torch.nn as nn
from jaxtyping import Float, Int
from torch import Tensor, optim

# TODO: Should this be moved to the dataset?
Batch = tuple[
    Float[Tensor, "B n_ifos L"],
    Float[Tensor, "B n_ifos L"],
    Int[Tensor, "B n_obs"],
]


class DenoisingMSE(L.LightningModule):
    """Seq2seq denoising via MSE loss."""

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

    def forward(self, x: Float[Tensor, "B n_ifos L"]) -> Float[Tensor, "B n_ifos L"]:
        """Model forward pass. Transposes input to (B, L, n_ifos).

        x: (B, n_ifos, L) -> (B, L, n_ifos) -> model -> (B, L, n_ifos) -> (B, n_ifos, L)
        """
        return self.model(x.transpose(1, 2)).transpose(1, 2)

    def _step(self, batch: Batch):
        """Apply model and loss to a batch."""
        X, y, _ = batch
        loss = self.criterion(self(X), y)
        return loss

    def training_step(self, batch: Batch):
        """Train step: compute loss and log it."""
        loss = self._step(batch)
        self.log("train/loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch: Batch):
        """Validation step: compute loss and log it."""
        loss = self._step(batch)
        self.log("val/loss", loss, on_step=False, on_epoch=True, prog_bar=True)

    def test_step(self, batch: Batch):
        """Test step: compute loss and log it."""
        loss = self._step(batch)
        self.log("test/loss", loss, on_step=False, on_epoch=True, prog_bar=True)

    def configure_optimizers(self):  # type: ignore
        """Configure optimizer and learning rate scheduler."""
        optimizer = optim.AdamW(self.parameters(), lr=self.lr)
        scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=self.lr_decay)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "interval": "epoch"},
        }
