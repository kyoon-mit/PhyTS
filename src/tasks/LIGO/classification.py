"""Classification task for LIGO gravitational wave strain data.

Model recieves the strain (H1 + L1) and predicts the class label (e.g. binary merger vs noise).


TODO: Complete the implementation and docstring
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


class Classification(L.LightningModule):
    """Binary classification via cross-entropy loss."""

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
        self.criterion = nn.BCEWithLogitsLoss()

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
