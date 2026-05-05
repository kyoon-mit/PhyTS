"""Regression task for LIGO gravitational wave strain data.

The model receives the strain (H1 + L1) and predicts scalar target variables
(e.g. chirp_mass). Any pooling nn.Module with signature
    forward(x: (B, L, d_input)) -> (B, d_output)
can be plugged in via the `model` argument.

Batch layout (from LIGODataset.__getitem__):
    X  (B, n_ifos, L)  injected strain (H1 + L1)  -> model sees (B, L, n_ifos)
    y  (B, n_targets)  target variables (e.g. chirp_mass)
    z  (B, n_obs)      observed variables (e.g. snr) -- unused
"""

import lightning as L
import torch.nn as nn
from jaxtyping import Float, Int
from torch import Tensor, optim

Batch = tuple[
    Float[Tensor, "B n_ifos L"],
    Float[Tensor, "B n_targets"],
    Int[Tensor, "B n_obs"],
]


class RegressionMSE(L.LightningModule):
    """Pooled regression from (B, n_ifos, L) LIGO strain to (B, n_targets) via MSE."""

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

    def forward(self, x: Float[Tensor, "B n_ifos L"]) -> Float[Tensor, "B n_targets"]:
        """x: (B, n_ifos, L) -> (B, L, n_ifos) -> model -> (B, n_targets)"""
        return self.model(x.transpose(1, 2))

    def _step(self, batch: Batch):
        X, y, _ = batch
        return self.criterion(self(X), y)

    def training_step(self, batch: Batch):
        loss = self._step(batch)
        self.log("train/loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch: Batch):
        loss = self._step(batch)
        self.log("val/loss", loss, on_step=False, on_epoch=True, prog_bar=True)

    def test_step(self, batch: Batch):
        loss = self._step(batch)
        self.log("test/loss", loss, on_step=False, on_epoch=True, prog_bar=True)

    def configure_optimizers(self):  # type: ignore
        optimizer = optim.AdamW(self.parameters(), lr=self.lr)
        scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=self.lr_decay)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "interval": "epoch"},
        }
