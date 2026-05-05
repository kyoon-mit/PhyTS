"""Classification task for the TESS variable-star dataset. (taken from Allison's)

The batch convention is:
    flux  (B, L)
    mask  (B, L)
    label (B,)

Any model with signature forward(x: (B, L, 1), mask: (B, L)) -> (B, C)
can be plugged in via the `model` argument.
"""

from __future__ import annotations

import lightning as L
import torch
import torch.nn as nn
from torch import Tensor, optim


class TESSClassificationCE(L.LightningModule):
    """Multi-class classification via cross-entropy loss."""

    def __init__(
        self,
        model: nn.Module,
        num_classes: int,
        lr: float = 1e-3,
        lr_decay: float = 0.99,
    ):
        super().__init__()
        self.save_hyperparameters(ignore=["model"])
        self.model = model
        self.num_classes = num_classes
        self.lr = lr
        self.lr_decay = lr_decay
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, x: Tensor, mask: Tensor) -> Tensor:
        return self.model(x.unsqueeze(-1), mask=mask)

    def _step(self, batch: tuple[Tensor, Tensor, Tensor]) -> tuple[Tensor, Tensor, Tensor]:
        flux, mask, label = batch
        logits = self(flux, mask)
        loss = self.criterion(logits, label)
        return loss, logits.argmax(dim=-1), label

    def training_step(self, batch, batch_idx):
        loss, preds, labels = self._step(batch)
        acc = (preds == labels).float().mean()
        self.log("train/loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("train/acc", acc, on_step=False, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        loss, preds, labels = self._step(batch)
        acc = (preds == labels).float().mean()
        self.log("val/loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val/acc", acc, on_step=False, on_epoch=True)

    def on_test_epoch_start(self):
        self._test_preds: list[Tensor] = []
        self._test_labels: list[Tensor] = []

    def test_step(self, batch, batch_idx):
        loss, preds, labels = self._step(batch)
        acc = (preds == labels).float().mean()
        self.log("test/loss", loss, on_step=False, on_epoch=True)
        self.log("test/acc", acc, on_step=False, on_epoch=True)
        self._test_preds.append(preds.detach().cpu())
        self._test_labels.append(labels.detach().cpu())

    def on_test_epoch_end(self):
        preds = torch.cat(self._test_preds)
        labels = torch.cat(self._test_labels)
        for c in range(self.num_classes):
            mask = labels == c
            if mask.any():
                self.log(f"test/acc_class_{c}", (preds[mask] == c).float().mean())

    def configure_optimizers(self):
        optimizer = optim.AdamW(self.parameters(), lr=self.lr)
        scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=self.lr_decay)
        return {"optimizer": optimizer, "lr_scheduler": {"scheduler": scheduler, "interval": "epoch"}}
