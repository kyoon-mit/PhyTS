"""Multi-class classification task for TESS stellar variability light curves.

The model receives a padded flux time series and a boolean mask, and predicts
one of 8 variability classes (see dataloader.tess_dataloader.Label).

Any model with signature  forward(x: (B,L,1)) -> (B, num_classes)
can be plugged in via the `model` argument (e.g. S4Model with mean pooling).

Because TESS light curves have variable length and are zero-padded, the task
applies a masked mean pool before the model's decoder when `use_mask` is True.
Set `use_mask=False` if the model handles masking internally.

Batch convention (from TESSDataset.__getitem__):
    flux   (B, seq_len)  float32   padded/truncated flux
    mask   (B, seq_len)  bool      True = real data, False = padding
    label  (B,)          int64     integer class label

Usage (LightningCLI YAML):
    model:
      class_path: tasks.TESS.classification.TESSClassification
      init_args:
        lr: 1.0e-3
        model:
          class_path: models.s4d.S4Model
          init_args:
            d_input: 1
            d_output: 8
            d_model: 64
            n_layers: 4
"""

import logging

import torch
import torch.nn as nn
from torch import Tensor, optim
import lightning as L
from torchmetrics.classification import (
    MulticlassAccuracy,
    MulticlassF1Score,
    MulticlassConfusionMatrix,
)

log = logging.getLogger(__name__)

from dataloader.tess_dataloader import NUM_CLASSES, Label

Batch = tuple[Tensor, Tensor, Tensor]  # (flux, mask, label)


class TESSClassification(L.LightningModule):
    """Multi-class stellar variability classification via cross-entropy loss.

    Logs accuracy and macro-F1 on train/val/test.
    Logs per-class accuracy at test time.
    """

    def __init__(
        self,
        model: nn.Module,
        num_classes: int = NUM_CLASSES,
        lr: float = 1e-3,
        lr_decay: float = 0.99,
        weight_decay: float = 1e-4,
        use_mask: bool = True,
        class_weights: list[float] | None = None,
    ):
        super().__init__()
        self.save_hyperparameters(ignore=["model"])
        self.model = model
        self.lr = lr
        self.lr_decay = lr_decay
        self.weight_decay = weight_decay
        self.num_classes = num_classes
        self.use_mask = use_mask

        weight = torch.tensor(class_weights, dtype=torch.float32) if class_weights else None
        self.criterion = nn.CrossEntropyLoss(weight=weight)

        # Metrics
        metric_kwargs = dict(num_classes=num_classes, average="macro")
        self.train_acc = MulticlassAccuracy(**metric_kwargs)
        self.val_acc = MulticlassAccuracy(**metric_kwargs)
        self.val_f1 = MulticlassF1Score(**metric_kwargs)
        self.test_acc = MulticlassAccuracy(**metric_kwargs)
        self.test_f1 = MulticlassF1Score(**metric_kwargs)
        self.test_acc_per_class = MulticlassAccuracy(num_classes=num_classes, average="none")

    def forward(self, flux: Tensor, mask: Tensor) -> Tensor:
        """
        flux: (B, L) -> unsqueeze -> (B, L, 1) -> model -> (B, num_classes)
        mask: (B, L) bool — used to zero out padding and for masked pooling.
        """
        x = flux.unsqueeze(-1)  # (B, L, 1)
        if self.use_mask:
            x = x * mask.unsqueeze(-1).float()
            return self.model(x, mask=mask)
        return self.model(x)

    def _step(self, batch: Batch):
        flux, mask, label = batch
        logits = self(flux, mask)  # (B, num_classes)
        loss = self.criterion(logits, label)
        return loss, logits, label

    def training_step(self, batch: Batch, batch_idx: int):
        loss, logits, label = self._step(batch)
        self.train_acc(logits, label)
        self.log("train/loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("train/acc", self.train_acc, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def on_train_epoch_end(self):
        metrics = self.trainer.callback_metrics
        log.info(
            "Epoch %3d | train/loss: %.4f  train/acc: %.4f  val/loss: %.4f  val/acc: %.4f  val/f1: %.4f",
            self.current_epoch,
            metrics.get("train/loss", 0),
            metrics.get("train/acc", 0),
            metrics.get("val/loss", 0),
            metrics.get("val/acc", 0),
            metrics.get("val/f1_macro", 0),
        )

    def validation_step(self, batch: Batch, batch_idx: int):
        loss, logits, label = self._step(batch)
        self.val_acc(logits, label)
        self.val_f1(logits, label)
        self.log("val/loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val/acc", self.val_acc, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val/f1_macro", self.val_f1, on_step=False, on_epoch=True)

    def test_step(self, batch: Batch, batch_idx: int):
        loss, logits, label = self._step(batch)
        self.test_acc(logits, label)
        self.test_f1(logits, label)
        self.test_acc_per_class(logits, label)
        self.log("test/loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("test/acc", self.test_acc, on_step=False, on_epoch=True, prog_bar=True)
        self.log("test/f1_macro", self.test_f1, on_step=False, on_epoch=True)

    def on_test_epoch_end(self):
        per_class = self.test_acc_per_class.compute()
        for cls in Label:
            self.log(f"test/acc/{cls.name}", per_class[cls.value])
        self.test_acc_per_class.reset()

    def configure_optimizers(self):
        optimizer = optim.AdamW(
            self.parameters(), lr=self.lr, weight_decay=self.weight_decay
        )
        scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=self.lr_decay)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "interval": "epoch"},
        }
