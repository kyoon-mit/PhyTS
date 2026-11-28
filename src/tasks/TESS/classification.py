"""TESS variability classification: light curve -> one of 8 classes, with cross-entropy loss.

Batches are (flux, mask, label) from TESSClassificationDataModule. The model gets the flux as
(B, L, 1) and the padding mask, and returns (B, 8) logits. Validation and test report balanced
accuracy (mean per-class recall) over the whole split, the paper's metric. The test step also
writes the true and predicted label of every test light curve to csv_out, which
benchmarks/TESS/score.py reads.
"""

from pathlib import Path

import lightning as L
import numpy as np
import pandas as pd
import torch.nn.functional as F
from torch import nn, optim

from dataloader.tess_dataloader import CLASSES


def balanced_accuracy(true: np.ndarray, pred: np.ndarray) -> float:
    """Mean over the classes present in true of the fraction predicted correctly."""
    return float(np.mean([np.mean(pred[true == c] == c) for c in np.unique(true)]))


class ClassificationEval:
    """Balanced accuracy over the full val / test split, and the test-prediction CSV.

    Mixed into the PyTorch and the LinOSS task; their eval steps call _collect.
    """

    csv_out: str

    def _collect(self, true: np.ndarray, pred: np.ndarray) -> None:
        self._true.append(true)
        self._pred.append(pred)

    def _reset(self) -> None:
        self._true, self._pred = [], []

    def on_validation_epoch_start(self):
        self._reset()

    def on_test_epoch_start(self):
        self._reset()

    def on_validation_epoch_end(self):
        self.log("val/balanced_acc", balanced_accuracy(np.concatenate(self._true),
                                                       np.concatenate(self._pred)))

    def on_test_epoch_end(self):
        true, pred = np.concatenate(self._true), np.concatenate(self._pred)
        self.log("test/balanced_acc", balanced_accuracy(true, pred))
        out = Path(self.csv_out)
        out.parent.mkdir(parents=True, exist_ok=True)
        names = np.array(CLASSES)
        table = pd.DataFrame({"true_label": names[true], "pred_label": names[pred]})
        table.to_csv(out, index=False)
        print(f"Test predictions -> {out} ({len(true):,} rows)")


class TESSClassification(ClassificationEval, L.LightningModule):
    """Cross-entropy classifier around model(x (B, L, 1), mask (B, L)) -> (B, n_classes)."""

    def __init__(
        self,
        model: nn.Module,
        lr: float = 1e-3,
        lr_decay: float = 0.99,
        weight_decay: float = 1e-4,
        csv_out: str = "results/TESS/classification/test_predictions.csv",
    ):
        super().__init__()
        self.save_hyperparameters(ignore=["model"])
        self.model = model
        self.csv_out = csv_out

    def forward(self, flux, mask):
        return self.model(flux.unsqueeze(-1), mask=mask)

    def _step(self, batch, stage: str):
        flux, mask, label = batch
        logits = self(flux, mask)
        loss = F.cross_entropy(logits, label)
        self.log(f"{stage}/loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        if stage != "train":
            self._collect(label.cpu().numpy(), logits.argmax(-1).cpu().numpy())
        return loss

    def training_step(self, batch, batch_idx):
        return self._step(batch, "train")

    def validation_step(self, batch, batch_idx):
        self._step(batch, "val")

    def test_step(self, batch, batch_idx):
        self._step(batch, "test")

    def configure_optimizers(self):
        opt = optim.AdamW(self.parameters(), lr=self.hparams.lr,
                          weight_decay=self.hparams.weight_decay)
        sched = optim.lr_scheduler.ExponentialLR(opt, gamma=self.hparams.lr_decay)
        return {"optimizer": opt, "lr_scheduler": {"scheduler": sched, "interval": "epoch"}}
