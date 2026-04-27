"""Classification tasks for TESS lightcurve data.

TESSClassificationCE — end-to-end: flux → variability class label.
TESSFrozenBackboneClassificationCE — frozen seq2seq backbone + trainable MLP head.

Batch convention (from TESSClassificationDataset.__getitem__):
    flux   (B, L)   — z-score normalized, padded to seq_len
    mask   (B, L)   — True where cadence is valid
    label  (B,)     — int64 class index

Any model with signature forward(x: (B, L, 1)) -> (B, num_classes) works as drop-in.

Usage (LightningCLI YAML):
    model:
      class_path: tasks.TESS.tess_classification.TESSClassificationCE
      init_args:
        num_classes: 7          # update from label_map.json
        lr: 1.0e-3
        lr_decay: 0.99
        model:
          class_path: models.mlp.MLPRegressor
          init_args:
            seq_len: 1100
            d_output: 7         # must equal num_classes
"""

import importlib

import torch
import torch.nn as nn
from torch import Tensor, optim
import lightning as L
import yaml


def _load_seq2seq_backbone(ckpt_path: str, cfg_path: str) -> nn.Module:
    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)
    model_cfg = cfg["model"]["init_args"]["model"]
    module_name, class_name = model_cfg["class_path"].rsplit(".", 1)
    cls = getattr(importlib.import_module(module_name), class_name)
    model = cls(**model_cfg.get("init_args", {}))
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    sd = {
        k[len("model."):]: v
        for k, v in ckpt["state_dict"].items()
        if k.startswith("model.")
    }
    model.load_state_dict(sd)
    return model


# ── End-to-end classification ────────────────────────────────────────────────

class TESSClassificationCE(L.LightningModule):
    """End-to-end multi-class classification: forward(flux) → label via cross-entropy."""

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

    def forward(self, x: Tensor) -> Tensor:
        # x: (B, L) → (B, L, 1) → model → (B, num_classes)
        return self.model(x.unsqueeze(-1))

    def _step(self, batch: tuple) -> tuple[Tensor, Tensor, Tensor]:
        flux, mask, label = batch
        logits = self(flux)
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
        self._test_preds.append(preds.cpu())
        self._test_labels.append(labels.cpu())

    def on_test_epoch_end(self):
        preds = torch.cat(self._test_preds)
        labels = torch.cat(self._test_labels)
        for c in range(self.num_classes):
            mask = labels == c
            if mask.sum() > 0:
                self.log(f"test/acc_class_{c}", (preds[mask] == c).float().mean())

    def configure_optimizers(self):
        opt = optim.AdamW(self.parameters(), lr=self.lr)
        sched = optim.lr_scheduler.ExponentialLR(opt, gamma=self.lr_decay)
        return {"optimizer": opt, "lr_scheduler": {"scheduler": sched, "interval": "epoch"}}


# ── Frozen-backbone classification ───────────────────────────────────────────

class TESSFrozenBackboneClassificationCE(L.LightningModule):
    """Frozen seq2seq backbone + trainable MLP classification head.

    The backbone is loaded from a LightningCLI checkpoint and frozen.
    Only the head is trained.

    Forward: flux → backbone(flux) → reconstructed_flux → head → class logits

    Usage (LightningCLI YAML):
        model:
          class_path: tasks.TESS.tess_classification.TESSFrozenBackboneClassificationCE
          init_args:
            backbone_ckpt: checkpoints/tess_s4d_reconstruction/best.ckpt
            backbone_cfg: configs/TESS/train_tess_s4d_reconstruction.yaml
            num_classes: 7
            lr: 1.0e-3
            lr_decay: 0.99
            head:
              class_path: models.mlp.MLPRegressor
              init_args:
                seq_len: 1100
                d_output: 7
    """

    def __init__(
        self,
        backbone_ckpt: str,
        backbone_cfg: str,
        head: nn.Module,
        num_classes: int,
        lr: float = 1e-3,
        lr_decay: float = 0.99,
    ):
        super().__init__()
        self.save_hyperparameters(ignore=["head"])
        backbone = _load_seq2seq_backbone(backbone_ckpt, backbone_cfg)
        backbone.eval()
        for p in backbone.parameters():
            p.requires_grad_(False)
        self.backbone = backbone
        self.head = head
        self.num_classes = num_classes
        self.lr = lr
        self.lr_decay = lr_decay
        self.criterion = nn.CrossEntropyLoss()

    def train(self, mode: bool = True):
        # Lightning calls model.train() each epoch; keep backbone in eval mode
        # regardless so its dropout layers stay off during head training.
        super().train(mode)
        self.backbone.eval()
        return self

    def forward(self, x: Tensor) -> Tensor:
        # x: (B, L) → backbone → (B, L, 1) → head → (B, num_classes)
        with torch.no_grad():
            reconstructed = self.backbone(x.unsqueeze(-1))  # (B, L, 1)
        return self.head(reconstructed)

    def _step(self, batch: tuple) -> tuple[Tensor, Tensor, Tensor]:
        flux, mask, label = batch
        logits = self(flux)
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
        self._test_preds.append(preds.cpu())
        self._test_labels.append(labels.cpu())

    def on_test_epoch_end(self):
        preds = torch.cat(self._test_preds)
        labels = torch.cat(self._test_labels)
        for c in range(self.num_classes):
            mask = labels == c
            if mask.sum() > 0:
                self.log(f"test/acc_class_{c}", (preds[mask] == c).float().mean())

    def configure_optimizers(self):
        # Only head parameters are trainable
        opt = optim.AdamW(self.head.parameters(), lr=self.lr)
        sched = optim.lr_scheduler.ExponentialLR(opt, gamma=self.lr_decay)
        return {"optimizer": opt, "lr_scheduler": {"scheduler": sched, "interval": "epoch"}}
