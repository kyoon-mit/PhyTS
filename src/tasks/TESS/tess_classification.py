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
import lightning as L
import yaml
import importlib

from tasks.param_count import (
    attach_scalar_hyperparams,
    torch_model_parameter_hyper_dict,
    torch_module_bundle_prefixed,
    torch_nn_parameter_count,
)
from tasks.TESS.eval_plots import log_validation_plots_to_wandb
from tasks.TESS.classification_metrics import log_extended_metrics


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
        self._val_preds.append(preds.cpu())
        self._val_labels.append(labels.cpu())

    def on_validation_epoch_end(self):
        preds  = torch.cat(self._val_preds)
        labels = torch.cat(self._val_labels)
        self.log("val/acc", (preds == labels).float().mean())
        per_class = [
            (preds[labels == c] == c).float().mean().item()
            for c in range(self.num_classes) if (labels == c).any()
        ]
        if per_class:
            self.log("val/balanced_acc", sum(per_class) / len(per_class))
        log_extended_metrics(
            self,
            preds.numpy(),
            labels.numpy(),
            num_classes=self.num_classes,
            prefix="val",
        )
        log_validation_plots_to_wandb(
            self,
            kind="classification",
            y_true=labels.numpy(),
            y_hat=preds.numpy(),
        )

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
        self.log("test/acc", (preds == labels).float().mean())
        per_class = [
            (preds[labels == c] == c).float().mean().item()
            for c in range(self.num_classes) if (labels == c).any()
        ]
        if per_class:
            self.log("test/balanced_acc", sum(per_class) / len(per_class))
        log_extended_metrics(
            self,
            preds.numpy(),
            labels.numpy(),
            num_classes=self.num_classes,
            prefix="test",
        )
        for c in range(self.num_classes):
            mask = labels == c
            if mask.any():
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
            backbone_cfg: configs/TESS/other/train_tess_s4d_reconstruction.yaml
            num_classes: 8
            lr: 1.0e-3
            lr_decay: 0.99
            head:
              class_path: models.mlp.MLPRegressor
              init_args:
                seq_len: 1100
                d_output: 8
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
        bundled = dict(**torch_module_bundle_prefixed("backbone", backbone))
        bundled.update(torch_module_bundle_prefixed("head", head))
        bundled["backbone_head_total_num_parameters_nn"] = (
            torch_nn_parameter_count(backbone) + torch_nn_parameter_count(head)
        )
        self.save_hyperparameters(ignore=["head"])
        attach_scalar_hyperparams(self, bundled)

    def train(self, mode: bool = True):
        # Lightning calls model.train() each epoch; keep backbone in eval mode
        # regardless so its dropout layers stay off during head training.
        super().train(mode)
        self.backbone.eval()
        return self

    def forward(self, x: Tensor, mask: Tensor | None = None) -> Tensor:
        # x: (B, L) → backbone → (B, L, 1) → (optionally mask) → head → (B, num_classes)
        with torch.no_grad():
            reconstructed = self.backbone(x.unsqueeze(-1))  # (B, L, 1)
        if mask is not None:
            reconstructed = reconstructed * mask.unsqueeze(-1).float()
        return self.head(reconstructed, mask=mask)

    def _step(self, batch: tuple) -> tuple[Tensor, Tensor, Tensor]:
        flux, mask, label = batch
        logits = self(flux, mask)
        loss = self.criterion(logits, label)
        return loss, logits.argmax(dim=-1), label

    def on_train_epoch_start(self):
        self._train_preds: list[Tensor] = []
        self._train_labels: list[Tensor] = []

    def training_step(self, batch, batch_idx):
        loss, preds, labels = self._step(batch)
        self.log("train/loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        self._train_preds.append(preds.cpu())
        self._train_labels.append(labels.cpu())
        return loss

    def on_train_epoch_end(self):
        preds = torch.cat(self._train_preds)
        labels = torch.cat(self._train_labels)
        self.log("train/acc", (preds == labels).float().mean())

    def on_validation_epoch_start(self):
        self._val_preds: list[Tensor] = []
        self._val_labels: list[Tensor] = []

    def validation_step(self, batch, batch_idx):
        loss, preds, labels = self._step(batch)
        self.log("val/loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        self._val_preds.append(preds.cpu())
        self._val_labels.append(labels.cpu())

    def on_validation_epoch_end(self):
        preds  = torch.cat(self._val_preds)
        labels = torch.cat(self._val_labels)
        self.log("val/acc", (preds == labels).float().mean())
        per_class = [
            (preds[labels == c] == c).float().mean().item()
            for c in range(self.num_classes) if (labels == c).any()
        ]
        if per_class:
            self.log("val/balanced_acc", sum(per_class) / len(per_class))
        log_extended_metrics(
            self,
            preds.numpy(),
            labels.numpy(),
            num_classes=self.num_classes,
            prefix="val",
        )
        log_validation_plots_to_wandb(
            self,
            kind="classification",
            y_true=labels.numpy(),
            y_hat=preds.numpy(),
        )

    def on_test_epoch_start(self):
        self._test_preds: list[Tensor] = []
        self._test_labels: list[Tensor] = []

    def test_step(self, batch, batch_idx):
        loss, preds, labels = self._step(batch)
        self.log("test/loss", loss, on_step=False, on_epoch=True)
        self._test_preds.append(preds.cpu())
        self._test_labels.append(labels.cpu())

    def on_test_epoch_end(self):
        preds = torch.cat(self._test_preds)
        labels = torch.cat(self._test_labels)
        self.log("test/acc", (preds == labels).float().mean())
        per_class = [
            (preds[labels == c] == c).float().mean().item()
            for c in range(self.num_classes) if (labels == c).any()
        ]
        if per_class:
            self.log("test/balanced_acc", sum(per_class) / len(per_class))
        log_extended_metrics(
            self,
            preds.numpy(),
            labels.numpy(),
            num_classes=self.num_classes,
            prefix="test",
        )
        for c in range(self.num_classes):
            mask_c = labels == c
            if mask_c.sum() > 0:
                self.log(f"test/acc_class_{c}", (preds[mask_c] == c).float().mean())

    def configure_optimizers(self):
        # Only head parameters are trainable
        opt = optim.AdamW(self.head.parameters(), lr=self.lr)
        sched = optim.lr_scheduler.ExponentialLR(opt, gamma=self.lr_decay)
        return {"optimizer": opt, "lr_scheduler": {"scheduler": sched, "interval": "epoch"}}
