"""Regression tasks for TESS lightcurve data.

TESSRegressionMSE — end-to-end: flux → frot (rotation frequency).
TESSFrozenBackboneRegressionMSE — frozen seq2seq backbone + trainable MLP head.

Batch convention (from TESSRegressionDataset.__getitem__):
    flux  (B, L)   — z-score normalized, padded to seq_len
    mask  (B, L)   — True where cadence is valid
    frot  (B,)     — rotation frequency target

Any model with signature forward(x: (B, L, 1), mask: (B, L)) -> (B, 1) works as drop-in.

Usage (LightningCLI YAML):
    model:
      class_path: tasks.TESS.tess_regression.TESSRegressionMSE
      init_args:
        lr: 1.0e-3
        lr_decay: 0.99
        model:
          class_path: models.mlp.MLPRegressor
          init_args:
            seq_len: 1100
            d_output: 1
"""

import importlib

import torch
import torch.nn as nn
from torch import Tensor, optim
import lightning as L
import yaml

from tasks.param_count import (
    attach_scalar_hyperparams,
    torch_model_parameter_hyper_dict,
    torch_module_bundle_prefixed,
    torch_nn_parameter_count,
)
from tasks.TESS.eval_plots import log_test_plots_to_wandb, log_validation_plots_to_wandb


def _load_seq2seq_backbone(ckpt_path: str, cfg_path: str) -> nn.Module:
    """Instantiate a seq2seq model from a LightningCLI YAML and load its weights."""
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


# ── End-to-end regression ────────────────────────────────────────────────────

class TESSRegressionMSE(L.LightningModule):
    """End-to-end regression: forward(flux) → frot via MSE loss."""

    def __init__(
        self,
        model: nn.Module,
        lr: float = 1e-3,
        lr_decay: float = 0.99,
    ):
        super().__init__()
        self.model = model
        self.lr = lr
        self.lr_decay = lr_decay
        self.criterion = nn.MSELoss()
        self.save_hyperparameters(ignore=["model"])
        attach_scalar_hyperparams(self, torch_model_parameter_hyper_dict(model))

    def forward(self, x: Tensor, mask: Tensor | None = None) -> Tensor:
        # x: (B, L) → (B, L, 1) → model → (B, 1) → (B,)
        return self.model(x.unsqueeze(-1), mask=mask).squeeze(-1)

    def _step(self, batch: tuple) -> tuple[Tensor, Tensor, Tensor]:
        flux, mask, frot = batch
        y_hat = self(flux, mask)
        return self.criterion(y_hat, frot), y_hat, frot

    def on_train_epoch_start(self):
        self._train_preds: list[Tensor] = []
        self._train_labels: list[Tensor] = []

    def training_step(self, batch, batch_idx):
        loss, y_hat, y = self._step(batch)
        self.log("train/loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        self._train_preds.append(y_hat.detach().cpu())
        self._train_labels.append(y.detach().cpu())
        return loss

    def on_train_epoch_end(self):
        y_hat = torch.cat(self._train_preds)
        y = torch.cat(self._train_labels)
        self.log("train/rmse", (y_hat - y).pow(2).mean().sqrt())

    def on_validation_epoch_start(self):
        self._val_preds: list[Tensor] = []
        self._val_labels: list[Tensor] = []

    def validation_step(self, batch, batch_idx):
        loss, y_hat, y = self._step(batch)
        self.log("val/loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        self._val_preds.append(y_hat.detach().cpu())
        self._val_labels.append(y.detach().cpu())

    def on_validation_epoch_end(self):
        y_hat = torch.cat(self._val_preds)
        y = torch.cat(self._val_labels)
        ss_res = (y_hat - y).pow(2).sum()
        ss_tot = (y - y.mean()).pow(2).sum().clamp(min=1e-8)
        self.log("val/r2", 1.0 - ss_res / ss_tot)
        self.log("val/rmse", (y_hat - y).pow(2).mean().sqrt())
        self.log("val/mae", (y_hat - y).abs().mean())
        log_validation_plots_to_wandb(
            self,
            kind="regression",
            y_true=y.numpy(),
            y_hat=y_hat.numpy(),
        )

    def on_test_epoch_start(self):
        self._test_preds: list[Tensor] = []
        self._test_labels: list[Tensor] = []

    def test_step(self, batch, batch_idx):
        loss, y_hat, y = self._step(batch)
        self.log("test/loss", loss, on_step=False, on_epoch=True)
        self._test_preds.append(y_hat.detach().cpu())
        self._test_labels.append(y.detach().cpu())

    def on_test_epoch_end(self):
        y_hat = torch.cat(self._test_preds)
        y = torch.cat(self._test_labels)
        ss_res = (y_hat - y).pow(2).sum()
        ss_tot = (y - y.mean()).pow(2).sum().clamp(min=1e-8)
        self.log("test/r2", 1.0 - ss_res / ss_tot)
        self.log("test/rmse", (y_hat - y).pow(2).mean().sqrt())
        self.log("test/mae", (y_hat - y).abs().mean())
        log_test_plots_to_wandb(
            self,
            kind="regression",
            y_true=y.numpy(),
            y_hat=y_hat.numpy(),
        )

    def configure_optimizers(self):
        opt = optim.AdamW(self.parameters(), lr=self.lr)
        sched = optim.lr_scheduler.ExponentialLR(opt, gamma=self.lr_decay)
        return {"optimizer": opt, "lr_scheduler": {"scheduler": sched, "interval": "epoch"}}


# ── Frozen-backbone regression ───────────────────────────────────────────────

class TESSFrozenBackboneRegressionMSE(L.LightningModule):
    """Frozen seq2seq backbone + trainable MLP regression head.

    The backbone (pretrained seq2seq model, e.g. S4ModelSeq2Seq) is loaded
    from a LightningCLI checkpoint and frozen. Only the head is trained.

    Forward: flux → backbone(flux) → reconstructed_flux → head → frot

    Usage (LightningCLI YAML):
        model:
          class_path: tasks.TESS.tess_regression.TESSFrozenBackboneRegressionMSE
          init_args:
            backbone_ckpt: checkpoints/tess_s4d_reconstruction/best.ckpt
            backbone_cfg: configs/TESS/other/train_tess_s4d_reconstruction.yaml
            lr: 1.0e-3
            lr_decay: 0.99
            head:
              class_path: models.mlp.MLPRegressor
              init_args:
                seq_len: 1100
                d_output: 1
    """

    def __init__(
        self,
        backbone_ckpt: str,
        backbone_cfg: str,
        head: nn.Module,
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
        self.lr = lr
        self.lr_decay = lr_decay
        self.criterion = nn.MSELoss()
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
        # x: (B, L) → backbone → (B, L, 1) → (optionally mask) → head → (B, 1) → (B,)
        with torch.no_grad():
            reconstructed = self.backbone(x.unsqueeze(-1))  # (B, L, 1)
        if mask is not None:
            reconstructed = reconstructed * mask.unsqueeze(-1).float()
        return self.head(reconstructed, mask=mask).squeeze(-1)

    def _step(self, batch: tuple) -> tuple[Tensor, Tensor, Tensor]:
        flux, mask, frot = batch
        y_hat = self(flux, mask)
        return self.criterion(y_hat, frot), y_hat, frot

    def on_train_epoch_start(self):
        self._train_preds: list[Tensor] = []
        self._train_labels: list[Tensor] = []

    def training_step(self, batch, batch_idx):
        loss, y_hat, y = self._step(batch)
        self.log("train/loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        self._train_preds.append(y_hat.detach().cpu())
        self._train_labels.append(y.detach().cpu())
        return loss

    def on_train_epoch_end(self):
        y_hat = torch.cat(self._train_preds)
        y = torch.cat(self._train_labels)
        self.log("train/rmse", (y_hat - y).pow(2).mean().sqrt())

    def on_validation_epoch_start(self):
        self._val_preds: list[Tensor] = []
        self._val_labels: list[Tensor] = []

    def validation_step(self, batch, batch_idx):
        loss, y_hat, y = self._step(batch)
        self.log("val/loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        self._val_preds.append(y_hat.detach().cpu())
        self._val_labels.append(y.detach().cpu())

    def on_validation_epoch_end(self):
        y_hat = torch.cat(self._val_preds)
        y = torch.cat(self._val_labels)
        ss_res = (y_hat - y).pow(2).sum()
        ss_tot = (y - y.mean()).pow(2).sum().clamp(min=1e-8)
        self.log("val/r2", 1.0 - ss_res / ss_tot)
        self.log("val/rmse", (y_hat - y).pow(2).mean().sqrt())
        self.log("val/mae", (y_hat - y).abs().mean())
        log_validation_plots_to_wandb(
            self,
            kind="regression",
            y_true=y.numpy(),
            y_hat=y_hat.numpy(),
        )

    def on_test_epoch_start(self):
        self._test_preds: list[Tensor] = []
        self._test_labels: list[Tensor] = []

    def test_step(self, batch, batch_idx):
        loss, y_hat, y = self._step(batch)
        self.log("test/loss", loss, on_step=False, on_epoch=True)
        self._test_preds.append(y_hat.detach().cpu())
        self._test_labels.append(y.detach().cpu())

    def on_test_epoch_end(self):
        y_hat = torch.cat(self._test_preds)
        y = torch.cat(self._test_labels)
        ss_res = (y_hat - y).pow(2).sum()
        ss_tot = (y - y.mean()).pow(2).sum().clamp(min=1e-8)
        self.log("test/r2", 1.0 - ss_res / ss_tot)
        self.log("test/rmse", (y_hat - y).pow(2).mean().sqrt())
        self.log("test/mae", (y_hat - y).abs().mean())
        log_test_plots_to_wandb(
            self,
            kind="regression",
            y_true=y.numpy(),
            y_hat=y_hat.numpy(),
        )

    def configure_optimizers(self):
        # Only head parameters are trainable
        opt = optim.AdamW(self.head.parameters(), lr=self.lr)
        sched = optim.lr_scheduler.ExponentialLR(opt, gamma=self.lr_decay)
        return {"optimizer": opt, "lr_scheduler": {"scheduler": sched, "interval": "epoch"}}
