"""LoRA-fine-tuned regression probe for foundation models.

Mirror of :mod:`evaluators.classification_finetune` with a 1-d regression
head (Smooth-L1 / Huber loss) and best-val-MSE checkpoint selection.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import torch
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from torch import nn
from torch.utils.data import DataLoader

from ..finetuning.finetune import (
    LoRALinear,
    _find_backbone,
    _unfreeze_last_n,
    apply_lora,
)
from ..wrappers.base import BaseFoundationModel
from .classification_finetune import _embed_pooled, _get_backbone


class _RegressionHead(nn.Module):
    """MLP regression head (d_embed -> 256 -> 128 -> 1)."""

    def __init__(self, d_embed: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_embed, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(256, 128),
            nn.LayerNorm(128),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(128, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)  # (B,)


def finetune_regression(
    wrapper: BaseFoundationModel,
    *,
    train_loader: DataLoader,
    val_loader: DataLoader,
    test_loader: DataLoader,
    target_name: str = "target",
    target_mean: float = 0.0,
    target_std: float = 1.0,
    win_len: int = 512,
    pool: str = "mean",
    device: str = "cuda",
    epochs: int = 10,
    head_lr: float = 1e-3,
    backbone_lr: float = 1e-4,
    adapter: str = "lora",
    unfreeze_last_n: int = 2,
    lora_r: int = 8,
    lora_alpha: int = 16,
    lora_dropout: float = 0.0,
    lora_last_n_blocks: Optional[int] = None,
    weight_decay: float = 1e-4,
    loss: str = "huber",                 # "huber" | "mse"
    huber_beta: float = 1.0,
    patience: Optional[int] = None,
    max_train_batches: Optional[int] = None,
    max_test_batches: Optional[int] = None,
    seed: int = 42,
    verbose: bool = True,
) -> dict:
    """LoRA / last-N-blocks fine-tuning for scalar regression."""
    if not wrapper.supports_embed:
        raise ValueError(f"{wrapper.name} does not support embed.")

    torch.manual_seed(seed)
    np.random.seed(seed)

    # ── Adapt the backbone ──────────────────────────────────────────────
    backbone = _get_backbone(wrapper)
    trainable_backbone: list[nn.Parameter] = []

    if backbone is None:
        if verbose:
            print(f"  [finetune-reg] {wrapper.name}: no backbone discovered — head-only.")
    elif adapter == "frozen":
        for p in backbone.parameters():
            p.requires_grad_(False)
    elif adapter == "last_n":
        trainable_backbone = _unfreeze_last_n(backbone, unfreeze_last_n)
    elif adapter == "full":
        for p in backbone.parameters():
            p.requires_grad_(True)
        trainable_backbone = [p for p in backbone.parameters() if p.requires_grad]
    elif adapter == "lora":
        trainable_backbone = apply_lora(
            backbone, r=lora_r, alpha=lora_alpha, dropout=lora_dropout,
            last_n_blocks=lora_last_n_blocks,
        )
        try:
            bb_device = next(backbone.parameters()).device
            backbone.to(bb_device)
        except StopIteration:
            pass
        if verbose:
            n_mod = sum(1 for m in backbone.modules() if isinstance(m, LoRALinear))
            n = sum(p.numel() for p in trainable_backbone)
            print(f"  [finetune-reg] {wrapper.name}: LoRA r={lora_r} α={lora_alpha} into "
                  f"{n_mod} modules, {n / 1e6:.3f}M trainable params.")
    else:
        raise ValueError(f"Unknown adapter '{adapter}'")

    # ── Infer d_embed from one batch ────────────────────────────────────
    first_flux, _ = next(iter(train_loader))
    with torch.no_grad():
        flux_np = first_flux[:1].numpy().astype(np.float32)
        d_embed = _embed_pooled(wrapper, flux_np, win_len=win_len, pool=pool).shape[-1]
    if verbose:
        print(f"  [finetune-reg] {wrapper.name}: d_embed={d_embed}")

    head = _RegressionHead(d_embed).to(device)
    param_groups = [{"params": list(head.parameters()), "lr": head_lr}]
    if trainable_backbone:
        param_groups.append({"params": trainable_backbone, "lr": backbone_lr})
    optim = torch.optim.AdamW(param_groups, weight_decay=weight_decay)

    if loss == "mse":
        loss_fn = nn.MSELoss()
    elif loss == "huber":
        loss_fn = nn.SmoothL1Loss(beta=huber_beta)
    else:
        raise ValueError(f"Unknown loss '{loss}'")

    def _normalise(y: torch.Tensor) -> torch.Tensor:
        return (y - target_mean) / target_std

    def _denormalise(y: torch.Tensor) -> torch.Tensor:
        return y * target_std + target_mean

    def _run_forward(flux_tensor: torch.Tensor) -> torch.Tensor:
        flux_np = flux_tensor.numpy().astype(np.float32)
        emb = _embed_pooled(wrapper, flux_np, win_len=win_len, pool=pool)
        return head(emb)  # predicts on the normalised scale

    # ── Train / validate ────────────────────────────────────────────────
    best_val_mse = float("inf")
    best_epoch = 0
    epochs_since_best = 0
    best_head_state = {k: v.detach().clone() for k, v in head.state_dict().items()}
    best_backbone_state: Optional[list[torch.Tensor]] = (
        [p.detach().clone() for p in trainable_backbone] if trainable_backbone else None
    )

    history: list[dict] = []
    for ep in range(epochs):
        head.train()
        if backbone is not None:
            backbone.eval()

        tr_losses: list[float] = []
        for bi, (flux, target) in enumerate(train_loader):
            if max_train_batches is not None and bi >= max_train_batches:
                break
            y = target.to(device).float()
            y_norm = _normalise(y)
            pred = _run_forward(flux)
            loss_val = loss_fn(pred, y_norm)
            optim.zero_grad()
            loss_val.backward()
            optim.step()
            tr_losses.append(float(loss_val.item()))

        # Validation (MSE on the original scale for selection).
        head.eval()
        if backbone is not None:
            backbone.eval()
        val_sse = 0.0
        val_n = 0
        with torch.no_grad():
            for flux, target in val_loader:
                y = target.to(device).float()
                pred = _denormalise(_run_forward(flux))
                val_sse += float(((pred - y) ** 2).sum().item())
                val_n += int(y.numel())
        val_mse = val_sse / max(val_n, 1)
        tl = float(np.mean(tr_losses)) if tr_losses else 0.0
        if verbose:
            print(f"  [finetune-reg] epoch {ep + 1}/{epochs}  "
                  f"train_loss={tl:.4f}  val_mse={val_mse:.4f}")
        history.append({"epoch": ep + 1, "train_loss": tl, "val_mse": val_mse})

        if val_mse < best_val_mse:
            best_val_mse = val_mse
            best_epoch = ep + 1
            epochs_since_best = 0
            best_head_state = {k: v.detach().clone() for k, v in head.state_dict().items()}
            if trainable_backbone:
                best_backbone_state = [p.detach().clone() for p in trainable_backbone]
        else:
            epochs_since_best += 1

        if patience is not None and epochs_since_best >= patience:
            if verbose:
                print(f"  [finetune-reg] early stop at epoch {ep + 1}: "
                      f"no val improvement for {patience} epochs "
                      f"(best val_mse={best_val_mse:.4f} at epoch {best_epoch}).")
            break

    # Restore best
    head.load_state_dict(best_head_state)
    if best_backbone_state is not None:
        with torch.no_grad():
            for p, saved in zip(trainable_backbone, best_backbone_state):
                p.copy_(saved)

    # Test
    head.eval()
    if backbone is not None:
        backbone.eval()
    y_true_chunks, y_pred_chunks = [], []
    with torch.no_grad():
        for bi, (flux, target) in enumerate(test_loader):
            if max_test_batches is not None and bi >= max_test_batches:
                break
            pred = _denormalise(_run_forward(flux)).cpu().numpy()
            y_pred_chunks.append(pred)
            y_true_chunks.append(target.numpy())
    y_true = np.concatenate(y_true_chunks) if y_true_chunks else np.zeros(0, dtype=np.float32)
    y_pred = np.concatenate(y_pred_chunks) if y_pred_chunks else np.zeros(0, dtype=np.float32)

    def _safe_corr(fn, a, b):
        if len(a) < 2 or np.std(a) == 0 or np.std(b) == 0:
            return float("nan")
        return float(fn(a, b)[0])

    metrics = {
        "mse": float(mean_squared_error(y_true, y_pred)) if len(y_true) else 0.0,
        "mae": float(mean_absolute_error(y_true, y_pred)) if len(y_true) else 0.0,
        "r2": float(r2_score(y_true, y_pred)) if len(y_true) >= 2 else 0.0,
        "pearson_r": _safe_corr(pearsonr, y_true, y_pred),
        "spearman_r": _safe_corr(spearmanr, y_true, y_pred),
        "val_mse": float(best_val_mse),
    }
    return {
        "metrics": metrics,
        "target_name": target_name,
        "y_true": y_true, "y_pred": y_pred,
        "d_embed": int(d_embed),
        "history": history,
    }
