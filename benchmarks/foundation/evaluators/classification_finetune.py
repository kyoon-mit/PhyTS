"""LoRA-fine-tuned classification probe for foundation models.

Pipeline
--------
1. Inject LoRA adapters (or unfreeze last N blocks) into the wrapper's
   fine-tuning backbone, via :func:`apply_lora` / :func:`_unfreeze_last_n`.
2. Train LoRA params + an MLP classification head jointly on the training
   set.  Each light curve is sliced into fixed-length windows, embedded via
   ``wrapper.embed_torch``, pooled across windows, and classified.
3. Evaluate on the validation set after every epoch; keep the checkpoint
   with the best validation accuracy.
4. Report test-set metrics (accuracy, balanced accuracy, macro-F1,
   per-class F1, confusion matrix) using the best-val checkpoint.

Contract with the wrapper
-------------------------
The wrapper must expose:

  * ``embed_torch(signal, pool=...)``     — grad-preserving embedding.
  * ``get_finetune_backbone()``            — nn.Module to adapt.

Both are defined on :class:`BaseFoundationModel`; wrappers that don't
override them raise NotImplementedError and are skipped by the runner.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)

from ..wrappers.base import BaseFoundationModel
from ..finetuning.finetune import (
    LoRALinear,
    _find_backbone,
    _unfreeze_last_n,
    apply_lora,
)


def _window_offsets(lc_len: int, win_len: int) -> list[int]:
    """Minimum windows of length ``win_len`` that cover ``[0, lc_len)``.

    Mirrors ``evaluators.classification._window_offsets`` so the
    fine-tuned probe operates over the same windowing as zero-shot.
    """
    if lc_len <= win_len:
        return [0]
    n = int(np.ceil(lc_len / win_len))
    if n * win_len < lc_len:
        n += 1
    starts = np.linspace(0, lc_len - win_len, n).round().astype(int).tolist()
    seen, out = set(), []
    for s in starts:
        if s not in seen:
            out.append(s); seen.add(s)
    return out


class _ClassificationHead(nn.Module):
    """MLP classification head (d_embed -> 256 -> 128 -> n_classes)."""

    def __init__(self, d_embed: int, n_classes: int, dropout: float = 0.1) -> None:
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
            nn.Linear(128, n_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def _get_backbone(wrapper) -> Optional[nn.Module]:
    fn = getattr(wrapper, "get_finetune_backbone", None)
    if fn is not None:
        try:
            bb = fn()
            if isinstance(bb, nn.Module):
                return bb
        except Exception:
            pass
    return _find_backbone(wrapper)


def _embed_pooled(
    wrapper: BaseFoundationModel,
    flux_np: np.ndarray,
    *,
    win_len: int,
    pool: str,
) -> torch.Tensor:
    """Embed each window and mean-pool across windows.  Preserves grads."""
    offsets = _window_offsets(flux_np.shape[1], win_len)
    win_embs = []
    for s in offsets:
        chunk = flux_np[:, s:s + win_len]
        win_embs.append(wrapper.embed_torch(chunk, pool=pool))    # (B, d)
    if len(win_embs) == 1:
        return win_embs[0]
    return torch.stack(win_embs, dim=1).mean(dim=1)               # (B, d)


def finetune_classification(
    wrapper: BaseFoundationModel,
    *,
    train_loader: DataLoader,
    val_loader: DataLoader,
    test_loader: DataLoader,
    class_names: list[str],
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
    class_weights: Optional[str] = None,       # None | "balanced"
    patience: Optional[int] = None,            # early-stop after N epochs w/o val improvement
    max_train_batches: Optional[int] = None,
    max_test_batches: Optional[int] = None,
    seed: int = 42,
    verbose: bool = True,
) -> dict:
    """LoRA / last-N-blocks fine-tuning for classification.

    Returns a dict matching the zero-shot evaluator's schema plus a
    per-epoch ``history`` list, so the results can be dropped into the
    same ``metrics.json`` layout and rendered by ``make_kepler_table.py``.
    """
    if not wrapper.supports_embed:
        raise ValueError(f"{wrapper.name} does not support embed.")

    torch.manual_seed(seed)
    np.random.seed(seed)

    # ── Adapt the backbone ──────────────────────────────────────────────
    backbone = _get_backbone(wrapper)
    trainable_backbone: list[nn.Parameter] = []

    if backbone is None:
        if verbose:
            print(f"  [finetune-cls] {wrapper.name}: no backbone discovered — head-only.")
    elif adapter == "frozen":
        for p in backbone.parameters():
            p.requires_grad_(False)
        if verbose:
            print(f"  [finetune-cls] {wrapper.name}: backbone frozen (head-only baseline).")
    elif adapter == "last_n":
        trainable_backbone = _unfreeze_last_n(backbone, unfreeze_last_n)
        if verbose:
            n = sum(p.numel() for p in trainable_backbone)
            print(f"  [finetune-cls] {wrapper.name}: unfroze last {unfreeze_last_n} blocks "
                  f"({n / 1e6:.2f}M params).")
    elif adapter == "full":
        for p in backbone.parameters():
            p.requires_grad_(True)
        trainable_backbone = [p for p in backbone.parameters() if p.requires_grad]
        if verbose:
            n = sum(p.numel() for p in trainable_backbone)
            print(f"  [finetune-cls] {wrapper.name}: full fine-tune "
                  f"({n / 1e6:.2f}M backbone params trainable).")
    elif adapter == "lora":
        trainable_backbone = apply_lora(
            backbone,
            r=lora_r, alpha=lora_alpha, dropout=lora_dropout,
            last_n_blocks=lora_last_n_blocks,
        )
        # LoRA params are created on CPU; move to the backbone's device.
        try:
            bb_device = next(backbone.parameters()).device
            backbone.to(bb_device)
        except StopIteration:
            pass
        n_mod = sum(1 for m in backbone.modules() if isinstance(m, LoRALinear))
        n = sum(p.numel() for p in trainable_backbone)
        if verbose:
            scope = (f"last {lora_last_n_blocks} blocks"
                     if lora_last_n_blocks else "whole backbone")
            print(f"  [finetune-cls] {wrapper.name}: LoRA r={lora_r} α={lora_alpha} "
                  f"injected into {n_mod} modules ({scope}), {n / 1e6:.3f}M trainable params.")
            if n_mod == 0:
                print(f"  [finetune-cls] WARNING: no LoRA targets matched the default "
                      "attention-projection patterns — effectively head-only training. "
                      "Consider adapter='last_n' for this model.")
    else:
        raise ValueError(f"Unknown adapter '{adapter}'")

    # ── Infer d_embed from one batch ────────────────────────────────────
    first_flux, _ = next(iter(train_loader))
    with torch.no_grad():
        flux_np = first_flux[:1].numpy().astype(np.float32)
        d_embed = _embed_pooled(wrapper, flux_np, win_len=win_len, pool=pool).shape[-1]
    if verbose:
        print(f"  [finetune-cls] {wrapper.name}: d_embed={d_embed}")

    n_classes = len(class_names)
    head = _ClassificationHead(d_embed, n_classes).to(device)

    param_groups = [{"params": list(head.parameters()), "lr": head_lr}]
    if trainable_backbone:
        param_groups.append({"params": trainable_backbone, "lr": backbone_lr})
    optim = torch.optim.AdamW(param_groups, weight_decay=weight_decay)

    # Optional: inverse-frequency class weights, to counter imbalance in
    # balanced_accuracy / per-class F1 on low-support classes.
    ce_weight: Optional[torch.Tensor] = None
    if class_weights == "balanced":
        label_counts = np.zeros(n_classes, dtype=np.int64)
        for bi, (_flux, label) in enumerate(train_loader):
            if max_train_batches is not None and bi >= max_train_batches:
                break
            y = label.numpy().astype(np.int64)
            for c in range(n_classes):
                label_counts[c] += int((y == c).sum())
        total = int(label_counts.sum())
        # Inverse-frequency, normalised to mean 1 so overall loss magnitude
        # stays comparable to unweighted CE.
        freq = np.maximum(label_counts, 1) / max(total, 1)
        w = 1.0 / freq
        w = w / w.mean()
        ce_weight = torch.tensor(w, dtype=torch.float32, device=device)
        if verbose:
            print(f"  [finetune-cls] class weights (balanced): "
                  f"{[round(float(x), 2) for x in w]}")
    elif class_weights not in (None, "none"):
        raise ValueError(f"Unknown class_weights='{class_weights}'")

    loss_fn = nn.CrossEntropyLoss(weight=ce_weight)

    def _run_forward(flux_tensor: torch.Tensor) -> torch.Tensor:
        flux_np = flux_tensor.numpy().astype(np.float32)
        emb = _embed_pooled(wrapper, flux_np, win_len=win_len, pool=pool)
        return head(emb)

    # ── Train / validate ────────────────────────────────────────────────
    best_val_acc = -1.0
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
            # Keep pretrained dropout / LayerNorm in eval mode so the
            # frozen base's running stats and stochastic layers don't
            # destabilise training.  LoRA params still receive gradients.
            backbone.eval()

        tr_losses: list[float] = []
        for bi, (flux, label) in enumerate(train_loader):
            if max_train_batches is not None and bi >= max_train_batches:
                break
            y = label.to(device).long()
            logits = _run_forward(flux)
            loss = loss_fn(logits, y)

            optim.zero_grad()
            loss.backward()
            optim.step()
            tr_losses.append(float(loss.item()))

        # Validation
        head.eval()
        if backbone is not None:
            backbone.eval()
        val_losses: list[float] = []
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for flux, label in val_loader:
                y = label.to(device).long()
                logits = _run_forward(flux)
                val_losses.append(float(loss_fn(logits, y).item()))
                val_correct += int((logits.argmax(dim=1) == y).sum().item())
                val_total += int(y.numel())
        val_acc = val_correct / max(val_total, 1)
        tl = float(np.mean(tr_losses)) if tr_losses else 0.0
        vl = float(np.mean(val_losses)) if val_losses else 0.0
        if verbose:
            print(f"  [finetune-cls] epoch {ep + 1}/{epochs}  "
                  f"train_loss={tl:.4f}  val_loss={vl:.4f}  val_acc={val_acc:.4f}")
        history.append({"epoch": ep + 1, "train_loss": tl, "val_loss": vl, "val_acc": val_acc})

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch = ep + 1
            epochs_since_best = 0
            best_head_state = {k: v.detach().clone() for k, v in head.state_dict().items()}
            if trainable_backbone:
                best_backbone_state = [p.detach().clone() for p in trainable_backbone]
        else:
            epochs_since_best += 1

        if patience is not None and epochs_since_best >= patience:
            if verbose:
                print(f"  [finetune-cls] early stop at epoch {ep + 1}: "
                      f"no val improvement for {patience} epochs "
                      f"(best val_acc={best_val_acc:.4f} at epoch {best_epoch}).")
            break

    # ── Restore best checkpoint ─────────────────────────────────────────
    head.load_state_dict(best_head_state)
    if best_backbone_state is not None:
        with torch.no_grad():
            for p, saved in zip(trainable_backbone, best_backbone_state):
                p.copy_(saved)

    # ── Test ────────────────────────────────────────────────────────────
    head.eval()
    if backbone is not None:
        backbone.eval()
    y_true_chunks, y_pred_chunks = [], []
    with torch.no_grad():
        for bi, (flux, label) in enumerate(test_loader):
            if max_test_batches is not None and bi >= max_test_batches:
                break
            logits = _run_forward(flux)
            y_pred_chunks.append(logits.argmax(dim=1).cpu().numpy())
            y_true_chunks.append(label.numpy())
    y_true = np.concatenate(y_true_chunks) if y_true_chunks else np.zeros(0, dtype=int)
    y_pred = np.concatenate(y_pred_chunks) if y_pred_chunks else np.zeros(0, dtype=int)

    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)) if len(y_true) else 0.0,
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)) if len(y_true) else 0.0,
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)) if len(y_true) else 0.0,
        "val_accuracy": float(best_val_acc),
    }
    cm = confusion_matrix(y_true, y_pred, labels=list(range(n_classes))) if len(y_true) else np.zeros((n_classes, n_classes), dtype=int)
    report = (
        classification_report(
            y_true, y_pred, labels=list(range(n_classes)),
            target_names=class_names, digits=3, zero_division=0, output_dict=True,
        ) if len(y_true) else {c: {"precision": 0.0, "recall": 0.0, "f1-score": 0.0, "support": 0} for c in class_names}
    )
    return {
        "metrics": metrics,
        "confusion_matrix": cm,
        "report": report,
        "d_embed": int(d_embed),
        "y_true": y_true,
        "y_pred": y_pred,
        "history": history,
    }
