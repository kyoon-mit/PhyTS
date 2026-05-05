"""Zero-shot classification linear probe for foundation models.

Pipeline
--------
1. Window each full-length time series into fixed-size chunks (default 512)
   covering the whole sequence with the minimum overlap needed.
2. Run `wrapper.embed()` on each window → (B, d_embed) per window.
3. Mean-pool embeddings across windows per sample → (N, d_embed).
4. Fit sklearn LogisticRegression (L2, multinomial) on train embeddings,
   evaluate on test.

Metrics: accuracy, balanced accuracy, macro-F1, per-class F1, confusion matrix.
"""

from __future__ import annotations

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from torch.utils.data import DataLoader

from ..wrappers.base import BaseFoundationModel


def _window_offsets(lc_len: int, win_len: int) -> list[int]:
    """Minimum-count windows of length `win_len` covering [0, lc_len).

    Windows are left-aligned except the last, which is right-aligned so all
    samples are covered. May overlap with the previous window.
    """
    if lc_len <= win_len:
        return [0]
    n = int(np.ceil(lc_len / win_len))
    if n * win_len < lc_len:
        n += 1
    starts = np.linspace(0, lc_len - win_len, n).round().astype(int).tolist()
    # Deduplicate while preserving order
    seen, out = set(), []
    for s in starts:
        if s not in seen:
            out.append(s); seen.add(s)
    return out


def extract_pooled_embeddings(
    wrapper: BaseFoundationModel,
    loader: DataLoader,
    *,
    win_len: int = 512,
    max_batches: int | None = None,
    pool: str = "mean",
) -> tuple[np.ndarray, np.ndarray]:
    """Returns (embeddings [N, d], labels [N]). `pool` is the per-window temporal pool."""
    embs, labels = [], []
    for b, (flux, label) in enumerate(loader):
        if max_batches is not None and b >= max_batches:
            break
        x = flux.numpy().astype(np.float32)               # (B, L)
        offsets = _window_offsets(x.shape[1], win_len)
        win_embs = []
        for s in offsets:
            chunk = x[:, s:s + win_len]
            win_embs.append(wrapper.embed(chunk, pool=pool).embeddings)  # (B, d)
        # mean-pool across windows (separate from per-window temporal pooling)
        pooled = np.stack(win_embs, axis=1).mean(axis=1)
        embs.append(pooled)
        labels.append(label.numpy())
    return np.concatenate(embs, axis=0), np.concatenate(labels, axis=0)


def evaluate_classification(
    wrapper: BaseFoundationModel,
    *,
    train_loader: DataLoader,
    val_loader: DataLoader,
    test_loader: DataLoader,
    class_names: list[str],
    win_len: int = 512,
    pool: str = "mean",
    max_train_batches: int | None = None,
    max_test_batches: int | None = None,
    C: float = 1.0,
) -> dict:
    """Linear probe on pooled embeddings. Returns metrics + embeddings."""
    print(f"  [{wrapper.name}] extracting train embeddings...")
    X_tr, y_tr = extract_pooled_embeddings(
        wrapper, train_loader, win_len=win_len, max_batches=max_train_batches, pool=pool,
    )
    print(f"  [{wrapper.name}] extracting val embeddings...")
    X_va, y_va = extract_pooled_embeddings(
        wrapper, val_loader, win_len=win_len, max_batches=max_train_batches, pool=pool,
    )
    print(f"  [{wrapper.name}] extracting test embeddings...")
    X_te, y_te = extract_pooled_embeddings(
        wrapper, test_loader, win_len=win_len, max_batches=max_test_batches, pool=pool,
    )
    print(f"  [{wrapper.name}] train={X_tr.shape} val={X_va.shape} test={X_te.shape}")

    # Standardise before logistic regression
    mu = X_tr.mean(axis=0, keepdims=True)
    sd = X_tr.std(axis=0, keepdims=True) + 1e-6
    X_tr_n = (X_tr - mu) / sd
    X_va_n = (X_va - mu) / sd
    X_te_n = (X_te - mu) / sd

    clf = LogisticRegression(
        C=C, penalty="l2", solver="lbfgs", max_iter=2000,
        multi_class="multinomial", n_jobs=-1,
    )
    clf.fit(X_tr_n, y_tr)
    y_pred = clf.predict(X_te_n)
    y_pred_val = clf.predict(X_va_n)

    metrics = {
        "accuracy": float(accuracy_score(y_te, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_te, y_pred)),
        "macro_f1": float(f1_score(y_te, y_pred, average="macro")),
        "val_accuracy": float(accuracy_score(y_va, y_pred_val)),
        "val_balanced_accuracy": float(balanced_accuracy_score(y_va, y_pred_val)),
    }
    cm = confusion_matrix(y_te, y_pred, labels=list(range(len(class_names))))
    report = classification_report(
        y_te, y_pred, labels=list(range(len(class_names))),
        target_names=class_names, digits=3, zero_division=0, output_dict=True,
    )
    return {
        "metrics": metrics,
        "confusion_matrix": cm,
        "report": report,
        "d_embed": int(X_tr.shape[1]),
        "y_true": y_te, "y_pred": y_pred,
    }
