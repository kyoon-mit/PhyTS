"""Multiclass classification metrics for TESS (NumPy).

Used by Lightning tasks and offline checkpoint evaluation. Macro averages are
over **all** ``num_classes`` (absent classes contribute zero precision/recall).

Weighted averages use support (row sums of the confusion matrix) as class weights.
"""

from __future__ import annotations

from typing import Any

import numpy as np


def multiclass_extended_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    *,
    num_classes: int,
) -> dict[str, float]:
    """Return macro/weighted precision, recall, and F1 (no accuracy — keep Lightning as-is).

    Parameters
    ----------
    y_true, y_pred
        Integer class indices in ``[0, num_classes)``.
    num_classes
        Full class count (including empty classes in the eval split).

    Returns
    -------
    dict
        Keys: ``precision_macro``, ``recall_macro``, ``f1_macro``,
        ``precision_weighted``, ``recall_weighted``, ``f1_weighted``.
    """
    y_true = np.asarray(y_true, dtype=np.int64).ravel()
    y_pred = np.asarray(y_pred, dtype=np.int64).ravel()
    if y_true.shape != y_pred.shape:
        raise ValueError("y_true and y_pred must have the same shape")

    idx = y_true * num_classes + y_pred
    cm_flat = np.bincount(idx, minlength=num_classes * num_classes)
    cm = cm_flat.reshape(num_classes, num_classes).astype(np.float64)

    prec_c = np.zeros(num_classes, dtype=np.float64)
    rec_c = np.zeros(num_classes, dtype=np.float64)
    f1_c = np.zeros(num_classes, dtype=np.float64)
    support = cm.sum(axis=1)

    for c in range(num_classes):
        tp = cm[c, c]
        fp = cm[:, c].sum() - tp
        fn = cm[c, :].sum() - tp
        p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        r = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        prec_c[c] = p
        rec_c[c] = r
        f1_c[c] = (2 * p * r / (p + r)) if (p + r) > 0 else 0.0

    total = float(support.sum())
    if total <= 0:
        raise ValueError("No samples in y_true")

    macro_p = float(prec_c.mean())
    macro_r = float(rec_c.mean())
    macro_f1 = float(f1_c.mean())

    w = support / total
    weighted_p = float((w * prec_c).sum())
    weighted_r = float((w * rec_c).sum())
    weighted_f1 = float((w * f1_c).sum())

    return {
        "precision_macro": macro_p,
        "recall_macro": macro_r,
        "f1_macro": macro_f1,
        "precision_weighted": weighted_p,
        "recall_weighted": weighted_r,
        "f1_weighted": weighted_f1,
    }


def log_extended_metrics(
    module: Any,
    preds_np: np.ndarray,
    labels_np: np.ndarray,
    *,
    num_classes: int,
    prefix: str,
) -> None:
    """``module.log(f\"{prefix}/precision_macro\", ...)`` for each extended metric."""
    ext = multiclass_extended_metrics(
        labels_np, preds_np, num_classes=num_classes
    )
    for k, v in ext.items():
        module.log(f"{prefix}/{k}", float(v))
