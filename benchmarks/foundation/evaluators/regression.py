"""Zero-shot regression linear probe for foundation models.

Mirror of :mod:`evaluators.classification` but with a Ridge regression head.
Used by ``run_tess_regression.py`` to evaluate FM embeddings against
continuous stellar parameters (e.g. rotation frequency ``frot``).
"""

from __future__ import annotations

import numpy as np
from scipy.stats import pearsonr, spearmanr
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from torch.utils.data import DataLoader

from ..wrappers.base import BaseFoundationModel
from .classification import extract_pooled_embeddings


def evaluate_regression(
    wrapper: BaseFoundationModel,
    *,
    train_loader: DataLoader,
    val_loader: DataLoader,
    test_loader: DataLoader,
    target_name: str = "target",
    win_len: int = 512,
    pool: str = "mean",
    max_train_batches: int | None = None,
    max_test_batches: int | None = None,
    alpha: float = 1.0,
) -> dict:
    """Linear-probe regression on pooled embeddings (Ridge, L2-regularised)."""
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

    # Standardise features (train-only stats).
    mu = X_tr.mean(axis=0, keepdims=True)
    sd = X_tr.std(axis=0, keepdims=True) + 1e-6
    X_tr_n = (X_tr - mu) / sd
    X_va_n = (X_va - mu) / sd
    X_te_n = (X_te - mu) / sd

    # Standardise target with train stats so Ridge regularisation is well-scaled.
    y_mean = float(y_tr.mean())
    y_std = float(y_tr.std() + 1e-8)
    y_tr_n = (y_tr - y_mean) / y_std

    reg = Ridge(alpha=alpha)
    reg.fit(X_tr_n, y_tr_n)
    y_pred_te = reg.predict(X_te_n) * y_std + y_mean
    y_pred_va = reg.predict(X_va_n) * y_std + y_mean

    def _safe_corr(fn, a, b):
        if len(a) < 2 or np.std(a) == 0 or np.std(b) == 0:
            return float("nan")
        return float(fn(a, b)[0])

    metrics = {
        "mse": float(mean_squared_error(y_te, y_pred_te)),
        "mae": float(mean_absolute_error(y_te, y_pred_te)),
        "r2": float(r2_score(y_te, y_pred_te)),
        "pearson_r": _safe_corr(pearsonr, y_te, y_pred_te),
        "spearman_r": _safe_corr(spearmanr, y_te, y_pred_te),
        "val_mse": float(mean_squared_error(y_va, y_pred_va)),
        "val_r2": float(r2_score(y_va, y_pred_va)),
    }
    return {
        "metrics": metrics,
        "target_name": target_name,
        "y_true": y_te, "y_pred": y_pred_te,
        "d_embed": int(X_tr.shape[1]),
        "y_mean": y_mean, "y_std": y_std,
    }
