"""
Embedding + regression evaluator.

Flow
----
1. Extract per-sample embeddings from train/val/test splits using
   `wrapper.embed()`.  Cache them to avoid recomputing for the regression
   head's training loop.
2. Train a lightweight MLP on the frozen embeddings to predict the physical
   parameters [amplitude, frequency_hz, phase_rad].  Mirrors the architecture
   of `src/models/mlp.py::MLPRegressor` but takes d_embed as input.
3. Evaluate on the test set with the same metrics the existing regression
   task logs (`src/tasks/toy/toy_regression.py`):
       - per-parameter RMSE
       - SNR-stratified RMSE using the same SNR_BINS
4. Also fit a Ridge regressor as a "linear probe" baseline -- a high linear-probe
   R² means the embedding has linearly-accessible physical information.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from dataloader.toy_dataloader import Param

from .metrics import SNR_BINS, stratified_rmse
from ..wrappers.base import BaseFoundationModel


# ────────────────────────────────────────────────────────────────────────────
# MLP regression head trained on frozen embeddings
# ────────────────────────────────────────────────────────────────────────────

class EmbeddingMLP(nn.Module):
    """Small MLP: [d_embed] -> [256, 128] -> [n_targets]."""

    def __init__(self, d_embed: int, n_targets: int, dropout: float = 0.1) -> None:
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
            nn.Linear(128, n_targets),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


@dataclass
class EmbeddingCache:
    """Precomputed embeddings and targets for a single split."""
    embeddings: np.ndarray      # (N, d_embed)
    targets: np.ndarray         # (N, n_targets)
    snr: np.ndarray             # (N,) ground-truth SNR

    @property
    def d_embed(self) -> int:
        return self.embeddings.shape[-1]


# ────────────────────────────────────────────────────────────────────────────
# Embedding extraction
# ────────────────────────────────────────────────────────────────────────────

def extract_embeddings(
    wrapper: BaseFoundationModel,
    dataloader: DataLoader,
    target_idx: list[int],
    *,
    use_clean_input: bool = False,
    max_batches: int | None = None,
) -> EmbeddingCache:
    """Run the wrapper's embed() on every sample in the loader.

    Parameters
    ----------
    wrapper : loaded BaseFoundationModel
    dataloader : yields (sig_bkg, sig, params)
    target_idx : which columns of params to use as targets, e.g. [0, 1, 2]
    use_clean_input : if True, embed the clean signal (oracle); else embed noisy
    max_batches : smoke-test limit
    """
    embs, targets, snrs = [], [], []
    for batch_idx, (sig_bkg, sig, params) in enumerate(dataloader):
        if max_batches is not None and batch_idx >= max_batches:
            break
        x = sig.numpy() if use_clean_input else sig_bkg.numpy()
        e = wrapper.embed(x).embeddings
        embs.append(e)
        targets.append(params[:, target_idx].numpy())
        snrs.append(params[:, int(Param.snr)].numpy())
    return EmbeddingCache(
        embeddings=np.concatenate(embs, axis=0).astype(np.float32),
        targets=np.concatenate(targets, axis=0).astype(np.float32),
        snr=np.concatenate(snrs, axis=0).astype(np.float32),
    )


# ────────────────────────────────────────────────────────────────────────────
# Head training
# ────────────────────────────────────────────────────────────────────────────

def train_mlp_head(
    train: EmbeddingCache,
    val: EmbeddingCache,
    *,
    epochs: int = 50,
    batch_size: int = 256,
    lr: float = 1e-3,
    lr_decay: float = 0.99,
    patience: int = 10,
    device: str = "cuda",
    seed: int = 42,
) -> EmbeddingMLP:
    """Train an MLP on frozen embeddings with early stopping on val MSE."""
    torch.manual_seed(seed)
    dev = torch.device(device)
    n_targets = train.targets.shape[-1]
    model = EmbeddingMLP(train.d_embed, n_targets).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.ExponentialLR(opt, gamma=lr_decay)

    Xtr = torch.from_numpy(train.embeddings).to(dev)
    ytr = torch.from_numpy(train.targets).to(dev)
    Xv = torch.from_numpy(val.embeddings).to(dev)
    yv = torch.from_numpy(val.targets).to(dev)

    best_val, best_state, no_improve = float("inf"), None, 0
    N = Xtr.shape[0]
    rng = np.random.default_rng(seed)

    for epoch in range(epochs):
        model.train()
        perm = rng.permutation(N)
        for start in range(0, N, batch_size):
            idx = perm[start:start + batch_size]
            opt.zero_grad()
            pred = model(Xtr[idx])
            loss = F.mse_loss(pred, ytr[idx])
            loss.backward()
            opt.step()
        sched.step()

        model.eval()
        with torch.no_grad():
            val_pred = model(Xv)
            val_loss = float(F.mse_loss(val_pred, yv))
        if val_loss < best_val - 1e-6:
            best_val, best_state, no_improve = val_loss, {
                k: v.detach().clone() for k, v in model.state_dict().items()
            }, 0
        else:
            no_improve += 1
            if no_improve >= patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    return model


# ────────────────────────────────────────────────────────────────────────────
# Linear probe (Ridge) -- measures how linearly accessible params are
# ────────────────────────────────────────────────────────────────────────────

def linear_probe_r2(
    train: EmbeddingCache, test: EmbeddingCache, alpha: float = 1.0
) -> np.ndarray:
    """Per-parameter R² of a Ridge regression probe.  Shape (n_targets,)."""
    # Small, numpy-only Ridge: w = (X^T X + alpha I)^-1 X^T y
    X = train.embeddings
    y = train.targets
    XtX = X.T @ X
    reg = alpha * np.eye(X.shape[1], dtype=X.dtype)
    w = np.linalg.solve(XtX + reg, X.T @ y)           # (d, n_targets)
    # Intercept: centre targets
    b = y.mean(axis=0) - X.mean(axis=0) @ w
    pred = test.embeddings @ w + b
    ss_res = np.sum((test.targets - pred) ** 2, axis=0)
    ss_tot = np.sum((test.targets - test.targets.mean(axis=0)) ** 2, axis=0)
    return 1.0 - ss_res / (ss_tot + 1e-12)


# ────────────────────────────────────────────────────────────────────────────
# End-to-end evaluator
# ────────────────────────────────────────────────────────────────────────────

def evaluate_embedding_regression(
    wrapper: BaseFoundationModel,
    *,
    train_loader: DataLoader,
    val_loader: DataLoader,
    test_loader: DataLoader,
    target_params: list[str],
    device: str = "cuda",
    epochs: int = 50,
    max_train_batches: int | None = None,
    max_test_batches: int | None = None,
) -> dict:
    """Extract embeddings, train a regression head, evaluate on test.

    Returns a dict:
      per_param_metrics : {param: {rmse, mae, r2, rmse_snr_low/mid/high, linear_probe_r2}}
      predictions       : {'y_true': (N, P), 'y_pred': (N, P), 'snr': (N,)}
      meta              : {d_embed, n_train, n_test}
    """
    target_idx = [int(Param[p]) for p in target_params]

    train_cache = extract_embeddings(
        wrapper, train_loader, target_idx, max_batches=max_train_batches
    )
    val_cache = extract_embeddings(
        wrapper, val_loader, target_idx, max_batches=max_train_batches
    )
    test_cache = extract_embeddings(
        wrapper, test_loader, target_idx, max_batches=max_test_batches
    )

    head = train_mlp_head(
        train_cache, val_cache, epochs=epochs, device=device
    )

    # Test evaluation
    dev = torch.device(device)
    head.eval()
    with torch.no_grad():
        y_pred = head(torch.from_numpy(test_cache.embeddings).to(dev)).cpu().numpy()
    y_true = test_cache.targets
    snr = test_cache.snr

    # Linear probe R²
    lp_r2 = linear_probe_r2(train_cache, test_cache)

    # Per-parameter metrics
    per_param: dict[str, dict[str, float]] = {}
    diff = y_pred - y_true
    for i, name in enumerate(target_params):
        rmse = float(np.sqrt(np.mean(diff[:, i] ** 2)))
        mae = float(np.mean(np.abs(diff[:, i])))
        ss_res = float(np.sum(diff[:, i] ** 2))
        ss_tot = float(np.sum((y_true[:, i] - y_true[:, i].mean()) ** 2))
        r2 = 1.0 - ss_res / (ss_tot + 1e-12)
        strat = stratified_rmse(y_pred[:, i], y_true[:, i], snr)
        per_param[name] = {
            "rmse": rmse,
            "mae": mae,
            "r2": r2,
            "linear_probe_r2": float(lp_r2[i]),
            **{f"rmse_snr_{k}": v for k, v in strat.items()},
        }

    return {
        "per_param_metrics": per_param,
        "predictions": {"y_true": y_true, "y_pred": y_pred, "snr": snr},
        "meta": {
            "d_embed": int(test_cache.d_embed),
            "n_train": int(train_cache.embeddings.shape[0]),
            "n_test": int(test_cache.embeddings.shape[0]),
        },
    }
