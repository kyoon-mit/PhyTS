"""Zero-shot regression on Project8 (energy from I/Q traces) via FM linear probe.

Usage
-----
    python benchmarks/foundation/run_project8.py \
        --models granite_ttm timemoe \
        --out_dir plots/Project8/linear_probe

The Project 8 batches are (B, cutoff, C=2) with channels [output_ts_I, output_ts_Q]
and one regression target (z-scored ``energy``).  Each channel is windowed and
embedded separately by the foundation model (which expects univariate (B, L) input);
the per-channel, per-window embeddings are mean-pooled into a single (N, d_embed*C)
feature vector for a Ridge linear-probe + small MLP head.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

_HERE = Path(__file__).resolve().parent
_REPO = _HERE.parent.parent
for p in (str(_REPO), str(_HERE.parent), str(_REPO / "src")):
    if p not in sys.path:
        sys.path.insert(0, p)

from dataloader.project8_dataloader import Project8DataModule

# Match run_kepler.py per-model defaults: encoders→mean, causal decoders→last.
_DEFAULT_POOL = {
    "moment":      "mean",
    "chronos":     "mean",
    "moirai":      "mean",
    "granite_ttm": "mean",
    "timemoe":     "last",
    "timesfm":     "last",
    "lagllama":    "last",
}


def _get_wrapper(name: str):
    name = name.lower()
    if name == "moment":
        from foundation.wrappers.moment_wrapper import MomentWrapper
        return MomentWrapper()
    if name == "chronos":
        from foundation.wrappers.chronos_wrapper import ChronosWrapper
        return ChronosWrapper()
    if name == "timesfm":
        from foundation.wrappers.timesfm_wrapper import TimesFMWrapper
        return TimesFMWrapper()
    if name == "timemoe":
        from foundation.wrappers.timemoe_wrapper import TimeMoEWrapper
        return TimeMoEWrapper()
    if name == "moirai":
        from foundation.wrappers.moirai_wrapper import MoiraiWrapper
        return MoiraiWrapper()
    if name in ("lagllama", "lag_llama", "lag-llama"):
        from foundation.wrappers.lagllama_wrapper import LagLlamaWrapper
        return LagLlamaWrapper()
    if name in ("granite_ttm", "granite-ttm", "ttm"):
        from foundation.wrappers.granite_ttm_wrapper import GraniteTTMWrapper
        return GraniteTTMWrapper()
    raise ValueError(f"Unknown model '{name}'")


def _window_offsets(L: int, win_len: int) -> list[int]:
    if L <= win_len:
        return [0]
    n = int(np.ceil(L / win_len))
    if n * win_len < L:
        n += 1
    starts = np.linspace(0, L - win_len, n).round().astype(int).tolist()
    seen, out = set(), []
    for s in starts:
        if s not in seen:
            out.append(s); seen.add(s)
    return out


def extract_pooled_embeddings(
    wrapper, loader: DataLoader, *,
    win_len: int, pool: str, max_batches: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """For each batch (B, L, C) and each channel c, window+embed and mean-pool
    across windows.  Concatenate channel embeddings to (N, d_embed * C)."""
    embs, ys = [], []
    for b, (x, y) in enumerate(loader):
        if max_batches is not None and b >= max_batches:
            break
        x = x.numpy().astype(np.float32)                    # (B, L, C)
        B, L, C = x.shape
        offsets = _window_offsets(L, win_len)
        per_channel = []
        for c in range(C):
            xc = x[..., c]                                  # (B, L)
            win_embs = [
                wrapper.embed(xc[:, s:s + win_len], pool=pool).embeddings
                for s in offsets
            ]
            per_channel.append(np.stack(win_embs, axis=1).mean(axis=1))   # (B, d)
        embs.append(np.concatenate(per_channel, axis=-1))                 # (B, d*C)
        ys.append(y.numpy().astype(np.float32))
    return np.concatenate(embs, axis=0), np.concatenate(ys, axis=0)


class EmbeddingMLP(nn.Module):
    """[d_embed] -> [256, 128] -> [n_targets].  Same head as toy regression."""
    def __init__(self, d_embed: int, n_targets: int, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_embed, 256), nn.LayerNorm(256), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(256, 128),     nn.LayerNorm(128), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(128, n_targets),
        )
    def forward(self, x): return self.net(x)


def _ridge(X_tr, y_tr, X_te, alpha: float = 1.0):
    """Closed-form Ridge: returns predictions on X_te, shape (N_te, T)."""
    X = X_tr - X_tr.mean(axis=0, keepdims=True)
    yc = y_tr - y_tr.mean(axis=0, keepdims=True)
    XtX = X.T @ X
    w = np.linalg.solve(XtX + alpha * np.eye(X.shape[1], dtype=X.dtype), X.T @ yc)
    b = y_tr.mean(axis=0) - X_tr.mean(axis=0) @ w
    return X_te @ w + b


def _train_mlp(X_tr, y_tr, X_va, y_va, *, epochs=50, bs=256, lr=1e-3, device="cuda", seed=42):
    torch.manual_seed(seed)
    dev = torch.device(device)
    head = EmbeddingMLP(X_tr.shape[1], y_tr.shape[1]).to(dev)
    opt = torch.optim.AdamW(head.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.ExponentialLR(opt, gamma=0.99)
    Xt, yt = torch.from_numpy(X_tr).to(dev), torch.from_numpy(y_tr).to(dev)
    Xv, yv = torch.from_numpy(X_va).to(dev), torch.from_numpy(y_va).to(dev)
    rng = np.random.default_rng(seed)
    best, best_state, no_improve = float("inf"), None, 0
    for _ in range(epochs):
        head.train()
        perm = rng.permutation(Xt.shape[0])
        for s in range(0, Xt.shape[0], bs):
            idx = perm[s:s + bs]
            opt.zero_grad()
            F.mse_loss(head(Xt[idx]), yt[idx]).backward()
            opt.step()
        sched.step()
        head.eval()
        with torch.no_grad():
            v = float(F.mse_loss(head(Xv), yv))
        if v < best - 1e-6:
            best, best_state, no_improve = v, {k: t.detach().clone() for k, t in head.state_dict().items()}, 0
        else:
            no_improve += 1
            if no_improve >= 10:
                break
    if best_state is not None:
        head.load_state_dict(best_state)
    return head


def _metrics(y_true, y_pred):
    diff = y_pred - y_true
    rmse = float(np.sqrt(np.mean(diff ** 2)))
    mae  = float(np.mean(np.abs(diff)))
    ss_res = float(np.sum(diff ** 2))
    ss_tot = float(np.sum((y_true - y_true.mean()) ** 2))
    r2 = 1.0 - ss_res / (ss_tot + 1e-12)
    return {"rmse": rmse, "mae": mae, "r2": r2}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data_root", default="data/Project8")
    p.add_argument("--out_dir",  default="plots/Project8/linear_probe")
    p.add_argument("--models", nargs="+", required=True)
    p.add_argument("--model_size", default="base")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--num_workers", type=int, default=0)
    p.add_argument("--prefetch_factor", type=int, default=None,
                   help="DataLoader prefetch_factor (only effective with num_workers>0)")
    p.add_argument("--pin_memory", action="store_true",
                   help="Pin DataLoader buffers (faster H2D copies on CUDA)")
    p.add_argument("--persistent_workers", action="store_true",
                   help="Keep DataLoader workers alive between epochs (saves fork cost)")
    p.add_argument("--win_len", type=int, default=512)
    p.add_argument("--pool", default=None, choices=[None, "mean", "last", "max"])
    p.add_argument("--noise_type", default="cav", choices=["cav", "gauss", "none"])
    p.add_argument("--freq_transform", default=None, choices=[None, "fft"])
    p.add_argument("--cutoff", type=int, default=24576,
                   help="Trace length per channel; full sample is 24576")
    p.add_argument("--alpha", type=float, default=1.0, help="Ridge regularisation")
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--smoke_test", action="store_true")
    p.add_argument("--max_train_batches", type=int, default=None)
    p.add_argument("--max_test_batches",  type=int, default=None)
    args = p.parse_args()

    if args.smoke_test:
        args.max_train_batches = args.max_train_batches or 2
        args.max_test_batches  = args.max_test_batches  or 2
        args.epochs = min(args.epochs, 3)

    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    root = Path(args.data_root)

    dm = Project8DataModule(
        train_dir = str(root / "train"),
        val_dir   = str(root / "val"),
        test_dir  = str(root / "test"),
        inputs    = ["output_ts_I", "output_ts_Q"],
        variables = ["energy_eV"],
        cutoff         = args.cutoff,
        noise_type     = args.noise_type,
        freq_transform = args.freq_transform,
        batch_size     = args.batch_size,
        num_workers    = args.num_workers,
        pin_memory     = args.pin_memory,
        prefetch_factor    = args.prefetch_factor,
        persistent_workers = args.persistent_workers,
    )
    dm.setup()
    mu_y, std_y = dm.mu, dm.stds  # (1,)
    print(f"Project 8: train_dir={root/'train'} val={root/'val'} test={root/'test'} "
          f"noise={args.noise_type} freq={args.freq_transform} "
          f"channels={dm.input_channels_ts} mu={mu_y} std={std_y}")

    summary = {}
    for model_name in args.models:
        print(f"\n{'=' * 70}\n[{model_name}] loading (size={args.model_size})\n{'=' * 70}")
        wrapper = _get_wrapper(model_name)
        load_kwargs = {"device": args.device, "model_size": args.model_size}
        if model_name.lower() == "moirai":
            load_kwargs["patch_size"] = 32
        if model_name.lower().replace("-", "_") in ("lagllama", "lag_llama"):
            import os as _os
            load_kwargs["ckpt_path"] = _os.environ.get(
                "LAGLLAMA_CKPT",
                "/esat/smcdata/users/kkontras/Image_Dataset/no_backup/checkpoints/lag-llama/lag-llama.ckpt",
            )
        wrapper.load(**load_kwargs)
        if not wrapper.supports_embed:
            print(f"  ! {model_name} does not support embed; skipping")
            wrapper.unload(); continue

        pool = args.pool or _DEFAULT_POOL.get(model_name.lower().replace("-", "_"), "mean")
        print(f"  [{model_name}] pool={pool} win_len={args.win_len}")

        print(f"  [{model_name}] extracting train embeddings...")
        X_tr, y_tr = extract_pooled_embeddings(
            wrapper, dm.train_dataloader(),
            win_len=args.win_len, pool=pool, max_batches=args.max_train_batches,
        )
        print(f"  [{model_name}] extracting val embeddings...")
        X_va, y_va = extract_pooled_embeddings(
            wrapper, dm.val_dataloader(),
            win_len=args.win_len, pool=pool, max_batches=args.max_train_batches,
        )
        print(f"  [{model_name}] extracting test embeddings...")
        X_te, y_te = extract_pooled_embeddings(
            wrapper, dm.test_dataloader(),
            win_len=args.win_len, pool=pool, max_batches=args.max_test_batches,
        )
        print(f"  [{model_name}] train={X_tr.shape} val={X_va.shape} test={X_te.shape}")

        # Standardise embeddings before head training
        mu  = X_tr.mean(axis=0, keepdims=True)
        sd  = X_tr.std(axis=0, keepdims=True) + 1e-6
        X_tr_n, X_va_n, X_te_n = (X_tr - mu) / sd, (X_va - mu) / sd, (X_te - mu) / sd

        # Ridge linear probe (on z-scored targets)
        y_pred_lp = _ridge(X_tr_n, y_tr, X_te_n, alpha=args.alpha)
        # MLP head
        head = _train_mlp(X_tr_n, y_tr, X_va_n, y_va, epochs=args.epochs, device=args.device, seed=args.seed)
        head.eval()
        with torch.no_grad():
            y_pred_mlp = head(torch.from_numpy(X_te_n).to(args.device)).cpu().numpy()

        # Metrics in normalised AND physical units (energy has its own (mu, std))
        m_lp_norm  = _metrics(y_te, y_pred_lp)
        m_mlp_norm = _metrics(y_te, y_pred_mlp)
        y_true_phys     = y_te         * std_y + mu_y
        y_pred_lp_phys  = y_pred_lp    * std_y + mu_y
        y_pred_mlp_phys = y_pred_mlp   * std_y + mu_y
        m_lp_phys  = _metrics(y_true_phys, y_pred_lp_phys)
        m_mlp_phys = _metrics(y_true_phys, y_pred_mlp_phys)

        print(f"\n[{model_name}] linear probe (Ridge)  norm: {m_lp_norm}")
        print(f"[{model_name}] linear probe (Ridge)   phys: {m_lp_phys}")
        print(f"[{model_name}] MLP head (z-scored)   norm: {m_mlp_norm}")
        print(f"[{model_name}] MLP head              phys: {m_mlp_phys}")

        model_dir = out_dir / model_name; model_dir.mkdir(parents=True, exist_ok=True)
        np.save(model_dir / "y_true_norm.npy",     y_te)
        np.save(model_dir / "y_pred_ridge_norm.npy", y_pred_lp)
        np.save(model_dir / "y_pred_mlp_norm.npy",   y_pred_mlp)
        np.save(model_dir / "energy_mu_std.npy",    np.stack([mu_y, std_y], axis=0))
        with open(model_dir / "metrics.json", "w") as f:
            json.dump({
                "linear_probe_norm":  m_lp_norm,
                "linear_probe_phys":  m_lp_phys,
                "mlp_head_norm":      m_mlp_norm,
                "mlp_head_phys":      m_mlp_phys,
                "d_embed_per_channel": int(X_tr.shape[1] // dm.input_channels_ts),
                "d_embed_total":       int(X_tr.shape[1]),
                "input_channels":      int(dm.input_channels_ts),
                "pool": pool,
                "args": vars(args),
            }, f, indent=2)
        summary[model_name] = {"linear_probe_phys_rmse": m_lp_phys["rmse"],
                               "mlp_head_phys_rmse":     m_mlp_phys["rmse"],
                               "mlp_head_phys_r2":       m_mlp_phys["r2"]}
        wrapper.unload()

    print(f"\n{'=' * 70}\nSummary (zero-shot linear probe, seed={args.seed})\n{'=' * 70}")
    print(f"{'model':>15s}  {'lp_rmse':>10s}  {'mlp_rmse':>10s}  {'mlp_r2':>8s}")
    for name, m in summary.items():
        print(f"{name:>15s}  {m['linear_probe_phys_rmse']:10.4f}  "
              f"{m['mlp_head_phys_rmse']:10.4f}  {m['mlp_head_phys_r2']:8.4f}")
    with open(out_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nWrote: {out_dir}")


if __name__ == "__main__":
    main()
