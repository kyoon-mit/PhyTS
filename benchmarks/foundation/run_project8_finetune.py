"""LoRA / last-N / full fine-tuned starting-energy regression on Project 8.

Mirrors :mod:`run_project8` (zero-shot linear probe) but trains a small MLP
head together with LoRA adapters / last-N-block unfreezing / full backbone
fine-tuning on the foundation-model backbone.  Model selection uses the
held-out validation MSE; final metrics are reported on the test split.

Project 8 batches are multivariate (B, L, C=2) [I, Q]; we embed each
channel separately (mirroring the zero-shot probe in run_project8.py) and
concatenate channel embeddings before the regression head.

Usage
-----
    python benchmarks/foundation/run_project8_finetune.py \\
        --models moment chronos timesfm timemoe moirai \\
        --adapter lora --epochs 10 \\
        --out_dir plots/Project8/lora

Notes
-----
* granite_ttm has no attention projections — LoRA matches zero modules; use
  ``--adapter full`` or ``last_n`` instead.  Skipped automatically for LoRA.
* lagllama's grad-flowing forecast path is hopelessly slow at full length;
  not run by default.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

_HERE = Path(__file__).resolve().parent
_REPO = _HERE.parent.parent
for p in (str(_REPO), str(_HERE.parent), str(_REPO / "src")):
    if p not in sys.path:
        sys.path.insert(0, p)

from dataloader.project8_dataloader import Project8DataModule
from foundation.finetuning.finetune import (
    LoRALinear, _find_backbone, _unfreeze_last_n, apply_lora,
)


_DEFAULT_POOL = {
    "moment":      "mean",
    "chronos":     "mean",
    "moirai":      "mean",
    "granite_ttm": "mean",
    "timemoe":     "last",
    "timesfm":     "last",
    "lagllama":    "last",
}

_SUPPORTED = {"moment", "chronos", "timemoe", "granite_ttm", "timesfm", "moirai"}


def _get_wrapper(name: str):
    key = name.lower().replace("-", "_")
    if key == "moment":
        from foundation.wrappers.moment_wrapper import MomentWrapper
        return MomentWrapper()
    if key == "chronos":
        from foundation.wrappers.chronos_wrapper import ChronosWrapper
        return ChronosWrapper()
    if key == "timemoe":
        from foundation.wrappers.timemoe_wrapper import TimeMoEWrapper
        return TimeMoEWrapper()
    if key in ("granite_ttm", "ttm"):
        from foundation.wrappers.granite_ttm_wrapper import GraniteTTMWrapper
        return GraniteTTMWrapper()
    if key == "timesfm":
        from foundation.wrappers.timesfm_wrapper import TimesFMWrapper
        return TimesFMWrapper()
    if key == "moirai":
        from foundation.wrappers.moirai_wrapper import MoiraiWrapper
        w = MoiraiWrapper()
        _orig = w.load
        def _patched_load(*a, **kw):
            kw.setdefault("patch_size", 32)
            return _orig(*a, **kw)
        w.load = _patched_load
        return w
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


def _embed_pooled_multichannel(wrapper, x_np: np.ndarray, *, win_len: int, pool: str):
    """x_np: (B, L, C) → (B, C*d_embed) torch tensor on the wrapper's device.

    For each channel we window, call wrapper.embed_torch (grad-preserving),
    and mean-pool across windows; channel embeddings are then concatenated.
    Identical structure to the zero-shot probe in run_project8.py.
    """
    B, L, C = x_np.shape
    offsets = _window_offsets(L, win_len)
    per_channel = []
    for c in range(C):
        xc = np.ascontiguousarray(x_np[..., c])           # (B, L)
        win_embs = [
            wrapper.embed_torch(xc[:, s:s + win_len], pool=pool)
            for s in offsets
        ]
        per_channel.append(
            torch.stack(win_embs, dim=1).mean(dim=1) if len(win_embs) > 1 else win_embs[0]
        )  # (B, d)
    return torch.cat(per_channel, dim=-1)                  # (B, C*d)


class _RegressionHead(nn.Module):
    """MLP head: d_in → 256 → 128 → 1.  Same shape as zero-shot MLP head."""
    def __init__(self, d_in: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_in, 256), nn.LayerNorm(256), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(256, 128),  nn.LayerNorm(128), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(128, 1),
        )
    def forward(self, x):
        return self.net(x).squeeze(-1)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data_root", default="data/Project8")
    p.add_argument("--out_dir",   default="plots/Project8/lora")
    p.add_argument("--models", nargs="+", required=True)
    p.add_argument("--model_size", default="base")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")

    # Dataloader
    p.add_argument("--batch_size",  type=int, default=8)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--prefetch_factor", type=int, default=4)
    p.add_argument("--pin_memory",  action="store_true")
    p.add_argument("--persistent_workers", action="store_true")
    p.add_argument("--cutoff",      type=int, default=24576)
    p.add_argument("--noise_type",  default="cav", choices=["cav", "gauss", "none"])
    p.add_argument("--win_len",     type=int, default=512)
    p.add_argument("--pool",        default=None, choices=[None, "mean", "last", "max"])

    # Fine-tuning knobs (mirror run_kepler_finetune)
    p.add_argument("--adapter", default="lora",
                   choices=["lora", "last_n", "full", "frozen"])
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--head_lr", type=float, default=1e-3)
    p.add_argument("--backbone_lr", type=float, default=1e-4)
    p.add_argument("--unfreeze_last_n", type=int, default=2)
    p.add_argument("--lora_r", type=int, default=8)
    p.add_argument("--lora_alpha", type=int, default=16)
    p.add_argument("--lora_dropout", type=float, default=0.0)
    p.add_argument("--lora_last_n_blocks", type=int, default=None)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--loss", default="huber", choices=["huber", "mse"])
    p.add_argument("--huber_beta", type=float, default=1.0)
    p.add_argument("--patience", type=int, default=None,
                   help="Early-stop after N epochs without val_mse improvement.")

    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--smoke_test", action="store_true")
    p.add_argument("--max_train_batches", type=int, default=None)
    p.add_argument("--max_test_batches",  type=int, default=None)
    args = p.parse_args()

    if args.smoke_test:
        args.max_train_batches = args.max_train_batches or 2
        args.max_test_batches  = args.max_test_batches  or 2
        args.epochs = min(args.epochs, 2)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    root = Path(args.data_root)

    dm = Project8DataModule(
        train_dir=str(root/"train"), val_dir=str(root/"val"), test_dir=str(root/"test"),
        inputs=["output_ts_I", "output_ts_Q"], variables=["energy_eV"],
        cutoff=args.cutoff, noise_type=args.noise_type, freq_transform=None,
        batch_size=args.batch_size, num_workers=args.num_workers,
        pin_memory=args.pin_memory, prefetch_factor=args.prefetch_factor,
        persistent_workers=args.persistent_workers,
    )
    dm.setup()
    mu_y, std_y = float(dm.mu[0]), float(dm.stds[0])
    print(f"Project 8: noise={args.noise_type} cutoff={args.cutoff} "
          f"channels={dm.input_channels_ts} target=energy_eV "
          f"mu={mu_y:.2f} std={std_y:.4f}")

    summary: dict = {}
    for model_name in args.models:
        key = model_name.lower().replace("-", "_")
        if key not in _SUPPORTED:
            print(f"  ! {model_name}: not supported for fine-tune; skipping.")
            continue
        if key == "granite_ttm" and args.adapter == "lora":
            print(f"  ! {model_name}: LoRA = no-op on TTM (no attention projections); "
                  "use --adapter full or last_n. Skipping.")
            continue

        print(f"\n{'=' * 70}\n[{model_name}] adapter={args.adapter}  size={args.model_size}\n"
              f"{'=' * 70}")
        wrapper = _get_wrapper(model_name)
        wrapper.load(device=args.device, model_size=args.model_size)
        if not wrapper.supports_embed:
            print(f"  ! {model_name}: no embed; skipping."); wrapper.unload(); continue
        pool = args.pool or _DEFAULT_POOL.get(key, "mean")
        print(f"  [{model_name}] pool={pool}  win_len={args.win_len}")

        # ── Backbone adaptation ───────────────────────────────────────────
        backbone = _find_backbone(wrapper)
        trainable_bb: list[nn.Parameter] = []
        if backbone is None:
            print(f"  [{model_name}] no backbone discovered — head-only.")
        elif args.adapter == "frozen":
            for p_ in backbone.parameters(): p_.requires_grad_(False)
        elif args.adapter == "last_n":
            trainable_bb = _unfreeze_last_n(backbone, args.unfreeze_last_n)
        elif args.adapter == "full":
            for p_ in backbone.parameters(): p_.requires_grad_(True)
            trainable_bb = [p_ for p_ in backbone.parameters() if p_.requires_grad]
        elif args.adapter == "lora":
            trainable_bb = apply_lora(
                backbone, r=args.lora_r, alpha=args.lora_alpha,
                dropout=args.lora_dropout, last_n_blocks=args.lora_last_n_blocks,
            )
            try:
                bb_dev = next(backbone.parameters()).device; backbone.to(bb_dev)
            except StopIteration:
                pass
            n_mod = sum(1 for m in backbone.modules() if isinstance(m, LoRALinear))
            n_par = sum(p_.numel() for p_ in trainable_bb)
            print(f"  [{model_name}] LoRA r={args.lora_r} α={args.lora_alpha} "
                  f"into {n_mod} modules; {n_par/1e6:.3f}M trainable.")

        # Probe d_embed once on a one-event batch
        with torch.no_grad():
            x0, _ = next(iter(dm.train_dataloader()))
            d_in = _embed_pooled_multichannel(
                wrapper, x0[:1].numpy().astype(np.float32),
                win_len=args.win_len, pool=pool,
            ).shape[-1]
        print(f"  [{model_name}] d_in (concat I+Q) = {d_in}")

        head = _RegressionHead(d_in).to(args.device)
        groups = [{"params": list(head.parameters()), "lr": args.head_lr}]
        if trainable_bb:
            groups.append({"params": trainable_bb, "lr": args.backbone_lr})
        opt = torch.optim.AdamW(groups, weight_decay=args.weight_decay)

        loss_fn = (nn.SmoothL1Loss(beta=args.huber_beta)
                   if args.loss == "huber" else nn.MSELoss())

        # ── Training loop ─────────────────────────────────────────────────
        best_val_mse = float("inf"); best_epoch = 0; epochs_since_best = 0
        best_head = {k: v.detach().clone() for k, v in head.state_dict().items()}
        best_bb   = ([p_.detach().clone() for p_ in trainable_bb] if trainable_bb else None)
        history: list[dict] = []

        for ep in range(args.epochs):
            head.train()
            if backbone is not None: backbone.eval() if args.adapter == "frozen" else backbone.train()

            tr_losses = []
            for bi, (x, y) in enumerate(dm.train_dataloader()):
                if args.max_train_batches and bi >= args.max_train_batches: break
                # Project8DataModule already z-scores `y`; just squeeze the trailing dim.
                y_norm = y.to(args.device).float().squeeze(-1)
                emb = _embed_pooled_multichannel(
                    wrapper, x.numpy().astype(np.float32),
                    win_len=args.win_len, pool=pool,
                )
                pred_norm = head(emb)
                loss_val = loss_fn(pred_norm, y_norm)
                opt.zero_grad(); loss_val.backward(); opt.step()
                tr_losses.append(float(loss_val.item()))

            # Validation (MSE on physical scale for selection)
            head.eval()
            if backbone is not None: backbone.eval()
            val_sse = 0.0; val_n = 0
            with torch.no_grad():
                for x, y in dm.val_dataloader():
                    # y from loader is z-scored; un-normalise to physical eV.
                    y_phys = (y.to(args.device).float().squeeze(-1) * std_y) + mu_y
                    emb = _embed_pooled_multichannel(
                        wrapper, x.numpy().astype(np.float32),
                        win_len=args.win_len, pool=pool,
                    )
                    pred_phys = head(emb) * std_y + mu_y
                    val_sse += float(((pred_phys - y_phys) ** 2).sum().item())
                    val_n += int(y_phys.numel())
            val_mse = val_sse / max(val_n, 1)
            tl = float(np.mean(tr_losses)) if tr_losses else 0.0
            print(f"  [{model_name}] epoch {ep+1}/{args.epochs}  "
                  f"train_loss={tl:.4f}  val_mse_phys={val_mse:.4f} "
                  f"(rmse={np.sqrt(val_mse):.3f} eV)")
            history.append({"epoch": ep+1, "train_loss": tl, "val_mse_phys": val_mse})

            if val_mse < best_val_mse:
                best_val_mse = val_mse; best_epoch = ep+1; epochs_since_best = 0
                best_head = {k: v.detach().clone() for k, v in head.state_dict().items()}
                if trainable_bb:
                    best_bb = [p_.detach().clone() for p_ in trainable_bb]
            else:
                epochs_since_best += 1
            if args.patience is not None and epochs_since_best >= args.patience:
                print(f"  [{model_name}] early stop at ep {ep+1}: "
                      f"no improvement for {args.patience} epochs "
                      f"(best at ep {best_epoch}, rmse={np.sqrt(best_val_mse):.3f})")
                break

        # Restore best
        head.load_state_dict(best_head)
        if best_bb is not None:
            with torch.no_grad():
                for p_, saved in zip(trainable_bb, best_bb): p_.copy_(saved)

        # ── Test ──────────────────────────────────────────────────────────
        head.eval()
        if backbone is not None: backbone.eval()
        y_true_chunks, y_pred_chunks = [], []
        with torch.no_grad():
            for bi, (x, y) in enumerate(dm.test_dataloader()):
                if args.max_test_batches and bi >= args.max_test_batches: break
                # y from loader is z-scored; un-normalise to physical eV.
                y_phys = (y.float().squeeze(-1).numpy() * std_y) + mu_y
                emb = _embed_pooled_multichannel(
                    wrapper, x.numpy().astype(np.float32),
                    win_len=args.win_len, pool=pool,
                )
                pred_phys = (head(emb) * std_y + mu_y).cpu().numpy()
                y_true_chunks.append(y_phys); y_pred_chunks.append(pred_phys)
        y_true = np.concatenate(y_true_chunks)
        y_pred = np.concatenate(y_pred_chunks)
        diff = y_pred - y_true
        rmse = float(np.sqrt(np.mean(diff**2)))
        mae  = float(np.mean(np.abs(diff)))
        ss_res = float(np.sum(diff**2))
        ss_tot = float(np.sum((y_true - y_true.mean())**2))
        r2 = 1.0 - ss_res / (ss_tot + 1e-12)
        print(f"\n[{model_name}] TEST (best-val ckpt) phys: "
              f"rmse={rmse:.3f} eV  mae={mae:.3f}  r2={r2:.4f}  best_epoch={best_epoch}")

        m_dir = out_dir / model_name; m_dir.mkdir(parents=True, exist_ok=True)
        np.save(m_dir / "y_true.npy", y_true)
        np.save(m_dir / "y_pred.npy", y_pred)
        np.save(m_dir / "energy_mu_std.npy", np.asarray([mu_y, std_y], dtype=np.float64))
        with open(m_dir / "metrics.json", "w") as fh:
            json.dump({
                "metrics": {"rmse": rmse, "mae": mae, "r2": r2,
                            "best_val_mse_phys": best_val_mse,
                            "best_epoch": best_epoch},
                "d_embed": int(d_in), "pool": pool,
                "history": history, "args": vars(args),
            }, fh, indent=2)
        summary[model_name] = {"rmse": rmse, "mae": mae, "r2": r2,
                               "best_epoch": best_epoch}
        wrapper.unload()

    print(f"\n{'=' * 70}\nSummary ({args.adapter} fine-tune, seed={args.seed})\n{'=' * 70}")
    print(f"{'model':>15s}  {'rmse [eV]':>10s}  {'mae':>8s}  {'r2':>7s}  {'best_ep':>7s}")
    for name, m in summary.items():
        print(f"{name:>15s}  {m['rmse']:10.3f}  {m['mae']:8.3f}  {m['r2']:7.4f}  "
              f"{m['best_epoch']:7d}")
    with open(out_dir / "summary.json", "w") as fh:
        json.dump(summary, fh, indent=2)
    print(f"\nWrote: {out_dir}")


if __name__ == "__main__":
    main()
