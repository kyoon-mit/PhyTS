"""LoRA fine-tuned regression on PhyTS-bench TESS (target: ``frot``).

Mirrors :mod:`run_tess_finetune` but for the scalar rotation-frequency
target. Group-split by TIC. Drops non-positive ``frot`` sentinels.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

_HERE = Path(__file__).resolve().parent
_REPO = _HERE.parent.parent
for p in (str(_REPO), str(_HERE.parent), str(_REPO / "src")):
    if p not in sys.path:
        sys.path.insert(0, p)

from dataloader.tess_dataloader import make_loaders_regression
from foundation.evaluators.regression_finetune import finetune_regression
from run_tess_finetune import _DEFAULT_POOL, _SUPPORTED, _get_wrapper


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out_dir", default="plots/tess/lora_regression")
    p.add_argument("--models", nargs="+", required=True)
    p.add_argument("--model_size", default="base")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--num_workers", type=int, default=0)
    p.add_argument("--prefetch_factor", type=int, default=None)
    p.add_argument("--persistent_workers", action="store_true")
    p.add_argument("--pin_memory", action="store_true")
    p.add_argument("--win_len", type=int, default=512)
    p.add_argument("--pool", default=None, choices=[None, "mean", "last", "max"])

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
    p.add_argument("--patience", type=int, default=None)

    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--smoke_test", action="store_true")
    p.add_argument("--max_train_batches", type=int, default=None)
    p.add_argument("--max_test_batches", type=int, default=None)
    args = p.parse_args()

    if args.smoke_test:
        args.max_train_batches = args.max_train_batches or 2
        args.max_test_batches = args.max_test_batches or 2
        args.epochs = min(args.epochs, 2)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    train_loader, val_loader, test_loader, meta = make_loaders_regression(
        batch_size=args.batch_size, num_workers=args.num_workers, seed=args.seed,
        prefetch_factor=args.prefetch_factor,
        persistent_workers=args.persistent_workers,
        pin_memory=args.pin_memory,
    )
    print(f"TESS regression ({meta['target_name']}): {sum(meta['split_sizes'])} LCs "
          f"(train/val/test = {meta['split_sizes']}), "
          f"unique TICs per split = {meta['n_unique_stars']}, "
          f"target mean/std = {meta['target_mean']:.4f}/{meta['target_std']:.4f}, "
          f"lc_len={meta['lc_len']}")

    summary = {}
    for model_name in args.models:
        key = model_name.lower().replace("-", "_")
        if key not in _SUPPORTED:
            print(f"  ! {model_name}: fine-tuning not yet supported; skipping.")
            continue
        if key == "granite_ttm" and args.adapter == "lora":
            print(f"  ! {model_name}: LoRA is a no-op (no attention projections); "
                  f"use --adapter full or last_n. Skipping.")
            continue

        print(f"\n{'=' * 70}")
        print(f"[{model_name}] loading (size={args.model_size}, adapter={args.adapter})")
        print(f"{'=' * 70}")
        wrapper = _get_wrapper(model_name)
        wrapper.load(device=args.device, model_size=args.model_size)
        if not wrapper.supports_embed:
            print(f"  ! {model_name}: no embed capability; skipping.")
            wrapper.unload()
            continue

        pool = args.pool or _DEFAULT_POOL.get(key, "mean")
        print(f"  [{model_name}] pool={pool}")

        res = finetune_regression(
            wrapper,
            train_loader=train_loader, val_loader=val_loader, test_loader=test_loader,
            target_name=meta["target_name"],
            target_mean=meta["target_mean"], target_std=meta["target_std"],
            win_len=args.win_len, pool=pool, device=args.device,
            epochs=args.epochs,
            head_lr=args.head_lr, backbone_lr=args.backbone_lr,
            adapter=args.adapter, unfreeze_last_n=args.unfreeze_last_n,
            lora_r=args.lora_r, lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout, lora_last_n_blocks=args.lora_last_n_blocks,
            weight_decay=args.weight_decay,
            loss=args.loss, huber_beta=args.huber_beta,
            patience=args.patience,
            max_train_batches=args.max_train_batches,
            max_test_batches=args.max_test_batches,
            seed=args.seed,
        )

        print(f"\n[{model_name}] test metrics (best-val checkpoint):")
        for k, v in res["metrics"].items():
            print(f"  {k}: {v:.4f}")
        print(f"  d_embed: {res['d_embed']}")

        model_dir = out_dir / model_name
        model_dir.mkdir(parents=True, exist_ok=True)
        np.save(model_dir / "y_true.npy", res["y_true"])
        np.save(model_dir / "y_pred.npy", res["y_pred"])
        with open(model_dir / "metrics.json", "w") as f:
            json.dump(
                {"metrics": res["metrics"], "target_name": res["target_name"],
                 "d_embed": res["d_embed"], "pool": pool,
                 "history": res["history"], "args": vars(args),
                 "target_mean": meta["target_mean"], "target_std": meta["target_std"]},
                f, indent=2,
            )
        summary[model_name] = res["metrics"]
        wrapper.unload()

    print(f"\n{'=' * 70}")
    print(f"Summary (LoRA fine-tune regression, seed={args.seed}, adapter={args.adapter})")
    print(f"{'=' * 70}")
    print(f"{'model':>15s}  {'mse':>8s}  {'mae':>8s}  {'r2':>8s}  {'pearson':>8s}  {'spearman':>8s}")
    for name, m in summary.items():
        print(f"{name:>15s}  {m['mse']:.4f}  {m['mae']:.4f}  "
              f"{m['r2']:.4f}  {m['pearson_r']:.4f}  {m['spearman_r']:.4f}")
    with open(out_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nWrote: {out_dir}")


if __name__ == "__main__":
    main()
