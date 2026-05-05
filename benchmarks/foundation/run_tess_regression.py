"""Zero-shot regression on PhyTS-bench TESS via FM linear probe.

Predicts the rotation frequency (``frot``) of each TESS light curve from
foundation-model embeddings, with Ridge regression on top.  Splits are
performed by TIC so the same star never appears in two splits.

Usage
-----
    python benchmarks/foundation/run_tess_regression.py \\
        --models granite_ttm timemoe \\
        --out_dir plots/tess/linear_probe_regression
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
from foundation.evaluators.regression import evaluate_regression
from run_tess import _DEFAULT_POOL, _get_wrapper


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out_dir", default="plots/tess/linear_probe_regression")
    p.add_argument("--models", nargs="+", required=True)
    p.add_argument("--model_size", default="base")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--num_workers", type=int, default=0)
    p.add_argument("--win_len", type=int, default=512)
    p.add_argument("--pool", default=None, choices=[None, "mean", "last", "max"])
    p.add_argument("--alpha", type=float, default=1.0,
                   help="Ridge regression L2 strength.")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--smoke_test", action="store_true")
    p.add_argument("--max_train_batches", type=int, default=None)
    p.add_argument("--max_test_batches", type=int, default=None)
    args = p.parse_args()

    if args.smoke_test:
        args.max_train_batches = args.max_train_batches or 2
        args.max_test_batches = args.max_test_batches or 2

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    train_loader, val_loader, test_loader, meta = make_loaders_regression(
        batch_size=args.batch_size, num_workers=args.num_workers, seed=args.seed,
    )
    print(f"TESS regression ({meta['target_name']}): {sum(meta['split_sizes'])} LCs "
          f"(train/val/test = {meta['split_sizes']}), "
          f"unique TICs per split = {meta['n_unique_stars']}, "
          f"target mean/std = {meta['target_mean']:.4f}/{meta['target_std']:.4f}, "
          f"lc_len={meta['lc_len']}")

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
            wrapper.unload()
            continue

        pool = args.pool or _DEFAULT_POOL.get(model_name.lower().replace("-", "_"), "mean")
        print(f"  [{model_name}] pool={pool}")
        res = evaluate_regression(
            wrapper,
            train_loader=train_loader, val_loader=val_loader, test_loader=test_loader,
            target_name=meta["target_name"],
            win_len=args.win_len, pool=pool,
            max_train_batches=args.max_train_batches,
            max_test_batches=args.max_test_batches,
            alpha=args.alpha,
        )

        print(f"\n[{model_name}] test metrics:")
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
                 "y_mean": res["y_mean"], "y_std": res["y_std"],
                 "args": vars(args)},
                f, indent=2,
            )
        summary[model_name] = res["metrics"]
        wrapper.unload()

    print(f"\n{'=' * 70}\nSummary (zero-shot regression, seed={args.seed})\n{'=' * 70}")
    print(f"{'model':>15s}  {'mse':>8s}  {'mae':>8s}  {'r2':>8s}  {'pearson':>8s}  {'spearman':>8s}")
    for name, m in summary.items():
        print(f"{name:>15s}  {m['mse']:.4f}  {m['mae']:.4f}  "
              f"{m['r2']:.4f}  {m['pearson_r']:.4f}  {m['spearman_r']:.4f}")
    with open(out_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nWrote: {out_dir}")


if __name__ == "__main__":
    main()
