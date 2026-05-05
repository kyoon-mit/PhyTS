"""Zero-shot classification on PhyTS-bench TESS via FM linear probe.

Mirrors :mod:`run_kepler` but for the HuggingFace TESS dataset.  The TESS
loader splits **by TIC** (TESS Input Catalog ID) so a star never crosses
splits — most TICs in this set have ≥2 light curves and ignoring this
would leak star-specific signal across train/test.

Usage
-----
    python benchmarks/foundation/run_tess.py \\
        --models granite_ttm timemoe \\
        --out_dir plots/tess/linear_probe
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

from dataloader.tess_dataloader import make_loaders_classification, CLASS_NAMES
from foundation.evaluators.classification import evaluate_classification


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


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out_dir", default="plots/tess/linear_probe")
    p.add_argument("--models", nargs="+", required=True)
    p.add_argument("--model_size", default="base")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--num_workers", type=int, default=0)
    p.add_argument("--win_len", type=int, default=512)
    p.add_argument("--pool", default=None,
                   choices=[None, "mean", "last", "max"],
                   help="Per-window temporal pool. If omitted, uses per-model default.")
    p.add_argument("--C", type=float, default=1.0,
                   help="Logistic regression inverse regularisation.")
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

    train_loader, val_loader, test_loader, meta = make_loaders_classification(
        batch_size=args.batch_size, num_workers=args.num_workers, seed=args.seed,
    )
    print(f"TESS classification: {sum(meta['split_sizes'])} LCs "
          f"(train/val/test = {meta['split_sizes']}), "
          f"unique TICs per split = {meta['n_unique_stars']}, "
          f"n_classes={meta['n_classes']}, lc_len={meta['lc_len']}")

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
        res = evaluate_classification(
            wrapper,
            train_loader=train_loader, val_loader=val_loader, test_loader=test_loader,
            class_names=CLASS_NAMES,
            win_len=args.win_len,
            pool=pool,
            max_train_batches=args.max_train_batches,
            max_test_batches=args.max_test_batches,
            C=args.C,
        )

        print(f"\n[{model_name}] test metrics:")
        for k, v in res["metrics"].items():
            print(f"  {k}: {v:.4f}")
        print(f"  d_embed: {res['d_embed']}")
        print(f"\nPer-class F1:")
        for cname in CLASS_NAMES:
            f1 = res["report"][cname]["f1-score"]
            support = res["report"][cname]["support"]
            print(f"  {cname:18s} f1={f1:.3f}  n={int(support)}")
        print("\nConfusion matrix (rows=true, cols=pred):")
        print("  " + " ".join(f"{c[:4]:>5s}" for c in CLASS_NAMES))
        for i, row in enumerate(res["confusion_matrix"]):
            print(f"  {CLASS_NAMES[i][:4]:>4s} " + " ".join(f"{v:5d}" for v in row))

        model_dir = out_dir / model_name
        model_dir.mkdir(parents=True, exist_ok=True)
        np.save(model_dir / "confusion_matrix.npy", res["confusion_matrix"])
        np.save(model_dir / "y_true.npy", res["y_true"])
        np.save(model_dir / "y_pred.npy", res["y_pred"])
        with open(model_dir / "metrics.json", "w") as f:
            json.dump(
                {"metrics": res["metrics"], "report": res["report"],
                 "d_embed": res["d_embed"], "class_names": CLASS_NAMES,
                 "pool": pool, "args": vars(args)},
                f, indent=2,
            )
        summary[model_name] = res["metrics"]
        wrapper.unload()

    print(f"\n{'=' * 70}\nSummary (zero-shot linear probe, seed={args.seed})\n{'=' * 70}")
    print(f"{'model':>15s}  {'acc':>6s}  {'bal_acc':>8s}  {'macro_f1':>9s}  {'val_acc':>8s}")
    for name, m in summary.items():
        print(f"{name:>15s}  {m['accuracy']:.4f}  {m['balanced_accuracy']:.4f}  "
              f"{m['macro_f1']:.4f}   {m['val_accuracy']:.4f}")
    with open(out_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nWrote: {out_dir}")


if __name__ == "__main__":
    main()
