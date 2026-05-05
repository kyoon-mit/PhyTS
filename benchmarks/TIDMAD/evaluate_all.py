#!/usr/bin/env python3
"""
evaluate_all.py — comprehensive TIDMAD denoising evaluation for all variants.

For each registered variant:
  1. Plots training + validation loss curves  -> results/{name}/loss_curve.png
     (skipped for zero-shot models like Chronos)
  2. Runs test-set inference on 8 held-out H5 files (unless --plot_only)
  3. Saves results/{name}/test_scores.csv      (per-window SNR values)
  4. Saves results/{name}/summary.json         (all metadata + final score)
  5. Writes aggregated                         results/master_results.csv

Usage:
    cd /path/to/TimeSeriesPhysics
    PYTHONPATH=src python benchmarks/TIDMAD/evaluate_all.py
    PYTHONPATH=src python benchmarks/TIDMAD/evaluate_all.py --plot_only
    PYTHONPATH=src python benchmarks/TIDMAD/evaluate_all.py --variants conv_l_full chronos_tiny
    PYTHONPATH=src python benchmarks/TIDMAD/evaluate_all.py --variants chronos_tiny chronos_base \\
        --chronos_max_windows 50
"""

import argparse
import csv
import glob
import importlib
import json
import os
import re
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import yaml

_REPO = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_REPO / "src"))
sys.path.insert(0, str(Path(__file__).parent))

import torch
from run_inference import (
    TEST_FILES,
    score_file,
    compute_benchmark_score,
    load_conv_denoiser,
    load_linoss_denoiser,
    load_chronos_denoiser,
    load_moment_denoiser,
)

# ---------------------------------------------------------------------------
# Variant registry — all paths relative to repo root
# ---------------------------------------------------------------------------

# Approximate parameter counts for zero-shot models (published values)
_CHRONOS_PARAMS = {"tiny": 8_000_000, "small": 46_000_000,
                   "base": 200_000_000, "large": 710_000_000}
_MOMENT_PARAMS  = {"small": 40_000_000, "base": 125_000_000, "large": 385_000_000}

VARIANTS = {
    # ---- ConvAE ablations (5 epochs each) — loss curves only, no inference ---
    # Ablations were used to select the best architecture; benchmark scores are
    # only meaningful for the full training runs below.
    "conv_abl_s": {
        "display_name": "ConvAE-S",
        "category": "conv_ablation",
        "model_type": "conv",
        "metrics_csv": "logs/ablation/conv_s/version_0/metrics.csv",
        "cfg":         "configs/TIDMAD/ablation/conv_abl_s.yaml",
        "arch":        "n_layers=3, channels=16, kernel=5",
        "plot_only":   True,
    },
    "conv_abl_m": {
        "display_name": "ConvAE-M",
        "category": "conv_ablation",
        "model_type": "conv",
        "metrics_csv": "logs/ablation/conv_m/version_0/metrics.csv",
        "cfg":         "configs/TIDMAD/ablation/conv_abl_m.yaml",
        "arch":        "n_layers=4, channels=32, kernel=9",
        "plot_only":   True,
    },
    "conv_abl_l": {
        "display_name": "ConvAE-L",
        "category": "conv_ablation",
        "model_type": "conv",
        "metrics_csv": "logs/ablation/conv_l/version_0/metrics.csv",
        "cfg":         "configs/TIDMAD/ablation/conv_abl_l.yaml",
        "arch":        "n_layers=5, channels=64, kernel=9",
        "plot_only":   True,
    },
    "conv_abl_w": {
        "display_name": "ConvAE-W",
        "category": "conv_ablation",
        "model_type": "conv",
        "metrics_csv": "logs/ablation/conv_w/version_0/metrics.csv",
        "cfg":         "configs/TIDMAD/ablation/conv_abl_w.yaml",
        "arch":        "n_layers=3, channels=64, kernel=5",
        "plot_only":   True,
    },
    "conv_abl_xl": {
        "display_name": "ConvAE-XL",
        "category": "conv_ablation",
        "model_type": "conv",
        # version_1 has better val/loss (52.415) than version_0 (52.467)
        "metrics_csv": "logs/ablation/conv_xl/version_1/metrics.csv",
        "cfg":         "configs/TIDMAD/ablation/conv_abl_xl.yaml",
        "arch":        "n_layers=5, channels=128, kernel=9",
        "plot_only":   True,
    },
    # ---- ConvAE-L full training — MSE loss (baseline, for comparison) -------
    "conv_l_mse": {
        "display_name": "ConvAE-L (MSE)",
        "category": "full_training",
        "model_type": "conv",
        "metrics_csv": "logs/tidmad_conv_l/version_0/metrics.csv",
        "ckpt":        "checkpoints/tidmad_conv_l/best.ckpt",
        "cfg":         "configs/TIDMAD/train_tidmad_conv_l_denoising.yaml",
        "arch":        "n_layers=5, channels=64, kernel=9, loss=MSE",
    },
    # ---- ConvAE-L full training — PSD loss ----------------------------------
    "conv_l_full": {
        "display_name": "ConvAE-L (PSD)",
        "category": "full_training",
        "model_type": "conv",
        "metrics_csv": "logs/tidmad_conv_l_psd/version_0/metrics.csv",
        "ckpt":        "checkpoints/tidmad_conv_l_psd/best.ckpt",
        "cfg":         "configs/TIDMAD/train_tidmad_conv_l_denoising.yaml",
        "arch":        "n_layers=5, channels=64, kernel=9, loss=PSD",
    },
    # ---- LinOSS ablation (4 epochs, stable) — loss curves only, no inference -
    "linoss_abl_s": {
        "display_name": "LinOSS-S",
        "category": "linoss_ablation",
        "model_type": "linoss",
        "metrics_csv": "logs/ablation/linoss_s/version_3/metrics.csv",
        "cfg":         "configs/TIDMAD/ablation/linoss_abl_s.yaml",
        "arch":        "blocks=1, ssm_size=16, H=32, damped_IMEX",
        "plot_only":   True,
    },
    # ---- LinOSS-190K full training — MSE loss (baseline, for comparison) ----
    "linoss_190k_mse": {
        "display_name": "LinOSS-190K (MSE)",
        "category": "full_training",
        "model_type": "linoss",
        "metrics_csv": "logs/tidmad_linoss_190k/version_0/metrics.csv",
        "ckpt_dir":    "logs/tidmad_linoss_190k/version_0/checkpoints",
        "cfg":         "configs/TIDMAD/train_tidmad_linoss_190k_denoising.yaml",
        "arch":        "blocks=4, ssm_size=32, H=128, damped_IMEX, loss=MSE",
    },
    # ---- LinOSS-190K full training — PSD loss -------------------------------
    "linoss_190k_full": {
        "display_name": "LinOSS-190K (PSD)",
        "category": "full_training",
        "model_type": "linoss",
        "metrics_csv": "logs/tidmad_linoss_190k_psd/version_0/metrics.csv",
        "ckpt_dir":    "logs/tidmad_linoss_190k_psd/version_0/checkpoints",
        "cfg":         "configs/TIDMAD/train_tidmad_linoss_190k_denoising.yaml",
        "arch":        "blocks=4, ssm_size=32, H=128, damped_IMEX, loss=PSD",
    },
    # ---- Chronos zero-shot baselines ----------------------------------------
    # No checkpoint, no training curves. n_params from published model sizes.
    # Use --chronos_max_windows N to limit windows per file for faster eval.
    "chronos_tiny": {
        "display_name": "Chronos-Tiny (zero-shot)",
        "category": "baseline",
        "model_type": "chronos",
        "model_size":  "tiny",
        "chunk_size":  128,
        "arch":        "zero-shot, T5-tiny encoder-decoder",
        "n_params":    _CHRONOS_PARAMS["tiny"],
    },
    "chronos_base": {
        "display_name": "Chronos-Base (zero-shot)",
        "category": "baseline",
        "model_type": "chronos",
        "model_size":  "base",
        "chunk_size":  128,
        "arch":        "zero-shot, T5-base encoder-decoder",
        "n_params":    _CHRONOS_PARAMS["base"],
    },
    # ---- MOMENT zero-shot baselines -----------------------------------------
    # Native reconstruction head; 512-sample chunks, single forward pass.
    "moment_small": {
        "display_name": "MOMENT-Small (zero-shot)",
        "category": "baseline",
        "model_type": "moment",
        "model_size":  "small",
        "arch":        "zero-shot, patch-based transformer, reconstruction head",
        "n_params":    _MOMENT_PARAMS["small"],
    },
    "moment_base": {
        "display_name": "MOMENT-Base (zero-shot)",
        "category": "baseline",
        "model_type": "moment",
        "model_size":  "base",
        "arch":        "zero-shot, patch-based transformer, reconstruction head",
        "n_params":    _MOMENT_PARAMS["base"],
    },
}

# ---------------------------------------------------------------------------
# Metrics parsing
# ---------------------------------------------------------------------------

def load_metrics(csv_path: str):
    """Parse a Lightning metrics CSV (4-col conv or 6-col LinOSS format).

    Returns: train_epochs, train_losses, val_epochs, val_losses (lists)
    """
    df = pd.read_csv(csv_path)
    if "train/loss" in df.columns:
        train_col, val_col = "train/loss", "val/loss"
    elif "train/loss_epoch" in df.columns:
        train_col, val_col = "train/loss_epoch", "val/loss_epoch"
    else:
        return [], [], [], []

    train_df = df[df[train_col].notna()][["epoch", train_col]].copy()
    val_df   = df[df[val_col].notna()][["epoch", val_col]].copy()
    return (
        train_df["epoch"].tolist(),
        train_df[train_col].tolist(),
        val_df["epoch"].tolist(),
        val_df[val_col].tolist(),
    )


# ---------------------------------------------------------------------------
# Loss curve plotting
# ---------------------------------------------------------------------------

def plot_loss_curve(display_name, train_e, train_l, val_e, val_l,
                    out_path, best_val_loss=None):
    fig, ax = plt.subplots(figsize=(7, 4))
    if train_e:
        ax.plot(train_e, train_l, marker="o", ms=4, lw=1.5,
                label="train loss", color="#2c7bb6")
    if val_e:
        ax.plot(val_e, val_l, marker="s", ms=4, lw=1.5,
                label="val loss", color="#d7191c")
        if best_val_loss is not None:
            ax.axhline(best_val_loss, ls="--", lw=1, color="#d7191c", alpha=0.6,
                       label=f"best val = {best_val_loss:.3f}")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("MSE Loss")
    ax.set_title(f"{display_name} — Loss Curves")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  Saved loss curve: {out_path}")


# ---------------------------------------------------------------------------
# Parameter counting
# ---------------------------------------------------------------------------

def count_params_conv(cfg_path: str) -> int:
    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)
    mc = cfg["model"]["init_args"]["model"]
    mod, cls = mc["class_path"].rsplit(".", 1)
    model = getattr(importlib.import_module(mod), cls)(**mc.get("init_args", {}))
    return sum(p.numel() for p in model.parameters())


def count_params_linoss(cfg_path: str) -> int:
    import jax
    import equinox as eqx
    from models.linoss import LinOSS

    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)
    ia = cfg["model"]["init_args"]
    model = LinOSS(
        num_blocks=ia["num_blocks"], N=1,
        ssm_size=ia["ssm_size"], H=ia["H"],
        output_dim=1, task="denoising", output_step=1,
        discretization=ia.get("discretization", "IMEX"),
        seed=ia.get("seed", 0),
    )
    leaves = jax.tree_util.tree_leaves(eqx.filter(model, eqx.is_inexact_array))
    return int(sum(x.size for x in leaves))


def count_params(variant_cfg: dict, repo_root: Path) -> int:
    model_type = variant_cfg["model_type"]
    if model_type in ("chronos", "moment"):
        return variant_cfg["n_params"]          # known from published model card
    try:
        cfg_path = str(repo_root / variant_cfg["cfg"])
        if model_type == "conv":
            return count_params_conv(cfg_path)
        else:
            return count_params_linoss(cfg_path)
    except Exception as e:
        print(f"  WARNING: could not count params: {e}")
        return -1


# ---------------------------------------------------------------------------
# Checkpoint resolution
# ---------------------------------------------------------------------------

def select_best_eqx(ckpt_dir: str, metrics_csv: str | None = None) -> str | None:
    """Pick the best .eqx checkpoint.

    If a metrics_csv is provided, finds the step of the epoch with the lowest
    val/loss_epoch and returns the checkpoint whose step number is closest.
    Falls back to the checkpoint with the highest step number (latest).
    """
    files = glob.glob(os.path.join(ckpt_dir, "*.eqx"))
    if not files:
        return None

    def _step(f):
        m = re.search(r"step_(\d+)", os.path.basename(f))
        return int(m.group(1)) if m else 0

    steps = {f: _step(f) for f in files}

    if metrics_csv and os.path.exists(metrics_csv):
        try:
            df = pd.read_csv(metrics_csv)
            val_df = df[df["val/loss_epoch"].notna()][["step", "val/loss_epoch"]]
            if not val_df.empty:
                best_step = int(val_df.loc[val_df["val/loss_epoch"].idxmin(), "step"])
                # pick the checkpoint whose step is closest to best_step
                best_file = min(files, key=lambda f: abs(steps[f] - best_step))
                print(f"  Best val at step {best_step} -> using {os.path.basename(best_file)}")
                return best_file
        except Exception:
            pass  # fall through to latest

    latest = max(files, key=lambda f: steps[f])
    print(f"  Falling back to latest checkpoint: {os.path.basename(latest)}")
    return latest


def resolve_ckpt(variant_cfg: dict, repo_root: Path) -> str | None:
    if "ckpt" in variant_cfg:
        p = repo_root / variant_cfg["ckpt"]
        return str(p) if p.exists() else None
    if "ckpt_dir" in variant_cfg:
        metrics_csv = (str(repo_root / variant_cfg["metrics_csv"])
                       if "metrics_csv" in variant_cfg else None)
        return select_best_eqx(str(repo_root / variant_cfg["ckpt_dir"]), metrics_csv)
    return None     # Chronos and other zero-shot models


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------

def run_inference(variant_cfg: dict, ckpt_path: str | None, repo_root: Path,
                  data_dir: str, scale_path: str,
                  batch_size: int, device: torch.device,
                  max_windows: int | None = None,
                  stride: int = 1):
    """Load model and score all 8 test H5 files. Returns (score, snr_ch2, snr_ch1)."""
    model_type = variant_cfg["model_type"]

    if model_type == "conv":
        denoise_fn = load_conv_denoiser(ckpt_path, str(repo_root / variant_cfg["cfg"]), device)
    elif model_type == "linoss":
        denoise_fn = load_linoss_denoiser(ckpt_path, str(repo_root / variant_cfg["cfg"]))
    elif model_type == "chronos":
        denoise_fn = load_chronos_denoiser(
            variant_cfg["model_size"], device,
            chunk_size=variant_cfg.get("chunk_size", 128),
        )
    elif model_type == "moment":
        denoise_fn = load_moment_denoiser(variant_cfg["model_size"], device)
    else:
        raise ValueError(f"Unknown model_type: {model_type}")

    scaling = float(np.load(scale_path))
    all_ch2, all_ch1 = [], []
    for fname in TEST_FILES:
        fpath = os.path.join(data_dir, fname)
        if not os.path.exists(fpath):
            print(f"  WARNING: {fname} not found, skipping")
            continue
        print(f"  Scoring {fname} ...")
        ch2, ch1 = score_file(fpath, denoise_fn, scaling, batch_size, max_windows, stride)
        all_ch2.extend(ch2)
        all_ch1.extend(ch1)
        print(f"    -> {len(ch2)} windows")

    snr_ch2 = np.array(all_ch2)
    snr_ch1 = np.array(all_ch1)
    score   = compute_benchmark_score(snr_ch2, snr_ch1)
    return score, snr_ch2, snr_ch1


# ---------------------------------------------------------------------------
# Per-variant evaluation
# ---------------------------------------------------------------------------

def evaluate_variant(name: str, variant_cfg: dict, args, repo_root: Path):
    print(f"\n{'='*60}")
    print(f"Variant: {name}  ({variant_cfg['display_name']})")
    print(f"{'='*60}")

    out_dir = Path(args.out_dir) / name
    out_dir.mkdir(parents=True, exist_ok=True)

    is_zero_shot = variant_cfg["model_type"] in ("chronos", "moment")
    is_chronos = is_zero_shot  # kept for backward compat with downstream uses

    result = {
        "variant":          name,
        "display_name":     variant_cfg["display_name"],
        "category":         variant_cfg["category"],
        "model_type":       variant_cfg["model_type"],
        "arch":             variant_cfg.get("arch", ""),
        "timestamp":        datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "n_params":         None,
        "best_val_loss":    None,
        "best_val_epoch":   None,
        "n_epochs":         None,
        "final_train_loss": None,
        "benchmark_score":  None,
        "ckpt_path":        None,
        "error":            None,
    }

    # --- 1. Loss curves (skipped for zero-shot models) -----------------------
    metrics_path = variant_cfg.get("metrics_csv")
    if metrics_path and not is_chronos:
        full_metrics_path = str(repo_root / metrics_path)
        if os.path.exists(full_metrics_path):
            train_e, train_l, val_e, val_l = load_metrics(full_metrics_path)
            bvl = min(val_l) if val_l else float("nan")
            result["best_val_loss"]    = round(float(bvl), 5) if val_l else None
            result["best_val_epoch"]   = int(val_e[val_l.index(min(val_l))]) if val_l else None
            result["n_epochs"]         = int(max(max(train_e, default=0),
                                                  max(val_e,   default=0))) + 1
            result["final_train_loss"] = round(float(train_l[-1]), 5) if train_l else None
            plot_loss_curve(
                variant_cfg["display_name"],
                train_e, train_l, val_e, val_l,
                str(out_dir / "loss_curve.png"),
                best_val_loss=result["best_val_loss"],
            )
        else:
            print(f"  WARNING: metrics CSV not found: {full_metrics_path}")
    elif is_chronos:
        print("  No loss curves (zero-shot model, no training)")

    # --- 2. Count parameters -------------------------------------------------
    n_params = count_params(variant_cfg, repo_root)
    result["n_params"] = n_params
    print(f"  Parameters: {n_params:,}")

    # --- 3. Inference ---------------------------------------------------------
    if args.plot_only or variant_cfg.get("plot_only", False):
        print("  Skipping inference (--plot_only)")
    else:
        max_windows = args.chronos_max_windows if is_chronos else None
        if max_windows:
            print(f"  Chronos: capping at {max_windows} windows/file")
        stride = args.stride

        if is_chronos:
            ckpt_path = None   # no checkpoint for zero-shot
        else:
            ckpt_path = resolve_ckpt(variant_cfg, repo_root)
            if ckpt_path is None or not os.path.exists(ckpt_path):
                print("  WARNING: checkpoint not found, skipping inference")
                result["error"] = "checkpoint_not_found"
                _save_summary(result, out_dir)
                return result
            result["ckpt_path"] = ckpt_path
            print(f"  Checkpoint: {ckpt_path}")

        device = torch.device(args.device)
        try:
            score, snr_ch2, snr_ch1 = run_inference(
                variant_cfg, ckpt_path, repo_root,
                args.data_dir, args.scale_path,
                args.batch_size, device,
                max_windows=max_windows,
                stride=stride,
            )
            result["benchmark_score"] = round(score, 6)
            print(f"  Benchmark score: {score:.4f}")

            scores_csv = out_dir / "test_scores.csv"
            with open(scores_csv, "w", newline="") as f:
                w = csv.writer(f)
                w.writerow(["window_idx", "snr_ch2_reference", "snr_ch1_denoised"])
                for i, (c2, c1) in enumerate(zip(snr_ch2, snr_ch1)):
                    w.writerow([i, float(c2), float(c1)])
            print(f"  Saved test scores: {scores_csv}")

        except Exception as e:
            print(f"  ERROR during inference: {e}")
            result["error"] = str(e)

    # --- 4. Summary JSON ------------------------------------------------------
    _save_summary(result, out_dir)
    return result


def _save_summary(result: dict, out_dir: Path):
    summary_path = out_dir / "summary.json"
    with open(summary_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"  Saved summary: {summary_path}")


# ---------------------------------------------------------------------------
# Master CSV
# ---------------------------------------------------------------------------

MASTER_COLUMNS = [
    "variant", "display_name", "category", "model_type", "arch",
    "n_params", "n_epochs", "best_val_loss", "best_val_epoch",
    "final_train_loss", "benchmark_score", "ckpt_path", "timestamp", "error",
]


def write_master_csv(results: list[dict], out_path: str):
    """Write results to master CSV, merging with any existing rows by variant key."""
    existing = {}
    if os.path.exists(out_path):
        with open(out_path, newline="") as f:
            for row in csv.DictReader(f):
                existing[row["variant"]] = row

    # New results overwrite same-key rows; others are kept
    for r in results:
        existing[r["variant"]] = r

    with open(out_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=MASTER_COLUMNS, extrasaction="ignore")
        w.writeheader()
        for row in existing.values():
            w.writerow(row)
    print(f"\nMaster CSV written: {out_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="TIDMAD evaluate_all")
    parser.add_argument("--variants", nargs="*", default=None,
                        help="Subset of variants to run (default: all).")
    parser.add_argument("--plot_only", action="store_true",
                        help="Only generate loss curves, skip inference.")
    parser.add_argument("--data_dir",   default="data/TIDMAD/original")
    parser.add_argument("--scale_path", default="data/TIDMAD/preprocessed/scale.npy")
    parser.add_argument("--out_dir",    default="benchmarks/TIDMAD/results")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--device",     default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--chronos_max_windows", type=int, default=None,
                        help="Max windows per file for Chronos (slow). Default: all.")
    parser.add_argument("--stride", type=int, default=1,
                        help="Process every Nth window (default: 1=all). stride=10 matches original --coarse mode.")
    args = parser.parse_args()

    repo_root = _REPO
    for attr in ("data_dir", "scale_path", "out_dir"):
        v = getattr(args, attr)
        if not os.path.isabs(v):
            setattr(args, attr, str(repo_root / v))

    selected = args.variants or list(VARIANTS.keys())
    unknown  = [v for v in selected if v not in VARIANTS]
    if unknown:
        parser.error(f"Unknown variants: {unknown}. Available: {list(VARIANTS.keys())}")

    print(f"Running: {selected}")
    print(f"Device:  {args.device}")
    print(f"Output:  {args.out_dir}")

    all_results = []
    for name in selected:
        try:
            r = evaluate_variant(name, VARIANTS[name], args, repo_root)
            all_results.append(r)
        except Exception as e:
            print(f"ERROR evaluating {name}: {e}")
            all_results.append({"variant": name, "error": str(e)})

    os.makedirs(args.out_dir, exist_ok=True)
    write_master_csv(all_results, os.path.join(args.out_dir, "master_results.csv"))

    # Summary table
    print(f"\n{'Variant':<26} {'Params':>10} {'Best Val':>10} {'Score':>10}")
    print("-" * 60)
    for r in all_results:
        params = f"{r['n_params']:,}" if r.get("n_params") and r["n_params"] > 0 else "?"
        bval   = f"{r['best_val_loss']:.3f}" if r.get("best_val_loss") else "n/a"
        score  = f"{r['benchmark_score']:.4f}" if r.get("benchmark_score") is not None else "?"
        print(f"{r['variant']:<26} {params:>10} {bval:>10} {score:>10}")


if __name__ == "__main__":
    main()
