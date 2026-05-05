"""
Plotting for the foundation-model benchmark.

Loads the aggregated `summary.csv` produced by `run_benchmark.py` and
generates a small set of comparison figures:

1. ``forecast_mse_bar.png``        -- MSE vs clean (models + baselines)
2. ``forecast_multimetric.png``    -- grouped bars for MSE / MAE / CRPS / SFER
3. ``forecast_peak_freq.png``      -- peak-frequency error (Hz)
4. ``embedding_rmse_by_param.png`` -- per-parameter RMSE grouped by model
5. ``embedding_r2_by_param.png``   -- per-parameter R²
6. ``embedding_snr_stratified.png``-- RMSE × SNR bin for each parameter

All outputs land in ``out_dir``.  Plots are saved as PNG at 150 DPI.

Usage
-----
    python benchmarks/foundation/results/plotting.py \
        --summary plots/toy/foundation/summary.csv \
        --out_dir plots/toy/foundation/plots
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# Stable color map so the same model is drawn in the same color across figures.
_COLORS: Dict[str, str] = {
    "moment":    "#1f77b4",
    "chronos":   "#ff7f0e",
    "timemoe":   "#2ca02c",
    "timesfm":   "#d62728",
    "moirai":    "#9467bd",
    "lagllama":  "#8c564b",
    "baseline_zero":           "#7f7f7f",
    "baseline_repeat_last":    "#bcbd22",
    "baseline_seasonal_naive": "#17becf",
}

_FOUNDATION = ["moment", "chronos", "timemoe", "timesfm", "moirai", "lagllama"]
_BASELINES  = ["baseline_zero", "baseline_repeat_last", "baseline_seasonal_naive"]


# ───────────────────────────────────────────────────────────────────────────
# Helpers
# ───────────────────────────────────────────────────────────────────────────

def _color(name: str) -> str:
    return _COLORS.get(name, "#888888")


def _ordered_models(df: pd.DataFrame, include_baselines: bool = True) -> List[str]:
    present = set(df["model"].unique())
    order = [m for m in _FOUNDATION if m in present]
    if include_baselines:
        order += [m for m in _BASELINES if m in present]
    return order


def _get_metric(df: pd.DataFrame, model: str, task: str, metric: str):
    row = df[(df["model"] == model) & (df["task"] == task) & (df["metric"] == metric)]
    if row.empty:
        return None
    return row.iloc[0]


# ───────────────────────────────────────────────────────────────────────────
# Forecasting plots
# ───────────────────────────────────────────────────────────────────────────

def plot_forecast_mse(df: pd.DataFrame, out_path: Path) -> None:
    """Horizontal bar of MSE vs clean, with 95% CI error bars."""
    models = _ordered_models(df, include_baselines=True)
    vals, lo, hi = [], [], []
    for m in models:
        row = _get_metric(df, m, "forecasting", "mse_vs_clean")
        if row is None:
            continue
        vals.append(float(row["value"]))
        lo.append(float(row["ci_lower"]) if row["ci_lower"] else float(row["value"]))
        hi.append(float(row["ci_upper"]) if row["ci_upper"] else float(row["value"]))
    fig, ax = plt.subplots(figsize=(8, max(3, 0.45 * len(models))))
    y = np.arange(len(models))
    errs = np.vstack([np.array(vals) - np.array(lo), np.array(hi) - np.array(vals)])
    bar_colors = [_color(m) for m in models]
    ax.barh(y, vals, xerr=errs, color=bar_colors, edgecolor="black", alpha=0.85)
    ax.set_yticks(y)
    ax.set_yticklabels(models)
    ax.invert_yaxis()
    ax.set_xlabel("MSE vs clean (lower is better)")
    ax.set_title("Forecasting — MSE against clean signal (95% bootstrap CI)")
    ax.grid(axis="x", alpha=0.3)
    # Highlight the zero baseline vertical line
    zero_row = _get_metric(df, "baseline_zero", "forecasting", "mse_vs_clean")
    if zero_row is not None:
        ax.axvline(float(zero_row["value"]), color="#7f7f7f", ls="--", lw=1.0,
                   label="zero baseline (noise floor)")
        ax.legend(loc="lower right", fontsize=9)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_forecast_multimetric(df: pd.DataFrame, out_path: Path) -> None:
    """Grouped bars of multiple forecasting metrics across foundation models."""
    metrics = ["mse_vs_clean", "mae_vs_clean", "crps_vs_clean", "sfer"]
    labels  = ["MSE",          "MAE",          "CRPS",          "SFER"]
    models = _ordered_models(df, include_baselines=False)
    if not models:
        return
    data = np.full((len(models), len(metrics)), np.nan)
    for i, m in enumerate(models):
        for j, k in enumerate(metrics):
            row = _get_metric(df, m, "forecasting", k)
            if row is not None:
                data[i, j] = float(row["value"])
    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(models))
    width = 0.8 / len(metrics)
    for j, (lbl, col) in enumerate(zip(labels, ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"])):
        ax.bar(x + (j - len(metrics) / 2 + 0.5) * width, data[:, j],
               width, label=lbl, color=col, edgecolor="black", alpha=0.85)
    ax.set_xticks(x)
    ax.set_xticklabels(models)
    ax.set_ylabel("metric value")
    ax.set_title("Forecasting metrics — foundation models only")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_forecast_peak_freq(df: pd.DataFrame, out_path: Path) -> None:
    """Peak-frequency error (Hz) bar chart."""
    models = _ordered_models(df, include_baselines=True)
    vals = []
    names = []
    for m in models:
        row = _get_metric(df, m, "forecasting", "peak_freq_error")
        if row is None:
            continue
        names.append(m)
        vals.append(float(row["value"]))
    if not names:
        return
    fig, ax = plt.subplots(figsize=(8, max(3, 0.45 * len(names))))
    ax.barh(np.arange(len(names)), vals,
            color=[_color(n) for n in names], edgecolor="black", alpha=0.85)
    ax.set_yticks(np.arange(len(names)))
    ax.set_yticklabels(names)
    ax.invert_yaxis()
    ax.set_xlabel("|peak-FFT(forecast) − true frequency| (Hz)  — lower is better")
    ax.set_title("Forecasting — peak frequency error")
    ax.grid(axis="x", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


# ───────────────────────────────────────────────────────────────────────────
# Embedding plots
# ───────────────────────────────────────────────────────────────────────────

def _embedding_params(df: pd.DataFrame) -> List[str]:
    params = set()
    for m in df[df["task"] == "embedding"]["metric"]:
        if "/" in m:
            params.add(m.split("/")[0])
    return sorted(params)


def plot_embedding_rmse(df: pd.DataFrame, out_path: Path) -> None:
    """Grouped bar of per-param RMSE across models."""
    params = _embedding_params(df)
    models = [m for m in _FOUNDATION if m in df[df["task"] == "embedding"]["model"].unique()]
    if not (params and models):
        return
    data = np.full((len(models), len(params)), np.nan)
    for i, m in enumerate(models):
        for j, p in enumerate(params):
            row = _get_metric(df, m, "embedding", f"{p}/rmse")
            if row is not None:
                data[i, j] = float(row["value"])
    fig, ax = plt.subplots(figsize=(9, 5))
    x = np.arange(len(params))
    width = 0.8 / max(1, len(models))
    for i, m in enumerate(models):
        ax.bar(x + (i - len(models) / 2 + 0.5) * width, data[i], width,
               label=m, color=_color(m), edgecolor="black", alpha=0.85)
    ax.set_xticks(x)
    ax.set_xticklabels(params)
    ax.set_ylabel("RMSE (lower is better)")
    ax.set_title("Embedding → regression — per-parameter RMSE")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_embedding_r2(df: pd.DataFrame, out_path: Path) -> None:
    """Grouped bar of per-param R²."""
    params = _embedding_params(df)
    models = [m for m in _FOUNDATION if m in df[df["task"] == "embedding"]["model"].unique()]
    if not (params and models):
        return
    data = np.full((len(models), len(params)), np.nan)
    for i, m in enumerate(models):
        for j, p in enumerate(params):
            row = _get_metric(df, m, "embedding", f"{p}/r2")
            if row is not None:
                data[i, j] = float(row["value"])
    fig, ax = plt.subplots(figsize=(9, 5))
    x = np.arange(len(params))
    width = 0.8 / max(1, len(models))
    for i, m in enumerate(models):
        ax.bar(x + (i - len(models) / 2 + 0.5) * width, data[i], width,
               label=m, color=_color(m), edgecolor="black", alpha=0.85)
    ax.set_xticks(x)
    ax.set_xticklabels(params)
    ax.set_ylabel("R²  (higher is better; 0 = random)")
    ax.set_title("Embedding → regression — per-parameter R²")
    ax.axhline(0, color="black", lw=0.6)
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_embedding_snr_stratified(df: pd.DataFrame, out_path: Path) -> None:
    """For each parameter, plot RMSE in low / mid / high SNR bins per model."""
    params = _embedding_params(df)
    models = [m for m in _FOUNDATION if m in df[df["task"] == "embedding"]["model"].unique()]
    if not (params and models):
        return
    bins = ["rmse_snr_low", "rmse_snr_mid", "rmse_snr_high"]
    bin_labels = ["low SNR", "mid SNR", "high SNR"]
    fig, axes = plt.subplots(1, len(params),
                             figsize=(4.5 * len(params), 4.5), sharey=False)
    if len(params) == 1:
        axes = [axes]
    for ax, p in zip(axes, params):
        x = np.arange(len(bin_labels))
        width = 0.8 / max(1, len(models))
        for i, m in enumerate(models):
            vals = []
            for b in bins:
                row = _get_metric(df, m, "embedding", f"{p}/{b}")
                vals.append(float(row["value"]) if row is not None else np.nan)
            ax.bar(x + (i - len(models) / 2 + 0.5) * width, vals, width,
                   label=m, color=_color(m), edgecolor="black", alpha=0.85)
        ax.set_xticks(x)
        ax.set_xticklabels(bin_labels)
        ax.set_title(p)
        ax.grid(axis="y", alpha=0.3)
    axes[0].set_ylabel("RMSE (lower is better)")
    axes[-1].legend(loc="upper right", fontsize=8)
    fig.suptitle("Embedding regression — SNR-stratified RMSE")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


# ───────────────────────────────────────────────────────────────────────────
# CLI
# ───────────────────────────────────────────────────────────────────────────

def generate_all(summary_csv: Path, out_dir: Path) -> list[Path]:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(summary_csv)

    generated: list[Path] = []
    fns = [
        ("forecast_mse_bar.png",         plot_forecast_mse),
        ("forecast_multimetric.png",     plot_forecast_multimetric),
        ("forecast_peak_freq.png",       plot_forecast_peak_freq),
        ("embedding_rmse_by_param.png",  plot_embedding_rmse),
        ("embedding_r2_by_param.png",    plot_embedding_r2),
        ("embedding_snr_stratified.png", plot_embedding_snr_stratified),
    ]
    for fname, fn in fns:
        path = out_dir / fname
        try:
            fn(df, path)
            if path.exists():
                generated.append(path)
                print(f"  wrote {path}")
        except Exception as e:
            print(f"  SKIP {fname}: {e}")
    return generated


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--summary", default="plots/toy/foundation/summary.csv",
                   help="Aggregated summary.csv from run_benchmark.py")
    p.add_argument("--out_dir", default="plots/toy/foundation/plots",
                   help="Directory to write PNG plots")
    args = p.parse_args()
    generate_all(Path(args.summary), Path(args.out_dir))


if __name__ == "__main__":
    main()
