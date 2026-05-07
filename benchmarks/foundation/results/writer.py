"""CSV output for the foundation-model benchmark.

Output directories
------------------
{out_dir}/{model}_{mode}/results.csv   -- denoising
{out_dir}/{model}/forecast_results.csv -- forecasting
{out_dir}/{model}/embedding_results.csv -- embedding+regression
{out_dir}/summary.csv                  -- aggregated comparison table
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


# ────────────────────────────────────────────────────────────────────────────
# Denoising: compare_pipelines.py-compatible CSV
# ────────────────────────────────────────────────────────────────────────────

def save_denoise_csv(
    out_dir: Path,
    *,
    y_true: np.ndarray,
    y_raw: np.ndarray,
    y_den: np.ndarray,
    y_signal: np.ndarray,
    snr_gt: np.ndarray,
    snr_raw: np.ndarray,
    snr_denoised: np.ndarray,
    param_names: list[str],
) -> Path:
    """Write denoising `results.csv` matching the existing schema exactly.

    The column order is: snr_gt, snr_raw, snr_denoised,
    then per-parameter {name}_true, {name}_raw, {name}_denoised, {name}_signal.

    Writes to `{out_dir}/results.csv`.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    data: dict[str, np.ndarray] = {
        "snr_gt":       snr_gt,
        "snr_raw":      snr_raw,
        "snr_denoised": snr_denoised,
    }
    for i, name in enumerate(param_names):
        data[f"{name}_true"]     = y_true[:, i]
        data[f"{name}_raw"]      = y_raw[:, i]
        data[f"{name}_denoised"] = y_den[:, i]
        data[f"{name}_signal"]   = y_signal[:, i]
    path = out_dir / "results.csv"
    pd.DataFrame(data).to_csv(path, index=False)
    return path


# ────────────────────────────────────────────────────────────────────────────
# Forecasting CSV (per-sample detailed output)
# ────────────────────────────────────────────────────────────────────────────

def save_forecast_csv(
    out_dir: Path,
    *,
    model: str,
    mode: str,
    snr_gt: np.ndarray,
    freq_true: np.ndarray,
    per_sample_metrics: dict[str, np.ndarray],
) -> Path:
    """Write a per-sample forecasting CSV.

    `per_sample_metrics` maps metric name → (N,) array.  Typical keys:
        mse_vs_clean, mae_vs_clean, crps_vs_clean, spectral_mse,
        peak_freq_error, sfer, mse_vs_noisy, mae_vs_noisy
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    data = {"snr_gt": snr_gt, "freq_true": freq_true, **per_sample_metrics}
    df = pd.DataFrame(data)
    df.insert(0, "mode", mode)
    df.insert(0, "model", model)
    path = out_dir / "forecast_results.csv"
    df.to_csv(path, index=False)
    return path


# ────────────────────────────────────────────────────────────────────────────
# Embedding+regression CSV
# ────────────────────────────────────────────────────────────────────────────

def save_embedding_csv(
    out_dir: Path,
    *,
    model: str,
    mode: str,
    per_param_metrics: dict[str, dict[str, float]],
) -> Path:
    """Write one row per (param, metric) summary.

    `per_param_metrics[param] = {'rmse': ..., 'mae': ..., 'r2': ...,
                                 'rmse_snr_low': ..., 'rmse_snr_mid': ...,
                                 'rmse_snr_high': ...}`
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for param, metrics in per_param_metrics.items():
        row = {"model": model, "mode": mode, "param": param}
        row.update(metrics)
        rows.append(row)
    df = pd.DataFrame(rows)
    path = out_dir / "embedding_results.csv"
    df.to_csv(path, index=False)
    return path


# ────────────────────────────────────────────────────────────────────────────
# Summary: long-format table aggregating everything
# ────────────────────────────────────────────────────────────────────────────

def append_summary_row(
    summary_path: Path,
    *,
    model: str,
    task: str,
    mode: str,
    metric: str,
    value: float,
    ci_lower: float | None = None,
    ci_upper: float | None = None,
) -> None:
    """Append a single row to `summary.csv`, creating the file if needed."""
    summary_path = Path(summary_path)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    row = pd.DataFrame([{
        "model": model, "task": task, "mode": mode, "metric": metric,
        "value": value, "ci_lower": ci_lower, "ci_upper": ci_upper,
    }])
    header = not summary_path.exists()
    row.to_csv(summary_path, mode="a", header=header, index=False)


def save_summary(summary_path: Path, rows: Iterable[dict]) -> Path:
    """Write a full summary.csv from a list of row dicts (overwrites)."""
    summary_path = Path(summary_path)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(list(rows))
    df.to_csv(summary_path, index=False)
    return summary_path
