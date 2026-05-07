"""
Forecasting evaluator.

Splits each toy signal into a context window (first C samples) and a horizon
(last H samples).  Asks the wrapped foundation model to forecast the horizon
given only the noisy context, then measures how close the forecast is to the
CLEAN horizon -- i.e. whether the model has inferred the underlying
sinusoidal pattern despite the heavy noise (SNR ~ 0.14).

Naive baselines (repeat-last, zero, seasonal-naive) are computed in the same
loop so every model is compared against them.
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
from torch.utils.data import DataLoader

from dataloader.tidmad_dataloader import Param

from .metrics import (
    bootstrap_ci,
    crps,
    mae,
    mse,
    peak_freq_error,
    quantile_coverage,
    sfer,
    spectral_mse,
)
from ..wrappers.base import BaseFoundationModel, ForecastResult


# ────────────────────────────────────────────────────────────────────────────
# Naive baselines (computed without any foundation model)
# ────────────────────────────────────────────────────────────────────────────

def baseline_repeat_last(context: np.ndarray, horizon: int) -> np.ndarray:
    """Forecast = constant (last value of context).  Shape (B, horizon)."""
    return np.repeat(context[:, -1:], horizon, axis=-1)


def baseline_zero(context: np.ndarray, horizon: int) -> np.ndarray:
    """Forecast = 0.  For a zero-mean sinusoid, this is surprisingly strong at
    low SNR since the per-sample MSE collapses to the sinusoid's power (~ A^2/2).
    """
    return np.zeros((context.shape[0], horizon), dtype=np.float32)


def baseline_seasonal_naive(
    context: np.ndarray, horizon: int, sample_rate: float = 64.0
) -> np.ndarray:
    """Estimate period from FFT of context, then repeat the last period.

    If the detected period is longer than the context, fall back to repeat-last.
    """
    B, C = context.shape
    out = np.zeros((B, horizon), dtype=np.float32)
    freqs = np.fft.rfftfreq(C, d=1.0 / sample_rate)
    spec = np.abs(np.fft.rfft(context, axis=-1))
    # Zero-out the DC bin so we don't pick up the signal mean.
    spec[:, 0] = 0.0
    peak_idx = np.argmax(spec, axis=-1)                       # (B,)
    f_hat = freqs[peak_idx]                                   # (B,)
    for b in range(B):
        if f_hat[b] < 1e-6:
            out[b] = context[b, -1]
            continue
        period = int(round(sample_rate / f_hat[b]))
        if period < 1 or period > C:
            out[b] = context[b, -1]
            continue
        last_period = context[b, -period:]
        # Tile and truncate
        reps = int(np.ceil(horizon / period))
        tiled = np.tile(last_period, reps)[:horizon]
        out[b] = tiled
    return out


# ────────────────────────────────────────────────────────────────────────────
# Main evaluator
# ────────────────────────────────────────────────────────────────────────────

def evaluate_forecasting(
    wrapper: BaseFoundationModel,
    dataloader: DataLoader,
    *,
    context_len: int = 512,
    horizon: int = 128,
    sample_rate: float = 64.0,
    include_baselines: bool = True,
    baseline_fns: dict[str, Callable] | None = None,
    max_batches: int | None = None,
) -> dict:
    """Run forecasting evaluation over the whole test set.

    Parameters
    ----------
    wrapper : loaded BaseFoundationModel
    dataloader : yields (sig_bkg, sig, params) with sig_bkg.shape (B, L>=C+H)
    context_len : context window C
    horizon : forecast horizon H (C + H <= L)
    sample_rate : Hz, for peak-frequency metrics
    include_baselines : also evaluate repeat-last / zero / seasonal
    baseline_fns : optional override of the baseline dict
    max_batches : stop after N batches (for smoke tests)

    Returns
    -------
    dict with keys:
      per_sample : {'<variant>': {metric: np.ndarray}}  per-sample arrays
      aggregate  : {'<variant>': {metric: {'mean': ..., 'ci_lower': ..., 'ci_upper': ...}}}
      meta       : {'context_len': ..., 'horizon': ..., 'n_samples': ...}
    """
    variants: dict[str, dict[str, list[np.ndarray]]] = {
        wrapper.name: {k: [] for k in (
            "mse_vs_clean", "mae_vs_clean", "crps_vs_clean",
            "spectral_mse", "peak_freq_error", "sfer",
            "mse_vs_noisy", "mae_vs_noisy",
        )},
    }
    if include_baselines:
        if baseline_fns is None:
            baseline_fns = {
                "baseline_repeat_last": baseline_repeat_last,
                "baseline_zero": baseline_zero,
                "baseline_seasonal_naive": lambda c, h: baseline_seasonal_naive(c, h, sample_rate),
            }
        for name in baseline_fns:
            variants[name] = {k: [] for k in variants[wrapper.name]}

    snr_gt_chunks: list[np.ndarray] = []
    freq_true_chunks: list[np.ndarray] = []

    for batch_idx, (sig_bkg, sig, params) in enumerate(dataloader):
        if max_batches is not None and batch_idx >= max_batches:
            break
        sig_bkg_np = sig_bkg.numpy()                  # (B, L)
        sig_np = sig.numpy()                          # (B, L)
        params_np = params.numpy()                    # (B, 5)

        if sig_bkg_np.shape[-1] < context_len + horizon:
            raise ValueError(
                f"Signal length {sig_bkg_np.shape[-1]} < context_len ({context_len}) "
                f"+ horizon ({horizon})."
            )
        context_noisy = sig_bkg_np[:, :context_len]   # (B, C)
        horizon_clean = sig_np[:, context_len:context_len + horizon]    # (B, H)
        horizon_noisy = sig_bkg_np[:, context_len:context_len + horizon]  # (B, H)

        snr_gt_chunks.append(params_np[:, int(Param.snr)])
        freq_true_chunks.append(params_np[:, int(Param.frequency_hz)])

        # ── Foundation model forecast ────────────────────────────────────
        result: ForecastResult = wrapper.forecast(context_noisy, horizon=horizon)
        _accumulate(
            variants[wrapper.name],
            pred=result.point_forecast,
            samples=result.samples,
            horizon_clean=horizon_clean,
            horizon_noisy=horizon_noisy,
            freq_true=params_np[:, int(Param.frequency_hz)],
            sample_rate=sample_rate,
        )

        # ── Baselines ────────────────────────────────────────────────────
        if include_baselines:
            for name, fn in baseline_fns.items():
                pred = fn(context_noisy, horizon)
                _accumulate(
                    variants[name],
                    pred=pred,
                    samples=None,
                    horizon_clean=horizon_clean,
                    horizon_noisy=horizon_noisy,
                    freq_true=params_np[:, int(Param.frequency_hz)],
                    sample_rate=sample_rate,
                )

    # Concatenate per-sample arrays and compute aggregates.
    snr_gt = np.concatenate(snr_gt_chunks)
    freq_true = np.concatenate(freq_true_chunks)
    per_sample: dict[str, dict[str, np.ndarray]] = {}
    aggregate: dict[str, dict[str, dict[str, float]]] = {}
    for v_name, metric_lists in variants.items():
        per_sample[v_name] = {
            m: np.concatenate(lst) for m, lst in metric_lists.items() if lst
        }
        aggregate[v_name] = {
            m: _aggregate(arr) for m, arr in per_sample[v_name].items()
        }
    return {
        "per_sample": per_sample,
        "aggregate": aggregate,
        "snr_gt": snr_gt,
        "freq_true": freq_true,
        "meta": {
            "context_len": context_len,
            "horizon": horizon,
            "n_samples": int(len(snr_gt)),
            "sample_rate": sample_rate,
        },
    }


# ────────────────────────────────────────────────────────────────────────────
# Helpers
# ────────────────────────────────────────────────────────────────────────────

def _accumulate(
    target: dict[str, list[np.ndarray]],
    *,
    pred: np.ndarray,
    samples: np.ndarray | None,
    horizon_clean: np.ndarray,
    horizon_noisy: np.ndarray,
    freq_true: np.ndarray,
    sample_rate: float,
) -> None:
    """Append per-sample metric arrays into `target`."""
    target["mse_vs_clean"].append(mse(pred, horizon_clean))
    target["mae_vs_clean"].append(mae(pred, horizon_clean))
    target["spectral_mse"].append(spectral_mse(pred, horizon_clean))
    target["peak_freq_error"].append(
        peak_freq_error(pred, freq_true, sample_rate)
    )
    target["sfer"].append(sfer(pred, horizon_clean))
    target["mse_vs_noisy"].append(mse(pred, horizon_noisy))
    target["mae_vs_noisy"].append(mae(pred, horizon_noisy))
    if samples is not None:
        target["crps_vs_clean"].append(crps(samples, horizon_clean))
    else:
        # CRPS of a deterministic forecast reduces to MAE.
        target["crps_vs_clean"].append(mae(pred, horizon_clean))


def _aggregate(arr: np.ndarray) -> dict[str, float]:
    """Return mean + 95% bootstrap CI of a per-sample metric array."""
    mean_val = float(np.mean(arr))
    lo, hi = bootstrap_ci(arr, n_boot=500, alpha=0.05)
    return {"mean": mean_val, "ci_lower": lo, "ci_upper": hi}
