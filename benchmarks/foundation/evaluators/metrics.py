"""
Shared metric functions for forecasting / denoising / embedding evaluation.

All functions accept numpy float arrays and return either a scalar or a
per-sample 1-D array (so downstream code can aggregate however it likes).

SNR convention: power(reference) / power(prediction - reference).  Higher is better.
"""

from __future__ import annotations

import numpy as np


# ────────────────────────────────────────────────────────────────────────────
# Point-forecast / point-denoise metrics
# ────────────────────────────────────────────────────────────────────────────

def mse(pred: np.ndarray, true: np.ndarray) -> np.ndarray:
    """Per-sample MSE averaged over the last axis.  Shape (B,)."""
    return np.mean((pred - true) ** 2, axis=-1)


def mae(pred: np.ndarray, true: np.ndarray) -> np.ndarray:
    """Per-sample MAE averaged over the last axis.  Shape (B,)."""
    return np.mean(np.abs(pred - true), axis=-1)


def rmse(pred: np.ndarray, true: np.ndarray) -> np.ndarray:
    """Per-sample RMSE."""
    return np.sqrt(mse(pred, true))


def waveform_snr(signal: np.ndarray, reference: np.ndarray) -> np.ndarray:
    """Per-sample SNR: power(reference) / power(signal - reference).  Shape (B,).

    Higher is better; power(reference) / power(signal - reference).
    """
    ref_power = np.mean(reference ** 2, axis=-1)
    noise_power = np.mean((signal - reference) ** 2, axis=-1)
    return ref_power / (noise_power + 1e-10)


def snr_improvement(
    denoised: np.ndarray, noisy: np.ndarray, clean: np.ndarray
) -> np.ndarray:
    """Per-sample SNR improvement factor: SNR_out / SNR_in.  Shape (B,)."""
    return waveform_snr(denoised, clean) / (waveform_snr(noisy, clean) + 1e-10)


# ────────────────────────────────────────────────────────────────────────────
# Frequency-domain metrics
# ────────────────────────────────────────────────────────────────────────────

def spectral_mse(pred: np.ndarray, true: np.ndarray) -> np.ndarray:
    """Per-sample MSE between power spectral densities.  Shape (B,).

    Aligned with the PSD-based SNR benchmark metric (TIDMAD Benchmark 1).
    """
    psd_pred = np.abs(np.fft.rfft(pred, axis=-1)) ** 2
    psd_true = np.abs(np.fft.rfft(true, axis=-1)) ** 2
    return np.mean((psd_pred - psd_true) ** 2, axis=-1)


def peak_freq_error(
    pred: np.ndarray, true_freq: np.ndarray, sample_rate: float
) -> np.ndarray:
    """|dominant frequency of pred - true frequency|.  Shape (B,).

    Parameters
    ----------
    pred : (B, L) predicted signal in the time domain.
    true_freq : (B,) ground-truth sinusoid frequency in Hz.
    sample_rate : sampling rate used to interpret FFT bins.
    """
    B, L = pred.shape
    freqs = np.fft.rfftfreq(L, d=1.0 / sample_rate)    # (L//2+1,)
    spec = np.abs(np.fft.rfft(pred, axis=-1))          # (B, L//2+1)
    peak_idx = np.argmax(spec, axis=-1)                # (B,)
    f_peak = freqs[peak_idx]                           # (B,)
    return np.abs(f_peak - true_freq)


# ────────────────────────────────────────────────────────────────────────────
# Probabilistic metrics
# ────────────────────────────────────────────────────────────────────────────

def crps(samples: np.ndarray, true: np.ndarray) -> np.ndarray:
    """Per-sample empirical CRPS from N forecast samples.  Shape (B,).

    CRPS = E|X - y| - 0.5 * E|X - X'|   where X, X' ~ forecast distribution.

    Parameters
    ----------
    samples : (B, N, H) forecast samples.
    true :    (B, H) ground truth.

    For deterministic forecasts (N=1), this reduces to MAE.
    """
    # Term 1: E|X - y|
    term1 = np.mean(np.abs(samples - true[:, None, :]), axis=1)   # (B, H)
    # Term 2: 0.5 * E|X - X'|
    # Use the unbiased empirical estimator: mean over all pairs (i != j).
    # For tractability when N can be large, use the sorted-absolute trick:
    #   E|X - X'| = 2 * sum_{i<j} |X_i - X_j| / N^2  (approximation; exact for
    #   iid samples when N is large).  We implement the direct O(N^2) computation,
    #   which is fine for N<=100.
    N = samples.shape[1]
    if N == 1:
        # Degenerate: CRPS reduces to MAE.
        return np.mean(term1, axis=-1)
    diff = np.abs(samples[:, :, None, :] - samples[:, None, :, :])  # (B, N, N, H)
    term2 = 0.5 * np.mean(diff, axis=(1, 2))                        # (B, H)
    return np.mean(term1 - term2, axis=-1)                          # (B,)


def quantile_coverage(
    samples: np.ndarray, true: np.ndarray, q_lo: float, q_hi: float
) -> float:
    """Fraction of true values inside the [q_lo, q_hi] quantile envelope.

    Parameters
    ----------
    samples : (B, N, H)
    true :    (B, H)
    q_lo, q_hi : quantile levels in [0, 1], e.g. 0.1 and 0.9 for 80% coverage.

    Returns
    -------
    scalar float: fraction of time-steps × samples that fell inside.
    """
    lo = np.quantile(samples, q_lo, axis=1)   # (B, H)
    hi = np.quantile(samples, q_hi, axis=1)   # (B, H)
    inside = (true >= lo) & (true <= hi)
    return float(np.mean(inside))


# ────────────────────────────────────────────────────────────────────────────
# Signal-to-forecast-error ratio (denoising-style metric for forecasts)
# ────────────────────────────────────────────────────────────────────────────

def sfer(pred: np.ndarray, true: np.ndarray) -> np.ndarray:
    """Signal-to-Forecast-Error Ratio.  Alias for waveform_snr."""
    return waveform_snr(pred, true)


# ────────────────────────────────────────────────────────────────────────────
# Uncertainty quantification: bootstrap confidence intervals
# ────────────────────────────────────────────────────────────────────────────

def bootstrap_ci(
    values: np.ndarray,
    n_boot: int = 1000,
    alpha: float = 0.05,
    stat=np.mean,
    rng: np.random.Generator | None = None,
) -> tuple[float, float]:
    """Percentile bootstrap CI for a statistic of `values`.

    Parameters
    ----------
    values : (N,) array of per-sample metric values.
    n_boot : number of bootstrap resamples.
    alpha  : two-sided significance level (e.g. 0.05 for 95% CI).
    stat   : aggregation function (default: mean).

    Returns
    -------
    (lo, hi) : the (alpha/2, 1-alpha/2) quantiles of the bootstrap
    distribution of `stat`.
    """
    if rng is None:
        rng = np.random.default_rng(0)
    values = np.asarray(values)
    N = len(values)
    if N == 0:
        return float("nan"), float("nan")
    samples = rng.choice(values, size=(n_boot, N), replace=True)
    stats = np.array([stat(s) for s in samples])
    lo = float(np.quantile(stats, alpha / 2))
    hi = float(np.quantile(stats, 1 - alpha / 2))
    return lo, hi


# ────────────────────────────────────────────────────────────────────────────
# SNR-stratified aggregation
# ────────────────────────────────────────────────────────────────────────────

SNR_BINS: dict[str, tuple[float, float]] = {
    "low":  (0.0, 0.1),
    "mid":  (0.1, 0.3),
    "high": (0.3, float("inf")),
}


def stratified_rmse(
    pred: np.ndarray, true: np.ndarray, snr: np.ndarray
) -> dict[str, float]:
    """Per-SNR-bin RMSE aggregated over all samples in that bin.

    Parameters
    ----------
    pred, true : (N,) or (N, P) arrays
    snr :        (N,) ground-truth SNR per sample

    Returns
    -------
    {bin_name: rmse_value} for each bin in SNR_BINS.
    """
    out: dict[str, float] = {}
    diff = pred - true
    if diff.ndim == 1:
        sq = diff ** 2
    else:
        sq = np.mean(diff ** 2, axis=-1)
    for name, (lo, hi) in SNR_BINS.items():
        mask = (snr >= lo) & (snr < hi)
        if mask.sum() == 0:
            out[name] = float("nan")
        else:
            out[name] = float(np.sqrt(np.mean(sq[mask])))
    return out
