"""
Denoising evaluator.

Runs the wrapped foundation model on the full 640-sample noisy signal, passes
the result through the existing trained regressors (raw-input and clean-input)
so the output CSV matches the schema produced by
`benchmarks/toy/eval_pipeline.py` and can be consumed by the existing
`compare_pipelines.py` without changes.

For MOMENT, `wrapper.denoise(...)` hits the native reconstruction head.
For all other models it falls back to `denoise_via_forecast` (bidirectional).
"""

from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import DataLoader

from dataloader.toy_dataloader import Param

from .metrics import mse as _mse, spectral_mse as _spectral_mse, waveform_snr
from ..wrappers.base import BaseFoundationModel


@torch.no_grad()
def evaluate_denoising(
    wrapper: BaseFoundationModel,
    dataloader: DataLoader,
    *,
    regressor_raw: torch.nn.Module,
    regressor_clean: torch.nn.Module,
    target_idx: list[int],
    device: str = "cuda",
    max_batches: int | None = None,
) -> dict:
    """Evaluate foundation-model denoising and produce compare-compatible arrays.

    Parameters
    ----------
    wrapper : loaded BaseFoundationModel
    dataloader : yields (sig_bkg, sig, params)
    regressor_raw : trained regressor for noisy inputs (produces the `_raw` column)
    regressor_clean : trained regressor for clean inputs (used for `_denoised` & `_signal`)
    target_idx : indices into params for the target parameters, e.g. [0, 1, 2]
    device : where to run the regressors
    max_batches : smoke-test limit

    Returns
    -------
    dict with arrays ready for `writer.save_denoise_csv`:
      y_true, y_raw, y_den, y_signal  (N, n_targets)
      snr_gt, snr_raw, snr_denoised   (N,)
      plus aggregate metrics (mse, spectral_mse) per sample for summary.
    """
    regressor_raw.eval()
    regressor_clean.eval()
    dev = torch.device(device)

    y_true_list, y_raw_list, y_den_list, y_signal_list = [], [], [], []
    snr_gt_list, snr_raw_list, snr_den_list = [], [], []
    mse_list, psd_mse_list = [], []

    for batch_idx, (sig_bkg, sig, params) in enumerate(dataloader):
        if max_batches is not None and batch_idx >= max_batches:
            break
        sig_bkg = sig_bkg.to(dev)
        sig = sig.to(dev)
        params = params.to(dev)
        sig_bkg_np = sig_bkg.cpu().numpy()          # (B, L)
        sig_np = sig.cpu().numpy()                  # (B, L)

        # ── Foundation model denoising (numpy I/O) ───────────────────────
        den_np = wrapper.denoise(sig_bkg_np).denoised    # (B, L)
        den = torch.from_numpy(den_np).to(dev)

        # ── Feed into the three regressor paths ──────────────────────────
        # Existing regressors expect either (B, L) directly (MLPRegressor
        # flattens) or (B, L, 1) after unsqueeze in their forward.  We match
        # the convention used in `eval_pipeline.py::run_all_pipelines`:
        y_raw    = regressor_raw(sig_bkg)            # (B, n_targets)
        y_den    = regressor_clean(den.unsqueeze(-1))  # (B, n_targets)
        y_signal = regressor_clean(sig)                # (B, n_targets)

        y_true_list.append(params[:, target_idx].cpu())
        y_raw_list.append(y_raw.cpu())
        y_den_list.append(y_den.cpu())
        y_signal_list.append(y_signal.cpu())

        snr_gt_list.append(params[:, int(Param.snr)].cpu().numpy())
        snr_raw_list.append(waveform_snr(sig_bkg_np, sig_np))
        snr_den_list.append(waveform_snr(den_np, sig_np))

        mse_list.append(_mse(den_np, sig_np))
        psd_mse_list.append(_spectral_mse(den_np, sig_np))

    y_true   = torch.cat(y_true_list).numpy()
    y_raw    = torch.cat(y_raw_list).numpy()
    y_den    = torch.cat(y_den_list).numpy()
    y_signal = torch.cat(y_signal_list).numpy()
    snr_gt   = np.concatenate(snr_gt_list)
    snr_raw_arr  = np.concatenate(snr_raw_list)
    snr_den_arr  = np.concatenate(snr_den_list)
    mse_arr      = np.concatenate(mse_list)
    psd_mse_arr  = np.concatenate(psd_mse_list)

    return {
        "y_true": y_true,
        "y_raw": y_raw,
        "y_den": y_den,
        "y_signal": y_signal,
        "snr_gt": snr_gt,
        "snr_raw": snr_raw_arr,
        "snr_denoised": snr_den_arr,
        "mse_per_sample": mse_arr,
        "psd_mse_per_sample": psd_mse_arr,
    }
