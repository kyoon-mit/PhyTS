"""
Benchmark: Raw vs Denoised Regression on the Toy Dataset.

Loads a pre-trained denoiser and a pre-trained regressor, runs both pipelines
on the test set, and produces a comparison plot of RMSE vs SNR bin.

Usage:
    python benchmarks/toy/eval_pipeline.py \
        --denoiser_ckpt  <path/to/denoising.ckpt> \
        --regressor_ckpt <path/to/regression.ckpt> \
        --data_dir       data/toy/sinusoidal_signal_white_noise \
        --out_dir        benchmarks/toy \
        --device         cuda
"""

import argparse
from pathlib import Path

import numpy as np
import torch
import matplotlib.pyplot as plt

from dataloader.toy_dataloader import ToyDataModule, Param
from tasks.toy_denoising import DenoisingMSE as Denoiser
from tasks.toy_regression import RegressionMSE as Regressor, SNR_BINS


# ─────────────────────────────────────────────────────────────────────────────

def load_model(cls, ckpt_path: str, device: torch.device):
    model = cls.load_from_checkpoint(ckpt_path, map_location=device)
    model.eval()
    model.to(device)
    return model


@torch.no_grad()
def run_pipeline(denoiser, regressor, dataloader, device):
    """Returns arrays: y_true (N, n_targets), y_raw (N, n_targets),
    y_denoised (N, n_targets), snr (N,)."""
    y_true_list, y_raw_list, y_denoised_list, snr_list = [], [], [], []

    for X, _, params in dataloader:
        X      = X.to(device)
        params = params.to(device)

        # Raw regression
        y_hat_raw = regressor(X)                              # (B, n_targets)

        # Denoised regression
        X_denoised = denoiser(X)                              # (B, L)
        y_hat_den  = regressor(X_denoised)                    # (B, n_targets)

        target_idx = regressor.target_idx
        y_true_list.append(params[:, target_idx].cpu())
        y_raw_list.append(y_hat_raw.cpu())
        y_denoised_list.append(y_hat_den.cpu())
        snr_list.append(params[:, int(Param.snr)].cpu())

    return (
        torch.cat(y_true_list).numpy(),
        torch.cat(y_raw_list).numpy(),
        torch.cat(y_denoised_list).numpy(),
        torch.cat(snr_list).numpy(),
    )


def compute_snr_binned_rmse(y_true, y_pred, snr, bins):
    """Returns dict: bin_name -> per-param RMSE array (n_targets,)."""
    results = {}
    for bin_name, (lo, hi) in bins.items():
        mask = (snr >= lo) & (snr < hi)
        if mask.sum() == 0:
            results[bin_name] = np.full(y_true.shape[1], np.nan)
        else:
            rmse = np.sqrt(((y_pred[mask] - y_true[mask]) ** 2).mean(axis=0))
            results[bin_name] = rmse
    return results


def plot_comparison(rmse_raw, rmse_den, param_names, bin_names, out_path):
    n_params = len(param_names)
    x = np.arange(len(bin_names))
    width = 0.35

    fig, axes = plt.subplots(1, n_params, figsize=(5 * n_params, 4), sharey=False)
    if n_params == 1:
        axes = [axes]

    for ax, name in zip(axes, param_names):
        raw_vals = [rmse_raw[b][param_names.index(name)] for b in bin_names]
        den_vals = [rmse_den[b][param_names.index(name)] for b in bin_names]
        ax.bar(x - width / 2, raw_vals, width, label='raw',      color='steelblue', alpha=0.85)
        ax.bar(x + width / 2, den_vals, width, label='denoised', color='tomato',    alpha=0.85)
        ax.set_xticks(x)
        ax.set_xticklabels(bin_names)
        ax.set_xlabel('SNR bin')
        ax.set_ylabel('RMSE')
        ax.set_title(name)
        ax.legend(fontsize=8)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'Saved {out_path}')


# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description='Raw vs denoised regression benchmark.')
    parser.add_argument('--denoiser_ckpt',  required=True)
    parser.add_argument('--regressor_ckpt', required=True)
    parser.add_argument('--data_dir', default='data/toy/sinusoidal_signal_white_noise')
    parser.add_argument('--out_dir',  default='benchmarks/toy')
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    args = parser.parse_args()

    device   = torch.device(args.device)
    out_dir  = Path(args.out_dir)

    # Load models
    denoiser  = load_model(Denoiser,  args.denoiser_ckpt,  device)
    regressor = load_model(Regressor, args.regressor_ckpt, device)

    # Load test data
    dm = ToyDataModule(data_dir=args.data_dir, batch_size=args.batch_size)
    dm.setup('test')
    loader = dm.test_dataloader()

    # Run both pipelines
    y_true, y_raw, y_den, snr = run_pipeline(denoiser, regressor, loader, device)

    # Compute SNR-binned RMSE
    bin_names    = list(SNR_BINS.keys())
    param_names  = regressor.target_params
    rmse_raw = compute_snr_binned_rmse(y_true, y_raw, snr, SNR_BINS)
    rmse_den = compute_snr_binned_rmse(y_true, y_den, snr, SNR_BINS)

    # Print summary
    print(f'\n{"Param":<20} {"SNR bin":<8} {"Raw RMSE":>12} {"Denoised RMSE":>15}')
    print('-' * 58)
    for bin_name in bin_names:
        for i, name in enumerate(param_names):
            print(f'{name:<20} {bin_name:<8} {rmse_raw[bin_name][i]:>12.4f} {rmse_den[bin_name][i]:>15.4f}')

    # Plot
    plot_comparison(rmse_raw, rmse_den, param_names, bin_names,
                    out_dir / 'regression_benchmark.png')


if __name__ == '__main__':
    main()
