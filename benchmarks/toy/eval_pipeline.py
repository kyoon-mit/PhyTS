"""
Benchmark: Raw vs Denoised Regression on the Toy Dataset.

Three pipelines compared on the test set:
  raw      — regressor_raw(sig_bkg)           regressor trained on noisy signals
  denoised — regressor_clean(denoiser(sig_bkg)) regressor trained on clean signals, applied after denoising
  oracle   — regressor_clean(sig)             regressor trained on clean signals, applied to ground truth

Usage:
    python benchmarks/toy/eval_pipeline.py \
        --denoiser_ckpt        checkpoints/toy_s4d_denoising/best.ckpt \
        --regressor_raw_ckpt   checkpoints/toy_s4d_regression_raw/best.ckpt \
        --regressor_clean_ckpt checkpoints/toy_s4d_regression_clean/best.ckpt \
        --data_dir             data/toy/sinusoidal_signal_white_noise \
        --out_dir              benchmarks/toy
"""

import argparse
from pathlib import Path

import numpy as np
import torch
import matplotlib.pyplot as plt

from dataloader.toy_dataloader import ToyDataModule, Param
from tasks.toy.toy_denoising import DenoisingMSE as Denoiser
from tasks.toy.toy_regression import RegressionMSE as Regressor, SNR_BINS


# ─────────────────────────────────────────────────────────────────────────────

def load_model(cls, ckpt_path: str, device: torch.device):
    model = cls.load_from_checkpoint(ckpt_path, map_location=device)
    model.eval()
    model.to(device)
    return model


@torch.no_grad()
def run_all_pipelines(denoiser, reg_raw, reg_clean, dataloader, device):
    """Returns: y_true, y_raw, y_denoised, y_oracle, snr  — all (N, n_targets) or (N,)."""
    y_true_list, y_raw_list, y_den_list, y_oracle_list, snr_list = [], [], [], [], []

    for sig_bkg, sig, params in dataloader:
        sig_bkg = sig_bkg.to(device)
        sig     = sig.to(device)
        params  = params.to(device)

        X_denoised = denoiser(sig_bkg)                         # (B, L)

        y_raw    = reg_raw(sig_bkg)                            # raw pipeline
        y_den    = reg_clean(X_denoised)                       # denoised pipeline
        y_oracle = reg_clean(sig)                              # oracle (clean ground truth)

        target_idx = reg_raw.target_idx
        y_true_list.append(params[:, target_idx].cpu())
        y_raw_list.append(y_raw.cpu())
        y_den_list.append(y_den.cpu())
        y_oracle_list.append(y_oracle.cpu())
        snr_list.append(params[:, int(Param.snr)].cpu())

    return (
        torch.cat(y_true_list).numpy(),
        torch.cat(y_raw_list).numpy(),
        torch.cat(y_den_list).numpy(),
        torch.cat(y_oracle_list).numpy(),
        torch.cat(snr_list).numpy(),
    )


def compute_snr_binned_rmse(y_true, y_pred, snr, bins):
    results = {}
    for bin_name, (lo, hi) in bins.items():
        mask = (snr >= lo) & (snr < hi)
        if mask.sum() == 0:
            results[bin_name] = np.full(y_true.shape[1], np.nan)
        else:
            results[bin_name] = np.sqrt(((y_pred[mask] - y_true[mask]) ** 2).mean(axis=0))
    return results


def plot_comparison(rmse_dict, param_names, bin_names, out_path):
    """rmse_dict: {'raw': {...}, 'denoised': {...}, 'oracle': {...}}"""
    colors = {'raw': 'steelblue', 'denoised': 'tomato', 'oracle': 'seagreen'}
    n_params    = len(param_names)
    n_pipelines = len(rmse_dict)
    x     = np.arange(len(bin_names))
    width = 0.8 / n_pipelines

    fig, axes = plt.subplots(1, n_params, figsize=(5 * n_params, 4), sharey=False)
    if n_params == 1:
        axes = [axes]

    for ax, name in zip(axes, param_names):
        i = param_names.index(name)
        for j, (label, rmse) in enumerate(rmse_dict.items()):
            vals = [rmse[b][i] for b in bin_names]
            offset = (j - n_pipelines / 2 + 0.5) * width
            ax.bar(x + offset, vals, width, label=label,
                   color=colors[label], alpha=0.85)
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
    parser = argparse.ArgumentParser()
    parser.add_argument('--denoiser_ckpt',        required=True)
    parser.add_argument('--regressor_raw_ckpt',   required=True)
    parser.add_argument('--regressor_clean_ckpt', required=True)
    parser.add_argument('--data_dir',   default='data/toy/sinusoidal_signal_white_noise')
    parser.add_argument('--out_dir',    default='benchmarks/toy')
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--device',     default='cuda' if torch.cuda.is_available() else 'cpu')
    args = parser.parse_args()

    device  = torch.device(args.device)
    out_dir = Path(args.out_dir)

    denoiser  = load_model(Denoiser,  args.denoiser_ckpt,        device)
    reg_raw   = load_model(Regressor, args.regressor_raw_ckpt,   device)
    reg_clean = load_model(Regressor, args.regressor_clean_ckpt, device)

    dm = ToyDataModule(data_dir=args.data_dir, batch_size=args.batch_size)
    dm.setup('test')
    loader = dm.test_dataloader()

    y_true, y_raw, y_den, y_oracle, snr = run_all_pipelines(
        denoiser, reg_raw, reg_clean, loader, device
    )

    bin_names   = list(SNR_BINS.keys())
    param_names = reg_raw.target_params

    rmse_dict = {
        'raw':      compute_snr_binned_rmse(y_true, y_raw,    snr, SNR_BINS),
        'denoised': compute_snr_binned_rmse(y_true, y_den,    snr, SNR_BINS),
        'oracle':   compute_snr_binned_rmse(y_true, y_oracle, snr, SNR_BINS),
    }

    # Print summary
    print(f'\n{"Param":<20} {"SNR bin":<8} {"Raw":>10} {"Denoised":>10} {"Oracle":>10}')
    print('-' * 62)
    for bin_name in bin_names:
        for i, name in enumerate(param_names):
            print(f'{name:<20} {bin_name:<8}'
                  f' {rmse_dict["raw"][bin_name][i]:>10.4f}'
                  f' {rmse_dict["denoised"][bin_name][i]:>10.4f}'
                  f' {rmse_dict["oracle"][bin_name][i]:>10.4f}')

    plot_comparison(rmse_dict, param_names, bin_names,
                    out_dir / 'regression_benchmark.png')


if __name__ == '__main__':
    main()
