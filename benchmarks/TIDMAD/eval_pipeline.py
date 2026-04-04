"""
Benchmark: Raw vs Denoised Regression on TIDMAD.

Three pipelines compared on the validation set:
  raw      — regressor_raw(noisy)              regressor trained on noisy signals
  denoised — regressor_clean(denoiser(noisy))  regressor trained on clean signals, applied after denoising
  signal   — regressor_clean(clean)            regressor trained on clean signals, applied to ground truth

Usage:
    python benchmarks/TIDMAD/eval_pipeline.py \
        --denoiser_ckpt        checkpoints/tidmad_s4d_denoising/best.ckpt \
        --denoiser_cfg         configs/TIDMAD/train_tidmad_s4d_denoising.yaml \
        --regressor_raw_ckpt   checkpoints/tidmad_mlp_regression_raw/best.ckpt \
        --regressor_raw_cfg    configs/TIDMAD/train_tidmad_mlp_regression_raw.yaml \
        --regressor_clean_ckpt checkpoints/tidmad_mlp_regression_clean/best.ckpt \
        --regressor_clean_cfg  configs/TIDMAD/train_tidmad_mlp_regression_clean.yaml \
        --data_dir             data/TIDMAD \
        --out_dir              plots/TIDMAD/s4d_mse
"""

import argparse
from pathlib import Path

import yaml
import importlib

import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt

from dataloader.tidmad_dataloader import TIDMADDataModule, Param


# ── Histogram bins per variable (signed residuals: predicted − true) ──────────
BINS = {
    'frequency_hz': np.linspace(-500.0, 500.0, 61),   # Hz
    'amplitude':    np.linspace(-5.0,   5.0,   61),   # mV
}

LATEX = {
    'frequency_hz': r'$\hat{f} - f$ [Hz]',
    'amplitude':    r'$\hat{A} - A$ [mV]',
}

COLORS = {'raw': 'steelblue', 'denoised': 'tomato', 'signal': 'seagreen'}


# ─────────────────────────────────────────────────────────────────────────────

def load_model(ckpt_path: str, cfg_path: str, device: torch.device):
    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)

    model_cfg  = cfg['model']['init_args']['model']
    class_path = model_cfg['class_path']
    init_args  = model_cfg.get('init_args', {})

    module_name, class_name = class_path.rsplit('.', 1)
    cls   = getattr(importlib.import_module(module_name), class_name)
    model = cls(**init_args)

    ckpt       = torch.load(ckpt_path, map_location=device, weights_only=True)
    state_dict = {k.replace('model.', ''): v for k, v in ckpt['state_dict'].items()
                  if k.startswith('model.')}
    model.load_state_dict(state_dict)
    model.eval()
    return model.to(device)


def _waveform_snr(signal: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
    """Per-sample SNR: power(reference) / power(signal - reference). Shape (B,)."""
    ref_power   = reference.pow(2).mean(-1)
    noise_power = (signal - reference).pow(2).mean(-1)
    return ref_power / (noise_power + 1e-10)


@torch.no_grad()
def run_all_pipelines(denoiser, reg_raw, reg_clean, dataloader, device, target_idx):
    """
    Returns:
        y_true       (N, n_targets)  ground-truth parameter values
        y_raw        (N, n_targets)  predictions from raw noisy input
        y_den        (N, n_targets)  predictions from denoised input
        y_signal     (N, n_targets)  predictions from clean signal (upper bound)
        snr_gt       (N,)            ground-truth SNR stored in dataset params
        snr_raw      (N,)            waveform SNR of raw noisy signal vs clean
        snr_denoised (N,)            waveform SNR of denoised signal vs clean
    """
    y_true_list, y_raw_list, y_den_list, y_signal_list = [], [], [], []
    snr_gt_list, snr_raw_list, snr_den_list = [], [], []

    for noisy, clean, params in dataloader:
        noisy  = noisy.to(device)
        clean  = clean.to(device)
        params = params.to(device)

        X_denoised = denoiser(noisy.unsqueeze(-1))             # (B, L, 1)
        x_den_sq   = X_denoised.squeeze(-1)                   # (B, L)

        y_raw    = reg_raw(noisy)                              # (B, n_targets)
        y_den    = reg_clean(X_denoised)                       # (B, n_targets)
        y_signal = reg_clean(clean)                            # (B, n_targets)

        y_true_list.append(params[:, target_idx].cpu())
        y_raw_list.append(y_raw.cpu())
        y_den_list.append(y_den.cpu())
        y_signal_list.append(y_signal.cpu())

        snr_gt_list.append(params[:, int(Param.snr)].cpu())
        snr_raw_list.append(_waveform_snr(noisy, clean).cpu())
        snr_den_list.append(_waveform_snr(x_den_sq, clean).cpu())

    return (
        torch.cat(y_true_list).numpy(),
        torch.cat(y_raw_list).numpy(),
        torch.cat(y_den_list).numpy(),
        torch.cat(y_signal_list).numpy(),
        torch.cat(snr_gt_list).numpy(),
        torch.cat(snr_raw_list).numpy(),
        torch.cat(snr_den_list).numpy(),
    )


def plot_residuals(residuals_dict, param_names, out_dir):
    """One PNG per variable: overlaid step histograms of (predicted − true)."""
    for i, name in enumerate(param_names):
        fig, ax = plt.subplots(figsize=(5, 4))
        bins = BINS.get(name, np.linspace(-1, 1, 61))
        for label, res in residuals_dict.items():
            ax.hist(res[:, i], bins=bins, histtype='step',
                    label=label, color=COLORS[label], linewidth=1.5)
        ax.set_xlabel(LATEX.get(name, name))
        ax.set_ylabel('Count')
        ax.legend()
        plt.tight_layout()
        out = out_dir / f'residuals_{name}.png'
        plt.savefig(out, dpi=150, bbox_inches='tight')
        plt.close()
        print(f'Saved {out}')


def plot_snr_comparison(snr_gt, snr_raw, snr_denoised, out_dir):
    """Overlaid histograms: ground-truth SNR, raw-signal SNR, denoised-signal SNR.

    SNR is computed as power(clean) / power(predicted − clean), so higher is better.
    Log-scale x-axis handles the wide dynamic range.
    """
    snr_min = max(1e-2, min(snr_raw.min(), snr_denoised.min(), snr_gt.min()))
    snr_max = max(snr_raw.max(), snr_denoised.max(), snr_gt.max())
    bins = np.logspace(np.log10(snr_min), np.log10(snr_max), 61)

    fig, ax = plt.subplots(figsize=(5, 4))
    ax.hist(snr_gt,       bins=bins, histtype='step', color='gray',             linewidth=1.5, label='ground-truth SNR')
    ax.hist(snr_raw,      bins=bins, histtype='step', color=COLORS['raw'],      linewidth=1.5, label='raw')
    ax.hist(snr_denoised, bins=bins, histtype='step', color=COLORS['denoised'], linewidth=1.5, label='denoised')
    ax.set_xscale('log')
    ax.set_xlabel('SNR')
    ax.set_ylabel('Count')
    ax.legend()
    plt.tight_layout()
    out = out_dir / 'snr_comparison.png'
    plt.savefig(out, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'Saved {out}')


@torch.no_grad()
def get_sample_waveforms(denoiser, dataset, sample_idx: int, device: torch.device):
    """Return (noisy, clean, denoised) numpy arrays for one sample."""
    noisy, clean, _ = dataset[sample_idx]
    noisy_t = noisy.unsqueeze(0).to(device)                    # (1, L)
    clean_t = clean.unsqueeze(0).to(device)
    denoised_t = denoiser(noisy_t.unsqueeze(-1)).squeeze(-1)   # (1, L)
    return (
        noisy_t.squeeze(0).cpu().numpy(),
        clean_t.squeeze(0).cpu().numpy(),
        denoised_t.squeeze(0).cpu().numpy(),
    )


def plot_fft(noisy, clean, denoised, sample_rate: float, sample_idx: int, out_dir: Path):
    """FFT magnitude spectrum for one sample: noisy vs denoised vs clean signal."""
    N = len(noisy)
    freqs = np.fft.rfftfreq(N, d=1.0 / sample_rate)

    def mag(x):
        return 2.0 * np.abs(np.fft.rfft(x)) / N

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.semilogy(freqs, mag(noisy),    color=COLORS['raw'],      linewidth=0.8, label='raw',      alpha=0.7)
    ax.semilogy(freqs, mag(denoised), color=COLORS['denoised'], linewidth=1.5, label='denoised', alpha=0.9)
    ax.semilogy(freqs, mag(clean),    color=COLORS['signal'],   linewidth=1.5, label='signal',   alpha=0.9)
    ax.set_xlabel('Frequency [Hz]')
    ax.set_ylabel('Amplitude')
    ax.legend()
    plt.tight_layout()
    out = out_dir / f'fft_sample{sample_idx}.png'
    plt.savefig(out, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'Saved {out}')


def save_results_csv(y_true, y_raw, y_den, y_signal,
                     snr_gt, snr_raw, snr_denoised,
                     param_names, out_dir):
    """Save per-sample results to results.csv for use by compare_pipelines.py."""
    data = {'snr_gt': snr_gt, 'snr_raw': snr_raw, 'snr_denoised': snr_denoised}
    for i, name in enumerate(param_names):
        data[f'{name}_true']     = y_true[:, i]
        data[f'{name}_raw']      = y_raw[:, i]
        data[f'{name}_denoised'] = y_den[:, i]
        data[f'{name}_signal']   = y_signal[:, i]
    df = pd.DataFrame(data)
    out = out_dir / 'results.csv'
    df.to_csv(out, index=False)
    print(f'Saved {out}')


# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--denoiser_ckpt',        required=True)
    parser.add_argument('--denoiser_cfg',         required=True)
    parser.add_argument('--regressor_raw_ckpt',   required=True)
    parser.add_argument('--regressor_raw_cfg',    required=True)
    parser.add_argument('--regressor_clean_ckpt', required=True)
    parser.add_argument('--regressor_clean_cfg',  required=True)
    parser.add_argument('--data_dir',   default='data/TIDMAD/preprocessed')
    parser.add_argument('--out_dir',    default='plots/TIDMAD/s4d_mse')
    parser.add_argument('--batch_size',     type=int,   default=32)
    parser.add_argument('--device',         default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--fft_sample_idx', type=int,   default=0,
                        help='Index of test sample to use for FFT plot')
    parser.add_argument('--sample_rate',    type=float, default=10_000_000.0,
                        help='Sampling rate in Hz (10 MHz, no downsampling)')
    args = parser.parse_args()

    device  = torch.device(args.device)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    denoiser  = load_model(args.denoiser_ckpt,        args.denoiser_cfg,        device)
    reg_raw   = load_model(args.regressor_raw_ckpt,   args.regressor_raw_cfg,   device)
    reg_clean = load_model(args.regressor_clean_ckpt, args.regressor_clean_cfg, device)

    with open(args.regressor_raw_cfg) as f:
        reg_cfg = yaml.safe_load(f)
    target_params = reg_cfg['model']['init_args']['target_params']
    target_idx    = [int(Param[p]) for p in target_params]

    dm = TIDMADDataModule(data_dir=args.data_dir, batch_size=args.batch_size)
    dm.setup('test')
    loader  = dm.test_dataloader()
    dataset = dm.test

    y_true, y_raw, y_den, y_signal, snr_gt, snr_raw, snr_denoised = run_all_pipelines(
        denoiser, reg_raw, reg_clean, loader, device, target_idx
    )

    residuals = {
        'raw':      y_raw    - y_true,
        'denoised': y_den    - y_true,
        'signal':   y_signal - y_true,
    }

    plot_residuals(residuals, target_params, out_dir)
    plot_snr_comparison(snr_gt, snr_raw, snr_denoised, out_dir)
    save_results_csv(y_true, y_raw, y_den, y_signal,
                     snr_gt, snr_raw, snr_denoised,
                     target_params, out_dir)

    noisy_w, clean_w, den_w = get_sample_waveforms(
        denoiser, dataset, args.fft_sample_idx, device)
    plot_fft(noisy_w, clean_w, den_w, args.sample_rate, args.fft_sample_idx, out_dir)


if __name__ == '__main__':
    main()
