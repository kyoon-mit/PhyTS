"""
Compare eval results across denoiser pipelines.

Reads results.csv from each plots/TIDMAD/{pipeline}/ directory and produces:
  1. comparison_mse_{param}.png     — all MSE-loss models overlaid
  2. comparison_psd_{param}.png     — all PSD-loss models overlaid
  3. comparison_{model}_{param}.png — MSE vs PSD for each model that has both

Usage:
    python benchmarks/TIDMAD/compare_pipelines.py \
        --plots_dir plots/TIDMAD \
        --out_dir   plots/TIDMAD/comparison
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm


BINS = {
    'amplitude':    np.linspace(-1.0,   1.0,   61),
    'frequency_hz': np.linspace(-5.0,   5.0,   61),
    'phase_rad':    np.linspace(-np.pi, np.pi, 61),
}

LATEX = {
    'amplitude':    r'$\hat{A} - A$',
    'frequency_hz': r'$\hat{f} - f$ [Hz]',
    'phase_rad':    r'$\hat{\phi} - \phi$ [rad]',
}

PIPELINE_COLORS = {
    's4d_mse':           '#1f77b4',
    'conv_ae_mse':       '#ff7f0e',
    'conv_ae_psd':       '#ff7f0e',
    'conv_attn_ae_mse':  '#2ca02c',
    'conv_attn_ae_psd':  '#2ca02c',
    'rnn_mse':           '#d62728',
    'rnn_psd':           '#d62728',
    'mlp_denoiser_mse':  '#9467bd',
    'mlp_denoiser_psd':  '#9467bd',
    'classical':         '#8c564b',
}

LOSS_STYLE = {'mse': '-', 'psd': '--'}


def load_all(plots_dir: Path) -> dict[str, pd.DataFrame]:
    results = {}
    for csv in sorted(plots_dir.glob('*/results.csv')):
        name = csv.parent.name
        results[name] = pd.read_csv(csv)
    if not results:
        raise FileNotFoundError(f'No results.csv files found under {plots_dir}')
    return results


def _infer_params(df: pd.DataFrame) -> list[str]:
    return [c.replace('_true', '') for c in df.columns if c.endswith('_true')]


def _loss_of(name: str) -> str:
    if name.endswith('_psd'):
        return 'psd'
    if name.endswith('_mse'):
        return 'mse'
    return 'none'  # classical filter


def _model_of(name: str) -> str:
    for suffix in ('_mse', '_psd'):
        if name.endswith(suffix):
            return name[: -len(suffix)]
    return name


def _plot_group(group: dict[str, pd.DataFrame], param: str, pipeline: str,
                out_dir: Path, title: str):
    """Overlay residual histograms for a group of pipelines, one metric pipeline."""
    bins = BINS.get(param, np.linspace(-1, 1, 61))
    fig, axes = plt.subplots(1, 3, figsize=(13, 4), sharey=True)
    metrics = [('raw', f'{param}_raw'), ('denoised', f'{param}_denoised'), ('signal', f'{param}_signal')]

    for ax, (metric_label, col) in zip(axes, metrics):
        for name, df in group.items():
            if col not in df.columns or f'{param}_true' not in df.columns:
                continue
            res = df[col].values - df[f'{param}_true'].values
            loss = _loss_of(name)
            color = PIPELINE_COLORS.get(name, 'gray')
            style = LOSS_STYLE.get(loss, ':')
            ax.hist(res, bins=bins, histtype='step', linewidth=1.5,
                    linestyle=style, color=color, label=name)
        ax.set_xlabel(LATEX.get(param, param))
        ax.set_ylabel('Count' if ax is axes[0] else '')
        ax.set_title(metric_label)
        ax.legend(fontsize=7)

    plt.tight_layout()
    out = out_dir / f'comparison_{pipeline}_{param}.png'
    plt.savefig(out, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'Saved {out}')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--plots_dir', default='plots/TIDMAD')
    parser.add_argument('--out_dir',   default='plots/TIDMAD/comparison')
    args = parser.parse_args()

    plots_dir = Path(args.plots_dir)
    out_dir   = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    all_results = load_all(plots_dir)

    # Detect param names from first result
    params = _infer_params(next(iter(all_results.values())))

    # ── Group 1: all MSE models ────────────────────────────────────────────────
    mse_group = {k: v for k, v in all_results.items()
                 if _loss_of(k) in ('mse', 'none')}  # include classical (no loss)

    # ── Group 2: all PSD models ────────────────────────────────────────────────
    psd_group = {k: v for k, v in all_results.items() if _loss_of(k) == 'psd'}

    # ── Group 3: per-model MSE vs PSD ─────────────────────────────────────────
    models_with_both = {
        _model_of(k) for k in all_results
        if _loss_of(k) == 'mse'
    } & {
        _model_of(k) for k in all_results
        if _loss_of(k) == 'psd'
    }

    for param in params:
        if mse_group:
            _plot_group(mse_group, param, 'mse', out_dir, 'MSE-loss models')
        if psd_group:
            _plot_group(psd_group, param, 'psd', out_dir, 'PSD-loss models')
        for model in sorted(models_with_both):
            cross = {k: v for k, v in all_results.items() if _model_of(k) == model}
            _plot_group(cross, param, model, out_dir, f'{model}: MSE vs PSD')

    # ── SNR comparison across pipelines ───────────────────────────────────────
    if any('snr_denoised' in df.columns for df in all_results.values()):
        all_snr = {k: v for k, v in all_results.items() if 'snr_denoised' in v.columns}
        snr_vals = np.concatenate([df['snr_denoised'].values for df in all_snr.values()])
        snr_min = max(1e-2, snr_vals.min())
        snr_max = snr_vals.max()
        bins = np.logspace(np.log10(snr_min), np.log10(snr_max), 61)

        fig, ax = plt.subplots(figsize=(6, 4))
        # Plot raw SNR once (same for all pipelines)
        first_df = next(iter(all_snr.values()))
        ax.hist(first_df['snr_raw'].values, bins=bins, histtype='step',
                color='gray', linewidth=1.5, linestyle='-', label='raw (noisy)')
        for name, df in all_snr.items():
            color = PIPELINE_COLORS.get(name, 'gray')
            loss  = _loss_of(name)
            style = LOSS_STYLE.get(loss, ':')
            ax.hist(df['snr_denoised'].values, bins=bins, histtype='step',
                    color=color, linewidth=1.5, linestyle=style, label=name)
        ax.set_xscale('log')
        ax.set_xlabel('SNR (waveform)')
        ax.set_ylabel('Count')
        ax.legend(fontsize=7)
        plt.tight_layout()
        out = out_dir / 'snr_all_pipelines.png'
        plt.savefig(out, dpi=150, bbox_inches='tight')
        plt.close()
        print(f'Saved {out}')


if __name__ == '__main__':
    main()
