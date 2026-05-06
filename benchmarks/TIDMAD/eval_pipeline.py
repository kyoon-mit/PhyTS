"""
Benchmark: Raw vs Denoised Regression on TIDMAD.

Three pipelines compared on the validation set:
  raw      — regressor_raw(noisy)              regressor trained on noisy signals
  denoised — regressor_clean(denoiser(noisy))  regressor trained on clean signals, applied after denoising
  oracle   — regressor_clean(clean)            regressor trained on clean signals, applied to ground truth

Usage:
    python benchmarks/TIDMAD/eval_pipeline.py \
        --denoiser_ckpt        checkpoints/tidmad_s4d_denoising/best.ckpt \
        --denoiser_cfg         configs/TIDMAD/train_tidmad_s4d_denoising.yaml \
        --regressor_raw_ckpt   checkpoints/tidmad_mlp_regression_raw/best.ckpt \
        --regressor_raw_cfg    configs/TIDMAD/train_tidmad_mlp_regression_raw.yaml \
        --regressor_clean_ckpt checkpoints/tidmad_mlp_regression_clean/best.ckpt \
        --regressor_clean_cfg  configs/TIDMAD/train_tidmad_mlp_regression_clean.yaml \
        --data_dir             data/TIDMAD \
        --out_dir              benchmarks/TIDMAD
"""

import argparse
import importlib
import io
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import yaml

from dataloader.tidmad_dataloader import TIDMADDataModule, Param


# ── Histogram bins per variable (signed residuals: predicted − true) ──────────
BINS = {
    'frequency_hz': np.linspace(-500.0, 500.0, 41),   # Hz; effective res ~1 Hz at 10k samples
    'amplitude':    np.linspace(-5.0,   5.0,   41),   # mV
}

# LaTeX x-axis labels
LATEX = {
    'frequency_hz': r'$\hat{f} - f$ [Hz]',
    'amplitude':    r'$\hat{A} - A$ [mV]',
}

COLORS = {'raw': 'steelblue', 'denoised': 'tomato', 'signal': 'seagreen'}


# ─────────────────────────────────────────────────────────────────────────────

def _is_jax_model(class_path: str) -> bool:
    """Return True if class_path points to an equinox Module (JAX model)."""
    try:
        import equinox as eqx
        module_name, class_name = class_path.rsplit('.', 1)
        cls = getattr(importlib.import_module(module_name), class_name)
        return issubclass(cls, eqx.Module)
    except Exception:
        return False


class _JAXDenoiserWrapper(nn.Module):
    """Thin nn.Module wrapper around a JAXLightningModule for eval inference."""

    def __init__(self, task):
        super().__init__()
        self._task = task

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        from models.utils.jax.utils import jax_to_tensor
        jax_out = self._task.forward(x)  # (B, L, 1) JAX array
        return jax_to_tensor(jax_out)    # (B, L, 1) PyTorch tensor


def _load_jax_model(ckpt_path: str, cfg_path: str) -> nn.Module:
    """Instantiate a JAXLightningModule and restore weights from a Lightning checkpoint."""
    import equinox as eqx

    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)

    task_cfg = cfg['model']
    task_class_path = task_cfg['class_path']
    task_init_args = dict(task_cfg.get('init_args', {}))

    # Build inner JAX model
    inner_cfg = task_init_args.pop('model')
    inner_module, inner_class = inner_cfg['class_path'].rsplit('.', 1)
    inner_cls = getattr(importlib.import_module(inner_module), inner_class)
    inner_model = inner_cls(**inner_cfg.get('init_args', {}))

    # Build task wrapper
    task_module, task_class = task_class_path.rsplit('.', 1)
    task_cls = getattr(importlib.import_module(task_module), task_class)
    task = task_cls(model=inner_model, **task_init_args)

    # Restore JAX weights from Lightning checkpoint
    ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    if 'jax_weights' in ckpt:
        buf = io.BytesIO(ckpt['jax_weights'])
        task.jax_model, task.jax_model_state = eqx.tree_deserialise_leaves(
            buf, (task.jax_model, task.jax_model_state)
        )

    return _JAXDenoiserWrapper(task)


def load_model(ckpt_path: str, cfg_path: str, device: torch.device):
    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)

    model_cfg  = cfg['model']['init_args']['model']
    class_path = model_cfg['class_path']

    if _is_jax_model(class_path):
        return _load_jax_model(ckpt_path, cfg_path)

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


@torch.no_grad()
def run_all_pipelines(denoiser, reg_raw, reg_clean, dataloader, device, target_idx):
    """Returns: y_true, y_raw, y_denoised, y_signal, snr  — all (N, n_targets) or (N,)."""
    y_true_list, y_raw_list, y_den_list, y_signal_list, snr_list = [], [], [], [], []

    for noisy, clean, params in dataloader:
        noisy  = noisy.to(device)
        clean  = clean.to(device)
        params = params.to(device)

        X_denoised = denoiser(noisy.unsqueeze(-1))             # (B, L, 1)

        y_raw    = reg_raw(noisy)                              # (B, L) → (B, n_targets)
        y_den    = reg_clean(X_denoised)                       # (B, L, 1) → squeeze → (B, n_targets)
        y_signal = reg_clean(clean)                            # (B, L) → (B, n_targets)

        y_true_list.append(params[:, target_idx].cpu())
        y_raw_list.append(y_raw.cpu())
        y_den_list.append(y_den.cpu())
        y_signal_list.append(y_signal.cpu())
        snr_list.append(params[:, int(Param.snr)].cpu())

    return (
        torch.cat(y_true_list).numpy(),
        torch.cat(y_raw_list).numpy(),
        torch.cat(y_den_list).numpy(),
        torch.cat(y_signal_list).numpy(),
        torch.cat(snr_list).numpy(),
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


def plot_snr_distribution(snr, out_dir):
    """Histogram of ground-truth SNR values in the test set."""
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.hist(snr, bins=np.linspace(snr.min(), snr.max(), 61),
            histtype='step', color='gray', linewidth=1.5)
    ax.set_xlabel('SNR')
    ax.set_ylabel('Count')
    plt.tight_layout()
    out = out_dir / 'snr_distribution.png'
    plt.savefig(out, dpi=150, bbox_inches='tight')
    plt.close()
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
    parser.add_argument('--data_dir',   default='data/TIDMAD')
    parser.add_argument('--out_dir',    default='benchmarks/TIDMAD')
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--device',     default='cuda' if torch.cuda.is_available() else 'cpu')
    args = parser.parse_args()

    device  = torch.device(args.device)
    out_dir = Path(args.out_dir)

    denoiser  = load_model(args.denoiser_ckpt,        args.denoiser_cfg,        device)
    reg_raw   = load_model(args.regressor_raw_ckpt,   args.regressor_raw_cfg,   device)
    reg_clean = load_model(args.regressor_clean_ckpt, args.regressor_clean_cfg, device)

    # Read target_params from YAML (load_model returns a raw nn.Module)
    with open(args.regressor_raw_cfg) as f:
        reg_cfg = yaml.safe_load(f)
    target_params = reg_cfg['model']['init_args']['target_params']
    target_idx    = [int(Param[p]) for p in target_params]

    dm = TIDMADDataModule(data_dir=args.data_dir, batch_size=args.batch_size)
    dm.setup('test')
    loader = dm.test_dataloader()

    y_true, y_raw, y_den, y_signal, snr = run_all_pipelines(
        denoiser, reg_raw, reg_clean, loader, device, target_idx
    )

    residuals = {
        'raw':      y_raw    - y_true,
        'denoised': y_den    - y_true,
        'signal':   y_signal - y_true,
    }

    plot_residuals(residuals, target_params, out_dir)
    plot_snr_distribution(snr, out_dir)


if __name__ == '__main__':
    main()
