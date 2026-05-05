"""
Apply a trained denoiser to the TIDMAD held-out test H5 files and compute
the official denoising benchmark score (Benchmark 1).

Score formula (mirrors tidmad_denoising.py):
    For each 100k-sample window:
        - Compute PSD of denoised ch1 and reference ch2
        - Measure SNR at the known injection frequency
    score = log_base_5.27( mean(norm_ch2_SNR * denoised_ch1_SNR) )

    Note: 100k samples at 10 MHz -> 100 Hz frequency resolution, which exactly
    aligns with the KNOWN_FREQS_HZ grid (minimum spacing = 100 Hz).

Test files (held out from NPY preprocessing, seed=42, never seen during training):
    abra_validation_0013.h5  abra_training_0014.h5
    abra_training_0013.h5    abra_validation_0026.h5
    abra_training_0036.h5    abra_validation_0025.h5
    abra_validation_0037.h5  abra_training_0008.h5

Usage:
    # ConvAE (PyTorch checkpoint)
    python benchmarks/TIDMAD/run_inference.py \\
        --model_type conv \\
        --ckpt       checkpoints/tidmad_conv_denoising/best.ckpt \\
        --cfg        configs/TIDMAD/train_tidmad_conv_denoising.yaml \\
        --data_dir   data/TIDMAD/original \\
        --scale_path data/TIDMAD/preprocessed/scale.npy \\
        --out_dir    benchmarks/TIDMAD/results/conv

    # LinOSS (JAX checkpoint)
    python benchmarks/TIDMAD/run_inference.py \\
        --model_type linoss \\
        --ckpt       logs/tidmad_linoss_denoising/version_0/checkpoints/best.eqx \\
        --cfg        configs/TIDMAD/train_tidmad_linoss_denoising.yaml \\
        --data_dir   data/TIDMAD/original \\
        --scale_path data/TIDMAD/preprocessed/scale.npy \\
        --out_dir    benchmarks/TIDMAD/results/linoss

    # Chronos zero-shot (no checkpoint or cfg needed)
    python benchmarks/TIDMAD/run_inference.py \\
        --model_type chronos \\
        --model_size base \\
        --data_dir   data/TIDMAD/original \\
        --scale_path data/TIDMAD/preprocessed/scale.npy \\
        --out_dir    benchmarks/TIDMAD/results/chronos

    # MOMENT zero-shot (native reconstruction head, no checkpoint or cfg needed)
    python benchmarks/TIDMAD/run_inference.py \\
        --model_type moment \\
        --model_size base \\
        --data_dir   data/TIDMAD/original \\
        --scale_path data/TIDMAD/preprocessed/scale.npy \\
        --out_dir    benchmarks/TIDMAD/results/moment_base
"""

import argparse
import csv
import importlib
import math
import os
import sys
from datetime import datetime
from pathlib import Path

import h5py
import numpy as np
import torch
import yaml

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, '..', '..', 'src'))
sys.path.insert(0, os.path.join(_HERE, '..'))   # for foundation.*

WINDOW_SIZE   = 100_000
FS            = 10_000_000
LABEL_WINDOW  = 10_000_000   # 1 s at 10 MHz — used for injection freq detection

# Known injection frequency grid (from TIDMAD benchmark specification)
KNOWN_FREQS_HZ = np.array(
    [*range(1_100,    10_000,    100),
     *range(10_000,  100_000,   1_000),
     *range(100_000, 1_000_000, 10_000),
     *range(1_000_000, 4_900_001, 100_000)],
    dtype=np.float32,
)

TEST_FILES = [
    'abra_validation_0013.h5',
    'abra_training_0014.h5',
    'abra_training_0013.h5',
    'abra_validation_0026.h5',
    'abra_training_0036.h5',
    'abra_validation_0025.h5',
    'abra_validation_0037.h5',
    'abra_training_0008.h5',
]


# ---------------------------------------------------------------------------
# Signal processing helpers
# ---------------------------------------------------------------------------

def get_injection_freq(ch2_segment: np.ndarray, scaling: float) -> float:
    """Estimate injection frequency from the first 1-second ch2 segment."""
    clean = ch2_segment.astype(np.float32) * scaling
    N = len(clean)
    fft_mag = 2.0 * np.abs(np.fft.rfft(clean)) / N
    freqs   = np.fft.rfftfreq(N, d=1.0 / FS)
    peak_idx = int(np.argmax(fft_mag[1:])) + 1
    raw_freq = freqs[peak_idx]
    return float(KNOWN_FREQS_HZ[np.argmin(np.abs(KNOWN_FREQS_HZ - raw_freq))])


def compute_snr(window: np.ndarray, target_freq: float) -> float:
    """Compute SNR at target_freq from a single 100k-sample window.

    PSD resolution = FS / WINDOW_SIZE = 10 MHz / 100k = 100 Hz, which exactly
    divides the KNOWN_FREQS_HZ grid so peaks fall on exact bins.
    """
    N  = len(window)
    dt = 1.0 / FS
    psd  = dt / N * np.abs(np.fft.rfft(window))[1:] ** 2
    freq = np.linspace(0, FS / 2, len(psd))

    center = int(np.argmin(np.abs(freq - target_freq)))
    sig_r, noise_r = 1, 50
    signal = np.sum(psd[max(0, center - sig_r)  : center + sig_r  + 1])
    noise  = np.sum(psd[max(0, center - noise_r): center + noise_r + 1]) - signal
    return float(signal / noise) if noise > 0 else 0.0


# ---------------------------------------------------------------------------
# Model loaders — return a callable  (B, L) float32 ndarray -> (B, L) ndarray
# ---------------------------------------------------------------------------

def load_conv_denoiser(ckpt_path: str, cfg_path: str, device: torch.device):
    """Load ConvAE from a Lightning checkpoint + YAML config."""
    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)

    model_cfg  = cfg['model']['init_args']['model']
    class_path = model_cfg['class_path']
    init_args  = model_cfg.get('init_args', {})
    mod_name, cls_name = class_path.rsplit('.', 1)
    model = getattr(importlib.import_module(mod_name), cls_name)(**init_args)

    ckpt = torch.load(ckpt_path, map_location=device, weights_only=True)
    state_dict = {k.replace('model.', '', 1): v
                  for k, v in ckpt['state_dict'].items()
                  if k.startswith('model.')}
    model.load_state_dict(state_dict)
    model.eval().to(device)

    @torch.no_grad()
    def denoise(x: np.ndarray) -> np.ndarray:
        t = torch.from_numpy(x).to(device)          # (B, L)
        y = model(t.unsqueeze(-1)).squeeze(-1)       # (B, L)
        return y.cpu().numpy()

    return denoise


def load_linoss_denoiser(ckpt_path: str, cfg_path: str):
    """Load LinOSS denoiser from a JAX (.eqx) checkpoint + YAML config."""
    import jax
    import equinox as eqx
    import optax
    from models.linoss import LinOSS
    from models.utils.jax.training import jax_inference
    from models.utils.jax.utils import tensor_to_jax, jax_to_tensor

    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)
    ia = cfg['model']['init_args']

    model = LinOSS(
        num_blocks=ia['num_blocks'],
        N=1,
        ssm_size=ia['ssm_size'],
        H=ia['H'],
        output_dim=1,
        task='denoising',
        output_step=1,
        discretization=ia.get('discretization', 'IMEX'),
        seed=ia.get('seed', 0),
    )
    state = eqx.nn.State(model)

    # Build dummy opt_state to match the saved 3-tuple (model, state, opt_state)
    lr = ia.get('lr', 1e-3)
    clip = ia.get('clip_grad_norm', 1.0)
    optimizer = optax.chain(
        optax.clip_by_global_norm(clip) if clip is not None else optax.identity(),
        optax.adamw(lr),
    )
    diff, _ = eqx.partition(model, eqx.is_inexact_array)
    dummy_opt_state = optimizer.init(diff)

    model, state, _ = eqx.tree_deserialise_leaves(
        ckpt_path, (model, state, dummy_opt_state)
    )

    def denoise(x: np.ndarray) -> np.ndarray:
        t   = torch.from_numpy(x).unsqueeze(-1)     # (B, L, 1)
        x_j = tensor_to_jax(t)
        keys = jax.random.split(jax.random.key(0), x.shape[0])
        y_j  = jax_inference(model, x_j, state, keys)
        return jax_to_tensor(y_j).squeeze(-1).numpy()  # (B, L)

    return denoise


def load_chronos_denoiser(model_size: str, device: torch.device, chunk_size: int = 128,
                          sub_batch: int = 512):
    """Zero-shot Chronos denoiser via batched bidirectional forecast on chunks.

    Each 100k-sample window is split into non-overlapping chunks of `chunk_size`
    (default 128 → prediction_length=64, within Chronos's recommended range).
    All chunks from a batch of windows are stacked into a single Chronos call
    (avoiding slow sequential calls). `sub_batch` limits the max batch size
    per Chronos call to avoid OOM.
    """
    from foundation.wrappers.chronos_wrapper import ChronosWrapper
    wrapper = ChronosWrapper()
    wrapper.load(device=str(device), model_size=model_size)
    print(f'Chronos ({model_size}) loaded, chunk_size={chunk_size}, sub_batch={sub_batch}')

    def denoise(x: np.ndarray) -> np.ndarray:
        # x: (B, L) float32
        B, L = x.shape
        n_chunks = L // chunk_size
        remainder = L % chunk_size

        # Stack all chunks: (B, n_chunks, chunk_size) -> (B*n_chunks, chunk_size)
        x_chunks = x[:, :n_chunks * chunk_size].reshape(B * n_chunks, chunk_size)

        # Denoise in sub-batches to avoid OOM
        denoised_chunks = np.zeros_like(x_chunks)
        for i in range(0, len(x_chunks), sub_batch):
            batch = x_chunks[i:i + sub_batch]
            denoised_chunks[i:i + sub_batch] = wrapper.denoise(batch).denoised

        out = np.zeros_like(x)
        out[:, :n_chunks * chunk_size] = denoised_chunks.reshape(B, n_chunks * chunk_size)
        # Keep any leftover samples (L not divisible by chunk_size) unchanged
        if remainder > 0:
            out[:, n_chunks * chunk_size:] = x[:, n_chunks * chunk_size:]
        return out

    return denoise


def load_moment_denoiser(model_size: str, device: torch.device, chunk_size: int = 512,
                         sub_batch: int = 512):
    """Zero-shot MOMENT denoiser via the native reconstruction head.

    Each 100k-sample window is split into non-overlapping chunks of `chunk_size`
    (default 512 = MOMENT's seq_len).  All chunks are forwarded in sub-batches
    of at most `sub_batch` to avoid OOM.  Unlike Chronos, MOMENT does a single
    forward pass per chunk (not autoregressive), so it is much faster.
    """
    from foundation.wrappers.moment_wrapper import MomentWrapper
    wrapper = MomentWrapper()
    wrapper.load(device=str(device), model_size=model_size, seq_len=chunk_size)
    print(f'MOMENT ({model_size}) loaded, chunk_size={chunk_size}, sub_batch={sub_batch}')

    def denoise(x: np.ndarray) -> np.ndarray:
        # x: (B, L) float32
        B, L = x.shape
        n_chunks = L // chunk_size
        remainder = L % chunk_size

        # Reshape to (B*n_chunks, chunk_size)
        x_chunks = x[:, :n_chunks * chunk_size].reshape(B * n_chunks, chunk_size)

        denoised_chunks = np.zeros_like(x_chunks)
        for i in range(0, len(x_chunks), sub_batch):
            batch = x_chunks[i:i + sub_batch]
            denoised_chunks[i:i + sub_batch] = wrapper.denoise(batch).denoised

        out = np.zeros_like(x)
        out[:, :n_chunks * chunk_size] = denoised_chunks.reshape(B, n_chunks * chunk_size)
        # Keep leftover samples (L not divisible by chunk_size) unchanged
        if remainder > 0:
            out[:, n_chunks * chunk_size:] = x[:, n_chunks * chunk_size:]
        return out

    return denoise


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------

def score_file(
    h5_path: str,
    denoise_fn,
    scaling: float,
    batch_size: int,
    max_windows: int | None = None,
    stride: int = 1,
) -> tuple[list[float], list[float]]:
    """Return (snr_ch2_list, snr_ch1_denoised_list) for sampled windows.

    stride: process every Nth window (1 = all, 10 = ~10x faster, matches original --coarse).
    Windows are sampled evenly across the full file so all injection frequencies are covered.
    """
    snr_ch2, snr_ch1 = [], []

    with h5py.File(h5_path, 'r') as f:
        ch1_ds = f['timeseries/channel0001/timeseries']
        ch2_ds = f['timeseries/channel0002/timeseries']
        n_samples = min(len(ch1_ds), len(ch2_ds))

        total_windows = n_samples // WINDOW_SIZE
        window_indices = list(range(0, total_windows, stride))
        if max_windows is not None:
            window_indices = window_indices[:max_windows]

        if stride > 1:
            print(f'  processing {len(window_indices)} windows (stride={stride}, out of {total_windows})')
        else:
            print(f'  processing {len(window_indices)} windows')

        for batch_start in range(0, len(window_indices), batch_size):
            batch_idx = window_indices[batch_start:batch_start + batch_size]
            B = len(batch_idx)

            if stride == 1:
                # contiguous read — fast path
                s = batch_idx[0] * WINDOW_SIZE
                e = (batch_idx[-1] + 1) * WINDOW_SIZE
                ch1_raw = ch1_ds[s:e].reshape(B, WINDOW_SIZE).astype(np.float32) * scaling
                ch2_raw = ch2_ds[s:e].reshape(B, WINDOW_SIZE).astype(np.float32) * scaling
            else:
                ch1_raw = np.stack([
                    ch1_ds[idx * WINDOW_SIZE:(idx + 1) * WINDOW_SIZE].astype(np.float32) * scaling
                    for idx in batch_idx
                ])
                ch2_raw = np.stack([
                    ch2_ds[idx * WINDOW_SIZE:(idx + 1) * WINDOW_SIZE].astype(np.float32) * scaling
                    for idx in batch_idx
                ])

            ch1_den = denoise_fn(ch1_raw)  # (B, L)

            for i in range(B):
                target_freq = get_injection_freq(ch2_raw[i], 1.0)  # per-window, already scaled
                snr_ch2.append(compute_snr(ch2_raw[i], target_freq))
                snr_ch1.append(compute_snr(ch1_den[i], target_freq))

    return snr_ch2, snr_ch1


def compute_benchmark_score(snr_ch2: np.ndarray, snr_ch1: np.ndarray) -> float:
    norm_ch2 = snr_ch2 / max(float(np.amax(snr_ch2)), 1e-30)
    raw = float(np.sum(norm_ch2 * snr_ch1)) / len(snr_ch1) + 1e-10
    return math.log(raw, 5.27)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def find_best_eqx(ckpt_dir: str) -> str:
    """Find the .eqx checkpoint with the lowest metric value in its filename."""
    import glob
    import re
    files = glob.glob(os.path.join(ckpt_dir, '*.eqx'))
    if not files:
        raise FileNotFoundError(f'No .eqx checkpoints found in {ckpt_dir}')

    def _metric(f):
        m = re.search(r'_metric_([\d.e+\-]+)', os.path.basename(f))
        return float(m.group(1)) if m else float('inf')

    best = min(files, key=_metric)
    print(f'Best checkpoint: {os.path.basename(best)}')
    return best


def main():
    parser = argparse.ArgumentParser(description='TIDMAD Denoising Benchmark Inference')
    parser.add_argument('--model_type', choices=['conv', 'linoss', 's4d', 'chronos', 'moment'], required=True)
    parser.add_argument('--ckpt',      default=None,
                        help='Checkpoint path (.ckpt or .eqx). Use --ckpt_dir for LinOSS.')
    parser.add_argument('--ckpt_dir',  default=None,
                        help='Directory of .eqx checkpoints; picks the one with lowest metric.')
    parser.add_argument('--cfg',        default=None,   help='Training config YAML (not needed for chronos)')
    # Chronos-specific
    parser.add_argument('--model_size', default='base', help='Chronos model size: tiny/small/base/large')
    parser.add_argument('--chunk_size', type=int, default=128,
                        help='Chunk size for Chronos (prediction_length=chunk_size//2, default 128 -> pred_len=64)')
    parser.add_argument('--max_windows', type=int, default=None,
                        help='Max windows per file (default: all). Use e.g. 100 for fast Chronos runs.')
    parser.add_argument('--stride', type=int, default=1,
                        help='Process every Nth window (default: 1=all). stride=10 matches original --coarse mode.')
    parser.add_argument('--data_dir',   default='data/TIDMAD/original')
    parser.add_argument('--scale_path', default='data/TIDMAD/preprocessed/scale.npy')
    parser.add_argument('--out_dir',    default='benchmarks/TIDMAD/results')
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--device',     default='cuda' if torch.cuda.is_available() else 'cpu')
    args = parser.parse_args()

    _zero_shot = args.model_type in ('chronos', 'moment')
    if not _zero_shot and args.ckpt is None and args.ckpt_dir is None:
        parser.error('one of --ckpt or --ckpt_dir is required for non-zero-shot models')
    if not _zero_shot and args.cfg is None:
        parser.error('--cfg is required for non-zero-shot models')

    scaling = float(np.load(args.scale_path))
    device  = torch.device(args.device)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.model_type == 'chronos':
        denoise_fn = load_chronos_denoiser(args.model_size, device, args.chunk_size)
    elif args.model_type == 'moment':
        denoise_fn = load_moment_denoiser(args.model_size, device)
    elif args.model_type == 'linoss':
        ckpt_path = args.ckpt or find_best_eqx(args.ckpt_dir)
        denoise_fn = load_linoss_denoiser(ckpt_path, args.cfg)
    else:  # conv or s4d — both are PyTorch Lightning checkpoints
        ckpt_path = args.ckpt or find_best_eqx(args.ckpt_dir)
        denoise_fn = load_conv_denoiser(ckpt_path, args.cfg, device)

    all_ch2_snr, all_ch1_snr = [], []

    for fname in TEST_FILES:
        fpath = os.path.join(args.data_dir, fname)
        if not os.path.exists(fpath):
            print(f'WARNING: {fname} not found, skipping')
            continue
        print(f'Processing {fname} ...')
        ch2_snrs, ch1_snrs = score_file(fpath, denoise_fn, scaling, args.batch_size, args.max_windows, args.stride)
        all_ch2_snr.extend(ch2_snrs)
        all_ch1_snr.extend(ch1_snrs)
        print(f'  windows processed: {len(ch2_snrs)}')

    snr_ch2 = np.array(all_ch2_snr)
    snr_ch1 = np.array(all_ch1_snr)
    score   = compute_benchmark_score(snr_ch2, snr_ch1)

    print(f'\n{"="*50}')
    print(f'Model: {args.model_type}')
    print(f'Denoising Benchmark Score: {score:.4f}')
    print(f'{"="*50}')

    model_tag = f'{args.model_type}_{args.model_size}' if _zero_shot else args.model_type
    out_csv = out_dir / f'benchmark_{model_tag}.csv'
    ckpt_info = f'{args.model_type}-{args.model_size} (zero-shot)' if _zero_shot else args.ckpt
    with open(out_csv, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['TIDMAD Benchmark 1: Denoising Score'])
        w.writerow(['Timestamp', datetime.now().strftime('%Y-%m-%d %H:%M:%S')])
        w.writerow(['Model', model_tag])
        w.writerow(['Checkpoint', ckpt_info])
        w.writerow(['Score', f'{score:.6f}'])
        w.writerow([])
        w.writerow(['Window Index', 'SNR ch2 (reference)', 'SNR ch1 (denoised)'])
        for i, (c2, c1) in enumerate(zip(snr_ch2, snr_ch1)):
            w.writerow([i, c2, c1])
    print(f'Results saved to {out_csv}')


if __name__ == '__main__':
    main()
