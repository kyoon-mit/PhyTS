"""
Export denoised H5 files in TIDMAD pre-chunked format for official scoring.

Reads raw ABRACADABRA H5 files, runs model inference, and writes output H5 files
compatible with the official tidmad_denoising.py scoring script:

    time_series_ch1  : (n_chunks, 10_000_000) float32 — denoised ch1
    time_series_ch2  : (n_chunks, 10_000_000) float32 — original ch2 (unchanged)
    signal_frequency : (n_chunks, 1)          float32 — injection freq per chunk (1 Hz res)
    attrs: sample_rate_hz=10_000_000, n_chunks

Inference is done in 100k sub-windows (model's native window size) then concatenated
into 10M-sample chunks.  signal_frequency is detected at 10M resolution (1 Hz) for
maximum accuracy before snapping to the KNOWN_FREQS_HZ grid.

With --stride 10 (default), every 10th chunk is exported — matches the official
--coarse mode and reduces output size ~10x (~200-400 MB/file compressed).

Usage:
    # ConvAE-L PSD (best conv model)
    PYTHONPATH=src python benchmarks/TIDMAD/export_denoised_h5.py \\
        --model_type conv \\
        --ckpt  checkpoints/tidmad_conv_l_psd/best.ckpt \\
        --cfg   configs/TIDMAD/train_tidmad_conv_l_denoising.yaml \\
        --out_dir benchmarks/TIDMAD/results/h5_export/conv_l_psd

    # LinOSS-190K PSD (best overall model — requires JAX)
    PYTHONPATH=src python benchmarks/TIDMAD/export_denoised_h5.py \\
        --model_type linoss \\
        --ckpt  logs/tidmad_linoss_190k_psd/version_0/checkpoints/checkpoint_step_213248.eqx \\
        --cfg   configs/TIDMAD/train_tidmad_linoss_190k_denoising.yaml \\
        --out_dir benchmarks/TIDMAD/results/h5_export/linoss_190k_psd

    # Chronos-Tiny (zero-shot baseline — no checkpoint needed)
    PYTHONPATH=src python benchmarks/TIDMAD/export_denoised_h5.py \\
        --model_type chronos \\
        --model_size tiny \\
        --out_dir benchmarks/TIDMAD/results/h5_export/chronos_tiny

    # MOMENT-Base (zero-shot baseline — no checkpoint needed)
    PYTHONPATH=src python benchmarks/TIDMAD/export_denoised_h5.py \\
        --model_type moment \\
        --model_size base \\
        --out_dir benchmarks/TIDMAD/results/h5_export/moment_base

Then score with the official script:
    python src/tasks/TIDMAD/tidmad_denoising.py \\
        --data_dir benchmarks/TIDMAD/results/h5_export/conv_l_psd \\
        --coarse
"""

import argparse
import os
import sys
from pathlib import Path

import h5py
import numpy as np
import torch
import yaml

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, '..', '..', 'src'))
sys.path.insert(0, _HERE)

from run_inference import (
    TEST_FILES,
    WINDOW_SIZE,
    FS,
    KNOWN_FREQS_HZ,
    load_conv_denoiser,
    load_linoss_denoiser,
    load_chronos_denoiser,
    load_moment_denoiser,
)

CHUNK_SIZE = 10_000_000          # 1 second at 10 MHz — official benchmark chunk size
WINDOWS_PER_CHUNK = CHUNK_SIZE // WINDOW_SIZE   # 100 sub-windows per chunk


def detect_chunk_freq(ch2_chunk: np.ndarray) -> float:
    """Detect injection frequency from a 10M-sample ch2 chunk at 1 Hz resolution."""
    N = len(ch2_chunk)
    fft_mag = 2.0 * np.abs(np.fft.rfft(ch2_chunk.astype(np.float32))) / N
    freqs   = np.fft.rfftfreq(N, d=1.0 / FS)
    peak_idx = int(np.argmax(fft_mag[1:])) + 1
    raw_freq = freqs[peak_idx]
    return float(KNOWN_FREQS_HZ[np.argmin(np.abs(KNOWN_FREQS_HZ - raw_freq))])


def export_file(h5_path: str, denoise_fn, scaling: float,
                out_dir: str, batch_size: int, stride: int):
    """Read one raw H5 file, denoise ch1, write pre-chunked output H5."""
    fname = os.path.basename(h5_path)
    out_path = os.path.join(out_dir, fname)

    with h5py.File(h5_path, 'r') as f:
        ch1_ds = f['timeseries/channel0001/timeseries']
        ch2_ds = f['timeseries/channel0002/timeseries']
        n_samples = min(len(ch1_ds), len(ch2_ds))

    total_chunks  = n_samples // CHUNK_SIZE
    chunk_indices = list(range(0, total_chunks, stride))
    n_out         = len(chunk_indices)
    print(f'  {fname}: {total_chunks} chunks total, exporting {n_out} (stride={stride})')

    signal_freqs    = np.zeros((n_out, 1), dtype=np.float32)
    ch1_out_chunks  = []
    ch2_out_chunks  = []

    with h5py.File(h5_path, 'r') as f:
        ch1_ds = f['timeseries/channel0001/timeseries']
        ch2_ds = f['timeseries/channel0002/timeseries']

        for out_idx, chunk_idx in enumerate(chunk_indices):
            s = chunk_idx * CHUNK_SIZE
            e = s + CHUNK_SIZE

            ch1_raw = ch1_ds[s:e].astype(np.float32) * scaling  # (10M,)
            ch2_raw = ch2_ds[s:e].astype(np.float32) * scaling  # (10M,)

            # Detect injection frequency at 1 Hz resolution from full 10M chunk
            signal_freqs[out_idx, 0] = detect_chunk_freq(ch2_raw)

            # Denoise ch1 in WINDOW_SIZE sub-windows, then reconstruct full chunk
            windows = ch1_raw.reshape(WINDOWS_PER_CHUNK, WINDOW_SIZE)  # (100, 100k)
            denoised_windows = np.zeros_like(windows)

            for batch_start in range(0, WINDOWS_PER_CHUNK, batch_size):
                batch_end = min(batch_start + batch_size, WINDOWS_PER_CHUNK)
                batch     = windows[batch_start:batch_end]           # (B, 100k)
                denoised_windows[batch_start:batch_end] = denoise_fn(batch)

            ch1_out_chunks.append(denoised_windows.reshape(CHUNK_SIZE))
            ch2_out_chunks.append(ch2_raw)

            if (out_idx + 1) % 10 == 0 or (out_idx + 1) == n_out:
                print(f'    chunk {out_idx + 1}/{n_out}  '
                      f'(inj={signal_freqs[out_idx, 0]/1e3:.1f} kHz)')

    ch1_arr = np.stack(ch1_out_chunks)   # (n_out, 10M)
    ch2_arr = np.stack(ch2_out_chunks)   # (n_out, 10M)

    with h5py.File(out_path, 'w') as out_f:
        out_f.attrs['sample_rate_hz'] = FS
        out_f.attrs['n_chunks']       = n_out
        out_f.create_dataset('time_series_ch1',  data=ch1_arr,
                             compression='gzip', compression_opts=4)
        out_f.create_dataset('time_series_ch2',  data=ch2_arr,
                             compression='gzip', compression_opts=4)
        out_f.create_dataset('signal_frequency', data=signal_freqs)

    size_mb = os.path.getsize(out_path) / 1e6
    print(f'  Saved: {out_path}  ({size_mb:.0f} MB)')


def main():
    parser = argparse.ArgumentParser(description='Export denoised H5 files for official scoring')
    parser.add_argument('--model_type', choices=['conv', 'linoss', 'chronos', 'moment'], required=True)
    parser.add_argument('--ckpt',       default=None,
                        help='Checkpoint path (.ckpt for conv, .eqx for linoss)')
    parser.add_argument('--cfg',        default=None, help='Training config YAML')
    parser.add_argument('--model_size', default='tiny',
                        help='Chronos model size: tiny/small/base/large (default: tiny)')
    parser.add_argument('--data_dir',   default='data/TIDMAD/original')
    parser.add_argument('--scale_path', default='data/TIDMAD/preprocessed/scale.npy')
    parser.add_argument('--out_dir',    required=True,
                        help='Output directory for denoised H5 files')
    parser.add_argument('--batch_size', type=int, default=64,
                        help='Sub-window batch size for inference (default: 64)')
    parser.add_argument('--stride',     type=int, default=10,
                        help='Export every Nth chunk (default: 10, matches --coarse)')
    parser.add_argument('--device',     default='cuda' if torch.cuda.is_available() else 'cpu')
    args = parser.parse_args()

    _zero_shot = args.model_type in ('chronos', 'moment')
    if not _zero_shot and args.ckpt is None:
        parser.error('--ckpt is required for conv and linoss models')
    if not _zero_shot and args.cfg is None:
        parser.error('--cfg is required for conv and linoss models')

    os.makedirs(args.out_dir, exist_ok=True)
    scaling = float(np.load(args.scale_path))
    device  = torch.device(args.device)

    print(f'Loading {args.model_type} model...')
    if args.model_type == 'conv':
        denoise_fn = load_conv_denoiser(args.ckpt, args.cfg, device)
    elif args.model_type == 'linoss':
        denoise_fn = load_linoss_denoiser(args.ckpt, args.cfg)
    elif args.model_type == 'chronos':
        denoise_fn = load_chronos_denoiser(args.model_size, device)
    else:  # moment
        denoise_fn = load_moment_denoiser(args.model_size, device)

    for fname in TEST_FILES:
        fpath = os.path.join(args.data_dir, fname)
        if not os.path.exists(fpath):
            print(f'WARNING: {fname} not found, skipping')
            continue
        print(f'\nProcessing {fname} ...')
        export_file(fpath, denoise_fn, scaling, args.out_dir, args.batch_size, args.stride)

    print(f'\nDone. Score with:')
    print(f'  python src/tasks/TIDMAD/tidmad_denoising.py \\')
    print(f'      --data_dir {args.out_dir} --coarse')


if __name__ == '__main__':
    main()
