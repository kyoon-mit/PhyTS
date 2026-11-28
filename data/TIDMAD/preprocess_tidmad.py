"""
Preprocesses TIDMAD H5 files into memory-mappable .npy arrays.

Reads tidmad_training_*.h5, tidmad_validation_*.h5 and tidmad_test_file.h5 from
--data_dir (which should point to the directory containing the Hugging Face H5
files), extracts random non-overlapping 0.01-second windows at full 10 MHz
resolution (100,000 samples), computes FFT-based labels, snaps frequencies to
KNOWN_FREQS_HZ, and saves train / val / test splits.

Each H5 file holds 200 one-second chunks of integer DAQ counts:
  time_series_ch1  float32  (200, 10_000_000)  noisy SQUID channel
  time_series_ch2  float32  (200, 10_000_000)  clean injected reference

Split: follows the Hugging Face file names:
  20 train files / 20 val files / 1 test file

Output files in --out_dir:
  {split}_ch1.npy     int8     (N, 100_000)  noisy SQUID channel
  {split}_ch2.npy     int8     (N, 100_000)  clean injected reference
  {split}_params.npy  float32  (N, 3)        [frequency_hz, amplitude, snr]
  scale.npy           float32  scalar        DAQ counts to mV (40 mV / 128)

Labels:
  frequency_hz  FFT peak frequency, snapped to KNOWN_FREQS_HZ grid
  amplitude     peak amplitude in mV  (2 * |FFT peak| / N)
  snr           signal power / noise power (±1 vs ±50 PSD bins around peak)
Labels are per chunk. About 20% of chunks have no injection; there the peak is a
noise line of ~0.02 mV rather than a signal.

Usage:
  python data/TIDMAD/preprocess_tidmad.py \\
      --data_dir  data/TIDMAD/original \\
      --out_dir   data/TIDMAD/preprocessed \\
      --n_per_file 1666 \\
      --seed      42
"""

import argparse
import glob
import os

import h5py
import numpy as np


# ---------------------------------------------------------------------------
# Known injected-frequency grid (from TIDMAD benchmark specification).
# Labels are snapped to the nearest value in this list.
# ---------------------------------------------------------------------------
KNOWN_FREQS_HZ = np.array(
    [*range(1_100,  10_000,    100),   # 1.1 kHz – 9.9 kHz  in 100 Hz steps  (89 values)
     *range(10_000, 100_000,  1_000),  # 10 kHz  – 99 kHz   in 1 kHz steps   (90 values)
     *range(100_000, 1_000_000, 10_000), # 100 kHz – 990 kHz in 10 kHz steps  (90 values)
     *range(1_000_000, 4_900_001, 100_000)], # 1 MHz – 4.9 MHz in 100 kHz steps (40 values)
    dtype=np.float32,
)  # 309 distinct frequencies total

FS          = 10_000_000   # native sampling rate (Hz)
WINDOW_SIZE = 100_000      # 0.01 s × 10 MHz
SCALE_MV    = 40.0 / 128   # DAQ counts to mV

SPLIT_FILES = {
    'train': 'tidmad_training_[0-9][0-9][0-9][0-9].h5',
    'val':   'tidmad_validation_[0-9][0-9][0-9][0-9].h5',
    'test':  'tidmad_test_file.h5',
}


def _compute_labels(ch2_1s: np.ndarray, scaling: float) -> np.ndarray:
    """Compute (frequency_hz, amplitude, snr) from a 1-second time_series_ch2 chunk.

    ch2_1s : DAQ counts, length 10_000_000
    scaling: DAQ counts to mV
    Returns float32 array of shape (3,).
    """
    clean = ch2_1s.astype(np.float32) * np.float32(scaling)
    N = len(clean)

    fft_mag = 2.0 * np.abs(np.fft.rfft(clean)) / N
    freqs   = np.fft.rfftfreq(N, d=1.0 / FS)

    # Peak (skip DC bin 0)
    peak_idx     = int(np.argmax(fft_mag[1:])) + 1
    raw_freq     = freqs[peak_idx]
    frequency_hz = float(KNOWN_FREQS_HZ[np.argmin(np.abs(KNOWN_FREQS_HZ - raw_freq))])
    amplitude    = float(fft_mag[peak_idx])

    # SNR in PSD units (mirrors official TIDMAD benchmark getSNR)
    psd         = fft_mag ** 2 / (2.0 * FS)
    sig_range   = 1
    noise_range = 50
    lo_s, hi_s  = max(0, peak_idx - sig_range),   peak_idx + sig_range   + 1
    lo_n, hi_n  = max(0, peak_idx - noise_range),  peak_idx + noise_range + 1
    signal_pwr  = float(np.sum(psd[lo_s:hi_s]))
    noise_pwr   = float(np.sum(psd[lo_n:hi_n])) - signal_pwr
    snr         = signal_pwr / max(noise_pwr, 1e-30)

    return np.array([frequency_hz, amplitude, snr], dtype=np.float32)


def _process_files(files, n_per_file, rng, out_ch1, out_ch2, out_params, offset, scaling):
    """Extract n_per_file random windows from each file, writing into pre-allocated arrays."""
    for i, path in enumerate(files):
        print(f'  [{i+1}/{len(files)}] {os.path.basename(path)}', flush=True)
        with h5py.File(path, 'r') as f:
            ch1_ds = f['time_series_ch1']
            ch2_ds = f['time_series_ch2']

            n_chunks, chunk_length = ch1_ds.shape
            windows_per_chunk = chunk_length // WINDOW_SIZE

            # Sample non-overlapping window start indices
            win_indices = rng.choice(n_chunks * windows_per_chunk, size=n_per_file, replace=False)
            win_indices.sort()  # sorted read order is faster for HDF5

            # Each chunk has its own injection, so labels are computed per chunk
            current_chunk = None
            for j, wi in enumerate(win_indices):
                chunk, win = divmod(int(wi), windows_per_chunk)
                if chunk != current_chunk:
                    current_chunk = chunk
                    ch2_chunk = ch2_ds[chunk]
                    params = _compute_labels(ch2_chunk, scaling)

                start = win * WINDOW_SIZE
                end   = start + WINDOW_SIZE
                out_ch1[offset + j] = ch1_ds[chunk, start:end]
                out_ch2[offset + j] = ch2_chunk[start:end]
                out_params[offset + j] = params

        offset += n_per_file
    return offset


def main():
    parser = argparse.ArgumentParser(description='Preprocess TIDMAD H5 → NPY')
    parser.add_argument('--data_dir',   default='data/TIDMAD/original',
                        help='Directory containing tidmad_training_*.h5, tidmad_validation_*.h5 '
                             'and tidmad_test_file.h5')
    parser.add_argument('--out_dir',    default='data/TIDMAD/preprocessed')
    parser.add_argument('--n_per_file', type=int, default=1666,
                        help='Windows sampled per file (default 1666 → ~68K total)')
    parser.add_argument('--seed',       type=int, default=42)
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    # -----------------------------------------------------------------------
    # Collect files for each split
    # -----------------------------------------------------------------------
    splits = {
        split: sorted(glob.glob(os.path.join(args.data_dir, pattern)))
        for split, pattern in SPLIT_FILES.items()
    }
    for split, files in splits.items():
        if not files:
            raise FileNotFoundError(f'No {SPLIT_FILES[split]} in {args.data_dir}')

    print(f'Windows per file: {args.n_per_file}')
    for split, files in splits.items():
        print(f'  {split}: {len(files)} files × {args.n_per_file} = '
              f'{len(files)*args.n_per_file:,} windows')

    np.save(os.path.join(args.out_dir, 'scale.npy'), np.float32(SCALE_MV))

    # -----------------------------------------------------------------------
    # Process each split
    # -----------------------------------------------------------------------
    for split_index, (split, files) in enumerate(splits.items()):
        n_windows = len(files) * args.n_per_file
        print(f'\n=== {split.upper()} ({n_windows:,} windows) ===')

        ch1_path    = os.path.join(args.out_dir, f'{split}_ch1.npy')
        ch2_path    = os.path.join(args.out_dir, f'{split}_ch2.npy')
        params_path = os.path.join(args.out_dir, f'{split}_params.npy')

        # Pre-allocate memory-mapped output arrays (written directly to disk)
        out_ch1    = np.lib.format.open_memmap(
            ch1_path,    mode='w+', dtype='int8',    shape=(n_windows, WINDOW_SIZE))
        out_ch2    = np.lib.format.open_memmap(
            ch2_path,    mode='w+', dtype='int8',    shape=(n_windows, WINDOW_SIZE))
        out_params = np.lib.format.open_memmap(
            params_path, mode='w+', dtype='float32', shape=(n_windows, 3))

        split_rng = np.random.default_rng([args.seed, split_index])
        _process_files(files, args.n_per_file, split_rng,
                       out_ch1, out_ch2, out_params, offset=0, scaling=SCALE_MV)

        # Flush to disk
        del out_ch1, out_ch2, out_params

        size_gb = n_windows * WINDOW_SIZE * 2 / 1e9
        print(f'  Saved {split} split  ({size_gb:.1f} GB int8 for ch1+ch2)')

    print('\nPreprocessing complete.')
    print(f'Output directory: {os.path.abspath(args.out_dir)}')


if __name__ == '__main__':
    main()
