"""
Preprocesses TIDMAD H5 files into memory-mappable .npy arrays.

Reads abra_training_*.h5 and abra_validation_*.h5 from --data_dir (which should
point to the directory containing the original H5 files), extracts random
non-overlapping 0.01-second windows at full 10 MHz resolution (100,000 samples),
computes FFT-based labels, snaps frequencies to KNOWN_FREQS_HZ, and saves
train / val / test splits.

Split: files are shuffled with --seed, then split 80/10/10 by file count:
  48 train files / 6 val files / 6 test files  (out of 60 total)

Output files in --out_dir:
  {split}_ch1.npy     int8     (N, 100_000)  noisy SQUID channel
  {split}_ch2.npy     int8     (N, 100_000)  clean injected reference
  {split}_params.npy  float32  (N, 3)        [frequency_hz, amplitude, snr]
  scale.npy           float32  scalar        voltage_range_mV / 256.0

Labels:
  frequency_hz  FFT peak frequency, snapped to KNOWN_FREQS_HZ grid
  amplitude     peak amplitude in mV  (2 * |FFT peak| / N)
  snr           signal power / noise power (±1 vs ±50 PSD bins around peak)

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
LABEL_WINDOW = 10_000_000  # 1 s — used for high-resolution FFT label computation


def _compute_labels(ch2_1s: np.ndarray, scaling: float) -> np.ndarray:
    """Compute (frequency_hz, amplitude, snr) from 1-second channel0002 segment.

    ch2_1s : int8 array, length 10_000_000
    scaling: voltage_range_mV / 256.0
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
            ch1_ds = f['timeseries/channel0001/timeseries']
            ch2_ds = f['timeseries/channel0002/timeseries']

            # Per-file window count (some files are 2.00B samples, others 2.01B)
            n_samples_file  = min(ch1_ds.shape[0], ch2_ds.shape[0])
            n_windows_total = n_samples_file // WINDOW_SIZE

            # Labels from the first full 1-second window at 10 MHz
            ch2_1s = ch2_ds[:LABEL_WINDOW]
            params = _compute_labels(ch2_1s, scaling)

            # Sample non-overlapping window start indices
            n_sample = min(n_per_file, n_windows_total)
            win_indices = rng.choice(n_windows_total, size=n_sample, replace=False)
            win_indices.sort()  # sorted read order is faster for HDF5

            for j, wi in enumerate(win_indices):
                start = int(wi) * WINDOW_SIZE
                end   = start + WINDOW_SIZE
                out_ch1[offset + j] = ch1_ds[start:end]
                out_ch2[offset + j] = ch2_ds[start:end]
                out_params[offset + j] = params

        offset += n_sample
    return offset


def main():
    parser = argparse.ArgumentParser(description='Preprocess TIDMAD H5 → NPY')
    parser.add_argument('--data_dir',   default='data/TIDMAD/original',
                        help='Directory containing abra_training_*.h5 and abra_validation_*.h5')
    parser.add_argument('--out_dir',    default='data/TIDMAD/preprocessed')
    parser.add_argument('--n_per_file', type=int, default=1666,
                        help='Windows sampled per file (default 1666 → ~100K total)')
    parser.add_argument('--seed',       type=int, default=42)
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    # -----------------------------------------------------------------------
    # Collect and split files
    # -----------------------------------------------------------------------
    # Accept both HuggingFace naming (tidmad_training_*.h5) and legacy naming
    # (abra_training_*.h5).  Match exactly 4-digit suffixes to exclude derived
    # files like abra_validation_denoised_s4denois_XXXX.h5.
    import re
    _4digit = re.compile(r'^(abra|tidmad)_(training|validation)_\d{4}\.h5$')
    all_h5 = sorted(glob.glob(os.path.join(args.data_dir, 'abra_*.h5')) +
                    glob.glob(os.path.join(args.data_dir, 'tidmad_*.h5')))
    train_h5 = [f for f in all_h5 if _4digit.match(os.path.basename(f)) and 'training' in f]
    val_h5   = [f for f in all_h5 if _4digit.match(os.path.basename(f)) and 'validation' in f]
    all_files = train_h5 + val_h5

    if not all_files:
        raise FileNotFoundError(
            f'No tidmad_training_*.h5 / abra_training_*.h5 in {args.data_dir}'
        )

    rng = np.random.default_rng(args.seed)
    shuffled = list(rng.permutation(len(all_files)))
    all_files = [all_files[i] for i in shuffled]

    n       = len(all_files)
    n_train = int(round(0.8 * n))
    n_val   = int(round(0.1 * n))
    n_test  = n - n_train - n_val

    splits = {
        'train': all_files[:n_train],
        'val':   all_files[n_train:n_train + n_val],
        'test':  all_files[n_train + n_val:],
    }

    print(f'Files: {n_train} train / {n_val} val / {n_test} test')
    print(f'Windows per file: {args.n_per_file}')
    for split, files in splits.items():
        print(f'  {split}: {len(files)} files × {args.n_per_file} = '
              f'{len(files)*args.n_per_file:,} windows')

    # -----------------------------------------------------------------------
    # Derive voltage scaling from first file (all files share the same range)
    # -----------------------------------------------------------------------
    with h5py.File(all_files[0], 'r') as f:
        voltage_range_mV = float(f['timeseries/channel0001'].attrs['voltage_range_mV'])
    scaling = voltage_range_mV / 256.0
    np.save(os.path.join(args.out_dir, 'scale.npy'), np.float32(scaling))
    print(f'Voltage scaling: {voltage_range_mV} mV / 256 = {scaling:.6f}')

    # -----------------------------------------------------------------------
    # Process each split
    # -----------------------------------------------------------------------
    for split, files in splits.items():
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

        split_rng = np.random.default_rng(args.seed + hash(split) % (2**31))
        _process_files(files, args.n_per_file, split_rng,
                       out_ch1, out_ch2, out_params, offset=0, scaling=scaling)

        # Flush to disk
        del out_ch1, out_ch2, out_params

        size_gb = n_windows * WINDOW_SIZE * 2 / 1e9
        print(f'  Saved {split} split  ({size_gb:.1f} GB int8 for ch1+ch2)')

    print('\nPreprocessing complete.')
    print(f'Output directory: {os.path.abspath(args.out_dir)}')


if __name__ == '__main__':
    main()
