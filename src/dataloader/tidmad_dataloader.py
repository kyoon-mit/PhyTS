"""
Dataloader for the TIDMAD dataset (ABRACADABRA dark matter experiment).

Each H5 file contains two channels at 10 MHz (int8):
  channel0001 — SQUID readout (noisy input)
  channel0002 — injected reference signal (clean ground truth)

Samples are non-overlapping 1-second windows (window_size=10_000_000 samples).
An optional downsample_factor reduces the sequence length presented to the model.

Labels are derived per-window from channel0002 via FFT:
  frequency_hz — dominant frequency peak
  amplitude    — peak amplitude in mV
  snr          — signal-to-noise ratio (signal power / noise power in PSD bins)

Normalization: int8 × (volt_range_mV / 256) → physical millivolts (float32).

Usage (LightningCLI YAML):
    data:
      class_path: dataloader.tidmad_dataloader.TIDMADDataModule
      init_args:
        data_dir: data/TIDMAD
        window_size: 10000000
        downsample_factor: 1000
        batch_size: 32
        num_workers: 0

Batch convention:
    noisy   (B, seq_len)  — channel0001, downsampled
    clean   (B, seq_len)  — channel0002, downsampled
    params  (B, 3)        — [frequency_hz, amplitude_mV, snr]
"""

import glob
import os
from enum import IntEnum

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import lightning as L


class Param(IntEnum):
    frequency_hz = 0
    amplitude    = 1
    snr          = 2


class TIDMADDataset(Dataset):
    """Windowed dataset over multiple TIDMAD H5 files.

    Labels (frequency_hz, amplitude, snr) are computed from channel0002 FFT
    for each window in __getitem__. File handles are opened lazily per worker.
    """

    FS = 10_000_000  # native sampling rate (Hz)

    def __init__(
        self,
        h5_files: list,
        window_size: int = 10_000_000,
        downsample_factor: int = 1000,
    ):
        self.h5_files        = h5_files
        self.window_size     = window_size
        self.downsample_factor = downsample_factor
        self.seq_len         = window_size // downsample_factor
        self._file_handles   = {}  # populated lazily per worker process

        # Build flat index: list of (file_path, window_start_sample, scaling, params)
        # Labels are constant per file (injected signal frequency/amplitude don't change),
        # so we compute them once from the first full window at native 10 MHz resolution.
        self.index = []
        for path in h5_files:
            with h5py.File(path, 'r') as f:
                ch1_len   = f['timeseries/channel0001/timeseries'].shape[0]
                ch2_len   = f['timeseries/channel0002/timeseries'].shape[0]
                n_samples = min(ch1_len, ch2_len)
                scaling   = np.float32(f['timeseries/channel0001'].attrs['voltage_range_mV'] / 256.0)
                ch2_ref   = f['timeseries/channel0002/timeseries'][:window_size].astype(np.float32) * scaling
            params = self._compute_labels(ch2_ref)
            n_windows = n_samples // window_size
            for w in range(n_windows):
                self.index.append((path, w * window_size, scaling, params))

    def __len__(self):
        return len(self.index)

    def _get_file(self, path):
        if path not in self._file_handles:
            self._file_handles[path] = h5py.File(path, 'r')
        return self._file_handles[path]

    def __getitem__(self, idx):
        path, start, scaling, params = self.index[idx]
        end = start + self.window_size
        step = self.downsample_factor

        f = self._get_file(path)
        ch1 = f['timeseries/channel0001/timeseries'][start:end:step].astype(np.float32) * scaling
        ch2 = f['timeseries/channel0002/timeseries'][start:end:step].astype(np.float32) * scaling

        return (
            torch.from_numpy(ch1),
            torch.from_numpy(ch2),
            torch.from_numpy(params),
        )

    def _compute_labels(self, clean: np.ndarray) -> np.ndarray:
        """Derive frequency_hz, amplitude_mV, snr from full-resolution clean signal via FFT."""
        N      = len(clean)
        fs_eff = self.FS  # always full-resolution (10 MHz)

        # One-sided amplitude spectrum
        fft_mag = 2.0 * np.abs(np.fft.rfft(clean)) / N
        freqs   = np.fft.rfftfreq(N, d=1.0 / fs_eff)

        # Peak (skip DC bin)
        peak_idx     = int(np.argmax(fft_mag[1:])) + 1
        frequency_hz = float(freqs[peak_idx])
        amplitude    = float(fft_mag[peak_idx])

        # SNR in PSD units (mirrors benchmark.py getSNR)
        psd        = fft_mag ** 2 / (2.0 * fs_eff)
        sig_range  = 1
        noise_range = 50
        lo_s, hi_s = max(0, peak_idx - sig_range), peak_idx + sig_range + 1
        lo_n, hi_n = max(0, peak_idx - noise_range), peak_idx + noise_range + 1
        signal_pwr = float(np.sum(psd[lo_s:hi_s]))
        noise_pwr  = float(np.sum(psd[lo_n:hi_n])) - signal_pwr
        snr        = signal_pwr / max(noise_pwr, 1e-30)

        return np.array([frequency_hz, amplitude, snr], dtype=np.float32)


class TIDMADDataModule(L.LightningDataModule):
    """Lightning DataModule for TIDMAD.

    Splits:
      train — abra_training_*.h5
      val   — abra_validation_*.h5
      test  — abra_validation_*.h5  (same files, different shuffle)
    """

    def __init__(
        self,
        data_dir: str,
        window_size: int = 10_000_000,
        downsample_factor: int = 1000,
        batch_size: int = 32,
        num_workers: int = 0,
    ):
        super().__init__()
        self.data_dir         = data_dir
        self.window_size      = window_size
        self.downsample_factor = downsample_factor
        self.batch_size       = batch_size
        self.num_workers      = num_workers

    def _glob(self, pattern):
        files = sorted(glob.glob(os.path.join(self.data_dir, pattern)))
        if not files:
            raise FileNotFoundError(f'No files found matching {os.path.join(self.data_dir, pattern)}')
        return files

    def _dataset(self, files):
        return TIDMADDataset(files, self.window_size, self.downsample_factor)

    def setup(self, stage=None):
        if stage == 'fit':
            self.train_dataset = self._dataset(self._glob('abra_training_*.h5'))
            self.val_dataset   = self._dataset(self._glob('abra_validation_*.h5'))
        elif stage in ('test', 'predict'):
            self.test_dataset  = self._dataset(self._glob('abra_validation_*.h5'))

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size,
                          shuffle=True, num_workers=self.num_workers)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size,
                          shuffle=False, num_workers=self.num_workers)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size,
                          shuffle=False, num_workers=self.num_workers)

    def predict_dataloader(self):
        return self.test_dataloader()
