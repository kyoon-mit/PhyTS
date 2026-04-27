"""
Dataloader for the Project 8 simulation dataset (CRES neutrino-mass experiment).

Splits are stored as separate directories of HDF5 files (one per split).  Each
HDF5 file contains:
  inputs       e.g. ['output_ts_I', 'output_ts_Q']  (T,)  float32  time series
  variables    e.g. ['energy', ...]                 ()    float32  regression targets
  noise        pre-generated noise traces, stored under one of two keys
               (gaussian / non-gaussian); the active key is selected by
               ``noise_type`` and the full array is preloaded once per split.

Two tasks are supported:
  Project8SimDataset           - joint task: ts (+ optional fft),
                                 normalised regression targets.
  Project8SimDenoisingDataset  - denoising task: (X_noisy, X_clean) pairs.

Window convention:
  cutoff samples per channel (default 4000), n_inputs channels (typically I/Q).

Usage (LightningCLI YAML):
    data:
      class_path: dataloader.project8_dataloader.Project8DataModule
      init_args:
        train_dir:  data/Project8Sim/hdf5/train
        val_dir:    data/Project8Sim/hdf5/val
        test_dir:   data/Project8Sim/hdf5/test
        inputs:     [output_ts_I, output_ts_Q]
        variables:  [energy]
        noise_type: gaussian
        noise_paths:
          gaussian:     noise_gaussian
          non_gaussian: noise_non_gaussian
        # Optional: skip _compute_norm_stats by passing precomputed values.
        mu:   null
        stds: null
        batch_size: 512
        cutoff:     4000

Batch convention:
    Project8DataModule (freq_transform='fft', the default):
        ts        (B, cutoff, C)        float32  time-series, per-channel std-norm
        fft       (B, cutoff, C)        float32  FFT(I,Q), real/imag stacked into channels
        var       (B, n_variables)      float32  z-scored regression targets

    Project8DenoisingDataModule:
        noisy     (B, cutoff, C)        float32  signal + noise, per-channel std-norm
        clean     (B, cutoff, C)        float32  signal, normalised by the *same* std
"""

from enum import IntEnum
from typing import Optional

import h5py
import lightning as L
import numpy as np
import torch
from scipy.fft import fft
from torch.utils.data import DataLoader, Dataset


def fft_func_IQ_complex_channels(signal_I, signal_Q):
    signal = signal_I + 1j * signal_Q
    yf = fft(signal)
    return yf.real, yf.imag


# ─── batch-tuple indices for the joint task ─────────────────────────────────
# Position depends on freq_transform; see Project8SimDataset.__getitem__.
class JointBatch(IntEnum):
    ts  = 0
    var = 1   # 2 if a freq tensor sits in front


class DenoisingBatch(IntEnum):
    noisy = 0
    clean = 1


# ─── default noise key mapping ──────────────────────────────────────────────
DEFAULT_NOISE_PATHS = {
    'gaussian':     'noise_gaussian',
    'non_gaussian': 'noise_non_gaussian',
}


# ─── HDF5 helpers ───────────────────────────────────────────────────────────
def _list_hdf5(data_dir: str) -> list[str]:
    """Return sorted list of .hdf5/.h5 paths under data_dir."""
    import os
    paths = sorted(
        os.path.join(data_dir, f)
        for f in os.listdir(data_dir)
        if f.endswith('.hdf5') or f.endswith('.h5')
    )
    if not paths:
        raise FileNotFoundError(f'No HDF5 files found in {data_dir!r}')
    return paths


def _build_index(paths: list[str], probe_key: str) -> list[tuple[str, int]]:
    """Map global idx → (file_path, local_row_idx) by reading row counts."""
    index: list[tuple[str, int]] = []
    for path in paths:
        with h5py.File(path, 'r') as f:
            n = f[probe_key].shape[0]
        index.extend((path, i) for i in range(n))
    return index


def _resolve_noise_key(noise_type: str, noise_paths: Optional[dict]) -> str:
    table = noise_paths if noise_paths is not None else DEFAULT_NOISE_PATHS
    if noise_type not in table:
        raise ValueError(
            f"noise_type must be one of {list(table)}, got {noise_type!r}"
        )
    return table[noise_type]


def _preload_noise(paths: list[str], noise_key: str) -> np.ndarray:
    """Concatenate the HDF5 dataset at ``noise_key`` across all files.

    Concatenation order matches ``_build_index``, so global idx aligns with
    the row-index produced for the inputs/variables.
    """
    chunks = []
    for path in paths:
        with h5py.File(path, 'r') as f:
            chunks.append(np.asarray(f[noise_key][:], dtype=np.float32))
    return np.concatenate(chunks, axis=0)


def hdf5_worker_init_fn(worker_id):
    """Required for num_workers > 0: HDF5 handles aren't fork-safe."""
    info = torch.utils.data.get_worker_info()
    if info is not None:
        info.dataset._file_handles = {}


# ─── joint-task dataset ─────────────────────────────────────────────────────
class Project8SimDataset(Dataset):
    """Project 8 simulation dataset.  Reads one HDF5 row per __getitem__.

    Noise traces are loaded once at construction time from a fixed HDF5 path
    (selected by ``noise_type``); per-item access is a pure index into the
    preloaded array.  Optionally computes the FFT of the I/Q channels.
    """

    def __init__(
        self,
        data_dir:       str,
        inputs:         list[str],
        variables:      list[str],
        cutoff:         int = 4000,
        norm:           bool = True,
        noise_type:     str = 'gaussian',
        noise_paths:    Optional[dict] = None,
        freq_transform: Optional[str] = 'fft',   # 'fft' | None
        mu:             Optional[np.ndarray] = None,
        stds:           Optional[np.ndarray] = None,
    ):
        super().__init__()
        self.paths          = _list_hdf5(data_dir)
        self.inputs         = inputs
        self.variables      = variables
        self.cutoff         = cutoff
        self.norm           = norm
        self.noise_type     = noise_type
        self.noise_paths    = noise_paths if noise_paths is not None else dict(DEFAULT_NOISE_PATHS)
        self.noise_key      = _resolve_noise_key(noise_type, self.noise_paths)
        self.freq_transform = freq_transform

        probe_key   = (variables + inputs)[0]
        self._index = _build_index(self.paths, probe_key)
        self._noise = _preload_noise(self.paths, self.noise_key)

        # Norm stats: use the supplied values when given, otherwise compute.
        if norm:
            if mu is not None and stds is not None:
                self.mu   = np.asarray(mu,   dtype=np.float32)
                self.stds = np.asarray(stds, dtype=np.float32)
            else:
                self.mu, self.stds = self._compute_norm_stats()
        else:
            self.mu, self.stds = None, None

        self._file_handles: dict = {}

    # ─── normalisation ──────────────────────────────────────────────────────
    def _compute_norm_stats(self):
        """Welford online mean/std over self.variables across all files."""
        n_vars = len(self.variables)
        count, mean, M2 = 0, np.zeros(n_vars, np.float64), np.zeros(n_vars, np.float64)

        for path in self.paths:
            with h5py.File(path, 'r') as f:
                data = np.stack([f[v][:] for v in self.variables], axis=1)
            for row in data:
                count += 1
                delta  = row - mean
                mean  += delta / count
                M2    += delta * (row - mean)

        return mean.astype(np.float32), np.sqrt(M2 / count).astype(np.float32)

    # ─── lazy HDF5 access ───────────────────────────────────────────────────
    def _get_handle(self, path):
        if path not in self._file_handles:
            self._file_handles[path] = h5py.File(path, 'r')
        return self._file_handles[path]

    def _read_row(self, path, local_idx, keys):
        f = self._get_handle(path)
        return {k: f[k][local_idx] for k in keys}

    # ─── frequency transforms ───────────────────────────────────────────────
    def _compute_fft(self, X_ts, iI, iQ):
        real, imag = fft_func_IQ_complex_channels(X_ts[:, iI], X_ts[:, iQ])
        stack    = np.stack([real, imag], axis=1)
        mu, std  = stack.mean(axis=0), stack.std(axis=0)
        X_fft    = np.zeros_like(X_ts)
        X_fft[:, iI] = (real - mu[0]) / (std[0] + 1e-8)
        X_fft[:, iQ] = (imag - mu[1]) / (std[1] + 1e-8)
        return X_fft

    # ─── Dataset protocol ───────────────────────────────────────────────────
    def __len__(self):
        return len(self._index)

    def __getitem__(self, idx):
        path, local_idx = self._index[idx]
        row = self._read_row(path, local_idx, self.inputs + self.variables)

        # inputs → (cutoff, n_inputs); add preloaded noise (broadcast 1-D → C).
        X_clean = np.stack([row[k] for k in self.inputs], axis=-1)[:self.cutoff]
        noise   = self._noise[idx]
        if noise.ndim == 1:
            noise = np.broadcast_to(noise[:, None], X_clean.shape)
        noise = noise[:self.cutoff]

        X_ts = (X_clean + noise).astype(np.float32, copy=True)
        if self.norm:
            for j in range(X_ts.shape[1]):
                X_ts[:, j] = X_ts[:, j] / (np.std(X_ts[:, j]) + 1e-8)

        y_raw    = np.array([row[v] for v in self.variables], dtype=np.float32)
        var_norm = (y_raw - self.mu) / (self.stds + 1e-8) if self.norm else y_raw

        ts  = torch.from_numpy(X_ts)
        var = torch.from_numpy(var_norm.astype(np.float32))

        if self.freq_transform is None:
            return ts, var

        if self.freq_transform == 'fft':
            iI = self.inputs.index('output_ts_I')
            iQ = self.inputs.index('output_ts_Q')
            fft = torch.from_numpy(self._compute_fft(X_ts, iI, iQ).astype(np.float32))
            return ts, fft, var

        raise ValueError(
            f"freq_transform must be 'fft' | None, got {self.freq_transform!r}"
        )

    # ─── shape helpers ──────────────────────────────────────────────────────
    def outdim(self):    return len(self.variables)
    def indim_ts(self):  return len(self.inputs)
    def indim_fft(self):
        if self.freq_transform == 'fft':
            return len(self.inputs)
        return None

    # ─── cleanup ────────────────────────────────────────────────────────────
    def close(self):
        for fh in self._file_handles.values():
            fh.close()
        self._file_handles.clear()

    def __del__(self):
        self.close()


# ─── denoising dataset ──────────────────────────────────────────────────────
class Project8SimDenoisingDataset(Dataset):
    """Project 8 denoising dataset.  Returns (X_noisy, X_clean) of shape
    (cutoff, n_inputs).  When norm=True both are divided by std(X_noisy) so
    they share a coordinate space.
    """

    def __init__(
        self,
        data_dir:    str,
        inputs:      list[str],
        cutoff:      int = 4000,
        norm:        bool = True,
        noise_type:  str = 'gaussian',
        noise_paths: Optional[dict] = None,
    ):
        super().__init__()
        self.paths       = _list_hdf5(data_dir)
        self.inputs      = inputs
        self.cutoff      = cutoff
        self.norm        = norm
        self.noise_type  = noise_type
        self.noise_paths = noise_paths if noise_paths is not None else dict(DEFAULT_NOISE_PATHS)
        self.noise_key   = _resolve_noise_key(noise_type, self.noise_paths)

        self._index = _build_index(self.paths, inputs[0])
        self._noise = _preload_noise(self.paths, self.noise_key)
        self._file_handles: dict = {}

    def _get_handle(self, path):
        if path not in self._file_handles:
            self._file_handles[path] = h5py.File(path, 'r')
        return self._file_handles[path]

    def _read_row(self, path, local_idx, keys):
        f = self._get_handle(path)
        return {k: f[k][local_idx] for k in keys}

    def __len__(self):
        return len(self._index)

    def __getitem__(self, idx):
        path, local_idx = self._index[idx]
        row = self._read_row(path, local_idx, self.inputs)

        X_clean = np.stack([row[k] for k in self.inputs], axis=-1)[:self.cutoff]
        noise   = self._noise[idx]
        if noise.ndim == 1:
            noise = np.broadcast_to(noise[:, None], X_clean.shape)
        noise = noise[:self.cutoff]

        X_noisy = X_clean + noise
        X_noisy_norm = np.zeros_like(X_clean, dtype=np.float32)
        X_clean_norm = np.zeros_like(X_clean, dtype=np.float32)
        for j in range(X_clean.shape[1]):
            if self.norm:
                s = np.std(X_noisy[:, j]) + 1e-8
                X_noisy_norm[:, j] = X_noisy[:, j] / s
                X_clean_norm[:, j] = X_clean[:, j] / s
            else:
                X_noisy_norm[:, j] = X_noisy[:, j]
                X_clean_norm[:, j] = X_clean[:, j]

        return (
            torch.from_numpy(X_noisy_norm),
            torch.from_numpy(X_clean_norm),
        )

    def indim(self):
        return len(self.inputs)

    def close(self):
        for fh in self._file_handles.values():
            fh.close()
        self._file_handles.clear()

    def __del__(self):
        self.close()


# ─── DataModules ────────────────────────────────────────────────────────────
class Project8DataModule(L.LightningDataModule):
    """Lightning DataModule for the joint Project 8 task.

    Splits are loaded from independent directories.  Norm stats default to
    placeholders (``None``); when not supplied, they are computed on the
    training split and reused for val/test.
    """

    def __init__(
        self,
        train_dir:      str,
        val_dir:        str,
        test_dir:       str,
        inputs:         list[str],
        variables:      list[str],
        cutoff:         int   = 4000,
        norm:           bool  = True,
        noise_type:     str   = 'gaussian',
        noise_paths:    Optional[dict] = None,
        freq_transform: Optional[str]  = 'fft',
        mu:             Optional[list] = None,
        stds:           Optional[list] = None,
        batch_size:     int   = 512,
        num_workers:    int   = 4,
        pin_memory:     bool  = False,
    ):
        super().__init__()
        self.save_hyperparameters()

    def _make_dataset(self, data_dir, mu, stds):
        return Project8SimDataset(
            data_dir       = data_dir,
            inputs         = self.hparams.inputs,
            variables      = self.hparams.variables,
            cutoff         = self.hparams.cutoff,
            norm           = self.hparams.norm,
            noise_type     = self.hparams.noise_type,
            noise_paths    = self.hparams.noise_paths,
            freq_transform = self.hparams.freq_transform,
            mu             = mu,
            stds           = stds,
        )

    def setup(self, stage: str | None = None):
        self.train = self._make_dataset(
            self.hparams.train_dir, self.hparams.mu, self.hparams.stds,
        )
        # Val/test reuse train's stats so all splits share the same coordinate space.
        mu, stds = self.train.mu, self.train.stds
        self.val   = self._make_dataset(self.hparams.val_dir,  mu, stds)
        self.test  = self._make_dataset(self.hparams.test_dir, mu, stds)

        self.mu, self.stds             = mu, stds
        self.input_channels_ts         = self.train.indim_ts()
        self.input_channels_fft        = self.train.indim_fft()

    def _loader(self, ds, shuffle):
        return DataLoader(
            ds,
            batch_size  = self.hparams.batch_size,
            num_workers = self.hparams.num_workers,
            pin_memory  = self.hparams.pin_memory,
            shuffle     = shuffle,
            worker_init_fn = hdf5_worker_init_fn if self.hparams.num_workers > 0 else None,
        )

    def train_dataloader(self):   return self._loader(self.train, shuffle=True)
    def val_dataloader(self):     return self._loader(self.val,   shuffle=False)
    def test_dataloader(self):    return self._loader(self.test,  shuffle=False)
    def predict_dataloader(self): return self.test_dataloader()


class Project8DenoisingDataModule(L.LightningDataModule):
    """Lightning DataModule for the S4D denoising task."""

    def __init__(
        self,
        train_dir:    str,
        val_dir:      str,
        test_dir:     str,
        inputs:       list[str],
        cutoff:       int   = 4000,
        norm:         bool  = True,
        noise_type:   str   = 'gaussian',
        noise_paths:  Optional[dict] = None,
        batch_size:   int   = 512,
        num_workers:  int   = 4,
        pin_memory:   bool  = False,
    ):
        super().__init__()
        self.save_hyperparameters()

    def _make_dataset(self, data_dir):
        return Project8SimDenoisingDataset(
            data_dir    = data_dir,
            inputs      = self.hparams.inputs,
            cutoff      = self.hparams.cutoff,
            norm        = self.hparams.norm,
            noise_type  = self.hparams.noise_type,
            noise_paths = self.hparams.noise_paths,
        )

    def setup(self, stage: str | None = None):
        self.train = self._make_dataset(self.hparams.train_dir)
        self.val   = self._make_dataset(self.hparams.val_dir)
        self.test  = self._make_dataset(self.hparams.test_dir)
        self.input_channels = self.train.indim()

    def _loader(self, ds, shuffle):
        return DataLoader(
            ds,
            batch_size  = self.hparams.batch_size,
            num_workers = self.hparams.num_workers,
            pin_memory  = self.hparams.pin_memory,
            shuffle     = shuffle,
            worker_init_fn = hdf5_worker_init_fn if self.hparams.num_workers > 0 else None,
        )

    def train_dataloader(self):   return self._loader(self.train, shuffle=True)
    def val_dataloader(self):     return self._loader(self.val,   shuffle=False)
    def test_dataloader(self):    return self._loader(self.test,  shuffle=False)
    def predict_dataloader(self): return self.test_dataloader()
