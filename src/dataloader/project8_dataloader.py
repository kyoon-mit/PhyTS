"""
Dataloader for the Project 8 simulation dataset (CRES neutrino-mass experiment).

Expects a directory of HDF5 files under data_dir, each containing per-event
datasets keyed by name:
  inputs       e.g. ['output_ts_I', 'output_ts_Q']  (T,)  float32  time series
  variables    e.g. ['energy', ...]                 ()    float32  regression targets
  observables  e.g. ['frequency', ...]              ()    float32  auxiliary labels

HDF5 files are opened lazily (one handle per worker, not fork-shared) and a
single row is read per __getitem__ call.  Splits are produced at module setup
time by a deterministic torch.Generator(seed=42) random_split with ratios
[0.8, 0.1, 0.1].

Two tasks are supported:
  Project8SimDataset           – joint task: ts (+ optional fft / qtransform),
                                 normalised regression targets, observables.
  Project8SimDenoisingDataset  – denoising task: (X_noisy, X_clean) pairs.

Window convention:
  cutoff samples per channel (default 4000), n_inputs channels (typically I/Q).

Usage (LightningCLI YAML):
    data:
      class_path: dataloader.project8_dataloader.Project8DataModule
      init_args:
        data_dir:    data/Project8Sim/hdf5
        inputs:      [output_ts_I, output_ts_Q]
        variables:   [energy]
        observables: [frequency]
        batch_size:  512
        cutoff:      4000

Batch convention:
    Project8DataModule (freq_transform='fft', the default):
        ts        (B, cutoff, C)        float32  time-series, per-channel std-norm
        fft       (B, cutoff, C)        float32  FFT(I,Q), real/imag stacked into channels
        var       (B, n_variables)      float32  z-scored regression targets
        obs       (B, n_observables)    float32  auxiliary labels

    Project8DenoisingDataModule:
        noisy     (B, cutoff, C)        float32  signal + noise, per-channel std-norm
        clean     (B, cutoff, C)        float32  signal, normalised by the *same* std

Note on style vs TIDMAD:
  TIDMAD's loader assumes preprocessed, pre-split NPY arrays that are
  memory-mapped and only multiplied by a scalar in __getitem__.  Project 8
  cannot work that way – noise is freshly sampled every epoch (and the noise
  level itself is mutable for curriculum learning), so the dataset must do
  real work in __getitem__.  This file mirrors TIDMAD's organisation
  (single module, top-level docstring, IntEnum index helper, setup(stage)
  branching, minimal DataModule kwargs) while preserving the runtime work.
"""

from enum import IntEnum
from typing import Optional

import h5py
import lightning as L
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, random_split

from src.utils.noise import noise_model, bandpass_filter
from src.utils.transforms import (
    fft_func_IQ_complex_channels,
    qtransform_func_IQ_complex_channels,
)


# ─── batch-tuple indices for the joint task ─────────────────────────────────
# Position depends on freq_transform; see Project8SimDataset.__getitem__.
class JointBatch(IntEnum):
    ts  = 0
    var = 1   # 2 if a freq tensor sits in front
    obs = 2   # 3 if a freq tensor sits in front


class DenoisingBatch(IntEnum):
    noisy = 0
    clean = 1


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


def hdf5_worker_init_fn(worker_id):
    """Required for num_workers > 0: HDF5 handles aren't fork-safe."""
    info = torch.utils.data.get_worker_info()
    if info is not None:
        info.dataset._file_handles = {}


# ─── joint-task dataset ─────────────────────────────────────────────────────
class Project8SimDataset(Dataset):
    """Project 8 simulation dataset.  Reads one HDF5 row per __getitem__.

    Adds Gaussian noise at runtime (level controlled by self.noise_const, which
    may be mutated externally for curriculum learning) and optionally computes
    FFT and/or Q-transform of the I/Q channels.
    """

    def __init__(
        self,
        data_dir:       str,
        inputs:         list[str],
        variables:      list[str],
        observables:    list[str],
        cutoff:         int = 4000,
        norm:           bool = True,
        noise_const:    float = 1.0,
        apply_filter:   bool = False,
        freq_transform: Optional[str] = 'fft',   # 'fft' | 'qtransform' | 'both' | None
        q_params:       Optional[dict] = None,
    ):
        super().__init__()
        self.paths          = _list_hdf5(data_dir)
        self.inputs         = inputs
        self.variables      = variables
        self.observables    = observables
        self.cutoff         = cutoff
        self.norm           = norm
        self.noise_const    = noise_const
        self.apply_filter   = apply_filter
        self.freq_transform = freq_transform
        self.q_params       = q_params or {
            'fs': 200e6, 'fmin': 1e6, 'fmax': 100e6,
            'q_value': 5.0, 'num_freqs': 100,
        }

        probe_key   = (variables + observables + inputs)[0]
        self._index = _build_index(self.paths, probe_key)

        self.mu, self.stds = self._compute_norm_stats() if norm else (None, None)
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

    def _compute_qtransform(self, X_ts, iI, iQ):
        real, imag = qtransform_func_IQ_complex_channels(
            X_ts[:, iI], X_ts[:, iQ], **self.q_params,
        )
        nf   = self.q_params['num_freqs']
        real = real.reshape(self.cutoff, nf)
        imag = imag.reshape(self.cutoff, nf)
        stack = np.stack([real, imag], axis=0)
        mu    = stack.mean(axis=(1, 2), keepdims=True)
        std   = stack.std (axis=(1, 2), keepdims=True)
        return np.stack(
            [(real - mu[0]) / (std[0] + 1e-8),
             (imag - mu[1]) / (std[1] + 1e-8)],
            axis=-1,
        )  # (cutoff, nf, 2)

    # ─── Dataset protocol ───────────────────────────────────────────────────
    def __len__(self):
        return len(self._index)

    def __getitem__(self, idx):
        path, local_idx = self._index[idx]
        row = self._read_row(
            path, local_idx, self.inputs + self.variables + self.observables,
        )

        # inputs → (cutoff, n_inputs); add fresh noise per channel
        X_clean = np.stack([row[k] for k in self.inputs], axis=-1)[:self.cutoff]
        X_ts    = X_clean.copy()
        for j in range(X_ts.shape[1]):
            Xn = X_clean[:, j] + noise_model(self.cutoff, self.noise_const)
            if self.apply_filter:
                Xn = bandpass_filter(Xn)
            X_ts[:, j] = Xn / (np.std(Xn) + 1e-8) if self.norm else Xn

        y_raw    = np.array([row[v] for v in self.variables],   dtype=np.float32)
        var_norm = (y_raw - self.mu) / (self.stds + 1e-8) if self.norm else y_raw
        obs      = np.array([row[o] for o in self.observables], dtype=np.float32)

        ts  = torch.from_numpy(X_ts.astype(np.float32))
        var = torch.from_numpy(var_norm.astype(np.float32))
        obs = torch.from_numpy(obs)

        if self.freq_transform is None:
            return ts, var, obs

        iI = self.inputs.index('output_ts_I')
        iQ = self.inputs.index('output_ts_Q')

        if self.freq_transform == 'fft':
            fft = torch.from_numpy(self._compute_fft(X_ts, iI, iQ).astype(np.float32))
            return ts, fft, var, obs

        if self.freq_transform == 'qtransform':
            qt = torch.from_numpy(self._compute_qtransform(X_ts, iI, iQ).astype(np.float32))
            return qt, var, obs

        if self.freq_transform == 'both':
            fft = torch.from_numpy(self._compute_fft(X_ts, iI, iQ).astype(np.float32))
            qt  = torch.from_numpy(self._compute_qtransform(X_ts, iI, iQ).astype(np.float32))
            return ts, fft, qt, var, obs

        raise ValueError(
            f"freq_transform must be 'fft' | 'qtransform' | 'both' | None, "
            f"got {self.freq_transform!r}"
        )

    # ─── shape helpers ──────────────────────────────────────────────────────
    def outdim(self):                return len(self.variables)
    def indim_ts(self):              return len(self.inputs)
    def indim_fft(self):
        if self.freq_transform in ('fft', 'both'):  return len(self.inputs)
        if self.freq_transform == 'qtransform':     return self.q_params['num_freqs'] * 2
    def indim_qtransform(self):
        if self.freq_transform in ('qtransform', 'both'):
            return self.q_params['num_freqs'] * 2

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
        data_dir:     str,
        inputs:       list[str],
        cutoff:       int = 4000,
        noise_const:  float = 1.0,
        apply_filter: bool = False,
        norm:         bool = True,
    ):
        super().__init__()
        self.paths        = _list_hdf5(data_dir)
        self.inputs       = inputs
        self.cutoff       = cutoff
        self.noise_const  = noise_const
        self.apply_filter = apply_filter
        self.norm         = norm

        self._index = _build_index(self.paths, inputs[0])
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
        X_noisy_norm = np.zeros_like(X_clean)
        X_clean_norm = np.zeros_like(X_clean)

        for j in range(X_clean.shape[1]):
            Xn = X_clean[:, j] + noise_model(self.cutoff, self.noise_const)
            if self.apply_filter:
                Xn = bandpass_filter(Xn)
            if self.norm:
                s = np.std(Xn) + 1e-8
                X_noisy_norm[:, j] = Xn / s
                X_clean_norm[:, j] = X_clean[:, j] / s
            else:
                X_noisy_norm[:, j] = Xn
                X_clean_norm[:, j] = X_clean[:, j]

        return (
            torch.from_numpy(X_noisy_norm.astype(np.float32)),
            torch.from_numpy(X_clean_norm.astype(np.float32)),
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
_SPLIT_RATIOS = (0.8, 0.1, 0.1)
_SPLIT_SEED   = 42


class Project8DataModule(L.LightningDataModule):
    """Lightning DataModule for the joint Project 8 task."""

    def __init__(
        self,
        data_dir:               str,
        inputs:                 list[str],
        variables:              list[str],
        observables:            list[str],
        cutoff:                 int   = 4000,
        norm:                   bool  = True,
        noise_const:            float = 1.0,
        apply_filter:           bool  = False,
        freq_transform:         Optional[str] = 'fft',
        q_params:               Optional[dict] = None,
        use_curriculum_learning: bool = False,
        batch_size:             int   = 512,
        num_workers:            int   = 4,
        pin_memory:             bool  = False,
    ):
        super().__init__()
        self.save_hyperparameters()

    def setup(self, stage: str | None = None):
        # Single underlying dataset → deterministic random_split.
        ds = Project8SimDataset(
            data_dir       = self.hparams.data_dir,
            inputs         = self.hparams.inputs,
            variables      = self.hparams.variables,
            observables    = self.hparams.observables,
            cutoff         = self.hparams.cutoff,
            norm           = self.hparams.norm,
            noise_const    = self.hparams.noise_const,
            apply_filter   = self.hparams.apply_filter,
            freq_transform = self.hparams.freq_transform,
            q_params       = self.hparams.q_params,
        )
        self.dataset = ds
        self.mu, self.stds = ds.mu, ds.stds
        self.input_channels_ts         = ds.indim_ts()
        self.input_channels_fft        = ds.indim_fft()
        self.input_channels_qtransform = ds.indim_qtransform()

        gen = torch.Generator().manual_seed(_SPLIT_SEED)
        self.train, self.val, self.test = random_split(ds, list(_SPLIT_RATIOS), generator=gen)

    # curriculum-learning hook (mutates the underlying dataset in-place)
    def set_noise_const(self, new_const: float):
        if self.hparams.use_curriculum_learning:
            self.dataset.noise_const = new_const

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
        data_dir:                str,
        inputs:                  list[str],
        cutoff:                  int   = 4000,
        norm:                    bool  = True,
        noise_const:             float = 1.0,
        apply_filter:            bool  = False,
        use_curriculum_learning: bool  = False,
        batch_size:              int   = 512,
        num_workers:             int   = 4,
        pin_memory:              bool  = False,
    ):
        super().__init__()
        self.save_hyperparameters()

    def setup(self, stage: str | None = None):
        ds = Project8SimDenoisingDataset(
            data_dir     = self.hparams.data_dir,
            inputs       = self.hparams.inputs,
            cutoff       = self.hparams.cutoff,
            noise_const  = self.hparams.noise_const,
            apply_filter = self.hparams.apply_filter,
            norm         = self.hparams.norm,
        )
        self.dataset = ds
        self.input_channels = ds.indim()

        gen = torch.Generator().manual_seed(_SPLIT_SEED)
        self.train, self.val, self.test = random_split(ds, list(_SPLIT_RATIOS), generator=gen)

    def set_noise_const(self, new_const: float):
        if self.hparams.use_curriculum_learning:
            self.dataset.noise_const = new_const

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
