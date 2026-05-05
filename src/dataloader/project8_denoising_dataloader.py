"""
Paired (noisy, clean) Project 8 dataset for supervised denoising.

The vendored upstream `Project8SimDataset` only returns the noisy trace + the
scalar physics target.  Denoising needs the clean ground-truth trace alongside,
so this module subclasses it and returns both tensors per item.

Both tensors are normalised by the **same** per-channel scale (the std of the
*noisy* trace), so the model's MSE target lives in the same coordinate space
as its input.

This is a downstream addition; not in the upstream repo (chreissel/.../main).
The upstream `DenoisingBatch` enum hints this was anticipated but not wired.
"""

from __future__ import annotations

from typing import Optional

import lightning as L
import numpy as np
import torch
from torch.utils.data import DataLoader

from .project8_dataloader import (
    Project8SimDataset,
    hdf5_worker_init_fn,
)


class Project8SimDenoisingDataset(Project8SimDataset):
    """Returns (noisy, clean) per item; both shape (cutoff, C), float32.

    No FFT path here — denoising operates in the time domain.
    """

    def __init__(
        self,
        data_dir:   str,
        inputs:     list[str],
        cutoff:     int  = 4000,
        norm:       bool = True,
        noise_type: str  = 'gauss',
    ):
        # `variables` is required by the parent for label normalisation; we
        # don't use it here, but a non-empty list keeps `_compute_norm_stats`
        # happy.  Pick a per-event scalar that exists in every shard.
        super().__init__(
            data_dir       = data_dir,
            inputs         = inputs,
            variables      = ['energy_eV'],   # unused, but required scaffolding
            cutoff         = cutoff,
            norm           = norm,
            noise_type     = noise_type,
            freq_transform = None,
        )

    def __getitem__(self, idx):
        path, local_idx = self._index[idx]
        row = self._read_row(path, local_idx, self.inputs + self.noise_keys)

        clean = np.stack([row[k] for k in self.inputs], axis=-1)[:self.cutoff].astype(np.float32, copy=True)
        if self.noise_keys:
            noise = np.stack([row[k] for k in self.noise_keys], axis=-1)[:self.cutoff]
            noisy = (clean + noise).astype(np.float32, copy=True)
        else:
            noisy = clean.copy()

        if self.norm:
            # Same per-channel scale for both → identical coordinate space.
            for j in range(noisy.shape[1]):
                s = float(np.std(noisy[:, j])) + 1e-8
                noisy[:, j] /= s
                clean[:, j] /= s

        return torch.from_numpy(noisy), torch.from_numpy(clean)


class Project8DenoisingDataModule(L.LightningDataModule):
    """Lightning DataModule yielding (noisy, clean) pairs for Project 8."""

    def __init__(
        self,
        train_dir:   str,
        val_dir:     str,
        test_dir:    str,
        inputs:      list[str] = None,
        cutoff:      int       = 4000,
        norm:        bool      = True,
        noise_type:  str       = 'gauss',
        batch_size:  int       = 32,
        num_workers: int       = 4,
        pin_memory:  bool      = False,
    ):
        super().__init__()
        if inputs is None:
            inputs = ['output_ts_I', 'output_ts_Q']
        self.save_hyperparameters()

    def _make(self, data_dir: str) -> Project8SimDenoisingDataset:
        return Project8SimDenoisingDataset(
            data_dir   = data_dir,
            inputs     = self.hparams.inputs,
            cutoff     = self.hparams.cutoff,
            norm       = self.hparams.norm,
            noise_type = self.hparams.noise_type,
        )

    def setup(self, stage: Optional[str] = None):
        self.train = self._make(self.hparams.train_dir)
        self.val   = self._make(self.hparams.val_dir)
        self.test  = self._make(self.hparams.test_dir)
        self.input_channels = self.train.indim_ts()

    def _loader(self, ds, shuffle: bool) -> DataLoader:
        return DataLoader(
            ds,
            batch_size     = self.hparams.batch_size,
            num_workers    = self.hparams.num_workers,
            pin_memory     = self.hparams.pin_memory,
            shuffle        = shuffle,
            worker_init_fn = hdf5_worker_init_fn if self.hparams.num_workers > 0 else None,
        )

    def train_dataloader(self):  return self._loader(self.train, shuffle=True)
    def val_dataloader(self):    return self._loader(self.val,   shuffle=False)
    def test_dataloader(self):   return self._loader(self.test,  shuffle=False)
