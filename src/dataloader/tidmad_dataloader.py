"""
Dataloader for the preprocessed TIDMAD dataset (ABRACADABRA dark matter experiment).

Expects the output of data/TIDMAD/preprocess_tidmad.py in data_dir:
  {split}_ch1.npy     int8     (N, 100_000)  noisy SQUID channel (channel0001)
  {split}_ch2.npy     int8     (N, 100_000)  clean injected reference (channel0002)
  {split}_params.npy  float32  (N, 3)        [frequency_hz, amplitude_mV, snr]
  scale.npy           float32  scalar        voltage_range_mV / 256.0

int8 arrays are memory-mapped (not loaded into RAM) and converted to float32
on-the-fly in __getitem__ by multiplying by scale.

Window convention:
  0.01 s × 10 MHz = 100,000 samples per window, no downsampling.
  FFT resolution: 10 MHz / 100k = 100 Hz (matches minimum frequency grid step).
  Nyquist: 5 MHz (covers full 1.1 kHz – 4.9 MHz injection range).

Usage (LightningCLI YAML):
    data:
      class_path: dataloader.tidmad_dataloader.TIDMADDataModule
      init_args:
        data_dir: data/TIDMAD/preprocessed
        batch_size: 32

Batch convention:
    noisy   (B, 100_000)  float32  channel0001
    clean   (B, 100_000)  float32  channel0002
    params  (B, 3)        float32  [frequency_hz, amplitude_mV, snr]
"""

from enum import IntEnum

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import lightning as L


class Param(IntEnum):
    frequency_hz = 0
    amplitude    = 1
    snr          = 2


class TIDMADDataset(Dataset):
    """Memory-mapped TIDMAD dataset.  Loads int8 arrays, scales on-the-fly."""

    def __init__(self, data_dir: str, split: str):
        self.scale  = float(np.load(f'{data_dir}/scale.npy'))
        # mmap_mode='r': array is not loaded into RAM; OS pages on demand
        self.ch1    = np.load(f'{data_dir}/{split}_ch1.npy',    mmap_mode='r')
        self.ch2    = np.load(f'{data_dir}/{split}_ch2.npy',    mmap_mode='r')
        self.params = torch.as_tensor(
                          np.load(f'{data_dir}/{split}_params.npy'))

    def __len__(self):
        return len(self.params)

    def __getitem__(self, idx):
        # astype copies the int8 slice into a new float32 array
        ch1 = torch.from_numpy(self.ch1[idx].astype(np.float32)) * self.scale
        ch2 = torch.from_numpy(self.ch2[idx].astype(np.float32)) * self.scale
        return ch1, ch2, self.params[idx]


class TIDMADDataModule(L.LightningDataModule):
    """Lightning DataModule for preprocessed TIDMAD NPY splits.

    Expects train / val / test splits produced by preprocess_tidmad.py.
    """

    def __init__(
        self,
        data_dir:    str,
        batch_size:  int = 32,
        num_workers: int = 0,
    ):
        super().__init__()
        self.data_dir    = data_dir
        self.batch_size  = batch_size
        self.num_workers = num_workers

    def setup(self, stage: str | None = None):
        if stage == 'fit':
            self.train = TIDMADDataset(self.data_dir, 'train')
            self.val   = TIDMADDataset(self.data_dir, 'val')
        elif stage in ('test', 'predict'):
            self.test  = TIDMADDataset(self.data_dir, 'test')

    def train_dataloader(self):
        return DataLoader(self.train, batch_size=self.batch_size,
                          shuffle=True,  num_workers=self.num_workers)

    def val_dataloader(self):
        return DataLoader(self.val,   batch_size=self.batch_size,
                          shuffle=False, num_workers=self.num_workers)

    def test_dataloader(self):
        return DataLoader(self.test,  batch_size=self.batch_size,
                          shuffle=False, num_workers=self.num_workers)

    def predict_dataloader(self):
        return self.test_dataloader()
