"""
Dataloader for the sinusoidal + white noise toy dataset.

Signal model:
    s(t) = A * sin(2*pi*f*t + phi)          clean signal
    x(t) = s(t) + eps(t),  eps ~ N(0, σ²)  observed signal

Each split file (train.npz / val.npz / test.npz) contains:
    sig      (N, 640)  float32  clean signal
    sig_bkg  (N, 640)  float32  signal + noise
    params   (N, 5)    float32  see Param enum below

Usage:
    from dataloader.toy_dataloader import ToyDataModule, Param

    dm = ToyDataModule(data_dir='data/toy/sinusoidal_signal_white_noise', batch_size=64)
    dm.setup('fit')

    for sig_bkg, sig, params in dm.train_dataloader():
        snr  = params[:, Param.snr]           # (B,)
        freq = params[:, Param.frequency_hz]  # (B,)
"""

from enum import IntEnum

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import lightning as L

class Param(IntEnum):
    amplitude       = 0
    frequency_hz    = 1
    phase_rad       = 2
    noise_amplitude = 3
    snr             = 4

class ToyDataset(Dataset):
    def __init__(self, data_dir: str, split: str):
        f = np.load(f'{data_dir}/{split}.npz')
        self.sig     = torch.as_tensor(f['sig'])      # (N, T)
        self.sig_bkg = torch.as_tensor(f['sig_bkg'])  # (N, T)
        self.params  = torch.as_tensor(f['params'])   # (N, 5): see Param enum

    def __len__(self):
        return len(self.sig)

    def __getitem__(self, idx):
        return self.sig_bkg[idx], self.sig[idx], self.params[idx]

class ToyDataModule(L.LightningDataModule):
    def __init__(
        self,
        data_dir: str,
        batch_size: int = 64,
        num_workers: int = 0,
    ):
        super().__init__()
        self.data_dir    = data_dir
        self.batch_size  = batch_size
        self.num_workers = num_workers

    def setup(self, stage: str | None = None):
        if stage == 'fit':
            self.train = ToyDataset(self.data_dir, 'train')
            self.val   = ToyDataset(self.data_dir, 'val')
        elif stage in ('test', 'predict'):
            self.test  = ToyDataset(self.data_dir, 'test')

    def train_dataloader(self):
        return DataLoader(self.train, batch_size=self.batch_size, shuffle=True,  num_workers=self.num_workers)

    def val_dataloader(self):
        return DataLoader(self.val,   batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)

    def test_dataloader(self):
        return DataLoader(self.test,  batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)

    def predict_dataloader(self):
        return DataLoader(self.test,  batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)
