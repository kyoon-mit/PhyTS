"""Dataloader for Project 8: electron energy from simulated CRES I/Q time series.

Reads the Hub's split folders, data/Project8/{train,valid,test}/*.hdf5 (see
data/Project8/README.md). Every event stores the clean in-phase and quadrature signal
(output_ts_I, output_ts_Q), one cavity and one Gaussian noise realisation per channel, and the
truth parameters. The model input is signal plus the chosen noise (the paper uses the cavity
noise), cut to the first `cutoff` samples and divided by each channel's standard deviation. The
target is z-scored with the training split's mean and standard deviation, which the tasks use
to report errors in eV.

Batches are (x, y): x (B, cutoff, 2) float32 I/Q, y (B, 1) float32 z-scored target.
"""

from glob import glob

import h5py
import lightning as L
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

CHANNELS = ("output_ts_I", "output_ts_Q")


def split_files(data_dir: str, split: str) -> list[str]:
    """Sorted HDF5 files of one split folder."""
    paths = sorted(glob(f"{data_dir}/{split}/*.hdf5"))
    if not paths:
        raise FileNotFoundError(f"no .hdf5 files in {data_dir}/{split}")
    return paths


def read_target(paths: list[str], target: str) -> np.ndarray:
    """Target of every event, in file and row order."""
    out = []
    for path in paths:
        with h5py.File(path, "r") as f:
            out.append(f[target][:])
    return np.concatenate(out).astype(np.float64)


class Project8Dataset(Dataset):
    """Noisy, normalised I/Q traces and the z-scored target, read one event at a time."""

    def __init__(self, paths: list[str], target: str, noise_type: str, cutoff: int,
                 target_mean: float, target_std: float):
        self.target = target
        self.noise_keys = [f"{c}_{noise_type}_noise" for c in CHANNELS]
        self.cutoff = cutoff
        self.target_mean = target_mean
        self.target_std = target_std
        self.index = []                      # event -> (file, row)
        for path in paths:
            with h5py.File(path, "r") as f:
                self.index += [(path, row) for row in range(f[target].shape[0])]
        self.handles: dict[str, h5py.File] = {}

    def __len__(self):
        return len(self.index)

    def __getitem__(self, idx):
        path, row = self.index[idx]
        if path not in self.handles:         # opened lazily, so each loader worker opens its own
            self.handles[path] = h5py.File(path, "r")
        f = self.handles[path]
        signal = np.stack([f[k][row, :self.cutoff] for k in CHANNELS], axis=-1)
        noise = np.stack([f[k][row, :self.cutoff] for k in self.noise_keys], axis=-1)
        x = signal.astype(np.float32) + noise.astype(np.float32)
        x /= x.std(axis=0) + 1e-8
        y = (f[self.target][row] - self.target_mean) / self.target_std
        return torch.from_numpy(x), torch.tensor([y], dtype=torch.float32)


class Project8DataModule(L.LightningDataModule):
    """Train / valid / test loaders; target_mean and target_std come from the training split."""

    def __init__(
        self,
        data_dir: str = "data/Project8",
        target: str = "energy_eV",
        noise_type: str = "cav",
        cutoff: int = 24576,
        batch_size: int = 64,
        num_workers: int = 4,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.datasets: dict[str, Project8Dataset] = {}

    def setup(self, stage: str | None = None):
        hp = self.hparams
        train_target = read_target(split_files(hp.data_dir, "train"), hp.target)
        self.target_mean, self.target_std = float(train_target.mean()), float(train_target.std())
        splits = {"fit": ("train", "valid"), "validate": ("valid",), "test": ("test",),
                  "predict": ("test",)}.get(stage, ("train", "valid", "test"))
        for split in splits:
            self.datasets[split] = Project8Dataset(
                split_files(hp.data_dir, split), hp.target, hp.noise_type, hp.cutoff,
                self.target_mean, self.target_std)

    def _loader(self, split: str, shuffle: bool = False) -> DataLoader:
        return DataLoader(self.datasets[split], batch_size=self.hparams.batch_size,
                          shuffle=shuffle, num_workers=self.hparams.num_workers)

    def train_dataloader(self):
        return self._loader("train", shuffle=True)

    def val_dataloader(self):
        return self._loader("valid")

    def test_dataloader(self):
        return self._loader("test")

    def predict_dataloader(self):
        return self._loader("test")
