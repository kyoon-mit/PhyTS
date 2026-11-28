"""Dataloaders for the TESS light-curve tasks: variability classification and rotation regression.

Reads the fixed 80/10/10 Hub splits, data/TESS/split/tess_{task}_{split}.parquet (see
data/TESS/README.md); no Gaia DR3 source appears in more than one split.

Light curves have 542-3,917 points (10- or 30-minute cadence). Each one is z-scored over all of
its points (median-centred, divided by its standard deviation), then cropped to its first
seq_len points or zero-padded to seq_len. Batches are (flux, mask, target):
    flux    (B, seq_len)  float32  z-scored, zero-padded flux
    mask    (B, seq_len)  bool     True on real cadences
    target  (B,)          int64 class index (classification) or float32 frot in 1/day (regression)
"""

import lightning as L
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

# Class index -> label as stored on the Hub.
CLASSES = ("APERIODIC", "CONTACT_ROT", "DSCT_BCEP", "ECLIPSE", "GDOR_SPB", "INSTRUMENT/JUNK",
           "RRLYR_CEPH", "SOLARLIKE")


class TESSDataset(Dataset):
    """Fixed-length, z-scored light curves with a padding mask."""

    def __init__(self, flux: list[np.ndarray], targets: torch.Tensor, seq_len: int):
        self.flux = flux
        self.targets = targets
        self.seq_len = seq_len

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, idx):
        raw = self.flux[idx]
        n = min(len(raw), self.seq_len)
        flux = np.zeros(self.seq_len, dtype=np.float32)
        flux[:n] = ((raw - np.median(raw)) / raw.std())[:n]
        mask = np.arange(self.seq_len) < n
        return torch.from_numpy(flux), torch.from_numpy(mask), self.targets[idx]


class TESSDataModule(L.LightningDataModule):
    """Loaders over the Hub splits of one task; subclasses set the task and its target."""

    task: str

    def __init__(
        self,
        data_dir: str = "data/TESS/split",
        seq_len: int = 1100,
        batch_size: int = 128,
        num_workers: int = 4,
    ):
        super().__init__()
        self.data_dir = data_dir
        self.seq_len = seq_len
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.datasets: dict[str, TESSDataset] = {}

    def targets(self, df: pd.DataFrame) -> torch.Tensor:
        """Target of every row of one split file."""
        raise NotImplementedError

    def setup(self, stage: str | None = None):
        splits = {"fit": ("train", "val"), "validate": ("val",), "test": ("test",),
                  "predict": ("test",)}.get(stage, ("train", "val", "test"))
        for split in splits:
            df = pd.read_parquet(f"{self.data_dir}/tess_{self.task}_{split}.parquet")
            self.datasets[split] = TESSDataset(list(df["flux"]), self.targets(df), self.seq_len)

    def _loader(self, split: str, shuffle: bool = False) -> DataLoader:
        return DataLoader(self.datasets[split], batch_size=self.batch_size, shuffle=shuffle,
                          num_workers=self.num_workers)

    def train_dataloader(self):
        return self._loader("train", shuffle=True)

    def val_dataloader(self):
        return self._loader("val")

    def test_dataloader(self):
        return self._loader("test")

    def predict_dataloader(self):
        return self._loader("test")


class TESSClassificationDataModule(TESSDataModule):
    """8-class variability classification; targets are indices into CLASSES."""

    task = "classification"

    def targets(self, df: pd.DataFrame) -> torch.Tensor:
        return torch.tensor(df["label"].map(CLASSES.index).to_numpy(), dtype=torch.long)


class TESSRegressionDataModule(TESSDataModule):
    """Near-core rotation frequency (frot, 1/day) of the GDOR_SPB stars."""

    task = "regression"

    def targets(self, df: pd.DataFrame) -> torch.Tensor:
        return torch.tensor(df["frot"].to_numpy(), dtype=torch.float32)
