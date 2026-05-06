"""
Dataloader for the TESS light-curve classification dataset (PhyTS-bench).

Source:  huggingface.co/datasets/PhyTS-team/PhyTS-bench  → TESS/tess_classification.parquet

Each row contains a variable-length TESS light curve (~542-3917 points at ~30 min cadence):
    GaiaID   int64            Gaia DR3 source identifier
    TIC      int64            TESS Input Catalog identifier
    sector   int64            TESS observation sector
    label    str              one of 8 variability classes (see Label enum)
    time     ndarray float64  BJD timestamps
    flux     ndarray float64  normalised flux

Because light curves have variable length, each sample is padded (or truncated)
to a fixed `seq_len` and a boolean padding mask is returned.

Splitting is done by *star* (TIC), not by light curve, to prevent data leakage
(the same star observed in different sectors must not appear in both train and test).

Batch convention:
    flux   (B, seq_len)  float32   padded/truncated flux
    mask   (B, seq_len)  bool      True = real data, False = padding
    label  (B,)          int64     integer class label

Usage (LightningCLI YAML):
    data:
      class_path: dataloader.tess_dataloader.TESSDataModule
      init_args:
        data_path: data/TESS/tess_classification.parquet
        seq_len: 3917
        batch_size: 64
"""

from enum import IntEnum

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
import lightning as L


class Label(IntEnum):
    APERIODIC = 0
    CONTACT_ROT = 1
    DSCT_BCEP = 2
    ECLIPSE = 3
    GDOR_SPB = 4
    INSTRUMENT_JUNK = 5
    RRLYR_CEPH = 6
    SOLARLIKE = 7


LABEL_STR_TO_INT = {
    "APERIODIC": Label.APERIODIC,
    "CONTACT_ROT": Label.CONTACT_ROT,
    "DSCT_BCEP": Label.DSCT_BCEP,
    "ECLIPSE": Label.ECLIPSE,
    "GDOR_SPB": Label.GDOR_SPB,
    "INSTRUMENT/JUNK": Label.INSTRUMENT_JUNK,
    "RRLYR_CEPH": Label.RRLYR_CEPH,
    "SOLARLIKE": Label.SOLARLIKE,
}

NUM_CLASSES = len(Label)


class TESSDataset(Dataset):
    """Fixed-length TESS light-curve dataset with padding mask."""

    def __init__(self, flux: list[np.ndarray], labels: np.ndarray, seq_len: int):
        self.flux = flux
        self.labels = torch.as_tensor(labels, dtype=torch.long)
        self.seq_len = seq_len

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        raw = self.flux[idx].astype(np.float32)
        n = len(raw)

        flux = np.zeros(self.seq_len, dtype=np.float32)
        mask = np.zeros(self.seq_len, dtype=np.bool_)

        if n >= self.seq_len:
            flux[:] = raw[: self.seq_len]
            mask[:] = True
        else:
            flux[:n] = raw
            mask[:n] = True

        return (
            torch.from_numpy(flux),
            torch.from_numpy(mask),
            self.labels[idx],
        )


class TESSDataModule(L.LightningDataModule):
    """Lightning DataModule for TESS classification.

    Splits by TIC (star-level) to avoid data leakage across sectors.
    """

    def __init__(
        self,
        data_path: str,
        seq_len: int = 3917,
        batch_size: int = 64,
        num_workers: int = 0,
        train_frac: float = 0.7,
        val_frac: float = 0.15,
        seed: int = 42,
    ):
        super().__init__()
        self.data_path = data_path
        self.seq_len = seq_len
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.train_frac = train_frac
        self.val_frac = val_frac
        self.seed = seed

    def setup(self, stage: str | None = None):
        df = pd.read_parquet(self.data_path)
        df["label_int"] = df["label"].map(LABEL_STR_TO_INT)

        # Split by unique TIC to prevent leakage
        tics = df["TIC"].unique()
        rng = np.random.default_rng(self.seed)
        rng.shuffle(tics)

        n = len(tics)
        n_train = int(n * self.train_frac)
        n_val = int(n * self.val_frac)

        train_tics = set(tics[:n_train])
        val_tics = set(tics[n_train : n_train + n_val])
        test_tics = set(tics[n_train + n_val :])

        def make_dataset(tic_set):
            mask = df["TIC"].isin(tic_set)
            sub = df[mask]
            return TESSDataset(
                flux=list(sub["flux"].values),
                labels=sub["label_int"].values,
                seq_len=self.seq_len,
            )

        if stage == "fit":
            self.train_ds = make_dataset(train_tics)
            self.val_ds = make_dataset(val_tics)
        elif stage in ("test", "predict"):
            self.test_ds = make_dataset(test_tics)
        else:
            self.train_ds = make_dataset(train_tics)
            self.val_ds = make_dataset(val_tics)
            self.test_ds = make_dataset(test_tics)

    def train_dataloader(self):
        return DataLoader(
            self.train_ds,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_ds,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_ds,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
        )

    def predict_dataloader(self):
        return self.test_dataloader()
