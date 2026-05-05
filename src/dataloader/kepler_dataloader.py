"""Dataloader for the Kepler Q9 v3 stellar-variability dataset.

Each .txt under the zip archive holds one Kepler Q9 light curve:
    col 1: time (days)
    col 2: flux
    col 3: flux error
and lives under a folder named after the star's variability class:
    APERIODIC/, CONSTANT/, CONTACT_ROT/, DSCT_BCEP/, ECLIPSE/,
    GDOR_SPB/, INSTRUMENT/, RRLYR_CEPHEID/, SOLARLIKE/

Most curves have 1341 samples. Curves with a different length are padded or
truncated on the right to LC_LEN.

The dataloader reads from the zip in-memory (no extraction) and caches parsed
arrays to an .npz on first load.
"""

from __future__ import annotations

import io
import zipfile
from pathlib import Path

import numpy as np
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset


CLASS_NAMES = [
    "APERIODIC", "CONSTANT", "CONTACT_ROT", "DSCT_BCEP", "ECLIPSE",
    "GDOR_SPB", "INSTRUMENT", "RRLYR_CEPHEID", "SOLARLIKE",
]
CLASS_TO_IDX = {c: i for i, c in enumerate(CLASS_NAMES)}
LC_LEN = 1341  # canonical length


def _fill_nans(x: np.ndarray) -> np.ndarray:
    """Linearly interpolate NaN values in a 1D array; fall back to 0 if all-NaN."""
    mask = np.isnan(x)
    if not mask.any():
        return x
    if mask.all():
        return np.zeros_like(x)
    idx = np.arange(x.shape[0])
    x = x.copy()
    x[mask] = np.interp(idx[mask], idx[~mask], x[~mask])
    return x


def _parse_zip_to_npz(zip_path: Path, cache_path: Path) -> None:
    fluxes, labels, names = [], [], []
    with zipfile.ZipFile(zip_path) as zf:
        for info in zf.infolist():
            if info.is_dir() or not info.filename.endswith(".txt"):
                continue
            parts = info.filename.split("/")
            if len(parts) != 2 or parts[0] not in CLASS_TO_IDX:
                continue
            cls, fname = parts
            name = fname.replace(".txt", "")
            with zf.open(info) as fh:
                arr = np.loadtxt(io.TextIOWrapper(fh, encoding="utf-8"), delimiter=None)
            # columns: time, flux, flux_err
            flux = arr[:, 1].astype(np.float32)
            flux = _fill_nans(flux)
            if flux.shape[0] >= LC_LEN:
                flux = flux[:LC_LEN]
            else:
                flux = np.concatenate([flux, np.full(LC_LEN - flux.shape[0], flux[-1], dtype=np.float32)])
            fluxes.append(flux)
            labels.append(CLASS_TO_IDX[cls])
            names.append(name)
    fluxes = np.stack(fluxes, axis=0)
    labels = np.asarray(labels, dtype=np.int64)
    names = np.asarray(names, dtype=object)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cache_path, flux=fluxes, label=labels, name=names)


def load_kepler_q9v3(
    zip_path: str | Path,
    cache_dir: str | Path = "data/.cache",
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (flux [N,LC_LEN], label [N], name [N]), building the cache if needed."""
    zip_path = Path(zip_path)
    cache_path = Path(cache_dir) / f"{zip_path.stem}.npz"
    if not cache_path.exists():
        _parse_zip_to_npz(zip_path, cache_path)
    d = np.load(cache_path, allow_pickle=True)
    return d["flux"], d["label"], d["name"]


def stratified_split(
    labels: np.ndarray,
    *,
    train: float = 0.70,
    val: float = 0.15,
    test: float = 0.15,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    assert abs(train + val + test - 1.0) < 1e-6
    idx = np.arange(len(labels))
    # First split off test
    idx_tv, idx_te = train_test_split(
        idx, test_size=test, stratify=labels, random_state=seed,
    )
    # Then split train/val from the remaining
    rel_val = val / (train + val)
    idx_tr, idx_va = train_test_split(
        idx_tv, test_size=rel_val, stratify=labels[idx_tv], random_state=seed,
    )
    return idx_tr, idx_va, idx_te


class KeplerQ9Dataset(Dataset):
    def __init__(self, flux: np.ndarray, label: np.ndarray):
        self.flux = torch.as_tensor(flux)   # (N, L) float32
        self.label = torch.as_tensor(label) # (N,) int64

    def __len__(self):
        return self.flux.shape[0]

    def __getitem__(self, idx):
        return self.flux[idx], self.label[idx]


def make_loaders(
    zip_path: str | Path,
    *,
    batch_size: int = 32,
    num_workers: int = 0,
    seed: int = 42,
    cache_dir: str | Path = "data/.cache",
) -> tuple[DataLoader, DataLoader, DataLoader, dict]:
    flux, label, _name = load_kepler_q9v3(zip_path, cache_dir=cache_dir)
    tr, va, te = stratified_split(label, seed=seed)
    loaders = []
    for split_idx, shuffle in [(tr, True), (va, False), (te, False)]:
        ds = KeplerQ9Dataset(flux[split_idx], label[split_idx])
        loaders.append(DataLoader(ds, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers))
    meta = {
        "n_classes": len(CLASS_NAMES),
        "class_names": CLASS_NAMES,
        "lc_len": LC_LEN,
        "split_sizes": (len(tr), len(va), len(te)),
    }
    return loaders[0], loaders[1], loaders[2], meta
