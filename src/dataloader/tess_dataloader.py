"""TESS lightcurve datasets and Lightning DataModules for PhyTS-bench.

Dataset files (from HuggingFace PhyTS-team/PhyTS-bench) must already live directly under
``data_dir`` (e.g. ``data/TESS/.cache/TESS/*.parquet`` after ``data/TESS/download_tess.py`` flattens
``TESS/split``). Missing shards raise ``FileNotFoundError``.

  tess_regression_{train,val,test}.parquet   — target: frot
  tess_classification_{train,val,test}.parquet — 8 string labels → int via label_map.json

Preprocessing (shared):
  - z-score normalization per sample (median centering, std scaling)
  - crop or pad to seq_len=1100 with 0.0; bool mask (True = valid cadence)

Batch formats:
  TESSRegressionDataset             → (flux, mask, frot)   (L,), (L,), float scalar
  TESSClassificationDataset         → (flux, mask, label)  (L,), (L,), int64
  TESSReconstructionDataset         → (noisy_flux, flux, mask)
"""

import json
from pathlib import Path

import numpy as np
import torch
from torch import Tensor
from torch.utils.data import DataLoader, Dataset
import lightning as L


def _dl_extra_kwargs(num_workers: int) -> dict:
    """DataLoader options that only apply when using background workers."""
    if num_workers <= 0:
        return {}
    return {"persistent_workers": True, "prefetch_factor": 2}


# ── Preprocessing helpers ────────────────────────────────────────────────────

def _normalize_flux(arr: np.ndarray) -> np.ndarray:
    arr = arr.astype(np.float64)
    center = np.nanmedian(arr)
    std = np.nanstd(arr)
    if not np.isfinite(center) or std < 1e-8:
        return np.zeros_like(arr, dtype=np.float32)
    normed = (arr - center) / std
    normed = np.where(np.isfinite(normed), normed, 0.0)
    return normed.astype(np.float32)


def _pad_and_mask(arr: np.ndarray, seq_len: int) -> tuple[np.ndarray, np.ndarray]:
    """Crop to seq_len if longer; pad with 0.0 if shorter. Returns (float32 (seq_len,), bool (seq_len,))."""
    n = len(arr)
    out = np.zeros(seq_len, dtype=np.float32)
    mask = np.zeros(seq_len, dtype=bool)
    valid = min(n, seq_len)
    out[:valid] = arr[:valid]
    mask[:valid] = True
    return out, mask


def _table_to_flux_masks(table, seq_len: int) -> tuple[np.ndarray, np.ndarray]:
    """From a PyArrow table with column ``flux`` (list<float> per row), build padded tensors.

    Returns
    -------
    fluxes : ndarray
        Shape ``(N, seq_len)``, dtype float32.
    masks : ndarray
        Shape ``(N, seq_len)``, dtype bool.
    """
    flux_col = table["flux"]
    n = len(flux_col)
    fluxes = np.zeros((n, seq_len), dtype=np.float32)
    masks = np.zeros((n, seq_len), dtype=bool)
    for i in range(n):
        raw = np.asarray(flux_col[i].as_py(), dtype=np.float64)
        normed = _normalize_flux(raw)
        fluxes[i], masks[i] = _pad_and_mask(normed, seq_len)
    return fluxes, masks


def _read_presplit_parquet(
    data_dir: Path,
    split: str,
    columns: list[str],
    *,
    shard_prefix: str,
):
    """Read PhyTS Hub-style shard ``{shard_prefix}_{split}.parquet`` (or ``...-*.parquet`` parts)."""
    import pyarrow as pa
    import pyarrow.parquet as pq

    prefixed = data_dir / f"{shard_prefix}_{split}.parquet"
    if prefixed.is_file():
        return pq.read_table(prefixed, columns=columns)
    shard_pat = sorted(data_dir.glob(f"{shard_prefix}_{split}-*.parquet"))
    if shard_pat:
        return pa.concat_tables([pq.read_table(f, columns=columns) for f in shard_pat])

    msg = (
        f"No parquet for split={split!r} in {data_dir}: expected "
        f"{shard_prefix}_{split}.parquet or {shard_prefix}_{split}-*.parquet"
    )
    if (data_dir / "split").is_dir():
        msg += (
            f". Shards must be directly under {data_dir} (not only in {data_dir / 'split'}); "
            "run data/TESS/download_tess.py to flatten the Hub layout."
        )
    raise FileNotFoundError(msg)


# ── Regression ───────────────────────────────────────────────────────────────

class TESSRegressionDataset(Dataset):
    """TESS regression: flux → frot (stellar rotation frequency).

    Reads HuggingFace-style shards ``tess_regression_{split}.parquet`` under ``data_dir``.
    """

    def __init__(self, data_dir: str, split: str, seq_len: int = 1100):
        data_dir = Path(data_dir)
        table = _read_presplit_parquet(
            data_dir, split, columns=["flux", "frot"], shard_prefix="tess_regression"
        )
        self._frot = np.array(table["frot"].to_pylist(), dtype=np.float32)
        self._fluxes, self._masks = _table_to_flux_masks(table, seq_len)

    def __len__(self) -> int:
        return len(self._frot)

    def __getitem__(self, i: int) -> tuple[Tensor, Tensor, Tensor]:
        return (
            torch.as_tensor(self._fluxes[i]),
            torch.as_tensor(self._masks[i]),
            torch.tensor(self._frot[i]),
        )


class TESSRegressionDataModule(L.LightningDataModule):
    def __init__(
        self,
        data_dir: str,
        batch_size: int = 32,
        num_workers: int = 0,
        seq_len: int = 1100,
    ):
        super().__init__()
        self.save_hyperparameters()

    def setup(self, stage: str | None = None):
        hp = self.hparams
        if stage in (None, "fit", "validate"):
            self.train = TESSRegressionDataset(hp.data_dir, "train", hp.seq_len)
            self.val = TESSRegressionDataset(hp.data_dir, "val", hp.seq_len)
        if stage in (None, "test", "predict"):
            self.test = TESSRegressionDataset(hp.data_dir, "test", hp.seq_len)

    def train_dataloader(self):
        nw = self.hparams.num_workers
        return DataLoader(
            self.train,
            batch_size=self.hparams.batch_size,
            shuffle=True,
            num_workers=nw,
            pin_memory=True,
            **_dl_extra_kwargs(nw),
        )

    def val_dataloader(self):
        nw = self.hparams.num_workers
        return DataLoader(
            self.val,
            batch_size=self.hparams.batch_size,
            shuffle=False,
            num_workers=nw,
            pin_memory=True,
            **_dl_extra_kwargs(nw),
        )

    def test_dataloader(self):
        nw = self.hparams.num_workers
        return DataLoader(
            self.test,
            batch_size=self.hparams.batch_size,
            shuffle=False,
            num_workers=nw,
            pin_memory=True,
            **_dl_extra_kwargs(nw),
        )

    def predict_dataloader(self):
        return self.test_dataloader()


# ── Reconstruction (backbone pretraining) ────────────────────────────────────

class TESSReconstructionDataset(Dataset):
    """TESS reconstruction pretraining: (noisy_flux, clean_flux, mask).

    Noise is sampled i.i.d. per element during __getitem__; noise_std is
    relative to 1.0 (flux is already z-score normalized, so std≈1).

    Uses the same HuggingFace split shards as the supervised tasks:
    ``tess_regression_{split}`` or ``tess_classification_{split}`` (flux only).
    """

    def __init__(
        self,
        data_dir: str,
        split: str,
        task: str = "regression",
        seq_len: int = 1100,
        noise_std: float = 0.3,
    ):
        data_dir = Path(data_dir)
        if task == "regression":
            prefix = "tess_regression"
        elif task == "classification":
            prefix = "tess_classification"
        else:
            raise ValueError(f"task must be 'regression' or 'classification'; got {task!r}")

        table = _read_presplit_parquet(data_dir, split, columns=["flux"], shard_prefix=prefix)
        self._fluxes, self._masks = _table_to_flux_masks(table, seq_len)
        self.noise_std = noise_std

    def __len__(self) -> int:
        return len(self._fluxes)

    def __getitem__(self, i: int) -> tuple[Tensor, Tensor, Tensor]:
        clean = torch.as_tensor(self._fluxes[i])
        mask = torch.as_tensor(self._masks[i])
        noise = torch.randn_like(clean) * self.noise_std * mask
        return clean + noise, clean, mask


class TESSReconstructionDataModule(L.LightningDataModule):
    def __init__(
        self,
        data_dir: str,
        batch_size: int = 32,
        num_workers: int = 0,
        seq_len: int = 1100,
        noise_std: float = 0.3,
        task: str = "regression",
    ):
        super().__init__()
        self.save_hyperparameters()

    def setup(self, stage: str | None = None):
        hp = self.hparams
        kw = dict(
            data_dir=hp.data_dir,
            task=hp.task,
            seq_len=hp.seq_len,
            noise_std=hp.noise_std,
        )
        if stage in (None, "fit", "validate"):
            self.train = TESSReconstructionDataset(split="train", **kw)
            self.val = TESSReconstructionDataset(split="val", **kw)
        if stage in (None, "test", "predict"):
            self.test = TESSReconstructionDataset(split="test", **kw)

    def train_dataloader(self):
        nw = self.hparams.num_workers
        return DataLoader(
            self.train,
            batch_size=self.hparams.batch_size,
            shuffle=True,
            num_workers=nw,
            pin_memory=True,
            **_dl_extra_kwargs(nw),
        )

    def val_dataloader(self):
        nw = self.hparams.num_workers
        return DataLoader(
            self.val,
            batch_size=self.hparams.batch_size,
            shuffle=False,
            num_workers=nw,
            pin_memory=True,
            **_dl_extra_kwargs(nw),
        )

    def test_dataloader(self):
        nw = self.hparams.num_workers
        return DataLoader(
            self.test,
            batch_size=self.hparams.batch_size,
            shuffle=False,
            num_workers=nw,
            pin_memory=True,
            **_dl_extra_kwargs(nw),
        )

    def predict_dataloader(self):
        return self.test_dataloader()


# ── Classification ───────────────────────────────────────────────────────────

class TESSClassificationDataset(Dataset):
    """TESS classification from HuggingFace shards ``tess_classification_{split}.parquet``.

    Columns: flux (list<float>), label (string). z-score normalization → crop/pad to seq_len → bool mask.

    The label map is derived alphabetically from the current split's unique
    labels; when split=="train" the map is written to {data_dir}/label_map.json
    so subsequent val/test loads can verify consistency.
    """

    def __init__(self, data_dir: str, split: str, seq_len: int = 1100):
        data_dir = Path(data_dir)
        table = _read_presplit_parquet(
            data_dir, split, columns=["flux", "label"], shard_prefix="tess_classification"
        )

        label_strings = table["label"].to_pylist()
        unique_labels = sorted(set(label_strings))
        label_map = {lbl: i for i, lbl in enumerate(unique_labels)}

        label_map_path = data_dir / "label_map.json"
        if split == "train":
            with open(label_map_path, "w") as f:
                json.dump(label_map, f, indent=2)
        elif label_map_path.is_file():
            with open(label_map_path) as f:
                saved = json.load(f)
            if saved != label_map:
                raise ValueError(
                    f"Label map mismatch between saved ({saved}) and {split} ({label_map}). "
                    "Load the train split first so label_map.json is written before val/test."
                )
        else:
            raise FileNotFoundError(
                f"{label_map_path} not found; load split='train' first (or run Trainer.fit) "
                "so class indices stay consistent with the training data."
            )

        self.label_map = label_map
        labels = np.array([label_map[l] for l in label_strings], dtype=np.int64)

        self._fluxes, self._masks = _table_to_flux_masks(table, seq_len)
        self._labels = labels

    @property
    def num_classes(self) -> int:
        return len(self.label_map)

    @property
    def label_names(self) -> list[str]:
        return [k for k, _ in sorted(self.label_map.items(), key=lambda x: x[1])]

    def __len__(self) -> int:
        return len(self._labels)

    def __getitem__(self, i: int) -> tuple[Tensor, Tensor, Tensor]:
        return (
            torch.as_tensor(self._fluxes[i]),
            torch.as_tensor(self._masks[i]),
            torch.tensor(self._labels[i]),
        )


class TESSClassificationDataModule(L.LightningDataModule):
    """LightningDataModule for pre-split TESS classification data.

    Parameters
    ----------
    data_dir:
        Same TESS root as regression (e.g. ``data/TESS/.cache/TESS``): classification
        shards ``tess_classification_{train,val,test}.parquet`` plus optional
        ``label_map.json``. Set ``TESS_DATA_DIR`` or pass explicitly.
    """

    def __init__(
        self,
        data_dir: str,
        batch_size: int = 64,
        num_workers: int = 0,
        seq_len: int = 1100,
    ):
        super().__init__()
        self.save_hyperparameters()

    def setup(self, stage: str | None = None):
        hp = self.hparams
        if stage in (None, "fit", "validate"):
            # Train first so label_map.json is written before val reads it.
            self.train = TESSClassificationDataset(hp.data_dir, "train", hp.seq_len)
            self.val = TESSClassificationDataset(hp.data_dir, "val", hp.seq_len)
        if stage in (None, "test", "predict"):
            self.test = TESSClassificationDataset(hp.data_dir, "test", hp.seq_len)

    def train_dataloader(self):
        nw = self.hparams.num_workers
        return DataLoader(
            self.train,
            batch_size=self.hparams.batch_size,
            shuffle=True,
            num_workers=nw,
            pin_memory=True,
            **_dl_extra_kwargs(nw),
        )

    def val_dataloader(self):
        nw = self.hparams.num_workers
        return DataLoader(
            self.val,
            batch_size=self.hparams.batch_size,
            shuffle=False,
            num_workers=nw,
            pin_memory=True,
            **_dl_extra_kwargs(nw),
        )

    def test_dataloader(self):
        nw = self.hparams.num_workers
        return DataLoader(
            self.test,
            batch_size=self.hparams.batch_size,
            shuffle=False,
            num_workers=nw,
            pin_memory=True,
            **_dl_extra_kwargs(nw),
        )

    def predict_dataloader(self):
        return self.test_dataloader()
