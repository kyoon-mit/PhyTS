"""TESS lightcurve datasets and Lightning DataModules for PhyTS-bench.

Dataset files (from HuggingFace PhyTS-team/PhyTS-bench):
  data/TESS/.cache/TESS/tess_regression.parquet      — 4,183 rows, target: frot
  data/TESS/.cache/TESS/tess_classification.parquet  — 25,935 rows, target: label

All three Dataset classes share the same preprocessing:
  - z-score normalization per sample (median centering, std scaling)
  - crop or pad to seq_len=1100 with 0.0; bool mask (True = valid cadence)
  - 70 / 15 / 15 train / val / test split, stratified by target (fixed seed=42):
      regression     — frot binned into 10 quantile deciles
      classification — label string (8 classes)

Batch formats:
  TESSRegressionDataset       → (flux, mask, frot)    floats (L,), bool (L,), float scalar
  TESSClassificationDataset   → (flux, mask, label)   floats (L,), bool (L,), int64 scalar
  TESSReconstructionDataset   → (noisy_flux, flux, mask)  floats (L,), floats (L,), bool (L,)
"""

import json
from pathlib import Path

import numpy as np
import torch
from torch import Tensor
from torch.utils.data import DataLoader, Dataset
import lightning as L


# ── Preprocessing helpers ────────────────────────────────────────────────────

def _stratified_splits(
    strata: np.ndarray,
    seed: int = 42,
    ratios: tuple = (0.70, 0.15, 0.15),
) -> dict[str, np.ndarray]:
    """Return train/val/test index arrays, balanced across each unique stratum value.

    Within each stratum the indices are shuffled; then the first ratios[0] fraction
    goes to train, the next ratios[1] to val, and the remainder to test.
    """
    rng = np.random.default_rng(seed)
    train, val, test = [], [], []
    for u in np.unique(strata):
        idx = np.where(strata == u)[0]
        idx = rng.permutation(idx)
        n = len(idx)
        nt = int(n * ratios[0])
        nv = int(n * ratios[1])
        train.append(idx[:nt])
        val.append(idx[nt : nt + nv])
        test.append(idx[nt + nv :])
    return {
        "train": np.concatenate(train),
        "val": np.concatenate(val),
        "test": np.concatenate(test),
    }


def _regression_splits(
    frot: np.ndarray,
    seed: int = 42,
    ratios: tuple = (0.70, 0.15, 0.15),
    n_bins: int = 10,
) -> dict[str, np.ndarray]:
    """Stratified split for continuous frot: bin into n_bins quantile deciles, then stratify."""
    edges = np.quantile(frot, np.linspace(0, 1, n_bins + 1))
    edges[-1] += 1e-10           # include the maximum value in the last bin
    strata = np.digitize(frot, edges[1:]).astype(np.int32)
    return _stratified_splits(strata, seed=seed, ratios=ratios)


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


def _load_flux_arrays(
    parquet_path: str,
    split_indices: np.ndarray,
    seq_len: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Load, normalize, and pad flux arrays for the given split indices.

    Returns:
        fluxes: float32 (N, seq_len)
        masks:  bool    (N, seq_len)
    """
    import pyarrow.parquet as pq

    table = pq.read_table(parquet_path, columns=["flux"])
    flux_col = table["flux"]

    n = len(split_indices)
    fluxes = np.zeros((n, seq_len), dtype=np.float32)
    masks = np.zeros((n, seq_len), dtype=bool)

    for i, row_idx in enumerate(split_indices):
        raw = np.asarray(flux_col[int(row_idx)].as_py(), dtype=np.float64)
        normed = _normalize_flux(raw)
        fluxes[i], masks[i] = _pad_and_mask(normed, seq_len)

    return fluxes, masks


# ── Regression ───────────────────────────────────────────────────────────────

class TESSRegressionDataset(Dataset):
    """TESS regression: flux → frot (stellar rotation frequency)."""

    def __init__(self, data_dir: str, split: str, seq_len: int = 1100, seed: int = 42):
        import pyarrow.parquet as pq

        parquet_path = str(Path(data_dir) / "tess_regression.parquet")
        table = pq.read_table(parquet_path, columns=["frot", "flux"])

        frot_all = np.array(table["frot"].to_pylist(), dtype=np.float32)
        splits = _regression_splits(frot_all, seed=seed)
        idx = splits[split]

        self._fluxes, self._masks = _load_flux_arrays(parquet_path, idx, seq_len)
        self._frot = frot_all[idx]

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
        seed: int = 42,
    ):
        super().__init__()
        self.save_hyperparameters()

    def setup(self, stage: str | None = None):
        hp = self.hparams
        if stage == "fit":
            self.train = TESSRegressionDataset(hp.data_dir, "train", hp.seq_len, hp.seed)
            self.val = TESSRegressionDataset(hp.data_dir, "val", hp.seq_len, hp.seed)
        elif stage in ("test", "predict"):
            self.test = TESSRegressionDataset(hp.data_dir, "test", hp.seq_len, hp.seed)

    def train_dataloader(self):
        return DataLoader(self.train, batch_size=self.hparams.batch_size,
                          shuffle=True, num_workers=self.hparams.num_workers, pin_memory=True)

    def val_dataloader(self):
        return DataLoader(self.val, batch_size=self.hparams.batch_size,
                          shuffle=False, num_workers=self.hparams.num_workers, pin_memory=True)

    def test_dataloader(self):
        return DataLoader(self.test, batch_size=self.hparams.batch_size,
                          shuffle=False, num_workers=self.hparams.num_workers, pin_memory=True)

    def predict_dataloader(self):
        return self.test_dataloader()


# ── Classification ───────────────────────────────────────────────────────────

class TESSClassificationDataset(Dataset):
    """TESS classification: flux → variability class label."""

    def __init__(self, data_dir: str, split: str, seq_len: int = 1100, seed: int = 42):
        import pyarrow.parquet as pq

        parquet_path = str(Path(data_dir) / "tess_classification.parquet")
        label_map_path = Path(data_dir) / "label_map.json"

        table = pq.read_table(parquet_path, columns=["label", "flux"])

        # Build or load label map (alphabetically sorted for determinism)
        if label_map_path.exists():
            with open(label_map_path) as f:
                self.label_map = json.load(f)
        else:
            unique_labels = sorted(set(table["label"].to_pylist()))
            self.label_map = {lbl: i for i, lbl in enumerate(unique_labels)}
            with open(label_map_path, "w") as f:
                json.dump(self.label_map, f, indent=2)

        label_col = table["label"].to_pylist()
        labels_all = np.array([self.label_map[l] for l in label_col], dtype=np.int64)
        splits = _stratified_splits(labels_all, seed=seed)
        idx = splits[split]

        self._fluxes, self._masks = _load_flux_arrays(parquet_path, idx, seq_len)
        self._labels = labels_all[idx]

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
    def __init__(
        self,
        data_dir: str,
        batch_size: int = 32,
        num_workers: int = 0,
        seq_len: int = 1100,
        seed: int = 42,
    ):
        super().__init__()
        self.save_hyperparameters()

    def setup(self, stage: str | None = None):
        hp = self.hparams
        if stage == "fit":
            self.train = TESSClassificationDataset(hp.data_dir, "train", hp.seq_len, hp.seed)
            self.val = TESSClassificationDataset(hp.data_dir, "val", hp.seq_len, hp.seed)
        elif stage in ("test", "predict"):
            self.test = TESSClassificationDataset(hp.data_dir, "test", hp.seq_len, hp.seed)

    def train_dataloader(self):
        return DataLoader(self.train, batch_size=self.hparams.batch_size,
                          shuffle=True, num_workers=self.hparams.num_workers, pin_memory=True)

    def val_dataloader(self):
        return DataLoader(self.val, batch_size=self.hparams.batch_size,
                          shuffle=False, num_workers=self.hparams.num_workers, pin_memory=True)

    def test_dataloader(self):
        return DataLoader(self.test, batch_size=self.hparams.batch_size,
                          shuffle=False, num_workers=self.hparams.num_workers, pin_memory=True)

    def predict_dataloader(self):
        return self.test_dataloader()


# ── Reconstruction (backbone pretraining) ────────────────────────────────────

class TESSReconstructionDataset(Dataset):
    """TESS reconstruction pretraining: (noisy_flux, clean_flux, mask).

    Noise is sampled i.i.d. per element during __getitem__; noise_std is
    relative to 1.0 (flux is already z-score normalized, so std≈1).
    Works with either parquet file since only the flux column is used.
    """

    def __init__(
        self,
        data_dir: str,
        split: str,
        task: str = "regression",
        seq_len: int = 1100,
        noise_std: float = 0.3,
        seed: int = 42,
    ):
        import pyarrow.parquet as pq

        parquet_file = f"tess_{task}.parquet"
        parquet_path = str(Path(data_dir) / parquet_file)

        # Use the same stratified split as the corresponding downstream dataset
        # so reconstruction and downstream tasks see identical train/val/test rows.
        if task == "regression":
            target_col = "frot"
            table = pq.read_table(parquet_path, columns=["flux", target_col])
            frot_all = np.array(table[target_col].to_pylist(), dtype=np.float32)
            splits = _regression_splits(frot_all, seed=seed)
        else:  # classification
            target_col = "label"
            table = pq.read_table(parquet_path, columns=["flux", target_col])
            label_map_path = Path(data_dir) / "label_map.json"
            if label_map_path.exists():
                with open(label_map_path) as f:
                    label_map = json.load(f)
            else:
                unique_labels = sorted(set(table[target_col].to_pylist()))
                label_map = {lbl: i for i, lbl in enumerate(unique_labels)}
                with open(label_map_path, "w") as f:
                    json.dump(label_map, f, indent=2)
            labels_all = np.array(
                [label_map[l] for l in table[target_col].to_pylist()], dtype=np.int64
            )
            splits = _stratified_splits(labels_all, seed=seed)

        idx = splits[split]
        self._fluxes, self._masks = _load_flux_arrays(parquet_path, idx, seq_len)
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
        seed: int = 42,
    ):
        super().__init__()
        self.save_hyperparameters()

    def setup(self, stage: str | None = None):
        hp = self.hparams
        kw = dict(data_dir=hp.data_dir, task=hp.task, seq_len=hp.seq_len,
                  noise_std=hp.noise_std, seed=hp.seed)
        if stage == "fit":
            self.train = TESSReconstructionDataset(split="train", **kw)
            self.val = TESSReconstructionDataset(split="val", **kw)
        elif stage in ("test", "predict"):
            self.test = TESSReconstructionDataset(split="test", **kw)

    def train_dataloader(self):
        return DataLoader(self.train, batch_size=self.hparams.batch_size,
                          shuffle=True, num_workers=self.hparams.num_workers, pin_memory=True)

    def val_dataloader(self):
        return DataLoader(self.val, batch_size=self.hparams.batch_size,
                          shuffle=False, num_workers=self.hparams.num_workers, pin_memory=True)

    def test_dataloader(self):
        return DataLoader(self.test, batch_size=self.hparams.batch_size,
                          shuffle=False, num_workers=self.hparams.num_workers, pin_memory=True)

    def predict_dataloader(self):
        return self.test_dataloader()
