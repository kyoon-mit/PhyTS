"""Dataloader for the PhyTS-bench TESS dataset.

The dataset lives on the HuggingFace Hub at ``PhyTS-team/PhyTS-bench`` under
``TESS/``:

  * ``tess_classification.parquet`` (25 935 rows) — 8 stellar-variability
    classes per light curve.
  * ``tess_regression.parquet`` (4 183 rows) — rotation frequency ``frot``.

Both parquets share the columns ``GaiaID, TIC, sector, time (list<f64>),
flux (list<f64>)``.  Many TICs (TESS Input Catalog IDs) appear more than
once because the same star is observed in multiple sectors — 85.8% of TICs
in the classification set have ≥2 light curves, and the max is 29.  We
therefore split **by TIC**: every star ends up in exactly one of
train/val/test.  For classification we additionally stratify the group
split by class label so each split sees every class.

Light curves vary in length (min 542, median ~1068, max ~3917).  We pad
or truncate to ``LC_LEN`` (1024) to mirror the Kepler loader and align
with foundation models that prefer power-of-two windows.

Optional: set env ``TESS_DOWNSAMPLE_LONG_LC`` to a truthy value (e.g. ``1``)
to decimate flux sequences longer than ``THRESHOLD_FOR_DOWNSAMPLING`` by
``DOWNSAMPLE_FACTOR`` (10 min → 30 min cadence). Only ``flux`` is cached;
that stride matches the cadence change. Uses a separate ``.npz`` cache file.
"""

from __future__ import annotations

import os
from pathlib import Path

import lightning as L
import numpy as np
import torch
from sklearn.model_selection import GroupShuffleSplit, StratifiedGroupKFold
from torch.utils.data import DataLoader, Dataset


# Canonical, alphabetically sorted class list (deterministic across runs).
CLASS_NAMES = [
    "APERIODIC", "CONTACT_ROT", "DSCT_BCEP", "ECLIPSE",
    "GDOR_SPB", "INSTRUMENT/JUNK", "RRLYR_CEPH", "SOLARLIKE",
]
CLASS_TO_IDX = {c: i for i, c in enumerate(CLASS_NAMES)}
LC_LEN = 1024  # canonical length (pad / truncate)

# Long light curves (TESS 10 min cadence): optional decimation to ~30 min cadence.
# On/off: env ``TESS_DOWNSAMPLE_LONG_LC`` truthy (see ``benchmarks/TESS/create_sweeps_and_submit.sh``).
# Uses a separate ``.npz`` cache filename when enabled.
THRESHOLD_FOR_DOWNSAMPLING = 1500
DOWNSAMPLE_FACTOR = 3

HF_REPO_ID = "PhyTS-team/PhyTS-bench"
HF_FILES = {
    "classification": "TESS/tess_classification.parquet",
    "regression":     "TESS/tess_regression.parquet",
}


def _env_downsample_long_lc() -> bool:
    """True if long light curves should be cadence-decimated (10 min → 30 min)."""
    v = os.environ.get("TESS_DOWNSAMPLE_LONG_LC", "").strip().lower()
    return v in ("1", "true", "yes", "on")


def _maybe_downsample_long(flux_1d: np.ndarray) -> np.ndarray:
    """Every ``DOWNSAMPLE_FACTOR``-th sample when enabled and length is above threshold."""
    if not _env_downsample_long_lc():
        return flux_1d
    if flux_1d.shape[0] <= THRESHOLD_FOR_DOWNSAMPLING:
        return flux_1d
    return flux_1d[::DOWNSAMPLE_FACTOR]


def _tess_npz_basename(task: str) -> str:
    """Base filename (no directory) for the materialised ``.npz`` cache."""
    if _env_downsample_long_lc():
        return (
            f"tess_{task}_gt{THRESHOLD_FOR_DOWNSAMPLING}_ds{DOWNSAMPLE_FACTOR}.npz"
        )
    return f"tess_{task}.npz"


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


def _pad_or_truncate(flux: np.ndarray, target: int = LC_LEN) -> np.ndarray:
    """Right-pad with the last value or truncate to length ``target``."""
    if flux.shape[0] >= target:
        return flux[:target]
    pad_val = flux[-1] if flux.shape[0] > 0 else 0.0
    return np.concatenate([flux, np.full(target - flux.shape[0], pad_val, dtype=flux.dtype)])


def _download_parquet(task: str, hf_cache: Path) -> Path:
    """Resolve / download the requested parquet via huggingface_hub.

    ``hf_cache`` is passed to ``hf_hub_download`` as ``cache_dir`` — files
    end up at ``<hf_cache>/datasets--PhyTS-team--PhyTS-bench/...`` and
    interrupted downloads resume automatically on retry.
    """
    from huggingface_hub import hf_hub_download
    hf_cache.mkdir(parents=True, exist_ok=True)
    path = hf_hub_download(
        repo_id=HF_REPO_ID,
        filename=HF_FILES[task],
        repo_type="dataset",
        cache_dir=str(hf_cache),
    )
    return Path(path)


def _parse_parquet_to_npz(parquet_path: Path, npz_path: Path, *, task: str) -> None:
    """Materialise parquet → padded/truncated flux array + labels/targets + IDs."""
    import pyarrow.parquet as pq

    cols = ["TIC", "flux"]
    cols += ["label"] if task == "classification" else ["frot"]
    table = pq.read_table(parquet_path, columns=cols)

    flux_lists = table["flux"].to_pylist()
    tics = np.asarray(table["TIC"].to_pylist(), dtype=np.int64)
    n_valid = np.empty(len(flux_lists), dtype=np.int32)
    fluxes = np.empty((len(flux_lists), LC_LEN), dtype=np.float32)
    for i, lst in enumerate(flux_lists):
        f = np.asarray(lst, dtype=np.float32)
        f = _fill_nans(f)
        f = _maybe_downsample_long(f)
        n_valid[i] = min(int(f.shape[0]), LC_LEN)
        fluxes[i] = _pad_or_truncate(f)

    npz_path.parent.mkdir(parents=True, exist_ok=True)
    if task == "classification":
        labels_str = table["label"].to_pylist()
        # Skip rows with unknown labels (defensive — current data has none).
        keep = np.array([c in CLASS_TO_IDX for c in labels_str])
        if not keep.all():
            print(f"[tess] dropping {(~keep).sum()} rows with unknown labels")
        labels = np.array([CLASS_TO_IDX[c] for c in labels_str if c in CLASS_TO_IDX], dtype=np.int64)
        np.savez_compressed(
            npz_path,
            flux=fluxes[keep],
            label=labels,
            tic=tics[keep],
            n_valid=n_valid[keep],
        )
    else:
        target = np.asarray(table["frot"].to_pylist(), dtype=np.float32)
        np.savez_compressed(
            npz_path, flux=fluxes, target=target, tic=tics, n_valid=n_valid,
        )


def load_tess(
    task: str = "classification",
    *,
    cache_dir: str | Path = "data/.cache",
) -> dict:
    """Return a dict with arrays for the requested task, downloading if needed.

    Classification keys: ``flux [N,LC_LEN], label [N], tic [N], n_valid [N]``.
    Regression keys: ``flux [N,LC_LEN], target [N], tic [N], n_valid [N]``.
    Rows of ``n_valid`` are in ``[1, LC_LEN]``: real cadence count per row after
    truncate/pad (mask construction in the PyTorch datasets).

    Legacy ``.npz`` files without ``n_valid`` are still loaded; missing entries
    are treated as full-length valid sequences (all ``LC_LEN`` timesteps).

    Cache path depends on ``TESS_DOWNSAMPLE_LONG_LC`` (see module constants).
    """
    if task not in HF_FILES:
        raise ValueError(f"task must be 'classification' or 'regression', got {task!r}")
    cache_dir = Path(cache_dir)
    npz_path = cache_dir / _tess_npz_basename(task)
    if not npz_path.exists():
        # Match the layout the earlier HF_HOME-based download wrote to
        # (data/.cache/hf/hub/...) so partial downloads resume cleanly.
        parquet_path = _download_parquet(task, cache_dir / "hf" / "hub")
        _parse_parquet_to_npz(parquet_path, npz_path, task=task)
    d = np.load(npz_path, allow_pickle=False)
    out = {"flux": d["flux"], "tic": d["tic"]}
    out["label" if task == "classification" else "target"] = (
        d["label"] if task == "classification" else d["target"]
    )
    if "n_valid" in d.files:
        out["n_valid"] = np.asarray(d["n_valid"], dtype=np.int32)
    else:
        # Legacy cache (before per-row valid lengths); treat every timestep as real.
        out["n_valid"] = np.full(len(d["flux"]), LC_LEN, dtype=np.int32)
    return out


def group_stratified_split(
    *,
    groups: np.ndarray,
    labels: np.ndarray | None,
    train: float = 0.70,
    val: float = 0.15,
    test: float = 0.15,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Group-aware split that ensures no group ID crosses split boundaries.

    If ``labels`` is given, splits are also stratified by label
    (``StratifiedGroupKFold``).  Otherwise a plain ``GroupShuffleSplit`` is
    used (suitable for regression).
    """
    assert abs(train + val + test - 1.0) < 1e-6
    idx = np.arange(len(groups))

    def _two_way(idx_pool: np.ndarray, labels_pool: np.ndarray | None,
                 groups_pool: np.ndarray, test_size: float, rs: int) -> tuple[np.ndarray, np.ndarray]:
        if labels_pool is not None:
            # Stratified group K-fold: choose K so one fold ≈ test_size.
            k = max(2, int(round(1.0 / test_size)))
            sgkf = StratifiedGroupKFold(n_splits=k, shuffle=True, random_state=rs)
            tr_idx, te_idx = next(sgkf.split(idx_pool, labels_pool, groups=groups_pool))
            return idx_pool[tr_idx], idx_pool[te_idx]
        gss = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=rs)
        tr_idx, te_idx = next(gss.split(idx_pool, groups=groups_pool))
        return idx_pool[tr_idx], idx_pool[te_idx]

    # 1) hold out test
    idx_tv, idx_te = _two_way(
        idx,
        labels[idx] if labels is not None else None,
        groups[idx], test_size=test, rs=seed,
    )
    # 2) split remaining into train / val
    rel_val = val / (train + val)
    idx_tr, idx_va = _two_way(
        idx_tv,
        labels[idx_tv] if labels is not None else None,
        groups[idx_tv], test_size=rel_val, rs=seed + 1,
    )

    # Defensive: assert no group leakage across splits.
    g_tr, g_va, g_te = set(groups[idx_tr]), set(groups[idx_va]), set(groups[idx_te])
    assert g_tr.isdisjoint(g_va), "TIC leakage between train and val"
    assert g_tr.isdisjoint(g_te), "TIC leakage between train and test"
    assert g_va.isdisjoint(g_te), "TIC leakage between val and test"
    return idx_tr, idx_va, idx_te


def _bool_mask_row(n_real: int, seq_len: int) -> torch.Tensor:
    """True on [0, n_real) capped to seq_len — matches pad/truncate in ``_pad_or_truncate``."""
    n = int(max(0, min(int(n_real), seq_len)))
    m = torch.zeros(seq_len, dtype=torch.bool)
    m[:n] = True
    return m


def _normalize_flux(arr: np.ndarray) -> np.ndarray:
    """Per-light-curve z-score (median, nanstd); ``False`` finite mask → zeros like invalid cadences."""
    arr = np.asarray(arr, dtype=np.float64)
    center = np.nanmedian(arr)
    std = np.nanstd(arr)
    if not np.isfinite(center) or std < 1e-8:
        return np.zeros(arr.shape[0], dtype=np.float32)
    normed = (arr - center) / std
    normed = np.where(np.isfinite(normed), normed, 0.0)
    return normed.astype(np.float32)


class TESSClassificationDataset(Dataset):
    """TESS variability classification; batches match ``TESSClassificationCE``."""

    label_names = CLASS_NAMES  # Iterable[str]; retest uses ``list(dm.val.label_names)``.

    def __init__(
        self,
        flux: np.ndarray,
        label: np.ndarray,
        n_valid: np.ndarray,
        *,
        normalize_flux: bool = False,
    ) -> None:
        self.flux = torch.as_tensor(flux)    # (N, L) float32
        self.label = torch.as_tensor(label)  # (N,) int64
        self.n_valid = np.asarray(n_valid, dtype=np.int32)
        self.normalize_flux = normalize_flux

    def __len__(self) -> int:
        return self.flux.shape[0]

    def __getitem__(self, idx):
        L = int(self.flux.shape[1])
        n = int(self.n_valid[idx])
        mask = _bool_mask_row(n, L)
        if self.normalize_flux:
            seg = self.flux[idx, :n].numpy()
            normed = _normalize_flux(seg) if n > 0 else np.zeros(0, dtype=np.float32)
            x = torch.zeros(L, dtype=torch.float32)
            if n > 0:
                x[:n] = torch.from_numpy(normed)
            return x, mask, self.label[idx]
        return self.flux[idx], mask, self.label[idx]


class TESSRegressionDataset(Dataset):
    """TESS frot regression; batches match ``TESSRegressionMSE``."""

    def __init__(
        self,
        flux: np.ndarray,
        target: np.ndarray,
        n_valid: np.ndarray,
        *,
        normalize_flux: bool = False,
    ) -> None:
        self.flux = torch.as_tensor(flux)                                # (N, L) float32
        self.target = torch.as_tensor(target.astype(np.float32))         # (N,) float32
        self.n_valid = np.asarray(n_valid, dtype=np.int32)
        self.normalize_flux = normalize_flux

    def __len__(self) -> int:
        return self.flux.shape[0]

    def __getitem__(self, idx):
        L = int(self.flux.shape[1])
        n = int(self.n_valid[idx])
        mask = _bool_mask_row(n, L)
        if self.normalize_flux:
            seg = self.flux[idx, :n].numpy()
            normed = _normalize_flux(seg) if n > 0 else np.zeros(0, dtype=np.float32)
            x = torch.zeros(L, dtype=torch.float32)
            if n > 0:
                x[:n] = torch.from_numpy(normed)
            return x, mask, self.target[idx]
        return self.flux[idx], mask, self.target[idx]


def _loader_kwargs(num_workers: int, prefetch_factor: int | None,
                   persistent_workers: bool, pin_memory: bool) -> dict:
    """Build ``DataLoader`` keyword args shared by helpers in this module.

    If ``TESS_DATALOADER_MP_START_METHOD`` is ``spawn`` or ``forkserver``,
    forwards that as ``multiprocessing_context`` so workers do not fork a
    process that already initialized CUDA (otherwise ``trainer.test`` /
    teardown can crash with CUDA init errors on some setups).
    """
    kw: dict = {"num_workers": num_workers, "pin_memory": pin_memory}
    if num_workers > 0:
        if prefetch_factor is not None:
            kw["prefetch_factor"] = prefetch_factor
        kw["persistent_workers"] = persistent_workers
        mp_method = os.environ.get("TESS_DATALOADER_MP_START_METHOD", "").strip().lower()
        if mp_method in ("spawn", "forkserver"):
            import multiprocessing as mp_module

            kw["multiprocessing_context"] = mp_module.get_context(mp_method)
    return kw


def make_loaders_classification(
    *,
    batch_size: int = 32,
    num_workers: int = 0,
    seed: int = 42,
    cache_dir: str | Path = "data/.cache",
    prefetch_factor: int | None = None,
    persistent_workers: bool = False,
    pin_memory: bool = False,
    normalize_flux: bool = False,
) -> tuple[DataLoader, DataLoader, DataLoader, dict]:
    d = load_tess("classification", cache_dir=cache_dir)
    flux, label, tic, n_valid = d["flux"], d["label"], d["tic"], d["n_valid"]
    tr, va, te = group_stratified_split(groups=tic, labels=label, seed=seed)

    loader_kw = _loader_kwargs(num_workers, prefetch_factor, persistent_workers, pin_memory)
    loaders = []
    for split_idx, shuffle in [(tr, True), (va, False), (te, False)]:
        ds = TESSClassificationDataset(
            flux[split_idx],
            label[split_idx],
            n_valid[split_idx],
            normalize_flux=normalize_flux,
        )
        loaders.append(DataLoader(ds, batch_size=batch_size, shuffle=shuffle, **loader_kw))

    meta = {
        "n_classes": len(CLASS_NAMES),
        "class_names": CLASS_NAMES,
        "lc_len": LC_LEN,
        "split_sizes": (len(tr), len(va), len(te)),
        "n_unique_stars": (
            int(len(np.unique(tic[tr]))),
            int(len(np.unique(tic[va]))),
            int(len(np.unique(tic[te]))),
        ),
    }
    return loaders[0], loaders[1], loaders[2], meta


def make_loaders_regression(
    *,
    batch_size: int = 32,
    num_workers: int = 0,
    seed: int = 42,
    cache_dir: str | Path = "data/.cache",
    target_name: str = "frot",
    prefetch_factor: int | None = None,
    persistent_workers: bool = False,
    pin_memory: bool = False,
    drop_nonpositive_target: bool = True,
    normalize_flux: bool = False,
) -> tuple[DataLoader, DataLoader, DataLoader, dict]:
    d = load_tess("regression", cache_dir=cache_dir)
    flux, target, tic, n_valid = d["flux"], d["target"], d["tic"], d["n_valid"]
    if drop_nonpositive_target:
        # frot is a stellar rotation frequency — non-positive values are
        # sentinels for unmeasured / failed fits (the raw data has 0 NaNs
        # but ~26 negatives concentrated below ~-0.05).
        keep = target > 0
        n_drop = int((~keep).sum())
        if n_drop:
            print(f"[tess-reg] dropping {n_drop} rows with non-positive {target_name}")
        flux, target, tic, n_valid = flux[keep], target[keep], tic[keep], n_valid[keep]
    tr, va, te = group_stratified_split(groups=tic, labels=None, seed=seed)

    # Train-only target normalisation stats (for callers that want them).
    train_target = target[tr].astype(np.float64)
    target_mean = float(train_target.mean())
    target_std = float(train_target.std() + 1e-8)

    loader_kw = _loader_kwargs(num_workers, prefetch_factor, persistent_workers, pin_memory)
    loaders = []
    for split_idx, shuffle in [(tr, True), (va, False), (te, False)]:
        ds = TESSRegressionDataset(
            flux[split_idx],
            target[split_idx],
            n_valid[split_idx],
            normalize_flux=normalize_flux,
        )
        loaders.append(DataLoader(ds, batch_size=batch_size, shuffle=shuffle, **loader_kw))

    meta = {
        "target_name": target_name,
        "target_mean": target_mean,
        "target_std": target_std,
        "lc_len": LC_LEN,
        "split_sizes": (len(tr), len(va), len(te)),
        "n_unique_stars": (
            int(len(np.unique(tic[tr]))),
            int(len(np.unique(tic[va]))),
            int(len(np.unique(tic[te]))),
        ),
    }
    return loaders[0], loaders[1], loaders[2], meta


def _check_seq_len(seq_len: int) -> None:
    if int(seq_len) != LC_LEN:
        raise ValueError(
            f"TESS cached flux has length LC_LEN={LC_LEN}; got seq_len={seq_len!r}. "
            "Align model input length with the cache (or change LC_LEN and rebuild the .npz)."
        )


class TESSClassificationDataModule(L.LightningDataModule):
    """Lightning wrapper: HF cache, group stratified split, loaders."""

    def __init__(
        self,
        data_dir: str | Path = "data/TESS/.cache/TESS",
        batch_size: int = 32,
        num_workers: int = 0,
        seq_len: int = LC_LEN,
        seed: int = 42,
        prefetch_factor: int | None = None,
        persistent_workers: bool = False,
        pin_memory: bool = False,
        normalize_flux: bool = False,
    ) -> None:
        super().__init__()
        _check_seq_len(seq_len)
        self.data_dir = Path(data_dir)
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.seed = seed
        self.prefetch_factor = prefetch_factor
        self.persistent_workers = persistent_workers
        self.pin_memory = pin_memory
        self.normalize_flux = normalize_flux
        self.train: TESSClassificationDataset | None = None
        self.val: TESSClassificationDataset | None = None
        self.test: TESSClassificationDataset | None = None

    def setup(self, stage: str | None = None) -> None:
        if self.train is not None:
            return
        d = load_tess("classification", cache_dir=self.data_dir)
        flux, label, tic, n_valid = d["flux"], d["label"], d["tic"], d["n_valid"]
        tr, va, te = group_stratified_split(groups=tic, labels=label, seed=self.seed)
        nf = self.normalize_flux
        self.train = TESSClassificationDataset(
            flux[tr], label[tr], n_valid[tr], normalize_flux=nf,
        )
        self.val = TESSClassificationDataset(
            flux[va], label[va], n_valid[va], normalize_flux=nf,
        )
        self.test = TESSClassificationDataset(
            flux[te], label[te], n_valid[te], normalize_flux=nf,
        )

    def train_dataloader(self) -> DataLoader:
        assert self.train is not None
        kw = _loader_kwargs(
            self.num_workers,
            self.prefetch_factor,
            self.persistent_workers,
            self.pin_memory,
        )
        return DataLoader(
            self.train, batch_size=self.batch_size, shuffle=True, **kw,
        )

    def val_dataloader(self) -> DataLoader:
        assert self.val is not None
        kw = _loader_kwargs(
            self.num_workers,
            self.prefetch_factor,
            self.persistent_workers,
            self.pin_memory,
        )
        return DataLoader(
            self.val, batch_size=self.batch_size, shuffle=False, **kw,
        )

    def test_dataloader(self) -> DataLoader:
        assert self.test is not None
        kw = _loader_kwargs(
            self.num_workers,
            self.prefetch_factor,
            self.persistent_workers,
            self.pin_memory,
        )
        return DataLoader(
            self.test, batch_size=self.batch_size, shuffle=False, **kw,
        )


class TESSRegressionDataModule(L.LightningDataModule):
    """Same TIC-based group split as classification (no label stratification)."""

    def __init__(
        self,
        data_dir: str | Path = "data/TESS/.cache/TESS",
        batch_size: int = 32,
        num_workers: int = 0,
        seq_len: int = LC_LEN,
        seed: int = 42,
        prefetch_factor: int | None = None,
        persistent_workers: bool = False,
        pin_memory: bool = False,
        drop_nonpositive_target: bool = True,
        normalize_flux: bool = False,
    ) -> None:
        super().__init__()
        _check_seq_len(seq_len)
        self.data_dir = Path(data_dir)
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.seed = seed
        self.prefetch_factor = prefetch_factor
        self.persistent_workers = persistent_workers
        self.pin_memory = pin_memory
        self.drop_nonpositive_target = drop_nonpositive_target
        self.normalize_flux = normalize_flux
        self.train: TESSRegressionDataset | None = None
        self.val: TESSRegressionDataset | None = None
        self.test: TESSRegressionDataset | None = None

    def setup(self, stage: str | None = None) -> None:
        if self.train is not None:
            return
        d = load_tess("regression", cache_dir=self.data_dir)
        flux, target, tic, n_valid = d["flux"], d["target"], d["tic"], d["n_valid"]
        if self.drop_nonpositive_target:
            keep = target > 0
            n_drop = int((~keep).sum())
            if n_drop:
                print(f"[tess-reg] dropping {n_drop} rows with non-positive frot")
            flux, target, tic, n_valid = flux[keep], target[keep], tic[keep], n_valid[keep]
        tr, va, te = group_stratified_split(groups=tic, labels=None, seed=self.seed)
        nf = self.normalize_flux
        self.train = TESSRegressionDataset(
            flux[tr], target[tr], n_valid[tr], normalize_flux=nf,
        )
        self.val = TESSRegressionDataset(
            flux[va], target[va], n_valid[va], normalize_flux=nf,
        )
        self.test = TESSRegressionDataset(
            flux[te], target[te], n_valid[te], normalize_flux=nf,
        )

    def train_dataloader(self) -> DataLoader:
        assert self.train is not None
        kw = _loader_kwargs(
            self.num_workers,
            self.prefetch_factor,
            self.persistent_workers,
            self.pin_memory,
        )
        return DataLoader(
            self.train, batch_size=self.batch_size, shuffle=True, **kw,
        )

    def val_dataloader(self) -> DataLoader:
        assert self.val is not None
        kw = _loader_kwargs(
            self.num_workers,
            self.prefetch_factor,
            self.persistent_workers,
            self.pin_memory,
        )
        return DataLoader(
            self.val, batch_size=self.batch_size, shuffle=False, **kw,
        )

    def test_dataloader(self) -> DataLoader:
        assert self.test is not None
        kw = _loader_kwargs(
            self.num_workers,
            self.prefetch_factor,
            self.persistent_workers,
            self.pin_memory,
        )
        return DataLoader(
            self.test, batch_size=self.batch_size, shuffle=False, **kw,
        )