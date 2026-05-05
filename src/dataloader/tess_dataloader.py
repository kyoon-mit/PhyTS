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
"""

from __future__ import annotations

from pathlib import Path

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

HF_REPO_ID = "PhyTS-team/PhyTS-bench"
HF_FILES = {
    "classification": "TESS/tess_classification.parquet",
    "regression":     "TESS/tess_regression.parquet",
}


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

    fluxes = np.empty((len(flux_lists), LC_LEN), dtype=np.float32)
    for i, lst in enumerate(flux_lists):
        f = np.asarray(lst, dtype=np.float32)
        f = _fill_nans(f)
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
            npz_path, flux=fluxes[keep], label=labels, tic=tics[keep],
        )
    else:
        target = np.asarray(table["frot"].to_pylist(), dtype=np.float32)
        np.savez_compressed(npz_path, flux=fluxes, target=target, tic=tics)


def load_tess(
    task: str = "classification",
    *,
    cache_dir: str | Path = "data/.cache",
) -> dict:
    """Return a dict with arrays for the requested task, downloading if needed.

    Classification keys: ``flux [N,LC_LEN], label [N], tic [N]``.
    Regression keys:     ``flux [N,LC_LEN], target [N], tic [N]``.
    """
    if task not in HF_FILES:
        raise ValueError(f"task must be 'classification' or 'regression', got {task!r}")
    cache_dir = Path(cache_dir)
    npz_path = cache_dir / f"tess_{task}.npz"
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


class TESSClassificationDataset(Dataset):
    def __init__(self, flux: np.ndarray, label: np.ndarray) -> None:
        self.flux = torch.as_tensor(flux)    # (N, L) float32
        self.label = torch.as_tensor(label)  # (N,) int64

    def __len__(self) -> int:
        return self.flux.shape[0]

    def __getitem__(self, idx):
        return self.flux[idx], self.label[idx]


class TESSRegressionDataset(Dataset):
    def __init__(self, flux: np.ndarray, target: np.ndarray) -> None:
        self.flux = torch.as_tensor(flux)                                # (N, L) float32
        self.target = torch.as_tensor(target.astype(np.float32))         # (N,) float32

    def __len__(self) -> int:
        return self.flux.shape[0]

    def __getitem__(self, idx):
        return self.flux[idx], self.target[idx]


def _loader_kwargs(num_workers: int, prefetch_factor: int | None,
                   persistent_workers: bool, pin_memory: bool) -> dict:
    kw: dict = {"num_workers": num_workers, "pin_memory": pin_memory}
    if num_workers > 0:
        if prefetch_factor is not None:
            kw["prefetch_factor"] = prefetch_factor
        kw["persistent_workers"] = persistent_workers
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
) -> tuple[DataLoader, DataLoader, DataLoader, dict]:
    d = load_tess("classification", cache_dir=cache_dir)
    flux, label, tic = d["flux"], d["label"], d["tic"]
    tr, va, te = group_stratified_split(groups=tic, labels=label, seed=seed)

    loader_kw = _loader_kwargs(num_workers, prefetch_factor, persistent_workers, pin_memory)
    loaders = []
    for split_idx, shuffle in [(tr, True), (va, False), (te, False)]:
        ds = TESSClassificationDataset(flux[split_idx], label[split_idx])
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
) -> tuple[DataLoader, DataLoader, DataLoader, dict]:
    d = load_tess("regression", cache_dir=cache_dir)
    flux, target, tic = d["flux"], d["target"], d["tic"]
    if drop_nonpositive_target:
        # frot is a stellar rotation frequency — non-positive values are
        # sentinels for unmeasured / failed fits (the raw data has 0 NaNs
        # but ~26 negatives concentrated below ~-0.05).
        keep = target > 0
        n_drop = int((~keep).sum())
        if n_drop:
            print(f"[tess-reg] dropping {n_drop} rows with non-positive {target_name}")
        flux, target, tic = flux[keep], target[keep], tic[keep]
    tr, va, te = group_stratified_split(groups=tic, labels=None, seed=seed)

    # Train-only target normalisation stats (for callers that want them).
    train_target = target[tr].astype(np.float64)
    target_mean = float(train_target.mean())
    target_std = float(train_target.std() + 1e-8)

    loader_kw = _loader_kwargs(num_workers, prefetch_factor, persistent_workers, pin_memory)
    loaders = []
    for split_idx, shuffle in [(tr, True), (va, False), (te, False)]:
        ds = TESSRegressionDataset(flux[split_idx], target[split_idx])
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
