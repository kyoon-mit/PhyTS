"""Download the full LIGO BNS dataset from HuggingFace (PhyTS-team/PhyTS-bench).

Normal usage — one condor job per subdirectory, then a merge job:
    condor_submit download.sub          # spawns 19 download jobs
    condor_submit merge.sub             # run after all downloads finish

Or download + merge in a single process (slower):
    uv run python data/LIGO/download.py --dest /fast/barmstrong/LIGO

Dataset layout on HuggingFace:
  train  LIGO/train/bns_snr_{5_10,10_15,...,45_50}/   9 bins × ~500 shards @ 39 MB  (~180 GB)
  val    LIGO/val/bns_snr_{5_10,10_15,...,45_50}/     9 bins × ~500 shards @ 39 MB  (~180 GB)
  test   LIGO/test/snr_5_50/                           single file                   (~73 GB)

Total: ~433 GB on disk before merging.
"""

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

REPO_ID = "PhyTS-team/PhyTS-bench"
REPO_ROOT = Path(__file__).resolve().parents[2]  # TimeSeriesPhysics/

SNR_BINS = [
    "bns_snr_5_10",
    "bns_snr_10_15",
    "bns_snr_15_20",
    "bns_snr_20_25",
    "bns_snr_25_30",
    "bns_snr_30_35",
    "bns_snr_35_40",
    "bns_snr_40_45",
    "bns_snr_45_50",
]

SPLITS: dict[str, list[str]] = {
    "train": [f"LIGO/train/{b}" for b in SNR_BINS],
    "val":   [f"LIGO/val/{b}"   for b in SNR_BINS],
    "test":  ["LIGO/test/snr_5_50"],
}

ALL_SUBDIRS = [subdir for subdirs in SPLITS.values() for subdir in subdirs]

OUT_DIR = REPO_ROOT / "data" / "LIGO" / "sample_dataset"


def download_one(dest: Path, subdir: str) -> None:
    """Download a single subdirectory — called by one condor job."""
    from huggingface_hub import snapshot_download

    hf_dir = dest / "hf"
    print(f"Downloading {REPO_ID}/{subdir} → {hf_dir}")
    snapshot_download(
        REPO_ID,
        repo_type="dataset",
        local_dir=str(hf_dir),
        allow_patterns=[f"{subdir}/*"],
    )
    print(f"Done: {subdir}")


def merge_shards(shard_paths: list[Path], out_path: Path) -> None:
    """Merge H5 shards into one file by appending along axis 0.

    Reads one shard at a time to keep peak memory proportional to a single shard.
    """
    import h5py

    sorted_paths = sorted(shard_paths)
    if not sorted_paths:
        raise FileNotFoundError(f"No shard files found for {out_path.name}")

    out_path.parent.mkdir(parents=True, exist_ok=True)

    with h5py.File(sorted_paths[0], "r") as f0:
        keys = list(f0.keys())

    print(f"  Merging {len(sorted_paths)} shard(s) → {out_path}")

    n_total = 0
    with h5py.File(out_path, "w") as out:
        datasets = {}

        with h5py.File(sorted_paths[0], "r") as f:
            for key in keys:
                data = f[key][:]
                maxshape = (None,) + data.shape[1:]
                datasets[key] = out.create_dataset(
                    key, data=data, maxshape=maxshape, chunks=True
                )
            n_total = datasets[keys[0]].shape[0]

        for path in sorted_paths[1:]:
            with h5py.File(path, "r") as f:
                for key in keys:
                    data = f[key][:]
                    n_new = data.shape[0]
                    n_cur = datasets[key].shape[0]
                    datasets[key].resize(n_cur + n_new, axis=0)
                    datasets[key][n_cur:] = data
                n_total += data.shape[0]

    print(f"    → {n_total} samples, keys: {keys}")


def _merge_split(split: str, subdirs: list[str], hf_dir: Path, merged_dir: Path) -> str:
    out_path = merged_dir / f"{split}.h5"
    if out_path.exists():
        print(f"  {split}.h5 already exists, skipping merge")
        return split

    shards: list[Path] = []
    for subdir in subdirs:
        shard_dir = hf_dir / subdir
        if shard_dir.exists():
            shards.extend(shard_dir.glob("*.h5"))
        else:
            print(f"  Warning: expected shard directory not found: {shard_dir}")

    merge_shards(sorted(shards), out_path)
    return split


def merge_only(dest: Path, workers: int = 3) -> None:
    """Merge already-downloaded shards into per-split H5 files."""
    hf_dir = dest / "hf"
    merged_dir = dest / "merged"
    merged_dir.mkdir(parents=True, exist_ok=True)

    print(f"Merging splits ({min(len(SPLITS), workers)} parallel workers)")
    with ThreadPoolExecutor(max_workers=min(len(SPLITS), workers)) as executor:
        futures = {
            executor.submit(_merge_split, split, subdirs, hf_dir, merged_dir): split
            for split, subdirs in SPLITS.items()
        }
        for future in as_completed(futures):
            split = futures[future]
            try:
                future.result()
            except Exception as exc:
                print(f"  ✗ merge {split}: {exc}")
                raise

    symlink(OUT_DIR, merged_dir)
    print("\nDone. Run training with:")
    print("  uv run python main.py fit --config configs/LIGO/train_ligo_linoss_regression.yaml")


def symlink(link: Path, target: Path) -> None:
    if link.is_symlink():
        print(f"  Skipping symlink — {link} already exists")
        return
    if link.is_dir():
        for src in target.glob("*.h5"):
            dst = link / src.name
            if not dst.exists():
                dst.symlink_to(src)
                print(f"  Symlinked {link.name}/{src.name} → {src}")
        return
    link.symlink_to(target)
    print(f"  Symlinked {link} → {target}")


def download_all(dest: Path) -> None:
    """Download all subdirs sequentially then merge (single-process fallback)."""
    hf_dir = dest / "hf"
    from huggingface_hub import snapshot_download

    allow_patterns = [f"{s}/*" for s in ALL_SUBDIRS]
    print(f"Downloading {REPO_ID} → {hf_dir}")
    snapshot_download(
        REPO_ID,
        repo_type="dataset",
        local_dir=str(hf_dir),
        allow_patterns=allow_patterns,
    )
    merge_only(dest)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--dest",
        default="/fast/barmstrong/LIGO",
        help="Root download directory (default: /fast/barmstrong/LIGO)",
    )

    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--subdir",
        choices=ALL_SUBDIRS,
        metavar="SUBDIR",
        help=f"Download exactly one subdirectory (used by condor jobs). Choices: {ALL_SUBDIRS}",
    )
    mode.add_argument(
        "--merge-only",
        action="store_true",
        help="Skip download; merge already-downloaded shards into per-split H5 files.",
    )

    args = parser.parse_args()
    dest = Path(args.dest)

    if args.subdir:
        download_one(dest, args.subdir)
    elif args.merge_only:
        merge_only(dest)
    else:
        download_all(dest)
