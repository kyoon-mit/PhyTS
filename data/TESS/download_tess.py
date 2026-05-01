"""
Download the pre-split PhyTS TESS parquet shards from Hugging Face
(`PhyTS-team/PhyTS-bench` → ``TESS/split/*.parquet``), then copy shards into
``TESS/*.parquet`` so ``data_dir`` is a single ``.cache/TESS`` root (same as
regression parquet and HuggingFace split file names).

Uses ``snapshot_download`` (Hub layout under ``TESS/split/``); promotion step
matches ``src/dataloader/tess_dataloader.TESSClassificationDataset``.
Needs ``uv sync --extra jax`` (pyarrow + huggingface_hub).

Usage
-----
    uv run --extra jax python data/TESS/download_tess.py
    uv run --extra jax python data/TESS/download_tess.py --cache-dir path/to/cache
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

try:
    from huggingface_hub import snapshot_download
    from huggingface_hub.errors import HfHubHTTPError
    import pyarrow.parquet as pq
except ImportError:  # pragma: no cover
    print("Install: uv sync --extra jax", file=sys.stderr)
    raise SystemExit(1) from None

REPO_ID = "PhyTS-team/PhyTS-bench"
ALLOW_PATTERN = "TESS/split/*"
_DEFAULT_CACHE_PARENT = Path(__file__).resolve().parent / ".cache"


def promote_split_parquets_to_tess_root(tess_root: Path) -> int:
    """Copy ``*.parquet`` from ``tess_root/split`` into ``tess_root`` (skip if dest exists).

    Returns
    -------
    int
        Number of files copied.
    """
    tess_root = Path(tess_root)
    split_sub = tess_root / "split"
    if not split_sub.is_dir():
        return 0
    n = 0
    for src in split_sub.glob("*.parquet"):
        dest = tess_root / src.name
        if dest.is_file():
            continue
        shutil.copy2(src, dest)
        n += 1
    return n


def _cell(value: object) -> str:
    """One cell: repr for scalars; for sequences, len + first few values."""
    if isinstance(value, (bool, int, float)) or value is None:
        return repr(value)
    if isinstance(value, str):
        return value if len(value) <= 100 else value[:97] + "..."
    if isinstance(value, bytes):
        s = repr(value)
        return s if len(s) <= 100 else s[:97] + "..."
    if hasattr(value, "tolist") and not isinstance(value, (str, bytes)):
        value = value.tolist()  # numpy
    if isinstance(value, (list, memoryview)) or (
        hasattr(value, "__len__") and not isinstance(value, (str, bytes))
    ):
        n = len(value)  # type: ignore[arg-type]
        if n == 0:
            return "len=0"
        head = [value[i] for i in range(min(4, n))]  # type: ignore[index]
        return f"len={n} first4={head}"
    return repr(value)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download PhyTS-bench TESS split shards and mirror them under TESS/."
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=_DEFAULT_CACHE_PARENT,
        help="Root directory passed to Hugging Face snapshot_download. "
             f"Creates TESS/split/ on the Hub layout and copies shards into TESS/. "
             f"Default: {_DEFAULT_CACHE_PARENT}",
    )
    args = parser.parse_args()
    cache_parent = Path(args.cache_dir).resolve()
    cache_parent.mkdir(parents=True, exist_ok=True)

    split_dir = cache_parent / "TESS" / "split"

    print(f"Fetching {ALLOW_PATTERN} from {REPO_ID} into {cache_parent} …")

    try:
        snapshot_download(
            repo_id=REPO_ID,
            repo_type="dataset",
            allow_patterns=[ALLOW_PATTERN],
            local_dir=str(cache_parent),
        )
    except (OSError, HfHubHTTPError) as e:
        print(f"Download failed: {e}", file=sys.stderr)
        raise SystemExit(1) from e

    if not split_dir.is_dir():
        print(f"Expected directory missing after snapshot: {split_dir}", file=sys.stderr)
        raise SystemExit(1)

    parquets = sorted(split_dir.glob("*.parquet"))
    if not parquets:
        print(f"No parquet files under {split_dir}", file=sys.stderr)
        raise SystemExit(1)

    tess_root = cache_parent / "TESS"
    n_promoted = promote_split_parquets_to_tess_root(tess_root)
    if n_promoted:
        print(f"\nPromoted {n_promoted} parquet shard(s) from {split_dir} → {tess_root}")

    for path in parquets:
        print()
        print("=" * 60)
        pf = pq.ParquetFile(path)
        n = pf.metadata.num_rows
        mb = path.stat().st_size / 1e6
        print(f"{path.name}  |  {n} rows  |  {mb:.1f} MB  |  {pf.num_row_groups} row group(s)")
        print("-" * 60)
        print(pf.schema_arrow)
        b = next(pf.iter_batches(batch_size=1))
        row = b.to_pydict()
        print("-" * 60)
        print("One example row (first row in file):")
        for col in b.schema.names:
            print(f"  {col}: {_cell(row[col][0])}")


if __name__ == "__main__":
    main()
