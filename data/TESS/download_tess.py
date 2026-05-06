"""
Download the pre-split PhyTS TESS parquet shards from Hugging Face
(`PhyTS-team/PhyTS-bench`). The Hub stores them under ``TESS/split/*.parquet``;
this script pulls that subtree, **moves** every shard into ``TESS/*.parquet``,
and deletes the local ``TESS/split`` directory so ``data_dir`` is only
``.cache/TESS`` with parquet files at the top level.

Uses ``snapshot_download`` for the remote ``TESS/split/*`` paths; the on-disk
layout matches ``src/dataloader/tess_dataloader.TESSClassificationDataset``.
Needs ``uv sync`` (pyarrow + huggingface_hub).

Usage
-----
    python data/TESS/download_tess.py
    python data/TESS/download_tess.py --cache-dir path/to/cache
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


def flatten_hub_split_into_tess_root(tess_root: Path) -> tuple[int, int]:
    """Move ``*.parquet`` from ``tess_root/split`` into ``tess_root``; remove ``split``.

    If a shard already exists in ``tess_root``, the duplicate under ``split`` is
    deleted so the tree can be removed.

    Returns
    -------
    moved : int
        Parquet files moved into ``tess_root``.
    dropped_duplicate : int
        Files only removed from ``split`` because ``tess_root`` already had them.
    """
    tess_root = Path(tess_root)
    split_sub = tess_root / "split"
    if not split_sub.is_dir():
        return (0, 0)
    moved = 0
    dropped_duplicate = 0
    for src in sorted(split_sub.glob("*.parquet")):
        dest = tess_root / src.name
        if dest.is_file():
            src.unlink()
            dropped_duplicate += 1
        else:
            shutil.move(str(src), str(dest))
            moved += 1
    shutil.rmtree(split_sub)
    return (moved, dropped_duplicate)


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
        description="Download PhyTS-bench TESS shards into cache/TESS/*.parquet (no TESS/split)."
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=None,
        help="Root directory for downloaded data. "
        "Final layout: cache_dir/TESS/*.parquet (Hub TESS/split is flattened). "
        f"Default: {_DEFAULT_CACHE_PARENT}",
    )
    args = parser.parse_args()
    if args.cache_dir is not None:
        cache_parent = Path(args.cache_dir).resolve()
    else:
        cache_parent = _DEFAULT_CACHE_PARENT.resolve()
    cache_parent.mkdir(parents=True, exist_ok=True)

    tess_root = cache_parent / "TESS"
    split_dir = tess_root / "split"

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
        if not any(tess_root.glob("*.parquet")):
            print(f"Expected directory missing after snapshot: {split_dir}", file=sys.stderr)
            raise SystemExit(1)
        print(
            f"No {split_dir} after snapshot; using existing parquet(s) under {tess_root}",
        )
    else:
        parquets_pre = sorted(split_dir.glob("*.parquet"))
        if not parquets_pre:
            print(f"No parquet files under {split_dir}", file=sys.stderr)
            raise SystemExit(1)
        moved, dropped_dup = flatten_hub_split_into_tess_root(tess_root)
        msg_parts = []
        if moved:
            msg_parts.append(f"moved {moved} into {tess_root}")
        if dropped_dup:
            msg_parts.append(f"removed {dropped_dup} duplicate(s) from split")
        if msg_parts:
            print(f"\nFlattened Hub layout: {', '.join(msg_parts)}; removed {split_dir}")

    parquets = sorted(tess_root.glob("*.parquet"))
    if not parquets:
        print(f"No parquet files under {tess_root}", file=sys.stderr)
        raise SystemExit(1)

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
