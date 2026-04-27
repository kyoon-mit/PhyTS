"""
Peek at PhyTS TESS Parquet on Hugging Face (``PhyTS-team/PhyTS-bench`` → ``TESS/``).
Needs: ``uv sync --extra jax`` (pyarrow + huggingface_hub).

Usage:
    uv run --extra jax python data/TESS/download_tess.py
    uv run --extra jax python data/TESS/download_tess.py --cache-dir path/to/cache
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

try:
    from huggingface_hub import hf_hub_download
    from huggingface_hub.errors import HfHubHTTPError
    import pyarrow.parquet as pq
except ImportError:  # pragma: no cover
    print("Install: uv sync --extra jax", file=sys.stderr)
    raise SystemExit(1) from None

# Smaller file first (quicker feedback), then the large classification set.
NAMES = ("tess_regression.parquet", "tess_classification.parquet")
REPO = "PhyTS-team/PhyTS-bench"
_DEFAULT_CACHE = Path(__file__).resolve().parent / ".cache"


def _cell(value: object) -> str:
    """One cell: ``repr`` for scalars; for sequences, ``len`` + first few values."""
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
    parser = argparse.ArgumentParser(description="Download TESS PhyTS-bench Parquet files.")
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=_DEFAULT_CACHE,
        help="Directory to download into (files land in <cache-dir>/TESS/). "
             f"Default: {_DEFAULT_CACHE}",
    )
    args = parser.parse_args()
    CACHE = args.cache_dir

    CACHE.mkdir(parents=True, exist_ok=True)
    for name in NAMES:
        print()
        print("=" * 60)
        try:
            path = Path(
                hf_hub_download(
                    repo_id=REPO,
                    repo_type="dataset",
                    filename=f"TESS/{name}",
                    local_dir=str(CACHE),
                )
            )
        except (OSError, HfHubHTTPError) as e:
            print(f"Download failed ({name}): {e}", file=sys.stderr)
            raise SystemExit(1) from e

        pf = pq.ParquetFile(path)
        n = pf.metadata.num_rows
        mb = path.stat().st_size / 1e6
        print(f"{name}  |  {n} rows  |  {mb:.1f} MB  |  {pf.num_row_groups} row group(s)")
        print("-" * 60)
        print(pf.schema_arrow)
        # One sample row
        b = next(pf.iter_batches(batch_size=1))
        row = b.to_pydict()
        print("-" * 60)
        print("One example row (first row in file):")
        for col in b.schema.names:
            print(f"  {col}: {_cell(row[col][0])}")


if __name__ == "__main__":
    main()
