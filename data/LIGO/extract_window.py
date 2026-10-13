"""Merge LIGO_2048Hz shards into one windowed HDF5 file per split.

The full 2048 Hz dataset stores 64 s segments (131072 samples per channel) across
thousands of shards.  Training only uses a 4 s window ending at coalescence, so
this script cuts that window out of every shard and concatenates all SNR bins
into a single file per split that ``LIGODataModule`` can read directly.

Usage:
    python data/LIGO/extract_window.py \
        --src /path/to/PhyTS-bench/LIGO_2048Hz --dest data/LIGO_2048Hz --split train

Output (per split): ``<dest>/<split>.h5`` with
    whitened_injected  (N, 2, (window_end - window_begin) * sample_rate)  float32
    <param>            (N,)                                                float32
and attrs ``sample_rate``, ``window_begin``, ``window_end``.
"""

import argparse
from pathlib import Path

import h5py
import numpy as np

SPLIT_GLOBS = {
    "train": "train/bns_snr_*/*.h5",
    "val": "val/bns_snr_*/*.h5",
    "test": "test/snr_5_50/*.h5",
}
STRAIN_KEYS = {"whitened_injected", "whitened_signal", "whitened_bkg"}


def extract(
    src: Path,
    dest: Path,
    split: str,
    strain_keys: list[str],
    sample_rate: int,
    window_begin: float,
    window_end: float,
) -> None:
    """Window and concatenate all shards of one split."""
    shards = sorted(src.glob(SPLIT_GLOBS[split]))
    if not shards:
        raise FileNotFoundError(f"No shards found for {split} under {src}")

    start, end = int(window_begin * sample_rate), int(window_end * sample_rate)
    with h5py.File(shards[0], "r") as f:
        param_keys = [k for k in f if k not in STRAIN_KEYS]
        n_ifos = f[strain_keys[0]].shape[1]
    sizes = []
    for p in shards:
        with h5py.File(p, "r") as f:
            sizes.append(f[strain_keys[0]].shape[0])
    n_total = sum(sizes)
    print(f"{split}: {len(shards)} shards, {n_total} samples, window [{start}:{end}]")

    dest.mkdir(parents=True, exist_ok=True)
    out_path = dest / f"{split}.h5"
    tmp_path = out_path.with_suffix(".h5.tmp")
    with h5py.File(tmp_path, "w") as out:
        out.attrs.update(sample_rate=sample_rate, window_begin=window_begin, window_end=window_end)
        for k in strain_keys:
            out.create_dataset(
                k, (n_total, n_ifos, end - start), dtype="float32", chunks=(1, n_ifos, end - start)
            )
        for k in param_keys:
            out.create_dataset(k, (n_total,), dtype="float32")

        offset = 0
        for i, (p, n) in enumerate(zip(shards, sizes)):
            with h5py.File(p, "r") as f:
                for k in strain_keys:
                    out[k][offset : offset + n] = f[k][:, :, start:end].astype(np.float32)
                for k in param_keys:
                    out[k][offset : offset + n] = f[k][:]
            offset += n
            if i % 100 == 0 or i == len(shards) - 1:
                print(f"  {i + 1}/{len(shards)} shards ({offset}/{n_total} samples)", flush=True)
    tmp_path.rename(out_path)
    print(f"Wrote {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--src", type=Path, required=True, help="LIGO_2048Hz download directory")
    parser.add_argument("--dest", type=Path, required=True, help="Output directory")
    parser.add_argument("--split", choices=list(SPLIT_GLOBS) + ["all"], default="all")
    parser.add_argument("--strain_keys", nargs="+", default=["whitened_injected"])
    parser.add_argument("--sample_rate", type=int, default=2048)
    parser.add_argument("--window_begin", type=float, default=59.0)
    parser.add_argument("--window_end", type=float, default=63.0)
    args = parser.parse_args()

    splits = list(SPLIT_GLOBS) if args.split == "all" else [args.split]
    for split in splits:
        extract(
            args.src, args.dest, split, args.strain_keys,
            args.sample_rate, args.window_begin, args.window_end,
        )


if __name__ == "__main__":
    main()
