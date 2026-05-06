"""Download the TIDMAD dataset from HuggingFace (PhyTS-team/PhyTS-bench).

Downloads training and validation H5 files to a local directory, then
symlinks data/TIDMAD/original and data/TIDMAD/preprocessed into the repo.

Usage:
    uv run python data/TIDMAD/download.py
    uv run python data/TIDMAD/download.py --dest /fast/barmstrong/TIDMAD
"""

import argparse
import os
import shutil
from pathlib import Path


REPO_ID = "PhyTS-team/PhyTS-bench"
REPO_ROOT = Path(__file__).resolve().parents[2]  # TimeSeriesPhysics/


def download(dest: Path):
    from huggingface_hub import snapshot_download

    hf_dir = dest / "hf"
    original_dir = dest / "original"
    preprocessed_dir = dest / "preprocessed"

    print(f"Downloading {REPO_ID} → {hf_dir}  (~159 GB, training + validation only)")
    snapshot_download(
        REPO_ID,
        repo_type="dataset",
        local_dir=str(hf_dir),
        allow_patterns=["TIDMAD/training/*", "TIDMAD/validation/*"],
    )

    print(f"\nFlattening into {original_dir}")
    original_dir.mkdir(parents=True, exist_ok=True)
    for subdir in ["training", "validation"]:
        src = hf_dir / "TIDMAD" / subdir
        if src.exists():
            for f in src.iterdir():
                dest_file = original_dir / f.name
                if not dest_file.exists():
                    shutil.move(str(f), str(dest_file))
            print(f"  Moved {subdir}/ files")

    preprocessed_dir.mkdir(parents=True, exist_ok=True)

    symlink("original",     REPO_ROOT / "data" / "TIDMAD" / "original",     original_dir)
    symlink("preprocessed", REPO_ROOT / "data" / "TIDMAD" / "preprocessed", preprocessed_dir)

    print("\nDone. Next step — preprocess the H5 files:")
    print(
        f"  uv run python data/TIDMAD/preprocess_tidmad.py"
        f" --data_dir data/TIDMAD/original"
        f" --out_dir  data/TIDMAD/preprocessed"
    )


def symlink(name: str, link: Path, target: Path):
    if link.exists() or link.is_symlink():
        print(f"  Skipping symlink for {name} — {link} already exists")
        return
    link.symlink_to(target)
    print(f"  Symlinked data/TIDMAD/{name} → {target}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dest",
        default="/fast/barmstrong/TIDMAD",
        help="Directory to download raw files into (default: /fast/barmstrong/TIDMAD)",
    )
    args = parser.parse_args()
    download(Path(args.dest))
