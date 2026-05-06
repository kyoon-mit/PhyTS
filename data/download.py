"""
Download PhyTS-bench datasets from Hugging Face (PhyTS-team/PhyTS-bench).

Sample files (~few GB total) are enough to verify the full pipeline and
satisfy NeurIPS reproducibility requirements.  Full datasets are large
(LIGO ~157 GB, TIDMAD ~163 GB, Project 8 ~44 GB, TESS ~194 MB).

Usage
-----
    # Sample data for all domains (quick pipeline verification):
    python data/download.py --sample

    # Sample for one domain:
    python data/download.py --sample --domain tess

    # Full dataset for one domain:
    python data/download.py --domain tess
    python data/download.py --domain ligo
    python data/download.py --domain project8
    python data/download.py --domain tidmad   # then run preprocess_tidmad.py

Notes
-----
LIGO   full: train/val are sharded across SNR bins; test is a single merged
             file.  The dataloader expects one file per split, so for train/val
             pass the directory to a concat step or use the sample for CI.
TIDMAD full: after download, run:
             python data/TIDMAD/preprocess_tidmad.py
             to produce the .npy arrays the dataloader expects.
"""
from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

try:
    from huggingface_hub import hf_hub_download, snapshot_download
    from huggingface_hub.errors import HfHubHTTPError
except ImportError:
    print("Install huggingface_hub: pip install huggingface_hub", file=sys.stderr)
    raise SystemExit(1)

REPO_ID = "PhyTS-team/PhyTS-bench"
DATA = Path(__file__).resolve().parent  # data/


# ---------------------------------------------------------------------------
# Sample downloads — small files for pipeline verification
# ---------------------------------------------------------------------------

def _hf_get(filename: str, dest: Path) -> None:
    """Download one file from the Hub if dest does not exist."""
    if dest.exists():
        print(f"  already present: {dest.relative_to(DATA.parent)}")
        return
    dest.parent.mkdir(parents=True, exist_ok=True)
    print(f"  downloading {filename} → {dest.relative_to(DATA.parent)}")
    tmp = hf_hub_download(
        repo_id=REPO_ID,
        repo_type="dataset",
        filename=filename,
    )
    shutil.copy2(tmp, dest)


def sample_ligo() -> None:
    """
    Downloads sample_LIGO.h5 and places it as all three splits so the
    LIGO dataloader (which expects one file per split) works out of the box.
    """
    src = "sample/sample_LIGO.h5"
    for split, name in [
        ("train", "sig_combined_train.h5"),
        ("val",   "sig_combined_val.h5"),
        ("test",  "sig_combined_test.h5"),
    ]:
        _hf_get(src, DATA / "LIGO" / split / name)


def sample_tess() -> None:
    _hf_get(
        "sample/sample_tess_classification.parquet",
        DATA / "TESS" / "tess_classification.parquet",
    )


def sample_tidmad() -> None:
    """
    Downloads the TIDMAD sample H5 file.  Run preprocess_tidmad.py afterwards:
        python data/TIDMAD/preprocess_tidmad.py \\
            --data_dir data/TIDMAD/original --out_dir data/TIDMAD/preprocessed
    """
    _hf_get(
        "sample/sample_tidmad.h5",
        DATA / "TIDMAD" / "original" / "tidmad_training_0000.h5",
    )
    print("  TIDMAD sample downloaded.")
    print("  Next step: python data/TIDMAD/preprocess_tidmad.py "
          "--data_dir data/TIDMAD/original --out_dir data/TIDMAD/preprocessed")


def sample_project8() -> None:
    """
    Downloads sample_Project8.hdf5 and copies it into all three split
    directories; the Project 8 dataloader lists all .hdf5 files in each dir.
    """
    src = "sample/sample_Project8.hdf5"
    for split in ("train", "valid", "test"):
        _hf_get(src, DATA / "Project8" / split / "sample_Project8.hdf5")


# ---------------------------------------------------------------------------
# Full downloads
# ---------------------------------------------------------------------------

def full_tess() -> None:
    dest = DATA / "TESS"
    dest.mkdir(parents=True, exist_ok=True)
    print(f"  downloading TESS/tess_classification.parquet …")
    tmp = hf_hub_download(
        repo_id=REPO_ID, repo_type="dataset",
        filename="TESS/tess_classification.parquet",
    )
    out = dest / "tess_classification.parquet"
    if not out.exists():
        shutil.copy2(tmp, out)
    print(f"  → {out.relative_to(DATA.parent)}")


def full_project8() -> None:
    dest = DATA / "Project8"
    dest.mkdir(parents=True, exist_ok=True)
    print(f"  downloading Project8/ (~44 GB) …")
    snapshot_download(
        repo_id=REPO_ID, repo_type="dataset",
        allow_patterns=["Project8/**"],
        local_dir=str(DATA.parent),
    )
    print(f"  → {dest.relative_to(DATA.parent)}/")


def full_ligo() -> None:
    dest = DATA / "LIGO"
    dest.mkdir(parents=True, exist_ok=True)
    print("  downloading LIGO/ (~157 GB) …")
    snapshot_download(
        repo_id=REPO_ID, repo_type="dataset",
        allow_patterns=["LIGO/**"],
        local_dir=str(DATA.parent),
    )
    # Test split is a single merged file; train/val are sharded by SNR bin.
    # Rename the test file to the path the dataloader expects.
    test_src = dest / "test" / "snr_5_50" / "test_00000-99999.h5"
    test_dst = dest / "test" / "sig_combined_test.h5"
    if test_src.exists() and not test_dst.exists():
        test_dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(test_src), str(test_dst))
        shutil.rmtree(dest / "test" / "snr_5_50", ignore_errors=True)
    print("  Note: LIGO train/val shards are in data/LIGO/{train,val}/bns_snr_*/")
    print("  The dataloader expects a single merged file per split.")
    print("  For full training, concatenate shards or point configs to individual shards.")


def full_tidmad() -> None:
    dest = DATA / "TIDMAD" / "original"
    dest.mkdir(parents=True, exist_ok=True)
    print("  downloading TIDMAD/ (~163 GB) …")
    snapshot_download(
        repo_id=REPO_ID, repo_type="dataset",
        allow_patterns=["TIDMAD/**"],
        local_dir=str(DATA.parent),
    )
    # Rename to abra_ prefix expected by preprocess_tidmad.py
    tidmad_dir = DATA / "TIDMAD"
    for subdir, prefix_out in [("training", "training"), ("validation", "validation")]:
        src_dir = tidmad_dir / subdir
        if src_dir.is_dir():
            for f in sorted(src_dir.glob("tidmad_*.h5")):
                new_name = "abra_" + f.name[len("tidmad_"):]
                dst = dest / new_name
                if not dst.exists():
                    shutil.move(str(f), str(dst))
            shutil.rmtree(src_dir, ignore_errors=True)
    test_src = tidmad_dir / "test" / "tidmad_test_file.h5"
    if test_src.exists():
        shutil.move(str(test_src), str(dest / "abra_test_file.h5"))
        shutil.rmtree(tidmad_dir / "test", ignore_errors=True)
    print("  TIDMAD downloaded and renamed.")
    print("  Next step: python data/TIDMAD/preprocess_tidmad.py "
          "--data_dir data/TIDMAD/original --out_dir data/TIDMAD/preprocessed")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

SAMPLE_FNS = {"ligo": sample_ligo, "tess": sample_tess,
               "tidmad": sample_tidmad, "project8": sample_project8}
FULL_FNS   = {"ligo": full_ligo, "tess": full_tess,
               "tidmad": full_tidmad, "project8": full_project8}

DOMAINS = list(SAMPLE_FNS)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download PhyTS-bench datasets from Hugging Face.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--domain", choices=DOMAINS + ["all"], default="all",
        help="Which domain to download (default: all)",
    )
    parser.add_argument(
        "--sample", action="store_true",
        help="Download small sample files instead of full datasets",
    )
    args = parser.parse_args()

    domains = DOMAINS if args.domain == "all" else [args.domain]
    fns = SAMPLE_FNS if args.sample else FULL_FNS

    mode = "sample" if args.sample else "full"
    print(f"Downloading {mode} data for: {', '.join(domains)}")
    print(f"Repository: {REPO_ID}\n")

    for domain in domains:
        print(f"[{domain}]")
        try:
            fns[domain]()
        except (OSError, HfHubHTTPError) as e:
            print(f"  ERROR: {e}", file=sys.stderr)
            raise SystemExit(1)
        print()

    print("Done.")


if __name__ == "__main__":
    main()
