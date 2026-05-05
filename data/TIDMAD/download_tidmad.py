#!/usr/bin/env python3
"""
TIDMAD Dataset Downloader
Downloads the TIDMAD dataset from Hugging Face (PhyTS-team/PhyTS-bench).

This script provides a convenient interface to download the full TIDMAD dataset
with progress reporting, error handling, and data validation.

Usage:
    python tidmad_downloader.py --output ./tidmad_data
    python tidmad_downloader.py --help
"""

import argparse
import os
import sys
from pathlib import Path
from typing import Optional

try:
    from huggingface_hub import snapshot_download, HfApi, LoginRequired
    from huggingface_hub.utils import RepositoryNotFoundError
except ImportError:
    print("Error: Required package 'huggingface_hub' not installed.")
    print("Install with: pip install huggingface-hub")
    sys.exit(1)


class TIDMADDownloader:
    """Handles downloading the TIDMAD dataset from Hugging Face."""

    # Hugging Face repository details
    REPO_ID = "PhyTS-team/PhyTS-bench"
    SUBFOLDER = "TIDMAD"
    REPO_TYPE = "dataset"

    def __init__(self, output_dir: str = "./tidmad_data", verbose: bool = True):
        """
        Initialize the downloader.

        Args:
            output_dir: Directory where data will be downloaded.
            verbose: Enable verbose logging.
        """
        self.output_dir = Path(output_dir).expanduser().resolve()
        self.verbose = verbose
        self.api = HfApi()

    def _log(self, message: str) -> None:
        """Print a message if verbose mode is enabled."""
        if self.verbose:
            print(f"[TIDMAD] {message}")

    def _ensure_output_dir(self) -> None:
        """Create output directory if it doesn't exist."""
        try:
            self.output_dir.mkdir(parents=True, exist_ok=True)
            self._log(f"Output directory: {self.output_dir}")
        except PermissionError:
            raise PermissionError(
                f"Permission denied: cannot write to {self.output_dir}. "
                "Check directory permissions."
            )
        except OSError as e:
            raise OSError(
                f"Failed to create output directory {self.output_dir}: {e}"
            )

    def _check_space_available(self) -> None:
        """
        Check if sufficient disk space is available.
        Warns if less than 100GB free (TIDMAD is ~800GB).
        """
        try:
            stat = os.statvfs(self.output_dir)
            free_space_gb = (stat.f_bavail * stat.f_frsize) / (1024**3)

            if free_space_gb < 100:
                self._log(
                    f"WARNING: Only {free_space_gb:.1f}GB free. "
                    "TIDMAD is ~800GB. Download may fail."
                )
        except Exception as e:
            self._log(f"Could not check disk space: {e}")

    def _validate_repo_exists(self) -> None:
        """Verify the repository exists and is accessible."""
        try:
            self._log(f"Verifying repository: {self.REPO_ID}")
            repo_info = self.api.repo_info(
                repo_id=self.REPO_ID,
                repo_type=self.REPO_TYPE
            )
            self._log(f"Repository found: {repo_info.repo_id}")
        except RepositoryNotFoundError:
            raise RepositoryNotFoundError(
                f"Repository '{self.REPO_ID}' not found on Hugging Face. "
                "Check the repository name and ensure it exists."
            )
        except LoginRequired:
            raise LoginRequired(
                "Authentication required. Run 'huggingface-cli login' "
                "or set HF_TOKEN environment variable."
            )
        except Exception as e:
            raise RuntimeError(
                f"Failed to verify repository: {e}. "
                "Check your internet connection and Hugging Face status."
            )

    def _count_files(self) -> Optional[int]:
        """
        Attempt to get a file count from the repository.
        Returns None if unable to determine.
        """
        try:
            files = self.api.list_repo_tree(
                repo_id=self.REPO_ID,
                repo_type=self.REPO_TYPE,
                recursive=True
            )
            return sum(1 for f in files if f.type == "file")
        except Exception:
            return None

    def download(
        self,
        resume_download: bool = True,
        splits: Optional[list] = None
    ) -> Path:
        """
        Download the TIDMAD dataset from Hugging Face.

        The TIDMAD dataset contains three splits:
        - training/    (~500GB, ~10M samples)
        - validation/  (~200GB, ~4M samples)
        - test/        (~100GB, ~2M samples)

        Directory structure is preserved during download.

        Args:
            resume_download: Resume incomplete downloads (default: True).
            splits: List of splits to download. Options: ["training", "validation", "test"].
                   Default (None) downloads all splits.

        Returns:
            Path to the downloaded dataset root directory.

        Raises:
            Various exceptions with descriptive error messages on failure.
        """
        try:
            self._ensure_output_dir()
            self._check_space_available()
            self._validate_repo_exists()

            file_count = self._count_files()
            if file_count:
                self._log(f"Repository contains ~{file_count} files")

            if splits is None:
                splits = ["training", "validation", "test"]
                self._log("Downloading all splits: training, validation, test")
            else:
                self._log(f"Downloading selected splits: {', '.join(splits)}")

            self._log(
                f"Starting download of {self.REPO_ID}/{self.SUBFOLDER}..."
            )
            self._log("This may take a considerable amount of time (~800GB for all splits)")

            # Build pattern list based on requested splits
            allow_patterns = []
            for split in splits:
                allow_patterns.extend([
                    f"{split}/*.h5",
                    f"{split}/*.hdf5",
                    f"{split}/*.csv",
                    f"{split}/*.json",
                    f"{split}/*.txt",
                ])

            # Download the dataset using snapshot_download
            local_dir = snapshot_download(
                repo_id=self.REPO_ID,
                repo_type=self.REPO_TYPE,
                subfolder=self.SUBFOLDER,
                local_dir=str(self.output_dir),
                resume_download=resume_download,
                allow_patterns=allow_patterns,
            )

            self._log(f"✓ Download completed successfully")
            return Path(local_dir)

        except PermissionError as e:
            self._handle_error(
                f"Permission Error: {e}\n"
                "Ensure you have write permissions for the output directory."
            )
        except LoginRequired as e:
            self._handle_error(
                f"Authentication Required: {e}\n"
                "The dataset may require authentication. Run:\n"
                "  huggingface-cli login\n"
                "Or set: export HF_TOKEN=your_token"
            )
        except RepositoryNotFoundError as e:
            self._handle_error(f"Repository Not Found: {e}")
        except Exception as e:
            self._handle_error(
                f"Download failed: {e}\n"
                "Please check:\n"
                "  1. Internet connection\n"
                "  2. Hugging Face is accessible\n"
                "  3. Disk space available (~800GB)\n"
                "  4. Output directory is writable"
            )

    @staticmethod
    def _handle_error(message: str) -> None:
        """Print error message and exit with error code."""
        print(f"\n❌ ERROR: {message}\n", file=sys.stderr)
        sys.exit(1)

    def verify_download(self) -> bool:
        """
        Verify the downloaded dataset by checking for expected files.

        Returns:
            True if dataset appears valid, False otherwise.
        """
        self._log("Verifying downloaded dataset...")

        if not self.output_dir.exists():
            self._log(f"✗ Output directory does not exist: {self.output_dir}")
            return False

        # Check for expected HDF5 or data files
        hdf5_files = list(self.output_dir.glob("**/*.h5"))
        hdf5_files.extend(self.output_dir.glob("**/*.hdf5"))

        if not hdf5_files:
            self._log("✗ No HDF5 files found in dataset")
            return False

        total_size_gb = sum(f.stat().st_size for f in hdf5_files) / (1024**3)
        self._log(f"✓ Found {len(hdf5_files)} HDF5 files ({total_size_gb:.1f}GB)")

        return True


def main():
    """Parse arguments and run the downloader."""
    parser = argparse.ArgumentParser(
        description="Download the TIDMAD dataset from Hugging Face",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Dataset Structure:
  TIDMAD contains three splits:
    - training/    (~500GB, ~10M samples)
    - validation/  (~200GB, ~4M samples)
    - test/        (~100GB, ~2M samples)

Examples:
  Download all splits (default):
    python tidmad_downloader.py --output ./tidmad_data

  Download only training and validation:
    python tidmad_downloader.py --splits training validation

  Download only test set:
    python tidmad_downloader.py --splits test

  Download with verification:
    python tidmad_downloader.py --verify
        """,
    )

    parser.add_argument(
        "--output",
        type=str,
        default="./tidmad_data",
        help="Output directory for downloaded data (default: ./tidmad_data)",
    )
    parser.add_argument(
        "--splits",
        nargs="+",
        choices=["training", "validation", "test"],
        help="Dataset splits to download (default: all)",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress verbose output",
    )
    parser.add_argument(
        "--verify",
        action="store_true",
        help="Verify download after completion",
    )
    parser.add_argument(
        "--no-resume",
        action="store_true",
        help="Don't resume incomplete downloads (start fresh)",
    )

    args = parser.parse_args()

    downloader = TIDMADDownloader(
        output_dir=args.output,
        verbose=not args.quiet,
    )

    try:
        downloader.download(
            resume_download=not args.no_resume,
            splits=args.splits
        )

        if args.verify:
            if downloader.verify_download():
                print("\n✓ Dataset verification passed")
            else:
                print("\n⚠ Dataset verification failed")
                sys.exit(1)

        print(f"\n✓ Dataset ready at: {downloader.output_dir}")

    except KeyboardInterrupt:
        print("\n⚠ Download interrupted by user", file=sys.stderr)
        sys.exit(130)


if __name__ == "__main__":
    main()