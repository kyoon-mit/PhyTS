"""Score denoised TIDMAD test files with the official denoising score (paper Table 2).

Reads every results/TIDMAD/<model>/tidmad_test_file.h5 written by denoise.py and scores it with
src/tasks/TIDMAD/tidmad_denoising.py, the official TIDMAD Benchmark 1 scorer.

Usage: python benchmarks/TIDMAD/score.py [--results_root results/TIDMAD]
"""

import argparse
import sys
from pathlib import Path
from types import SimpleNamespace

_REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO / "src"))

from tasks.TIDMAD.tidmad_denoising import calculateBenchmark  # noqa: E402


def main():
    """Print the official denoising score of every denoised test file."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--results_root", default="results/TIDMAD", type=Path)
    ap.add_argument("--num_workers", default=8, type=int)
    args = ap.parse_args()

    opts = SimpleNamespace(coarse=False, parallel=args.num_workers > 1,
                           num_workers=args.num_workers)
    files = sorted(args.results_root.glob("*/tidmad_test_file.h5"))
    if not files:
        raise SystemExit(f"no tidmad_test_file.h5 under {args.results_root}")
    scores = {f.parent.name: calculateBenchmark(str(f.parent), [f.name], opts)[0] for f in files}
    print(f"{'model':<16} denoising score")
    for model, score in scores.items():
        print(f"{model:<16} {score:.2f}")


if __name__ == "__main__":
    main()
