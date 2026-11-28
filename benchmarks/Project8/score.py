"""Score Project 8 test predictions (paper Table 2): RMSE in eV and R^2 of the electron energy.

Reads every results/Project8/*/test_predictions.csv written by `main.py test`
(<model>_gaussnll_regression) and by fm_zeroshot_project8.py (fm_<model>). The mean baseline
predicts the training-split mean energy.

Usage: python benchmarks/Project8/score.py [--results_root ...] [--data_dir data/Project8]
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from dataloader.project8_dataloader import read_target, split_files


def scores(true: np.ndarray, pred: np.ndarray) -> dict[str, float]:
    """RMSE and R^2 of pred against true."""
    sq_err = np.sum((pred - true) ** 2)
    return {"rmse_eV": float(np.sqrt(sq_err / len(true))),
            "r2": float(1 - sq_err / np.sum((true - true.mean()) ** 2))}


def main():
    """Print RMSE and R^2 per model, plus the mean baseline."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--results_root", default="results/Project8", type=Path)
    ap.add_argument("--data_dir", default="data/Project8")
    args = ap.parse_args()

    rows = {}
    for f in sorted(args.results_root.glob("*/test_predictions.csv")):
        t = pd.read_csv(f)
        rows[f.parent.name.removesuffix("_gaussnll_regression")] = scores(
            t["true_energy_eV"].to_numpy(), t["pred_energy_eV_mean"].to_numpy())
    if not rows:
        raise SystemExit(f"no test_predictions.csv under {args.results_root}")

    train, test = (read_target(split_files(args.data_dir, split), "energy_eV")
                   for split in ("train", "test"))
    rows["mean baseline"] = scores(test, np.full_like(test, train.mean()))
    print(pd.DataFrame(rows).T.to_string(float_format=lambda x: f"{x:.4f}"))


if __name__ == "__main__":
    main()
