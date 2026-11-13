"""Score LIGO test predictions: RMSE and R^2 of the chirp-mass point estimate (paper Table 2).

Reads every results/LIGO/*/test_predictions.csv written by `main.py test` (supervised models) and
fm_zeroshot_ligo.py (fm_<model>). The mean baseline predicts the test-set mean chirp mass.

Usage: python benchmarks/LIGO/score.py [--results_root results/LIGO]
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results_root", default="results/LIGO", type=Path)
    args = ap.parse_args()

    rows, y = [], None
    for f in sorted(args.results_root.glob("*/test_predictions.csv")):
        t = pd.read_csv(f, usecols=["true_chirp_mass", "pred_chirp_mass_mean"])
        y = t["true_chirp_mass"].to_numpy()
        e = t["pred_chirp_mass_mean"].to_numpy() - y
        rows.append(dict(model=f.parent.name.replace("_gaussnll_regression", ""), n=len(t),
                         rmse=np.sqrt(np.mean(e ** 2)), r2=1 - np.sum(e ** 2) / np.sum((y - y.mean()) ** 2)))
    if not rows:
        raise SystemExit(f"no test_predictions.csv under {args.results_root}")
    rows.append(dict(model="mean baseline", n=len(y), rmse=np.std(y), r2=0.0))
    print(pd.DataFrame(rows).to_string(index=False, float_format=lambda x: f"{x:.4f}"))


if __name__ == "__main__":
    main()
