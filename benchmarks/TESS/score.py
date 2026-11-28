"""Score TESS test predictions (paper Table 2): balanced accuracy of the class and R^2 of frot.

Reads every results/TESS/<model>_<task>/test_predictions.csv written by `main.py test` and by
fm_zeroshot_tess.py (fm_<model>_<task>). The mean baseline predicts a single class, which scores
a balanced accuracy of 1/8, and the training-split mean frot.

Usage: python benchmarks/TESS/score.py [--results_root results/TESS] [--data_dir data/TESS/split]
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from dataloader.tess_dataloader import CLASSES
from tasks.TESS.classification import balanced_accuracy
from tasks.TESS.regression import r2_score


def main():
    """Print balanced accuracy and R^2 per model, plus the mean baseline."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--results_root", default="results/TESS", type=Path)
    ap.add_argument("--data_dir", default="data/TESS/split", type=Path)
    args = ap.parse_args()

    scores: dict[str, dict[str, float]] = {}
    for f in sorted(args.results_root.glob("*/test_predictions.csv")):
        model, task = f.parent.name.rsplit("_", 1)
        t = pd.read_csv(f)
        if task == "classification":
            score = {"bal_acc": balanced_accuracy(t["true_label"].to_numpy(),
                                                  t["pred_label"].to_numpy())}
        else:
            score = {"r2": r2_score(t["true_frot"].to_numpy(), t["pred_frot"].to_numpy())}
        scores.setdefault(model, {}).update(score)
    if not scores:
        raise SystemExit(f"no test_predictions.csv under {args.results_root}")

    train, test = (pd.read_parquet(args.data_dir / f"tess_regression_{split}.parquet")["frot"]
                   .to_numpy() for split in ("train", "test"))
    scores["mean baseline"] = {"bal_acc": 1 / len(CLASSES),
                               "r2": r2_score(test, np.full_like(test, train.mean()))}
    table = pd.DataFrame(scores).T.reindex(columns=["bal_acc", "r2"])
    print(table.to_string(float_format=lambda x: f"{x:.4f}"))


if __name__ == "__main__":
    main()
