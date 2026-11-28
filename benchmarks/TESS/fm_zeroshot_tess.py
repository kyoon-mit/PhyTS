"""Zero-shot foundation-model rows of Table 2 for TESS: frozen embeddings + linear probe.

Each light curve is right-padded with its last value, or truncated, to 1,024 points and split into
two 512-point windows, which the frozen backbone embeds (benchmarks/foundation/zeroshot.py); the
two embeddings are averaged. A logistic regression (C = 1) for the class and a ridge regression
(alpha = 1) for frot are fitted on the training split and evaluated on the test split, the same
Hub splits as the supervised models.

Usage (foundation-model env):
  benchmarks/foundation/.venv/bin/python benchmarks/TESS/fm_zeroshot_tess.py --model moment
  # smoke test
  benchmarks/foundation/.venv/bin/python benchmarks/TESS/fm_zeroshot_tess.py --model moment \
      --max_samples 256 --out_root /tmp/tess_fm_smoke
Writes <out_root>/fm_<model>_{classification,regression}/test_predictions.csv.
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "benchmarks"))
from foundation.zeroshot import CHUNK_LEN, MODELS, embed, load_wrapper, logistic_probe, ridge_probe

LC_LEN = 1024
TARGET = {"classification": "label", "regression": "frot"}


def windows(flux: pd.Series) -> np.ndarray:
    """Light curves -> (2N, 512): each padded with its last value or truncated to 1,024 points."""
    out = np.empty((len(flux), LC_LEN), dtype=np.float32)
    for i, f in enumerate(flux):
        n = min(len(f), LC_LEN)
        out[i, :n], out[i, n:] = f[:n], f[n - 1]
    return out.reshape(-1, CHUNK_LEN)


def main():
    """Embed the train and test light curves of both tasks with --model and fit the probes."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=MODELS)
    ap.add_argument("--data_dir", default="data/TESS/split", type=Path)
    ap.add_argument("--out_root", default="results/TESS", type=Path)
    ap.add_argument("--batch_size", default=128, type=int,
                    help="windows per model call (lower it if the GPU runs out of memory)")
    ap.add_argument("--max_samples", type=int, help="light curves per split (smoke test)")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    wrapper = load_wrapper(args.model, args.device)
    for task, target in TARGET.items():
        features, targets = {}, {}
        for split in ("train", "test"):
            df = pd.read_parquet(args.data_dir / f"tess_{task}_{split}.parquet",
                                 columns=["flux", target])
            if task == "regression":
                df = df[df["frot"] > 0]     # light curves with a measured frot
            df = df[:args.max_samples]
            emb = embed(wrapper, args.model, windows(df["flux"]), args.batch_size)
            features[split] = emb.reshape(len(df), -1, emb.shape[-1]).mean(axis=1)
            targets[split] = df[target].to_numpy()
            print(f"{task} {split}: {features[split].shape}", flush=True)
        probe = logistic_probe if task == "classification" else ridge_probe
        pred = probe(features["train"], targets["train"], features["test"])
        name = "label" if task == "classification" else "frot"
        out = args.out_root / f"fm_{args.model}_{task}" / "test_predictions.csv"
        out.parent.mkdir(parents=True, exist_ok=True)
        table = pd.DataFrame({f"true_{name}": targets["test"], f"pred_{name}": pred})
        table.to_csv(out, index=False)
        print(f"-> {out}")


if __name__ == "__main__":
    main()
