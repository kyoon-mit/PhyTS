"""Zero-shot foundation-model rows of Table 2 for Project 8: frozen embeddings + ridge probe.

Each event's noisy I and Q traces are prepared exactly as for the supervised models
(Project8Dataset: cavity noise, 24,576 samples, divided by their standard deviation). Each trace
is covered by 49 windows of 512 samples starting at linspace(0, 24064, 49), about 11 samples of
overlap, which the frozen backbone embeds (benchmarks/foundation/zeroshot.py). Window embeddings
are averaged per channel and the I and Q features concatenated. A ridge regression (alpha = 1)
is fitted on the training split and evaluated on the test split.

Usage (foundation-model env):
  benchmarks/foundation/.venv/bin/python benchmarks/Project8/fm_zeroshot_project8.py --model moment
  # smoke test
  benchmarks/foundation/.venv/bin/python benchmarks/Project8/fm_zeroshot_project8.py \
      --model moment --max_samples 64 --out_root /tmp/p8_fm_smoke
Writes <out_root>/fm_<model>/test_predictions.csv.
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Subset

_REPO = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(_REPO / "src"), str(_REPO / "benchmarks")]
from foundation.zeroshot import CHUNK_LEN, MODELS, embed, load_wrapper, ridge_probe

from dataloader.project8_dataloader import Project8Dataset, read_target, split_files

N_WINDOWS = 49


def embed_split(wrapper, args, split: str) -> tuple[np.ndarray, np.ndarray]:
    """(N, 2d) features and (N,) energies in eV of one split."""
    paths = split_files(args.data_dir, split)
    energy = read_target(paths, "energy_eV")
    data = Project8Dataset(paths, "energy_eV", "cav", args.cutoff, 0.0, 1.0)
    n = min(len(data), args.max_samples or len(data))
    loader = DataLoader(Subset(data, range(n)), batch_size=args.batch_size,
                        num_workers=args.num_workers)
    starts = np.linspace(0, args.cutoff - CHUNK_LEN, N_WINDOWS).round().astype(int)
    idx = starts[:, None] + np.arange(CHUNK_LEN)                   # (49, 512)
    features = []
    for x, _ in loader:
        x = x.numpy().transpose(0, 2, 1)                           # (B, 2, L)
        windows = x[:, :, idx]                                     # (B, 2, 49, 512)
        emb = embed(wrapper, args.model, windows.reshape(-1, CHUNK_LEN), args.embed_batch)
        features.append(emb.reshape(len(x), 2, N_WINDOWS, -1).mean(axis=2).reshape(len(x), -1))
        print(f"{split}: {sum(map(len, features))}/{n}", flush=True)
    return np.concatenate(features), energy[:n]


def main():
    """Embed the train and test events with --model and fit the ridge probe."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=MODELS)
    ap.add_argument("--data_dir", default="data/Project8")
    ap.add_argument("--out_root", default="results/Project8", type=Path)
    ap.add_argument("--cutoff", default=24576, type=int)
    ap.add_argument("--batch_size", default=16, type=int, help="events per loader batch")
    ap.add_argument("--embed_batch", default=256, type=int, help="windows per model call")
    ap.add_argument("--num_workers", default=4, type=int)
    ap.add_argument("--max_samples", type=int, help="events per split (smoke test)")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    wrapper = load_wrapper(args.model, args.device)
    X_train, y_train = embed_split(wrapper, args, "train")
    X_test, y_test = embed_split(wrapper, args, "test")
    out = args.out_root / f"fm_{args.model}" / "test_predictions.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    table = pd.DataFrame({"true_energy_eV": y_test,
                          "pred_energy_eV_mean": ridge_probe(X_train, y_train, X_test)})
    table.to_csv(out, index=False)
    print(f"-> {out}")


if __name__ == "__main__":
    main()
