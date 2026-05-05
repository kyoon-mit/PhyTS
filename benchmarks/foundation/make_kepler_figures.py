"""Render Kepler Q9 v3 linear-probe confusion matrices as a figure.

For each model under <results_dir>/<model>/, loads all
`[fold_*/]confusion_matrix.npy` files, averages them (row-normalised = per-class
recall), and draws a grid of heatmaps.

Directory layouts supported
---------------------------
  <results_dir>/<model>/confusion_matrix.npy                  # single fold
  <results_dir>/<model>/fold_{k}/confusion_matrix.npy         # k-fold CV

Usage
-----
    python benchmarks/foundation/make_kepler_figures.py \
        --results_dir plots/kepler_q9v3/linear_probe \
        --out plots/kepler_q9v3/linear_probe/confusion_matrices.png
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


MODEL_LABELS = {
    "moment":      "MOMENT",
    "chronos":     "Chronos",
    "timemoe":     "Time-MoE",
    "granite_ttm": "Granite-TTM",
    "timesfm":     "TimesFM",
    "moirai":      "MOIRAI",
    "lagllama":    "Lag-Llama",
}
MODEL_ORDER = ["moment", "chronos", "timemoe", "granite_ttm", "timesfm", "moirai", "lagllama"]


def _load_confusion_matrices(model_dir: Path) -> list[np.ndarray]:
    """Load per-fold confusion matrices from a model dir.

    Precedence: fold_*/confusion_matrix.npy, else confusion_matrix.npy.
    """
    fold_cms = sorted(model_dir.glob("fold_*/confusion_matrix.npy"))
    if fold_cms:
        return [np.load(p) for p in fold_cms]
    single = model_dir / "confusion_matrix.npy"
    if single.exists():
        return [np.load(single)]
    return []


def _row_normalise(cm: np.ndarray) -> np.ndarray:
    """Row-normalise a confusion matrix so each row sums to 1 (per-class recall)."""
    cm = cm.astype(np.float64)
    row_sums = cm.sum(axis=1, keepdims=True)
    with np.errstate(divide="ignore", invalid="ignore"):
        out = np.where(row_sums > 0, cm / row_sums, 0.0)
    return out


def _average_confusions(cms: list[np.ndarray]) -> np.ndarray:
    """Row-normalise each fold then average → mean per-class recall matrix."""
    if not cms:
        raise ValueError("No confusion matrices to average.")
    stack = np.stack([_row_normalise(cm) for cm in cms], axis=0)
    return stack.mean(axis=0)


def _plot_heatmap(ax, cm_norm: np.ndarray, class_names: list[str], *, title: str, n_folds: int):
    im = ax.imshow(cm_norm, cmap="Blues", vmin=0.0, vmax=1.0, aspect="equal")
    ax.set_xticks(range(len(class_names)))
    ax.set_yticks(range(len(class_names)))
    # Shortened labels for density
    short = [c[:8] for c in class_names]
    ax.set_xticklabels(short, rotation=45, ha="right", fontsize=7)
    ax.set_yticklabels(short, fontsize=7)
    # Annotate each cell with the rounded percentage
    for i in range(cm_norm.shape[0]):
        for j in range(cm_norm.shape[1]):
            v = cm_norm[i, j]
            colour = "white" if v > 0.55 else "black"
            ax.text(j, i, f"{v:.2f}" if v >= 0.01 else "",
                    ha="center", va="center", color=colour, fontsize=6)
    suffix = f" (avg of {n_folds} folds)" if n_folds > 1 else ""
    ax.set_title(f"{title}{suffix}", fontsize=10)
    ax.set_xlabel("Predicted", fontsize=8)
    ax.set_ylabel("True", fontsize=8)
    return im


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--results_dir", default="plots/kepler_q9v3/linear_probe")
    p.add_argument("--out", default=None)
    p.add_argument("--ncols", type=int, default=3)
    p.add_argument("--dpi", type=int, default=150)
    args = p.parse_args()

    results_dir = Path(args.results_dir)

    # Discover class names
    class_names = None
    for sub in results_dir.iterdir():
        mj = sub / "metrics.json"
        if mj.exists():
            with open(mj) as f:
                class_names = json.load(f)["class_names"]
            break
        # Try fold subdirs
        for fold_mj in sub.glob("fold_*/metrics.json"):
            with open(fold_mj) as f:
                class_names = json.load(f)["class_names"]
            break
        if class_names is not None:
            break
    if class_names is None:
        raise SystemExit(f"No metrics.json found under {results_dir}")

    # Collect (model, averaged_cm, n_folds) in fixed order
    panels = []
    for m in MODEL_ORDER:
        d = results_dir / m
        if not d.is_dir():
            continue
        cms = _load_confusion_matrices(d)
        if not cms:
            continue
        panels.append((m, _average_confusions(cms), len(cms)))

    if not panels:
        raise SystemExit(f"No confusion matrices found under {results_dir}")

    ncols = min(args.ncols, len(panels))
    nrows = int(np.ceil(len(panels) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(5.0 * ncols, 4.5 * nrows))
    axes = np.atleast_2d(axes).reshape(nrows, ncols)

    im = None
    for idx, (m, cm_avg, n_folds) in enumerate(panels):
        r, c = divmod(idx, ncols)
        im = _plot_heatmap(axes[r, c], cm_avg, class_names,
                           title=MODEL_LABELS.get(m, m), n_folds=n_folds)
    # Hide any unused panels
    for idx in range(len(panels), nrows * ncols):
        r, c = divmod(idx, ncols)
        axes[r, c].axis("off")

    # Shared colorbar
    fig.subplots_adjust(right=0.92)
    cbar_ax = fig.add_axes([0.94, 0.15, 0.015, 0.7])
    fig.colorbar(im, cax=cbar_ax, label="Per-class recall")

    fig.suptitle("Kepler Q9 v3 — zero-shot linear-probe confusion matrices (row-normalised)",
                 fontsize=12, y=0.995)
    fig.tight_layout(rect=[0, 0, 0.92, 0.97])

    out = Path(args.out) if args.out else results_dir / "confusion_matrices.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=args.dpi, bbox_inches="tight")
    print(f"Wrote {out} ({nrows}x{ncols} panels, "
          f"{sum(n for _, _, n in panels)} folds across {len(panels)} models)")


if __name__ == "__main__":
    main()
