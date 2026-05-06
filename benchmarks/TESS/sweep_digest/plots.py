"""White→blue heatmaps (one metric per figure)."""

from __future__ import annotations

from pathlib import Path
from typing import Callable, Sequence

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import Colormap, LinearSegmentedColormap, Normalize

WHITE_BLUE = LinearSegmentedColormap.from_list(
    "WhiteBlue",
    ["#ffffff", "#bdd7e7", "#2171b5", "#08306b"],
)


def build_matrix(
    values: dict[tuple[str, str], float | None],
    *,
    model_order: Sequence[str],
    size_order: Sequence[str],
    fmt_cell: Callable[[float], str],
) -> tuple[np.ma.MaskedArray, list[list[str]]]:
    n_m, n_s = len(model_order), len(size_order)
    arr = np.full((n_m, n_s), np.nan, dtype=float)
    txt: list[list[str]] = [["" for _ in range(n_s)] for _ in range(n_m)]
    for i, m in enumerate(model_order):
        for j, s in enumerate(size_order):
            v = values.get((m, s))
            if v is None:
                continue
            fv = float(v)
            if not np.isfinite(fv):
                continue
            arr[i, j] = fv
            txt[i][j] = fmt_cell(fv)
    return np.ma.masked_invalid(arr), txt


def build_column_matrix(
    values: dict[str, float | None],
    *,
    model_order: Sequence[str],
    fmt_cell: Callable[[float], str],
) -> tuple[np.ma.MaskedArray, list[list[str]]]:
    n_m = len(model_order)
    arr = np.full((n_m, 1), np.nan, dtype=float)
    txt: list[list[str]] = [[""] for _ in range(n_m)]
    for i, m in enumerate(model_order):
        v = values.get(m)
        if v is None:
            continue
        fv = float(v)
        if not np.isfinite(fv):
            continue
        arr[i, 0] = fv
        txt[i][0] = fmt_cell(fv)
    return np.ma.masked_invalid(arr), txt


def _cell_annotation_color(cmap: Colormap, norm: Normalize, value: float) -> str:
    """Black on light cells, white on dark blue (same mapping as ``imshow``)."""
    if not np.isfinite(value):
        return "black"
    t = float(norm(value))
    t = max(0.0, min(1.0, t))
    r, g, b, _ = cmap(t)
    # sRGB relative luminance (good enough for text contrast on our ramps)
    L = 0.2126 * r + 0.7152 * g + 0.0722 * b
    return "white" if L < 0.45 else "black"


def _cmap_and_norm(
    data: np.ma.MaskedArray, *, higher_is_better: bool
) -> tuple[Colormap, Normalize]:
    """WHITE_BLUE: white=weak, dark=blue=strong. For metrics where lower is better, reverse cmap."""
    valid = np.ma.compressed(data)
    if valid.size == 0:
        return WHITE_BLUE, Normalize(0.0, 1.0)
    lo, hi = float(valid.min()), float(valid.max())
    if not np.isfinite(lo) or not np.isfinite(hi):
        return WHITE_BLUE, Normalize(0.0, 1.0)
    if lo == hi:
        lo, hi = lo - 1.0, hi + 1.0
    norm = Normalize(vmin=lo, vmax=hi)
    cmap = WHITE_BLUE if higher_is_better else WHITE_BLUE.reversed()
    return cmap, norm


def plot_heatmap_2d(
    *,
    title: str,
    data: np.ma.MaskedArray,
    cell_text: list[list[str]],
    out_path: Path,
    yticklabels: Sequence[str],
    xticklabels: Sequence[str],
    higher_is_better: bool,
    colorbar_label: str,
    dpi: int = 175,
) -> None:
    title_fs = 13
    tick_fs = 10
    cell_fs = 10
    cbar_fs = 10
    if np.ma.count(data) == 0:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig, ax = plt.subplots(figsize=(5.2, 4.0))
        ax.text(0.5, 0.5, "no data", ha="center", va="center", fontsize=12)
        ax.set_title(title, fontsize=title_fs, fontweight="600")
        fig.savefig(out_path, dpi=dpi)
        plt.close(fig)
        return

    w = max(6.0, 1.15 * len(xticklabels))
    h = max(5.0, 0.58 * len(yticklabels))
    fig, ax = plt.subplots(figsize=(w, h))
    cmap, nrm = _cmap_and_norm(data, higher_is_better=higher_is_better)
    im = ax.imshow(data, cmap=cmap, aspect="auto", norm=nrm)
    ax.set_xticks(np.arange(len(xticklabels)), labels=list(xticklabels), fontsize=tick_fs)
    ax.set_yticks(np.arange(len(yticklabels)), labels=list(yticklabels), fontsize=tick_fs)
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label(colorbar_label, fontsize=cbar_fs)
    cbar.ax.tick_params(labelsize=tick_fs - 1)
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            if data.mask[i, j]:
                continue
            t = cell_text[i][j] if i < len(cell_text) and j < len(cell_text[i]) else ""
            if t:
                v = float(data[i, j])
                tx_col = _cell_annotation_color(cmap, nrm, v)
                ax.text(j, i, t, ha="center", va="center", color=tx_col, fontsize=cell_fs)
    ax.set_title(title, fontsize=title_fs, fontweight="600", pad=12)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=dpi)
    plt.close(fig)
