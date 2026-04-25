"""
Plot and summarize cached PhyTS TESS Parquet (histograms, example light curves, time coverage).

Loads ``tess_regression.parquet`` and ``tess_classification.parquet`` from
``data/TESS/.cache/`` (or downloads via Hugging Face Hub).

Dependencies: ``uv sync --extra jax`` (pyarrow, huggingface_hub, matplotlib).

Example
-------
    python data/TESS/visualize_tess.py
"""

from __future__ import annotations

import argparse
import gc
import sys
from collections import Counter
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pyarrow.parquet as pq
from matplotlib.ticker import MaxNLocator

try:
    from huggingface_hub import hf_hub_download
    from huggingface_hub.errors import HfHubHTTPError
except ImportError:  # pragma: no cover
    print("Install: uv sync --extra jax", file=sys.stderr)
    raise SystemExit(1) from None

REPO = "PhyTS-team/PhyTS-bench"
REG_NAME = "tess_regression.parquet"
CLS_NAME = "tess_classification.parquet"
SCRIPT_DIR = Path(__file__).resolve().parent
CACHE = SCRIPT_DIR / ".cache"

N_EXAMPLE_CURVES = 6
RNG_SEED = 42
N_CADENCE_BINS = 1500
COVERAGE_DPI = 150
DT_SAMPLE_ROWS = 800
# Downsampled rasters for ``imshow`` only (logical cadence count unchanged on axes).
COVERAGE_MAX_DISPLAY_H = 2400
COVERAGE_MAX_DISPLAY_W = 1600
COVERAGE_PREVIEW_MAX_H = 900
COVERAGE_PREVIEW_MAX_W = 700


def resolve_parquet(filename: str) -> Path:
    """Return path to ``TESS/<filename>``, downloading from the Hub if missing.

    Parameters
    ----------
    filename
        Basename under the ``TESS/`` folder on the dataset repo.

    Returns
    -------
    Path
        Local Parquet file path.

    Raises
    ------
    SystemExit
        If download fails.
    """
    candidates = [
        CACHE / "TESS" / filename,
        CACHE / filename,
    ]
    for p in candidates:
        if p.is_file():
            return p
    CACHE.mkdir(parents=True, exist_ok=True)
    try:
        return Path(
            hf_hub_download(
                repo_id=REPO,
                repo_type="dataset",
                filename=f"TESS/{filename}",
                local_dir=str(CACHE),
            )
        )
    except (OSError, HfHubHTTPError) as e:
        print(f"Download failed ({filename}): {e}", file=sys.stderr)
        raise SystemExit(1) from e


def estimate_median_dt(
    path: Path,
    rng: np.random.Generator,
    *,
    n_sample_rows: int,
) -> float:
    """Median of per-row median positive time steps over a random row subset.

    Parameters
    ----------
    path
        Parquet file with a ``time`` list column.
    rng
        NumPy random generator (row subsample).
    n_sample_rows
        Number of distinct rows to draw for the estimate (capped by file size).

    Returns
    -------
    float
        ``dt`` in the same time units as the Parquet ``time`` column.
    """
    pf = pq.ParquetFile(path)
    n = pf.metadata.num_rows
    k = min(max(1, n_sample_rows), n)
    pick = set(int(x) for x in rng.choice(n, size=k, replace=False))
    medians: list[float] = []
    seen = 0
    for batch in pf.iter_batches(columns=["time"], batch_size=1024):
        for j in range(batch.num_rows):
            if seen in pick:
                t = np.asarray(batch.column("time")[j].as_py(), dtype=np.float64)
                if t.size >= 2:
                    d = np.diff(np.sort(t))
                    d = d[d > 0]
                    if d.size:
                        medians.append(float(np.median(d)))
            seen += 1
    if not medians:
        return 1e-6
    return float(np.median(medians))


def _plot_lightcurves_grid(
    times_list: list[list[float]],
    fluxes_list: list[list[float]],
    titles: list[str],
    suptitle: str,
    out_path: Path,
    *,
    median_dt: float | None,
) -> None:
    """Plot multiple light curves; NaN flux is shown as red markers at y=0.

    Parameters
    ----------
    times_list, fluxes_list
        Per-panel time and flux sequences (same length within each pair).
    titles
        Subplot titles.
    suptitle
        Figure title (first line).
    out_path
        Where to save the PNG.
    median_dt
        If given, a cadence line is added under the suptitle (median positive Δt
        in the same units as the plotted time axis).
    """
    n = len(times_list)
    ncols = min(3, n)
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(4.2 * ncols, 2.8 * nrows),
        squeeze=False,
        sharex=False,
        sharey=False,
    )
    axes_flat = axes.ravel()
    for ax in axes_flat[n:]:
        ax.set_visible(False)

    for ax, t, f, title in zip(axes_flat, times_list, fluxes_list, titles, strict=True):
        t_arr = np.asarray(t, dtype=np.float64)
        f_arr = np.asarray(f, dtype=np.float64)
        if t_arr.shape != f_arr.shape:
            ax.set_title(f"{title} (length mismatch)")
            continue
        valid = np.isfinite(f_arr)
        nan_f = np.isnan(f_arr)
        if np.any(valid):
            ax.plot(t_arr[valid], f_arr[valid], color="0.2", linewidth=0.8, alpha=0.9)
        if np.any(nan_f):
            ax.scatter(
                t_arr[nan_f],
                np.zeros(np.sum(nan_f)),
                c="red",
                s=8,
                zorder=5,
                label="NaN flux @ 0",
            )
        ax.set_title(title, fontsize=9)
        ax.set_xlabel("time")
        ax.set_ylabel("flux")
        if np.any(nan_f):
            ax.legend(loc="upper right", fontsize=7)

    if median_dt is not None:
        dt_min = median_dt * 24.0 * 60.0
        cadence_line = (
            f"Median sampling step Δt ≈ {median_dt:.6g} (time column units); "
            f"≈ {dt_min:.4g} min if time is days"
        )
        fig.suptitle(f"{suptitle}\n{cadence_line}", fontsize=10)
    else:
        fig.suptitle(suptitle)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _histograms_figure(
    cls_sectors: np.ndarray,
    cls_labels: list[str],
    reg_sectors: np.ndarray,
    frot: np.ndarray,
    out_path: Path,
) -> None:
    """Save a 2x2 figure: classification label/sectors, regression sector/frot.

    Parameters
    ----------
    cls_sectors
        Sector indices for the classification table.
    cls_labels
        String class labels (same length as ``cls_sectors``).
    reg_sectors
        Sector indices for the regression table.
    frot
        Rotation frequencies for the regression table.
    out_path
        Output PNG path.
    """
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))

    ctr = Counter(cls_labels)
    labels_sorted = sorted(ctr.keys(), key=lambda k: (-ctr[k], k))
    counts = [ctr[k] for k in labels_sorted]
    ax = axes[0, 0]
    y_pos = np.arange(len(labels_sorted))
    ax.barh(y_pos, counts, color="steelblue", edgecolor="none", alpha=0.85)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels_sorted, fontsize=8)
    ax.invert_yaxis()
    ax.set_xlabel("Count")
    ax.set_title("Classification: label")

    ax = axes[0, 1]
    ax.hist(cls_sectors, bins="auto", color="coral", edgecolor="white", alpha=0.9)
    ax.set_xlabel("sector")
    ax.set_ylabel("Count")
    ax.set_title("Classification: sector")

    ax = axes[1, 0]
    ax.hist(reg_sectors, bins="auto", color="seagreen", edgecolor="white", alpha=0.9)
    ax.set_xlabel("sector")
    ax.set_ylabel("Count")
    ax.set_title("Regression: sector")

    ax = axes[1, 1]
    frot_f = np.asarray(frot, dtype=np.float64)
    frot_f = frot_f[np.isfinite(frot_f)]
    ax.hist(frot_f, bins="auto", color="mediumpurple", edgecolor="white", alpha=0.9)
    ax.set_xlabel("frot")
    ax.set_ylabel("Count")
    ax.set_title("Regression: frot")

    fig.suptitle("TESS dataset summaries (PhyTS-bench)")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _row_occupancy(
    times: np.ndarray,
    dt: float,
    n_bins: int,
) -> np.ndarray:
    """Binary occupancy on ``n_bins`` cadence slots from first sample time."""
    if times.size == 0 or dt <= 0:
        return np.zeros(n_bins, dtype=np.uint8)
    t0 = float(np.min(times))
    idx = np.floor((times - t0) / dt).astype(np.int64)
    m = (idx >= 0) & (idx < n_bins)
    idx = idx[m]
    occ = np.zeros(n_bins, dtype=np.uint8)
    occ[idx] = 1
    return occ


def _downsample_nn_2d(
    idx: np.ndarray,
    *,
    max_h: int,
    max_w: int,
) -> tuple[np.ndarray, int, int]:
    """Decimate with nearest indices along each axis (``NaN`` preserved).

    The figure still uses the original ``h`` and ``n_bins`` in ``extent``; this
    only reduces array size for Matplotlib memory.
    """
    h, w = idx.shape
    step_r = max(1, int(np.ceil(h / max_h)))
    step_c = max(1, int(np.ceil(w / max_w)))
    out = np.ascontiguousarray(idx[::step_r, ::step_c])
    return out, step_r, step_c


def _save_combined_coverage_figure(
    idx_raster: np.ndarray,
    unique_sectors: list[int],
    title: str,
    out_path: Path,
    *,
    dpi: int,
    n_bins: int,
    h_logical: int,
    cadence_dt: float | None,
    downsample_note: str | None = None,
) -> None:
    """Save sampling raster with a discrete sector colorbar (``nan_counting`` style).

    Uses a scalar field + ``tab20``-resampled colormap and ``plt.colorbar`` with
    per-sector tick labels, matching ``plot_nan_mask_by_sector``. X-axis is in
    cadence-number space via ``extent`` (0 … ``n_bins``).
    """
    n_sec = len(unique_sectors)
    cmap = plt.colormaps["tab20"].resampled(max(n_sec, 2))
    cmap.set_bad(color="black")

    fig, ax_img = plt.subplots(figsize=(14, 10), dpi=dpi, layout="constrained")
    im = ax_img.imshow(
        idx_raster,
        origin="upper",
        aspect="auto",
        interpolation="nearest",
        cmap=cmap,
        vmin=0,
        vmax=max(n_sec - 1, 1),
        extent=(0, n_bins, h_logical, 0),
    )

    extra = f" | {downsample_note}" if downsample_note else ""
    if cadence_dt is not None:
        dt_min = cadence_dt * 24.0 * 60.0
        subtitle = (
            f"Row color = sector, black = no sample in cadence | "
            f"median Δt ≈ {cadence_dt:.6g} (time units); ≈ {dt_min:.4g} min if time is days"
            f"{extra}"
        )
        ax_img.set_title(f"{title}\n{subtitle}", fontsize=14)
    else:
        ax_img.set_title(
            f"{title}\nRow color = sector, black = no sample in cadence{extra}",
            fontsize=14,
        )

    ax_img.set_xlabel("Cadence #", fontsize=12)
    ax_img.set_ylabel(
        "Light curve row (sorted by sector, then segment start time)",
        fontsize=12,
    )
    ax_img.tick_params(axis="both", which="major", labelsize=11)
    ax_img.xaxis.set_major_locator(MaxNLocator(12, integer=True, min_n_ticks=4))
    ax_img.yaxis.set_major_locator(MaxNLocator(10, integer=True, min_n_ticks=4))

    ticks = list(range(n_sec))
    cbar = fig.colorbar(im, ax=ax_img, ticks=ticks)
    cbar.ax.set_yticklabels([f"Sector {s}" for s in unique_sectors])
    cbar.set_label("Sector", fontsize=12)
    cbar.ax.tick_params(labelsize=11)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=dpi)
    plt.close(fig)


def _build_time_coverage_layer(
    path: Path,
    dt: float,
    n_bins: int,
) -> tuple[np.ndarray, list[int], int]:
    """Two-pass scan: sort by (sector, start time) without storing per-row occupancy lists.

    Returns
    -------
    idx_layer, unique_sectors, h_logical
        ``idx_layer`` is ``float32`` with ``NaN`` = empty cadence; sector indices
        0 … n_sec−1 where occupied. ``h_logical`` equals row count.
    """
    pf = pq.ParquetFile(path)
    n = pf.metadata.num_rows
    sectors = np.empty(n, dtype=np.int32)
    t0s = np.empty(n, dtype=np.float64)
    row_i = 0
    for batch in pf.iter_batches(columns=["time", "sector"], batch_size=1024):
        secs = batch.column("sector")
        for j in range(batch.num_rows):
            t = np.asarray(batch.column("time")[j].as_py(), dtype=np.float64)
            sectors[row_i] = int(secs[j].as_py())
            t0s[row_i] = float(np.min(t)) if t.size else 0.0
            row_i += 1

    order = np.lexsort((t0s, sectors))
    inv = np.empty(n, dtype=np.int32)
    inv[order] = np.arange(n, dtype=np.int32)
    unique_sectors = sorted(np.unique(sectors).tolist())
    sector_to_idx = {s: i for i, s in enumerate(unique_sectors)}
    del t0s, order
    gc.collect()

    idx_layer = np.full((n, n_bins), np.nan, dtype=np.float32)
    row_i = 0
    for batch in pf.iter_batches(columns=["time", "sector"], batch_size=1024):
        for j in range(batch.num_rows):
            t = np.asarray(batch.column("time")[j].as_py(), dtype=np.float64)
            occ = _row_occupancy(t, dt, n_bins)
            dest = int(inv[row_i])
            si = float(sector_to_idx[int(sectors[row_i])])
            m = occ.astype(bool)
            idx_layer[dest, m] = si
            row_i += 1

    del sectors, inv
    gc.collect()
    return idx_layer, unique_sectors, n


def _render_time_coverage_dataset(
    path: Path,
    title: str,
    out_path: Path,
    *,
    dt: float,
    n_bins: int,
) -> None:
    """Build full logical grid, downsample for display only, write PNG + smaller preview."""
    idx_layer, unique_sectors, h = _build_time_coverage_layer(path, dt, n_bins)

    idx_disp, sr, sc = _downsample_nn_2d(
        idx_layer,
        max_h=COVERAGE_MAX_DISPLAY_H,
        max_w=COVERAGE_MAX_DISPLAY_W,
    )
    note = f"display downsample ×{sr} rows, ×{sc} cols (full sort: {h} LCs × {n_bins} cadences)"
    _save_combined_coverage_figure(
        idx_disp,
        unique_sectors,
        title,
        out_path,
        dpi=COVERAGE_DPI,
        n_bins=n_bins,
        h_logical=h,
        cadence_dt=dt,
        downsample_note=note,
    )
    del idx_disp
    gc.collect()

    idx_prev, sr_p, sc_p = _downsample_nn_2d(
        idx_layer,
        max_h=COVERAGE_PREVIEW_MAX_H,
        max_w=COVERAGE_PREVIEW_MAX_W,
    )
    del idx_layer
    gc.collect()
    prev_note = f"preview downsample ×{sr_p} rows, ×{sc_p} cols ({h} LCs × {n_bins} cadences)"
    preview_path = out_path.with_name(out_path.stem + "_preview.png")
    _save_combined_coverage_figure(
        idx_prev,
        unique_sectors,
        title + " (preview)",
        preview_path,
        dpi=COVERAGE_DPI,
        n_bins=n_bins,
        h_logical=h,
        cadence_dt=dt,
        downsample_note=prev_note,
    )
    del idx_prev
    gc.collect()


def main() -> None:
    parser = argparse.ArgumentParser(description="Write TESS exploration figures under --out-dir.")
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=SCRIPT_DIR / "figures",
        help="Output directory (default: data/TESS/figures)",
    )
    args = parser.parse_args()
    out_dir = args.out_dir.resolve()

    reg_path = resolve_parquet(REG_NAME)
    cls_path = resolve_parquet(CLS_NAME)

    rng = np.random.default_rng(RNG_SEED)
    rng_dt = np.random.default_rng(RNG_SEED + 31337)
    dt_reg = estimate_median_dt(reg_path, rng_dt, n_sample_rows=DT_SAMPLE_ROWS)
    dt_cls = estimate_median_dt(cls_path, rng_dt, n_sample_rows=DT_SAMPLE_ROWS)

    reg_meta = pq.read_table(reg_path, columns=["sector", "frot"])
    cls_meta = pq.read_table(cls_path, columns=["sector", "label"])
    reg_sectors = reg_meta["sector"].to_numpy(zero_copy_only=False)
    frot_col = reg_meta["frot"].to_numpy(zero_copy_only=False)
    cls_sectors = cls_meta["sector"].to_numpy(zero_copy_only=False)
    cls_labels = cls_meta["label"].to_pylist()
    _histograms_figure(
        cls_sectors,
        cls_labels,
        reg_sectors,
        frot_col,
        out_dir / "tess_histograms.png",
    )

    n_ex = max(1, N_EXAMPLE_CURVES)
    reg_full = pq.read_table(reg_path)
    n_reg = reg_full.num_rows
    pick_reg = rng.choice(n_reg, size=min(n_ex, n_reg), replace=False)
    reg_sub = reg_full.take(pick_reg)
    times_r = reg_sub["time"].to_pylist()
    fluxes_r = reg_sub["flux"].to_pylist()
    sectors_r = reg_sub["sector"].to_pylist()
    tics_r = reg_sub["TIC"].to_pylist()
    titles_r = [
        f"TIC={tic} sector={s} frot={fr:.5g}"
        for tic, s, fr in zip(tics_r, sectors_r, reg_sub["frot"].to_pylist(), strict=True)
    ]
    _plot_lightcurves_grid(
        times_r,
        fluxes_r,
        titles_r,
        "TESS regression — example light curves",
        out_dir / "tess_regression_lightcurves.png",
        median_dt=dt_reg,
    )

    pf_cls = pq.ParquetFile(cls_path)
    n_cls = pf_cls.metadata.num_rows
    pick_cls = rng.choice(n_cls, size=min(n_ex, n_cls), replace=False)
    want: set[int] = {int(i) for i in pick_cls.tolist()}
    by_row: dict[int, tuple[list[float], list[float], str]] = {}
    seen_idx = 0
    for batch in pf_cls.iter_batches(
        columns=["time", "flux", "TIC", "sector", "label"],
        batch_size=2048,
    ):
        for j in range(batch.num_rows):
            if seen_idx in want:
                t_py = batch.column("time")[j].as_py()
                f_py = batch.column("flux")[j].as_py()
                tic = batch.column("TIC")[j].as_py()
                sec = batch.column("sector")[j].as_py()
                lab = batch.column("label")[j].as_py()
                by_row[seen_idx] = (t_py, f_py, f"TIC={tic} sector={sec} label={lab}")
                if len(by_row) == len(want):
                    break
            seen_idx += 1
        if len(by_row) == len(want):
            break

    order_cls = [int(i) for i in pick_cls.tolist()]
    times_c = [by_row[i][0] for i in order_cls]
    fluxes_c = [by_row[i][1] for i in order_cls]
    titles_c = [by_row[i][2] for i in order_cls]

    _plot_lightcurves_grid(
        times_c,
        fluxes_c,
        titles_c,
        "TESS classification — example light curves",
        out_dir / "tess_classification_lightcurves.png",
        median_dt=dt_cls,
    )

    _render_time_coverage_dataset(
        reg_path,
        "TESS regression — sampling on fixed dt grid (colored by sector)",
        out_dir / "tess_regression_time_coverage.png",
        dt=dt_reg,
        n_bins=N_CADENCE_BINS,
    )
    _render_time_coverage_dataset(
        cls_path,
        "TESS classification — sampling on fixed dt grid (colored by sector)",
        out_dir / "tess_classification_time_coverage.png",
        dt=dt_cls,
        n_bins=N_CADENCE_BINS,
    )

    print(f"Figures written to {out_dir}")


if __name__ == "__main__":
    main()
