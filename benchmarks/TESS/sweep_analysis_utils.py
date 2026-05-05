"""Shared helpers for wandb sweep post-analysis (classification + regression)."""

from __future__ import annotations

from typing import Any, Callable, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import wandb


MODEL_ORDER: tuple[str, ...] = (
    "mlp",
    "s4d",
    "cnn",
    "cnn_attn",
    "transformer",
    "linoss_imex",
    "linoss_damped",
)
SIZE_ORDER: tuple[str, ...] = ("xs", "sm", "md", "lg")
SWEEP_HPARAM_KEYS: tuple[str, ...] = ("size", "lr", "dropout", "weight_decay", "batch_size", "seed")


def summary_get(summary: Any, key: str) -> float | None:
    """Read a scalar from run.summary (supports nested dicts from some clients)."""
    if summary is None:
        return None
    if key in summary:
        v = summary[key]
        if v is None:
            return None
        try:
            return float(v)
        except (TypeError, ValueError):
            return None
    if hasattr(summary, "get"):
        parts = key.split("/")
        cur: Any = summary
        for p in parts:
            if cur is None or not isinstance(cur, dict) or p not in cur:
                return None
            cur = cur[p]
        try:
            return float(cur)
        except (TypeError, ValueError):
            return None
    return None


def build_matrix(
    metric_map: dict[tuple[str, str], float | None],
    *,
    fmt_cell: Callable[[float], str],
    model_order: Sequence[str] = MODEL_ORDER,
    size_order: Sequence[str] = SIZE_ORDER,
) -> tuple[np.ma.MaskedArray, list[list[str]]]:
    """Build float matrix + string labels for matplotlib ``imshow``."""
    mat = np.full((len(model_order), len(size_order)), np.nan, dtype=float)
    text = [["" for _ in size_order] for _ in model_order]

    for i, mk in enumerate(model_order):
        for j, sz in enumerate(size_order):
            v = metric_map.get((mk, sz))
            if v is None:
                continue
            mat[i, j] = v
            text[i][j] = fmt_cell(v)

    masked = np.ma.masked_invalid(mat)
    return masked, text


def plot_heatmap(
    title: str,
    subtitle: str,
    masked_mat: np.ma.MaskedArray,
    cell_text: list[list[str]],
    out_png: Path,
    *,
    model_order: Sequence[str],
    size_order: Sequence[str],
    vmin: float,
    vmax: float,
    cmap: str,
    colorbar_label: str,
) -> None:
    fig, ax = plt.subplots(figsize=(7.5, 6.8))
    im = ax.imshow(masked_mat, cmap=cmap, aspect="auto", vmin=vmin, vmax=vmax)
    ax.set_xticks(range(masked_mat.shape[1]))
    ax.set_xticklabels(list(size_order))
    ax.set_yticks(range(masked_mat.shape[0]))
    ax.set_yticklabels(list(model_order))

    ax.set_title(f"{title}\n{subtitle}", fontsize=11)
    for i in range(masked_mat.shape[0]):
        for j in range(masked_mat.shape[1]):
            txt = "" if masked_mat.mask[i, j] else cell_text[i][j]
            ax.text(
                j,
                i,
                txt,
                ha="center",
                va="center",
                color="black" if masked_mat.mask[i, j] else "#111",
                fontsize=10,
                fontweight="bold",
            )
    plt.colorbar(im, ax=ax, shrink=0.7, label=colorbar_label)
    fig.tight_layout()
    fig.savefig(out_png, dpi=150)
    plt.close(fig)


def auto_vmin_vmax(vals: list[float]) -> tuple[float, float]:
    """Stretch color scale when all cell values are nearly equal."""
    if not vals:
        return 0.0, 1.0
    vmin = float(min(vals))
    vmax = float(max(vals))
    if vmax - vmin < 1e-6:
        vmin -= 1e-3
        vmax += 1e-3
    return vmin, vmax


def resolve_wandb_entity(api: wandb.Api, explicit: str | None) -> str:
    entity = explicit or os.environ.get("WANDB_ENTITY")
    if entity is None:
        viewer = getattr(api, "viewer", None)
        entity = getattr(viewer, "entity", None) if viewer else None
    if not entity:
        raise SystemExit(
            "Could not determine wandb entity; pass --entity or set WANDB_ENTITY / login with wandb."
        )
    return entity


def load_sweep_id_list(sweep_ids: list[str] | None, from_file: Path | None) -> list[str]:
    """Merge sweep ids from CLI and optional file (one id per line, ``#`` comments)."""
    out: list[str] = []
    if from_file is not None:
        text = from_file.read_text(encoding="utf-8")
        for line in text.splitlines():
            s = line.split("#", 1)[0].strip()
            if s:
                out.append(s)
    if sweep_ids:
        for s in sweep_ids:
            t = s.strip()
            if t:
                out.append(t)
    if not out:
        raise SystemExit("Provide one or more --sweep_id values and/or --from_file pointing to a newline list.")

    seen: set[str] = set()
    deduped: list[str] = []
    for sid in out:
        if sid in seen:
            continue
        seen.add(sid)
        deduped.append(sid)
    return deduped


def add_common_wandb_args(p: Any) -> None:
    """Register shared entity / project flags."""
    p.add_argument("--entity", default=None, help="WANDB_ENTITY (defaults to env var or ~/.netrc viewer)")
    p.add_argument("--project", default="TimeSeriesPhysics", help="wandb project name")


def add_sweep_id_args(p: Any) -> None:
    """Register ``--sweep_id`` repeat + ``--from_file``."""
    p.add_argument(
        "--sweep_id",
        "-s",
        action="append",
        default=None,
        metavar="ID",
        help="wandb sweep id (repeat for each sweep; include one per architecture/task batch)",
    )
    p.add_argument(
        "--from_file",
        type=Path,
        default=None,
        help="newline-separated sweep ids (``#`` line comments allowed)",
    )
