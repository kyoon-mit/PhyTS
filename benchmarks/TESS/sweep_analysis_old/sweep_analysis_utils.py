"""Shared helpers for personal (non-paper) TESS sweep summaries.

Training artifacts from agents live under :func:`sweep_utils.tess_sweep_artifact_dir`
(``logs/tess_sweeps/...`` by default). These reports default to
``benchmarks/TESS/sweep_reports/personal/...``.
"""

from __future__ import annotations

import csv
import html
import json
from collections import Counter
from pathlib import Path
from typing import Any, Callable, Iterable, Sequence

import numpy as np

# Documented hardware assumption for merged personal sweep reports (Engaging L40S pool).
ASSUMED_GPU_LABEL = "NVIDIA L40S"

# Selection policy (documented in report notes): no test leakage for "top" sets.
REGRESSION_VAL_SELECTION_LINE = (
    "Within each (model_type, size) cell, runs are ordered by validation loss: prefer val/MAE "
    "(lower is better); if missing (e.g. some LinOSS runs), fall back to val/RMSE then val/R²."
)
CLASSIFICATION_VAL_SELECTION_LINE = (
    "Within each cell, runs are ordered by val/acc (higher is better), not balanced accuracy."
)

HPARAM_AGGREGATION_LINE = (
    "Per cell, after taking the first top_k runs under the validation selection rule, each "
    "hyperparameter is summarized as: median (and min/max) when all top_k values parse as numeric; "
    "otherwise plurality mode (ties broken arbitrarily). This follows common HPO reporting "
    "(robust summary of the good region) rather than a single-gridsearch 'winner'."
)

# Numeric sweep keys to plot as heatmaps (exclude size — fixed per column; seed uses mode only).
HPARAM_HEATMAP_NUMERIC_KEYS: tuple[str, ...] = ("lr", "dropout", "weight_decay", "batch_size")


def regression_val_sort_key(run: Any) -> tuple[int, float]:
    """Sort ascending: best runs first (MAE, then RMSE, then R² via negation)."""
    mae = getattr(run, "val_mae", None)
    if mae is not None:
        return (0, float(mae))
    rmse = getattr(run, "val_rmse", None)
    if rmse is not None:
        return (1, float(rmse))
    r2 = getattr(run, "val_r2", None)
    if r2 is not None:
        return (2, -float(r2))
    return (9, 0.0)


def classification_val_acc_desc_key(run: Any) -> float:
    """Higher val/acc first when sorting with reverse=True."""
    v = getattr(run, "val_acc", None)
    return float(v) if v is not None else float("-inf")


def _try_float_hyperparam(x: Any) -> float | None:
    if x is None:
        return None
    if isinstance(x, bool):
        return None
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


def aggregate_hyperparams_for_runs(
    runs: Sequence[Any],
    hp_keys: Sequence[str],
) -> dict[str, Any]:
    """Summarize hyperparameters over ``runs`` (typically top-k after val selection).

    Returns
    -------
    dict
        Maps each key in ``hp_keys`` to a dict with ``rule``, ``value`` (float or str),
        counts, and diagnostic fields.
    """
    out: dict[str, Any] = {}
    for k in hp_keys:
        raw = [getattr(r, "hyperparams", {}).get(k) for r in runs]
        present = [v for v in raw if v is not None and str(v).strip() != ""]
        floats: list[float] = []
        all_numeric = bool(present)
        for v in present:
            f = _try_float_hyperparam(v)
            if f is None:
                all_numeric = False
                break
            floats.append(f)
        if all_numeric and present:
            arr = np.array(floats, dtype=float)
            out[k] = {
                "rule": "median_numeric",
                "value": float(np.median(arr)),
                "min": float(np.min(arr)),
                "max": float(np.max(arr)),
                "n": len(floats),
            }
        else:
            c = Counter(str(v) for v in present)
            if not c:
                out[k] = {"rule": "empty", "value": None, "n": 0}
            else:
                mode, sup = c.most_common(1)[0]
                out[k] = {
                    "rule": "plurality_mode",
                    "value": mode,
                    "support": int(sup),
                    "n": len(present),
                }
    return out


def numeric_value_for_hparam_heatmap(hp_key: str, summary: dict[str, Any]) -> float | None:
    """Single float per cell for heatmap; ``lr`` is log10-scaled for color stretch."""
    if summary.get("rule") != "median_numeric":
        return None
    v = summary.get("value")
    if v is None:
        return None
    x = float(v)
    if hp_key == "lr":
        return float(np.log10(max(x, 1e-12)))
    return x


def hyperparam_cell_display(nk: str, summary_cell: dict[str, Any]) -> str:
    """Human-readable representative hyperparameter value for grids / heatmap overlays."""
    rule = summary_cell.get("rule")
    if rule == "median_numeric" and summary_cell.get("value") is not None:
        v = float(summary_cell["value"])
        if nk == "batch_size":
            return f"{v:.0f}"
        return f"{v:.4g}"
    if rule == "plurality_mode" and summary_cell.get("value") is not None:
        return str(summary_cell["value"])[:10]
    return ""


def extract_runtime_s(run: Any) -> float | None:
    s = getattr(run, "summary", None) or {}
    if not isinstance(s, dict) or "_runtime" not in s:
        return None
    try:
        return float(s["_runtime"])
    except (TypeError, ValueError):
        return None


def total_params_from_summary(summary: Any) -> int | None:
    """Single scalar param count: torch ``param_count_torch_nn`` or JAX tallies."""
    if summary is None:
        return None
    get = _fget(summary)
    tn = get("param_count_torch_nn")
    if tn is not None:
        return int(tn)
    jm = get("param_count_jax_model_float")
    if jm is None:
        return None
    tot = int(jm)
    js = get("param_count_jax_state_arrays")
    if js is not None:
        tot += int(js)
    return tot


def _fget(summary: Any) -> Callable[[str], float | None]:
    from sweep_analysis_utils import summary_get

    def get(key: str) -> float | None:
        return summary_get(summary, key)

    return get


def first_summary_float(summary: Any, keys: Sequence[str]) -> float | None:
    from sweep_analysis_utils import summary_get

    for k in keys:
        v = summary_get(summary, k)
        if v is not None:
            return v
    return None


def macro_recall_from_class_accs(summary: Any, *, max_classes: int = 16) -> float | None:
    """Mean of ``test/acc_class_*`` (per-class recall). Matches ``test/balanced_acc`` when all classes appear."""
    from sweep_analysis_utils import summary_get

    vals: list[float] = []
    for c in range(max_classes):
        v = summary_get(summary, f"test/acc_class_{c}")
        if v is not None:
            vals.append(float(v))
    if not vals:
        return None
    return float(sum(vals) / len(vals))


def topk_metric_stats(
    top_runs: Sequence[Any],
    metric_getters: dict[str, Callable[[Any], float | None]],
    *,
    higher_is_better: dict[str, bool],
) -> dict[str, dict[str, float | int | None]]:
    """For each metric: best (by polarity), mean and std over non-null values in top_runs."""
    out: dict[str, dict[str, float | int | None]] = {}
    for name, getter in metric_getters.items():
        raw = [getter(r) for r in top_runs]
        vals = [float(v) for v in raw if v is not None]
        hi = higher_is_better.get(name, True)
        if not vals:
            out[name] = {"best": None, "mean": None, "std": None, "n": 0}
            continue
        best = max(vals) if hi else min(vals)
        out[name] = {
            "best": best,
            "mean": float(np.mean(vals)),
            "std": float(np.std(vals)) if len(vals) > 1 else 0.0,
            "n": len(vals),
        }
    return out


def bucket_by_model_size(
    rows: Iterable[Any],
    *,
    model_key: Callable[[Any], str],
    size_key: Callable[[Any], str],
) -> dict[tuple[str, str], list[Any]]:
    buckets: dict[tuple[str, str], list[Any]] = {}
    for r in rows:
        mk = model_key(r)
        sz = size_key(r)
        buckets.setdefault((mk, sz), []).append(r)
    return buckets


def write_csv(path: Path, fieldnames: Sequence[str], rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fp:
        w = csv.DictWriter(fp, fieldnames=list(fieldnames), extrasaction="ignore")
        w.writeheader()
        for row in rows:
            w.writerow(row)


def write_md_metric_grid(
    path: Path,
    *,
    title: str,
    cell_text: dict[tuple[str, str], str],
    model_order: Sequence[str],
    size_order: Sequence[str],
    size_column_headers: Sequence[str] | None = None,
) -> None:
    """Markdown table: rows = model_type, columns = size tiers. Cell strings pre-formatted."""
    path.parent.mkdir(parents=True, exist_ok=True)
    headers = list(size_column_headers) if size_column_headers is not None else list(size_order)
    if len(headers) != len(size_order):
        raise ValueError("size_column_headers must align with size_order length")
    header = "| model_type | " + " | ".join(headers) + " |\n"
    sep = "| " + " | ".join(["---"] * (1 + len(size_order))) + " |\n"
    body: list[str] = []
    for m in model_order:
        cells = [cell_text.get((m, s), "") for s in size_order]
        body.append("| " + m + " | " + " | ".join(cells) + " |\n")
    text = f"## {title}\n\n" + header + sep + "".join(body) + "\n"
    path.write_text(text, encoding="utf-8")


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=False) + "\n", encoding="utf-8")


def write_minimal_sweep_report_html(
    path: Path,
    *,
    page_title: str,
    h1: str,
    meta_paragraphs: Sequence[str],
    coverage_pre: str,
    relative_pngs: Sequence[tuple[str, str]],
    artifact_links: Sequence[tuple[str, str]],
) -> None:
    """Write a small index page: figures, links to CSV/MD/JSON, and coverage.

    Parameters
    ----------
    relative_pngs
        ``(href relative to this HTML file, caption)`` for each ``<img>``.
    artifact_links
        ``(href, link label)`` for the artifacts list.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    esc = html.escape
    parts: list[str] = [
        "<!DOCTYPE html>",
        '<html lang="en"><head><meta charset="utf-8"/>',
        f"<title>{esc(page_title)}</title></head><body>",
        f"<h1>{esc(h1)}</h1>",
    ]
    for para in meta_paragraphs:
        parts.append(f"<p>{esc(para)}</p>")
    parts.append("<h2>Figures (mean over top-k per cell)</h2>")
    for href, caption in relative_pngs:
        parts.append(f"<h3>{esc(caption)}</h3>")
        parts.append(
            f'<p><img src="{esc(href)}" alt="{esc(caption)}" style="max-width:100%"/></p>'
        )
    parts.append("<h2>Artifacts</h2><ul>")
    for href, label in artifact_links:
        parts.append(f'<li><a href="{esc(href)}">{esc(label)}</a></li>')
    parts.append("</ul>")
    parts.append("<h2>Coverage</h2>")
    parts.append(f"<pre>{esc(coverage_pre)}</pre>")
    parts.append("</body></html>")
    path.write_text("\n".join(parts), encoding="utf-8")
