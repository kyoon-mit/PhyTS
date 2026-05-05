#!/usr/bin/env python3
"""Summarize finished TESS *regression* wandb sweeps after all sweeps finish.

Merges runs from multiple sweep IDs into one architecture × size view and reports:

    * Mean **``test/r2``** (higher is better, green in default colormap).
    * Mean **``test/rmse``** (lower is better, uses reversed colormap).

Within each ``(model_type, size)`` cell runs are ranked by ``val/r2``—the sweep
objective in ``sweep_reg_*.yaml``—then averaged over the first ``k`` runs having
the reported test metric.

Writes ``all_runs.csv``, ``top_k_summary.csv``, ``best_hyperparameters.json``,
heatmap PNGs, and ``report.html``.

Examples
--------

    uv run python benchmarks/TESS/analyze_regression_sweep.py \\
        --sweep_id aaa111 --sweep_id bbb222 ...

    uv run python benchmarks/TESS/analyze_regression_sweep.py \\
        --from_file benchmarks/TESS/sweep_ids_regression.txt \\
        --out_dir benchmarks/TESS/sweep_reports/my_reg_summary
"""

from __future__ import annotations

import argparse
import csv
import html
import json
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import numpy as np
import wandb

from sweep_aggregate_utils import (
    MODEL_ORDER,
    SIZE_ORDER,
    SWEEP_HPARAM_KEYS,
    add_common_wandb_args,
    add_sweep_id_args,
    auto_vmin_vmax,
    build_matrix,
    load_sweep_id_list,
    plot_heatmap,
    resolve_wandb_entity,
    summary_get,
)


RANK_KEY = "val/r2"


@dataclass
class RunRow:
    run_id: str
    source_sweep_id: str
    model_type: str
    state: str
    val_r2: float | None
    test_r2: float | None
    test_rmse: float | None
    hyperparams: dict[str, Any]
    wandb_url: str | None


def _collect_rows_one_sweep(
    *,
    sweep: Any,
    source_sweep_id: str,
    project_name: str,
    entity_str: str,
) -> tuple[list[RunRow], Counter[str]]:
    skipped: Counter[str] = Counter()
    rows: list[RunRow] = []

    for run in sweep.runs:
        if run.state != "finished":
            skipped[f"not_finished:{run.state}"] += 1
            continue

        cfg = dict(run.config) if isinstance(run.config, dict) else {}
        model_type = cfg.get("model_type")
        if model_type not in MODEL_ORDER:
            skipped["bad_or_missing_model_type"] += 1
            continue

        summary = run.summary
        val_r2 = summary_get(summary, RANK_KEY)
        test_r2 = summary_get(summary, "test/r2")
        if val_r2 is None:
            skipped["missing_val_r2"] += 1
            continue
        if test_r2 is None:
            skipped["missing_test_r2"] += 1
            continue

        test_rmse = summary_get(summary, "test/rmse")
        hp = {k: cfg.get(k) for k in SWEEP_HPARAM_KEYS}
        url = getattr(run, "url", None) or (
            f"https://wandb.ai/{entity_str}/{project_name}/runs/{run.id}"
            if getattr(run, "id", None)
            else None
        )

        rows.append(
            RunRow(
                run_id=str(run.id),
                source_sweep_id=source_sweep_id,
                model_type=str(model_type),
                state=str(run.state),
                val_r2=val_r2,
                test_r2=test_r2,
                test_rmse=test_rmse,
                hyperparams=hp,
                wandb_url=url,
            )
        )

    return rows, skipped


def _collect_merged_rows(
    api: wandb.Api,
    *,
    entity: str,
    project: str,
    sweep_ids: list[str],
) -> tuple[list[RunRow], Counter[str]]:
    skipped_all: Counter[str] = Counter()
    merged: list[RunRow] = []
    seen_rid: set[str] = set()

    for sid in sweep_ids:
        sweep = api.sweep(f"{entity}/{project}/{sid}")
        chunk, skipped = _collect_rows_one_sweep(
            sweep=sweep,
            source_sweep_id=sid,
            project_name=project,
            entity_str=entity,
        )
        skipped_all.update(skipped)
        for r in chunk:
            if r.run_id in seen_rid:
                skipped_all[f"duplicate_run:{r.run_id}"] += 1
                continue
            seen_rid.add(r.run_id)
            merged.append(r)

    return merged, skipped_all


def _group_topk_metric(
    rows: list[RunRow],
    *,
    key: str,
    top_k: int,
) -> dict[tuple[str, str], float | None]:
    buckets: defaultdict[tuple[str, str], list[RunRow]] = defaultdict(list)
    for r in rows:
        size = str(r.hyperparams.get("size", ""))
        buckets[(r.model_type, size)].append(r)

    out: dict[tuple[str, str], float | None] = {}
    for mk in MODEL_ORDER:
        for sz in SIZE_ORDER:
            group = buckets.get((mk, sz), [])
            if not group:
                out[(mk, sz)] = None
                continue
            group_sorted = sorted(
                group,
                key=lambda rr: rr.val_r2 if rr.val_r2 is not None else float("-inf"),
                reverse=True,
            )
            vals: list[float] = []
            for r in group_sorted:
                if key == "test_r2" and r.test_r2 is not None:
                    vals.append(r.test_r2)
                elif key == "test_rmse" and r.test_rmse is not None:
                    vals.append(r.test_rmse)
                if len(vals) >= top_k:
                    break
            out[(mk, sz)] = float(np.mean(vals)) if vals else None

    return out


def _write_all_runs_csv(path: Path, rows: list[RunRow]) -> None:
    fieldnames = ["run_id", "source_sweep_id", "model_type", RANK_KEY, "test/r2", "test/rmse"] + list(
        SWEEP_HPARAM_KEYS
    ) + ["wandb_url"]

    path.parent.mkdir(parents=True, exist_ok=True)
    rows_sorted = sorted(
        rows,
        key=lambda r: (r.model_type, str(r.hyperparams.get("size", "")), -(r.val_r2 or -1e18)),
    )

    with path.open("w", newline="", encoding="utf-8") as fp:
        w = csv.DictWriter(fp, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        for r in rows_sorted:
            entry: dict[str, Any] = {
                "run_id": r.run_id,
                "source_sweep_id": r.source_sweep_id,
                "model_type": r.model_type,
                RANK_KEY: r.val_r2,
                "test/r2": r.test_r2,
                "test/rmse": r.test_rmse if r.test_rmse is not None else "",
            }
            for k in SWEEP_HPARAM_KEYS:
                entry[k] = r.hyperparams.get(k, "")
            entry["wandb_url"] = r.wandb_url or ""
            w.writerow(entry)


def _coverage_text(rows: list[RunRow], sweep_ids: list[str]) -> str:
    sources = Counter(r.source_sweep_id for r in rows)
    lines = [
        f"Merged sweep IDs ({len(sweep_ids)}): {', '.join(sweep_ids)}",
        "Runs harvested per upstream sweep:",
    ]
    for sid in sweep_ids:
        lines.append(f"  {sid}: {sources.get(sid, 0)}")
    lines.append(f"Total unique finished runs (with val + test r²): {len(rows)}")

    mt_counts = Counter(r.model_type for r in rows)
    lines.append("Runs per model_type:")
    for m in MODEL_ORDER:
        lines.append(f"  {m}: {mt_counts.get(m, 0)}")

    for key in ("batch_size", "size", "seed"):
        c = Counter(r.hyperparams.get(key, None) for r in rows)
        if len(c):
            lines.append(f"{key} distribution: {dict(c)}")

    r2s = [r.test_r2 for r in rows if r.test_r2 is not None]
    if r2s:
        lines.append(f"Observed test/r2 range: {min(r2s):.6g} … {max(r2s):.6g}")

    rm_list = [float(r.test_rmse) for r in rows if r.test_rmse is not None]
    if rm_list:
        lines.append(f"Observed test/rmse range: {min(rm_list):.6g} … {max(rm_list):.6g}")

    return "\n".join(lines) + "\n"


def _row_to_hp_dict(row: RunRow) -> dict[str, Any]:
    return {
        "run_id": row.run_id,
        "source_sweep_id": row.source_sweep_id,
        "model_type": row.model_type,
        RANK_KEY: row.val_r2,
        "test/r2": row.test_r2,
        "test/rmse": row.test_rmse,
        **row.hyperparams,
        "wandb_url": row.wandb_url,
    }


def _best_hyperparams_payload(rows: list[RunRow]) -> dict[str, Any]:
    if not rows:
        return {}
    glob = max(rows, key=lambda r: r.val_r2 if r.val_r2 is not None else float("-inf"))
    cells: dict[str, dict[str, Any]] = {}
    for mk in MODEL_ORDER:
        for sz in SIZE_ORDER:
            group = [rr for rr in rows if rr.model_type == mk and str(rr.hyperparams.get("size")) == sz]
            if not group:
                continue
            winner = max(group, key=lambda r: r.val_r2 if r.val_r2 is not None else float("-inf"))
            cells[f"{mk}/{sz}"] = _row_to_hp_dict(winner)
    return {"global_best_by_val_r2": _row_to_hp_dict(glob), "best_per_model_type_and_size": cells}


def _topk_table_csv(
    path: Path,
    r2_map: dict[tuple[str, str], float | None],
    rmse_map: dict[tuple[str, str], float | None],
    *,
    top_k: int,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fp:
        w = csv.writer(fp)
        w.writerow(
            [
                "model_type",
                "size",
                f"mean_test_r2_top_{top_k}",
                f"mean_test_rmse_top_{top_k}",
            ]
        )
        for mk in MODEL_ORDER:
            for sz in SIZE_ORDER:
                rr = r2_map.get((mk, sz))
                mm = rmse_map.get((mk, sz))
                w.writerow(
                    [
                        mk,
                        sz,
                        "" if rr is None else f"{rr:.6f}",
                        "" if mm is None else f"{mm:.6g}",
                    ]
                )


def _write_simple_html(report_path: Path, sweep_ids: list[str], top_k: int, fragments: dict[str, str]) -> None:
    title_ids = html.escape(", ".join(sweep_ids))
    parts = [
        "<!DOCTYPE html>",
        '<html lang="en"><meta charset="utf-8"/>',
        f"<head><title>TESS regression merges ({title_ids})</title></head><body>",
        "<h1>TESS regression — merged sweeps</h1>",
        f"<p><code>sweep_ids</code> ({len(sweep_ids)}): <code>{title_ids}</code></p>",
        f"<p>Top-{top_k}, ranked by <code>{html.escape(RANK_KEY)}</code>.</p>",
        fragments.get("figures", ""),
        "<h2>Coverage summary</h2>",
        f"<pre>{html.escape(fragments['coverage'])}</pre>",
        "<h2>Artifact files</h2><ul>",
        "<li><code>all_runs.csv</code></li>",
        "<li><code>best_hyperparameters.json</code></li>",
        "<li><code>top_k_summary.csv</code></li>",
        "</ul></body></html>",
    ]
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(parts), encoding="utf-8")


def _parse_args() -> argparse.Namespace:
    default_out = Path(__file__).resolve().parents[2] / "benchmarks/TESS/sweep_reports/regression_all_sweeps"
    p = argparse.ArgumentParser(description=__doc__)
    add_common_wandb_args(p)
    add_sweep_id_args(p)
    p.add_argument(
        "--out_dir",
        type=Path,
        default=None,
        help=f"report directory (default: {default_out})",
    )
    p.add_argument("--top_k", type=int, default=5)
    args = p.parse_args()
    if args.out_dir is None:
        args.out_dir = default_out
    return args


def main() -> None:
    args = _parse_args()
    sweep_ids = load_sweep_id_list(args.sweep_id, args.from_file)

    api = wandb.Api()
    entity = resolve_wandb_entity(api, args.entity)
    out_root = Path(args.out_dir).expanduser().resolve()
    out_root.mkdir(parents=True, exist_ok=True)

    rows, skipped = _collect_merged_rows(api, entity=entity, project=args.project, sweep_ids=sweep_ids)

    coverage = _coverage_text(rows, sweep_ids)
    if skipped:
        coverage += "\nDropped wandb trials:\n" + "\n".join(f"  {k}: {v}" for k, v in sorted(skipped.items()))

    r2_map = _group_topk_metric(rows, key="test_r2", top_k=args.top_k)
    rm_map = _group_topk_metric(rows, key="test_rmse", top_k=args.top_k)

    vals_r2 = [v for v in r2_map.values() if v is not None]
    vals_rmse = [v for v in rm_map.values() if v is not None]

    vmin_r2, vmax_r2 = auto_vmin_vmax(vals_r2) if vals_r2 else (-1.0, 1.0)
    vmin_rm, vmax_rm = auto_vmin_vmax(vals_rmse) if vals_rmse else (0.0, 1.0)

    def _fmt_r2(v: float) -> str:
        return f"{v:.4f}"

    def _fmt_rmse(v: float) -> str:
        return f"{v:.5g}"

    subtitle = (
        f"first {args.top_k} ranked runs having each metric ({RANK_KEY} desc); color spans table extrema"
    )
    masked_r2, txt_r2 = build_matrix(r2_map, fmt_cell=_fmt_r2)
    masked_rmse, txt_rmse = build_matrix(rm_map, fmt_cell=_fmt_rmse)

    png_r2 = out_root / f"heatmap_test_r2_top{args.top_k}.png"
    png_rmse = out_root / f"heatmap_test_rmse_top{args.top_k}.png"

    plot_heatmap(
        "Mean test/R² — higher warmer/greener where colormap ramps up",
        subtitle,
        masked_r2,
        txt_r2,
        png_r2,
        model_order=list(MODEL_ORDER),
        size_order=list(SIZE_ORDER),
        vmin=vmin_r2,
        vmax=vmax_r2,
        cmap="RdYlGn",
        colorbar_label=r"$R^2$",
    )
    plot_heatmap(
        "Mean test RMSE — lower greener (reversed cmap)",
        subtitle,
        masked_rmse,
        txt_rmse,
        png_rmse,
        model_order=list(MODEL_ORDER),
        size_order=list(SIZE_ORDER),
        vmin=vmin_rm,
        vmax=vmax_rm,
        cmap="RdYlGn_r",
        colorbar_label="RMSE",
    )

    _write_all_runs_csv(out_root / "all_runs.csv", rows)
    _topk_table_csv(out_root / "top_k_summary.csv", r2_map, rm_map, top_k=args.top_k)

    payload = _best_hyperparams_payload(rows)
    payload["regression_sweep_ids"] = sweep_ids
    payload["wandb_entity"] = entity
    payload["wandb_project"] = args.project
    payload["top_k_for_cell_averages"] = args.top_k
    payload["rank_key_for_top_k_selection"] = RANK_KEY

    def _json_default(o: object) -> object:
        if isinstance(o, np.generic):
            return o.item()
        raise TypeError

    with (out_root / "best_hyperparameters.json").open("w", encoding="utf-8") as fp:
        json.dump(payload, fp, indent=2, default=_json_default)

    fig_html = (
        f"<h2>Top-{args.top_k} aggregates</h2>"
        f'<p><img src="{html.escape(png_r2.name)}" alt="test r2" style="max-width:100%"/></p>'
        f'<p><img src="{html.escape(png_rmse.name)}" alt="test rmse" style="max-width:100%"/></p>'
    )
    _write_simple_html(out_root / "report.html", sweep_ids, args.top_k, {"coverage": coverage, "figures": fig_html})

    print(f"Wrote report to {out_root / 'report.html'}")
    print(f"Artifacts directory: {out_root}")
    if skipped:
        print("Dropped wandb trials:")
        for k, v in sorted(skipped.items()):
            print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
