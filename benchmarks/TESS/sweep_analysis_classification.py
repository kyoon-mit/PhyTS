#!/usr/bin/env python3
"""Summarize finished TESS *classification* wandb sweeps **after all sweeps** finish.

Loads runs from **one or many** sweep IDs (typically one YAML sweep per architecture),
merges them, and builds the full architecture × width-tier picture.

Writes:

    * Two heatmaps: mean **`test/acc`** and mean **`test_balanced`** (macro over
      logged ``test/acc_class_*``).
    * ``all_runs.csv``, ``top_k_summary.csv``, ``best_hyperparameters.json``.
    * ``report.html``.

Ranking within each plot cell uses ``val/balanced_acc`` (classification sweep tuning
metric). Aggregated metrics still suffer selection bias; use for exploration.

Examples
--------

    uv run python benchmarks/TESS/analyze_classification_sweep.py \\
        --sweep_id abc123 \\
        --sweep_id def456 \\
        ...

    uv run python benchmarks/TESS/analyze_classification_sweep.py \\
        --from_file benchmarks/TESS/sweep_ids_classification.txt \\
        --out_dir benchmarks/TESS/sweep_reports/my_cls_summary

Requires ``wandb`` login / ``WANDB_API_KEY``.
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


RANK_KEY = "val/balanced_acc"


def _macro_test_balanced(summary: Any, max_classes: int = 16) -> float | None:
    vals: list[float] = []
    for c in range(max_classes):
        v = summary_get(summary, f"test/acc_class_{c}")
        if v is not None:
            vals.append(v)
    if not vals:
        return None
    return sum(vals) / len(vals)


@dataclass
class RunRow:
    run_id: str
    source_sweep_id: str
    model_type: str
    state: str
    val_balanced_acc: float | None
    test_acc: float | None
    test_balanced: float | None
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
        test_acc = summary_get(summary, "test/acc")
        rank_val = summary_get(summary, RANK_KEY)
        if rank_val is None:
            skipped["missing_val_balanced_acc"] += 1
            continue
        if test_acc is None:
            skipped["missing_test_acc"] += 1
            continue

        tb = _macro_test_balanced(summary)
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
                val_balanced_acc=rank_val,
                test_acc=test_acc,
                test_balanced=tb,
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
        sweep_path = f"{entity}/{project}/{sid}"
        sweep = api.sweep(sweep_path)
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
                key=lambda rr: rr.val_balanced_acc if rr.val_balanced_acc is not None else float("-inf"),
                reverse=True,
            )
            vals: list[float] = []
            for r in group_sorted:
                if key == "test_acc" and r.test_acc is not None:
                    vals.append(r.test_acc)
                elif key == "test_balanced" and r.test_balanced is not None:
                    vals.append(r.test_balanced)
                if len(vals) >= top_k:
                    break
            out[(mk, sz)] = float(np.mean(vals)) if vals else None

    return out


def _write_all_runs_csv(path: Path, rows: list[RunRow]) -> None:
    fieldnames = (
        ["run_id", "source_sweep_id", "model_type", RANK_KEY, "test/acc", "test_balanced"]
        + list(SWEEP_HPARAM_KEYS)
        + ["wandb_url"]
    )

    path.parent.mkdir(parents=True, exist_ok=True)
    rows_sorted = sorted(
        rows,
        key=lambda r: (
            r.model_type,
            str(r.hyperparams.get("size", "")),
            -(r.val_balanced_acc or -1),
        ),
    )

    with path.open("w", newline="", encoding="utf-8") as fp:
        w = csv.DictWriter(fp, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        for r in rows_sorted:
            entry: dict[str, Any] = {
                "run_id": r.run_id,
                "source_sweep_id": r.source_sweep_id,
                "model_type": r.model_type,
                RANK_KEY: r.val_balanced_acc,
                "test/acc": r.test_acc,
                "test_balanced": r.test_balanced if r.test_balanced is not None else "",
            }
            for k in SWEEP_HPARAM_KEYS:
                entry[k] = r.hyperparams.get(k, "")
            entry["wandb_url"] = r.wandb_url or ""
            w.writerow(entry)


def _coverage_text(rows: list[RunRow], sweep_ids: list[str]) -> str:
    lines: list[str] = []

    sources = Counter(r.source_sweep_id for r in rows)
    lines.append(f"Merged sweep IDs ({len(sweep_ids)}): {', '.join(sweep_ids)}")
    lines.append("Runs harvested per upstream sweep:")
    for sid in sweep_ids:
        lines.append(f"  {sid}: {sources.get(sid, 0)}")
    lines.append(f"Total unique finished runs (val + test): {len(rows)}")

    mt_counts = Counter(r.model_type for r in rows)
    lines.append("Runs per model_type:")
    for m in MODEL_ORDER:
        lines.append(f"  {m}: {mt_counts.get(m, 0)}")

    for key in ("batch_size", "size", "seed"):
        c = Counter(r.hyperparams.get(key, None) for r in rows)
        if len(c) == 0:
            continue
        lines.append(f"{key} distribution: {dict(c)}")

    lrs = [float(r.hyperparams["lr"]) for r in rows if isinstance(r.hyperparams.get("lr"), (int, float))]
    if lrs:
        lines.append(f"lr range (observed): {min(lrs):.6g} … {max(lrs):.6g}")

    wds = [
        float(r.hyperparams["weight_decay"]) for r in rows if isinstance(r.hyperparams.get("weight_decay"), (int, float))
    ]
    if wds:
        lines.append(f"weight_decay range (observed): {min(wds):.6g} … {max(wds):.6g}")

    dps = [float(r.hyperparams["dropout"]) for r in rows if isinstance(r.hyperparams.get("dropout"), (int, float))]
    if dps:
        lines.append(f"dropout range (observed): {min(dps):.4g} … {max(dps):.4g}")

    tb_ct = sum(1 for r in rows if r.test_balanced is not None)
    lines.append(f"Runs with computed test_balanced: {tb_ct}/{len(rows)}")

    return "\n".join(lines) + "\n"


def _row_to_hp_dict(row: RunRow) -> dict[str, Any]:
    return {
        "run_id": row.run_id,
        "source_sweep_id": row.source_sweep_id,
        "model_type": row.model_type,
        RANK_KEY: row.val_balanced_acc,
        "test/acc": row.test_acc,
        "test_balanced": row.test_balanced,
        **row.hyperparams,
        "wandb_url": row.wandb_url,
    }


def _best_hyperparams_payload(rows: list[RunRow]) -> dict[str, Any]:
    if not rows:
        return {}

    glob = max(rows, key=lambda r: r.val_balanced_acc or float("-inf"))

    cells: dict[str, dict[str, Any]] = {}
    for mk in MODEL_ORDER:
        for sz in SIZE_ORDER:
            group = [rr for rr in rows if rr.model_type == mk and str(rr.hyperparams.get("size")) == sz]
            if not group:
                continue
            winner = max(group, key=lambda r: r.val_balanced_acc or float("-inf"))
            cells[f"{mk}/{sz}"] = _row_to_hp_dict(winner)

    return {"global_best_by_val_balanced_acc": _row_to_hp_dict(glob), "best_per_model_type_and_size": cells}


def _topk_table_csv(
    path: Path,
    test_acc_map: dict[tuple[str, str], float | None],
    test_bal_map: dict[tuple[str, str], float | None],
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
                f"mean_test_acc_top_{top_k}",
                f"mean_test_balanced_top_{top_k}",
            ]
        )
        for mk in MODEL_ORDER:
            for sz in SIZE_ORDER:
                a = test_acc_map.get((mk, sz))
                b = test_bal_map.get((mk, sz))
                w.writerow(
                    [
                        mk,
                        sz,
                        "" if a is None else f"{a:.6f}",
                        "" if b is None else f"{b:.6f}",
                    ]
                )


def _write_simple_html(report_path: Path, sweep_ids: list[str], top_k: int, fragments: dict[str, str]) -> None:
    title_ids = html.escape(", ".join(sweep_ids))
    parts = [
        "<!DOCTYPE html>",
        '<html lang="en"><meta charset="utf-8"/>',
        f"<head><title>TESS classification merges ({title_ids})</title></head><body>",
        "<h1>TESS classification — merged sweeps</h1>",
        f"<p><code>sweep_ids</code> ({len(sweep_ids)}): <code>{title_ids}</code></p>",
        f"<p>Top-{top_k} averaging keyed on <code>{html.escape(RANK_KEY)}</code></p>",
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
    repo_reports = Path(__file__).resolve().parents[2] / "benchmarks/TESS/sweep_reports/classification_all_sweeps"
    p = argparse.ArgumentParser(description=__doc__)
    add_common_wandb_args(p)
    add_sweep_id_args(p)
    p.add_argument(
        "--out_dir",
        type=Path,
        default=None,
        help=f"report directory (default: {repo_reports})",
    )
    p.add_argument("--top_k", type=int, default=5)
    args = p.parse_args()
    if args.out_dir is None:
        args.out_dir = repo_reports
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
        coverage += "\nDropped wandb trials (combined):\n" + "\n".join(
            f"  {k}: {v}" for k, v in sorted(skipped.items())
        )

    test_acc_map = _group_topk_metric(rows, key="test_acc", top_k=args.top_k)
    test_bal_map = _group_topk_metric(rows, key="test_balanced", top_k=args.top_k)

    vals_acc = [v for v in test_acc_map.values() if v is not None]
    vals_bal = [v for v in test_bal_map.values() if v is not None]

    vmin_acc, vmax_acc = (0.0, 1.0) if not vals_acc else auto_vmin_vmax(vals_acc)
    vmin_bal, vmax_bal = (0.0, 1.0) if not vals_bal else auto_vmin_vmax(vals_bal)

    def _fmt_acc(v: float) -> str:
        return f"{v:.4f}"

    subtitle = f"first {args.top_k} eligible runs ranked by {RANK_KEY}; color spans table min/max"
    masked_a, txt_a = build_matrix(test_acc_map, fmt_cell=_fmt_acc)
    masked_b, txt_b = build_matrix(test_bal_map, fmt_cell=_fmt_acc)

    png_acc = out_root / f"heatmap_test_acc_top{args.top_k}.png"
    png_bal = out_root / f"heatmap_test_balanced_top{args.top_k}.png"
    plot_heatmap(
        "Mean test/acc",
        subtitle,
        masked_a,
        txt_a,
        png_acc,
        model_order=list(MODEL_ORDER),
        size_order=list(SIZE_ORDER),
        vmin=vmin_acc,
        vmax=vmax_acc,
        cmap="RdYlGn",
        colorbar_label="accuracy",
    )
    plot_heatmap(
        "Mean test_balanced (macro class acc)",
        subtitle,
        masked_b,
        txt_b,
        png_bal,
        model_order=list(MODEL_ORDER),
        size_order=list(SIZE_ORDER),
        vmin=vmin_bal,
        vmax=vmax_bal,
        cmap="RdYlGn",
        colorbar_label="balanced acc",
    )

    _write_all_runs_csv(out_root / "all_runs.csv", rows)
    _topk_table_csv(out_root / "top_k_summary.csv", test_acc_map, test_bal_map, top_k=args.top_k)

    best_payload = _best_hyperparams_payload(rows)
    best_payload["classification_sweep_ids"] = sweep_ids
    best_payload["wandb_entity"] = entity
    best_payload["wandb_project"] = args.project
    best_payload["top_k_for_cell_averages"] = args.top_k
    best_payload["rank_key_for_top_k_selection"] = RANK_KEY

    def _json_default(o: object) -> object:
        if isinstance(o, np.generic):
            return o.item()
        raise TypeError

    with (out_root / "best_hyperparameters.json").open("w", encoding="utf-8") as fp:
        json.dump(best_payload, fp, indent=2, default=_json_default)

    fig_html = (
        f"<h2>Top-{args.top_k} mean test metrics</h2>"
        f'<p><img src="{html.escape(png_acc.name)}" alt="heatmap test acc" style="max-width:100%"/></p>'
        f'<p><img src="{html.escape(png_bal.name)}" alt="heatmap balanced test" style="max-width:100%"/></p>'
    )
    report_path = out_root / "report.html"
    _write_simple_html(report_path, sweep_ids, args.top_k, {"coverage": coverage, "figures": fig_html})

    print(f"Wrote report to {report_path}")
    print(f"Artifacts directory: {out_root}")
    if skipped:
        print("Dropped wandb trials (combined):")
        for k, v in sorted(skipped.items()):
            print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
