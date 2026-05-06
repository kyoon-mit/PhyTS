#!/usr/bin/env python3
"""Personal TESS *regression* sweep digest (not intended for the paper).

Within each ``(model_type, size)`` cell, finished runs are sorted by **validation
only**, to avoid optimistic bias from picking on the test set: prefer ``val/mae``
(lower is better); if unavailable (some LinOSS runs), fall back to ``val/rmse``
then ``val/r2``. The first ``--top_k`` runs under that ordering define means/stds;
**reported scores** use the held-out test metrics for those runs.

**Compute:** parameter counts from ``wandb.summary``. FLOPs are not logged.

**Artifacts:** training logs under ``logs/tess_sweeps/regression/...`` via
:func:`sweep_utils.tess_sweep_artifact_dir`. This script defaults to
``benchmarks/TESS/sweep_reports/personal/regression``. Outputs include Markdown
grids, **PNG heatmaps** (``png_mean_topk/``; RMSE/MAE use a reversed colormap),
and ``report.html``.

Examples
--------
    uv run python benchmarks/TESS/sweep_personal_regression.py \\
        --sweep_id abc --sweep_id def

    uv run python benchmarks/TESS/sweep_personal_regression.py \\
        --from_file benchmarks/TESS/sweep_ids_regression.txt --top_k 5
"""

from __future__ import annotations

import argparse
import os
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

_BENCH = Path(__file__).resolve().parent
if str(_BENCH) not in sys.path:
    sys.path.insert(0, str(_BENCH))

import wandb
from tqdm.auto import tqdm

from sweep_analysis_utils import (
    HEATMAP_SIZE_LABELS,
    MODEL_ORDER,
    SIZE_ORDER,
    SWEEP_HPARAM_KEYS,
    add_common_wandb_args,
    add_sweep_id_args,
    auto_vmin_vmax,
    build_matrix,
    iter_sweep_runs,
    load_sweep_id_list,
    plot_heatmap,
    resolve_wandb_entity,
    summary_get,
)
from sweep_personal_benchmark_common import (
    ASSUMED_GPU_LABEL,
    HPARAM_AGGREGATION_LINE,
    HPARAM_HEATMAP_NUMERIC_KEYS,
    REGRESSION_VAL_SELECTION_LINE,
    aggregate_hyperparams_for_runs,
    bucket_by_model_size,
    extract_runtime_s,
    hyperparam_cell_display,
    numeric_value_for_hparam_heatmap,
    regression_val_sort_key,
    topk_metric_stats,
    total_params_from_summary,
    write_csv,
    write_json,
    write_md_metric_grid,
    write_minimal_sweep_report_html,
)


@dataclass
class RunRow:
    run_id: str
    source_sweep_id: str
    model_type: str
    size: str
    hyperparams: dict[str, Any]
    wandb_url: str | None
    val_r2: float | None
    val_mae: float | None
    val_rmse: float | None
    test_r2: float | None
    test_mae: float | None
    test_rmse: float | None
    param_total: int | None
    runtime_s: float | None


def _collect_one_sweep(
    *,
    sweep: Any,
    source_sweep_id: str,
    project_name: str,
    entity_str: str,
) -> tuple[list[RunRow], Counter[str]]:
    skipped: Counter[str] = Counter()
    rows: list[RunRow] = []

    for run in iter_sweep_runs(sweep, desc=f"Sweep {source_sweep_id}"):
        if run.state != "finished":
            skipped[f"not_finished:{run.state}"] += 1
            continue

        cfg = dict(run.config) if isinstance(run.config, dict) else {}
        model_type = cfg.get("model_type")
        if model_type not in MODEL_ORDER:
            skipped["bad_or_missing_model_type"] += 1
            continue

        size = str(cfg.get("size", ""))
        if size not in SIZE_ORDER:
            skipped["bad_or_missing_size"] += 1
            continue

        summary = run.summary
        val_mae = summary_get(summary, "val/mae")
        val_rmse = summary_get(summary, "val/rmse")
        val_r2 = summary_get(summary, "val/r2")
        if val_mae is None and val_rmse is None and val_r2 is None:
            skipped["missing_val_for_selection"] += 1
            continue

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
                size=size,
                hyperparams=hp,
                wandb_url=url,
                val_r2=val_r2,
                val_mae=val_mae,
                val_rmse=val_rmse,
                test_r2=summary_get(summary, "test/r2"),
                test_mae=summary_get(summary, "test/mae"),
                test_rmse=summary_get(summary, "test/rmse"),
                param_total=total_params_from_summary(summary),
                runtime_s=extract_runtime_s(run),
            )
        )

    return rows, skipped


def _merge_sweeps(
    api: wandb.Api,
    *,
    entity: str,
    project: str,
    sweep_ids: list[str],
) -> tuple[list[RunRow], Counter[str]]:
    skipped_all: Counter[str] = Counter()
    merged: list[RunRow] = []
    seen_rid: set[str] = set()

    for sid in tqdm(sweep_ids, desc="Wandb sweeps", unit="sweep", dynamic_ncols=True):
        sweep = api.sweep(f"{entity}/{project}/{sid}")
        chunk, skipped = _collect_one_sweep(
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


def _parse() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Personal TESS regression sweep benchmark tables")
    add_common_wandb_args(p)
    add_sweep_id_args(p)
    p.add_argument(
        "--out_dir",
        type=Path,
        default=Path("benchmarks/TESS/sweep_reports/personal/regression"),
        help="Output directory for CSV/MD/JSON",
    )
    p.add_argument("--top_k", type=int, default=3)
    p.add_argument(
        "--ckpt_root",
        type=Path,
        default=None,
        help="Sweep checkpoints root (regression/{{model}}/{{run_id}}/); defaults to TESS_CKPT_DIR",
    )
    p.add_argument(
        "--data_dir",
        type=Path,
        default=None,
        help="TESS parquet root; defaults to TESS_DATA_DIR",
    )
    p.add_argument("--seq_len", type=int, default=1100)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument(
        "--save_winner_plots",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Save test-set scatter/residual PNGs per cell winner (needs best.ckpt / best.eqx on disk)",
    )
    p.add_argument(
        "--plot_device",
        default="auto",
        choices=["auto", "cpu", "cuda"],
        help="Device for winner-plot inference",
    )
    return p.parse_args()


def main() -> None:
    args = _parse()
    sweep_ids = load_sweep_id_list(args.sweep_id, args.from_file)
    api = wandb.Api()
    entity = resolve_wandb_entity(api, args.entity)

    rows, skipped = _merge_sweeps(api, entity=entity, project=args.project, sweep_ids=sweep_ids)
    out_dir = args.out_dir

    rank_caption = "val/MAE (else val/RMSE, else val/R²)"

    fields = (
        [
            "run_id",
            "source_sweep_id",
            "model_type",
            "size",
            "val/r2",
            "val/mae",
            "val/rmse",
            "test/r2",
            "test/mae",
            "test/rmse",
            "param_total",
            "runtime_s",
        ]
        + list(SWEEP_HPARAM_KEYS)
        + ["wandb_url"]
    )
    all_rows: list[dict[str, Any]] = []
    for r in sorted(
        rows,
        key=lambda x: (x.model_type, x.size, regression_val_sort_key(x)),
    ):
        all_rows.append(
            {
                "run_id": r.run_id,
                "source_sweep_id": r.source_sweep_id,
                "model_type": r.model_type,
                "size": r.size,
                "val/r2": r.val_r2 if r.val_r2 is not None else "",
                "val/mae": r.val_mae if r.val_mae is not None else "",
                "val/rmse": r.val_rmse if r.val_rmse is not None else "",
                "test/r2": r.test_r2 if r.test_r2 is not None else "",
                "test/mae": r.test_mae if r.test_mae is not None else "",
                "test/rmse": r.test_rmse if r.test_rmse is not None else "",
                "param_total": r.param_total if r.param_total is not None else "",
                "runtime_s": r.runtime_s if r.runtime_s is not None else "",
                **{k: r.hyperparams.get(k, "") for k in SWEEP_HPARAM_KEYS},
                "wandb_url": r.wandb_url or "",
            }
        )
    write_csv(out_dir / "all_runs.csv", fields, all_rows)

    buckets = bucket_by_model_size(rows, model_key=lambda r: r.model_type, size_key=lambda r: r.size)

    metric_getters = {
        "val/r2": lambda r: r.val_r2,
        "val/mae": lambda r: r.val_mae,
        "val/rmse": lambda r: r.val_rmse,
        "test/r2": lambda r: r.test_r2,
        "test/mae": lambda r: r.test_mae,
        "test/rmse": lambda r: r.test_rmse,
        "param_total": lambda r: float(r.param_total) if r.param_total is not None else None,
        "runtime_s": lambda r: r.runtime_s,
    }
    higher = {k: True for k in metric_getters}
    higher["val/mae"] = False
    higher["val/rmse"] = False
    higher["test/mae"] = False
    higher["test/rmse"] = False
    higher["runtime_s"] = False

    metric_names = [
        "val/r2",
        "val/mae",
        "val/rmse",
        "test/r2",
        "test/mae",
        "test/rmse",
        "param_total",
        "runtime_s",
    ]

    by_cell: list[dict[str, Any]] = []
    best_hp: dict[str, Any] = {}
    hp_by_cell: dict[str, Any] = {}
    grid_mean: dict[str, dict[tuple[str, str], str]] = {name: {} for name in metric_names}
    grid_best: dict[str, dict[tuple[str, str], str]] = {name: {} for name in metric_names}

    png_metrics = ("test/r2", "test/rmse", "test/mae")
    png_mean_maps: dict[str, dict[tuple[str, str], float | None]] = {m: {} for m in png_metrics}
    png_runtime_map: dict[tuple[str, str], float | None] = {}

    grids_hparam: dict[str, dict[tuple[str, str], str]] = {
        k: {} for k in SWEEP_HPARAM_KEYS if k != "size"
    }
    winners_by_cell: dict[tuple[str, str], str] = {}

    for mk in MODEL_ORDER:
        for sz in SIZE_ORDER:
            cell = (mk, sz)
            group = buckets.get(cell, [])
            sorted_g = sorted(group, key=regression_val_sort_key)
            top = sorted_g[: max(1, args.top_k)]
            st = topk_metric_stats(top, metric_getters, higher_is_better=higher)
            row_out: dict[str, Any] = {
                "model_type": mk,
                "size": sz,
                "n_runs_in_cell": len(group),
                "top_k_used": min(len(top), args.top_k),
            }
            for mn in metric_names:
                s = st[mn]
                row_out[f"{mn}__best"] = s["best"]
                row_out[f"{mn}__mean_topk"] = s["mean"]
                row_out[f"{mn}__std_topk"] = s["std"]
                row_out[f"{mn}__n_scored_topk"] = s["n"]
                if s["mean"] is not None:
                    grid_mean[mn][cell] = f"{s['mean']:.6g}"
                else:
                    grid_mean[mn][cell] = ""
                if s["best"] is not None:
                    grid_best[mn][cell] = f"{s['best']:.6g}"
                else:
                    grid_best[mn][cell] = ""

            for pm in png_metrics:
                png_mean_maps[pm][cell] = st[pm]["mean"]
            png_runtime_map[cell] = st["runtime_s"]["mean"]

            hk = f"{mk}/{sz}"
            agg_hp = aggregate_hyperparams_for_runs(top, SWEEP_HPARAM_KEYS)
            hp_by_cell[hk] = agg_hp
            for hk_inner in grids_hparam:
                grids_hparam[hk_inner][cell] = hyperparam_cell_display(hk_inner, agg_hp.get(hk_inner, {}))

            by_cell.append(row_out)

            if sorted_g:
                winner = sorted_g[0]
                winners_by_cell[(mk, sz)] = winner.run_id
                best_hp[hk] = {
                    "run_id": winner.run_id,
                    "source_sweep_id": winner.source_sweep_id,
                    "wandb_url": winner.wandb_url,
                    **winner.hyperparams,
                    "param_total": winner.param_total,
                    "val/r2": winner.val_r2,
                    "val/mae": winner.val_mae,
                    "val/rmse": winner.val_rmse,
                    "test/mae": winner.test_mae,
                    "test/r2": winner.test_r2,
                    "runtime_s": winner.runtime_s,
                    "selection_rule": rank_caption,
                }
                best_hp[hk]["topk_hyperparam_snapshot"] = [
                    {
                        **r.hyperparams,
                        "run_id": r.run_id,
                        "val/mae": r.val_mae,
                        "val/rmse": r.val_rmse,
                        "val/r2": r.val_r2,
                    }
                    for r in top
                ]

    stat_fields = ["model_type", "size", "n_runs_in_cell", "top_k_used"] + [
        f"{mn}__{suffix}"
        for mn in metric_names
        for suffix in ("best", "mean_topk", "std_topk", "n_scored_topk")
    ]
    write_csv(out_dir / "by_cell_stats.csv", stat_fields, by_cell)

    grids_dir = out_dir / "grids_mean_topk"
    grids_bdir = out_dir / "grids_best_topk"
    for mn in metric_names:
        safe = mn.replace("/", "_")
        write_md_metric_grid(
            grids_dir / f"{safe}.md",
            title=f"{mn} — mean over top-{args.top_k} ({rank_caption})",
            cell_text=grid_mean[mn],
            model_order=MODEL_ORDER,
            size_order=SIZE_ORDER,
            size_column_headers=HEATMAP_SIZE_LABELS,
        )
        write_md_metric_grid(
            grids_bdir / f"{safe}.md",
            title=f"{mn} — best among top-{args.top_k} ({rank_caption})",
            cell_text=grid_best[mn],
            model_order=MODEL_ORDER,
            size_order=SIZE_ORDER,
            size_column_headers=HEATMAP_SIZE_LABELS,
        )

    grids_hp_dir = out_dir / "grids_hyperparams"
    grids_hp_dir.mkdir(parents=True, exist_ok=True)
    for hk in grids_hparam:
        write_md_metric_grid(
            grids_hp_dir / f"{hk}_representative.md",
            title=f"{hk} — representative over val-top-{args.top_k}",
            cell_text=grids_hparam[hk],
            model_order=MODEL_ORDER,
            size_order=SIZE_ORDER,
            size_column_headers=HEATMAP_SIZE_LABELS,
        )

    write_json(out_dir / "by_cell_hyperparams.json", hp_by_cell)

    png_dir = out_dir / "png_mean_topk"
    png_dir.mkdir(parents=True, exist_ok=True)
    heat_subtitle = (
        f"Runs sorted by {rank_caption}; mean over first {args.top_k} per cell "
        "(test metrics are evaluation-only; colors span table min/max)"
    )
    relative_pngs: list[tuple[str, str]] = []

    def _heat_png(
        m: dict[tuple[str, str], float | None],
        fname: str,
        title: str,
        cbl: str,
        cmap: str,
        vmin: float,
        vmax: float,
        fmt: Callable[[float], str],
        caption_suffix: str,
    ) -> None:
        masked, txt = build_matrix(m, fmt_cell=fmt)
        plot_heatmap(
            title,
            heat_subtitle + "; " + caption_suffix,
            masked,
            txt,
            png_dir / fname,
            model_order=list(MODEL_ORDER),
            size_order=list(SIZE_ORDER),
            size_xticklabels=HEATMAP_SIZE_LABELS,
            vmin=vmin,
            vmax=vmax,
            cmap=cmap,
            colorbar_label=cbl,
        )
        relative_pngs.append((f"png_mean_topk/{fname}", title))

    r2_map = png_mean_maps["test/r2"]
    vals_r2 = [float(v) for v in r2_map.values() if v is not None]
    vmin_r2, vmax_r2 = (-1.0, 1.0) if not vals_r2 else auto_vmin_vmax(vals_r2)
    _heat_png(
        r2_map,
        f"heatmap_mean_test_r2_top{args.top_k}.png",
        r"Mean test $R^2$ — higher is greener",
        r"$R^2$",
        "RdYlGn",
        vmin_r2,
        vmax_r2,
        lambda v: f"{float(v):.4f}",
        "held-out test (models chosen on val)",
    )

    rmse_map = png_mean_maps["test/rmse"]
    vals_rm = [float(v) for v in rmse_map.values() if v is not None]
    vmin_rm, vmax_rm = (0.0, 1.0) if not vals_rm else auto_vmin_vmax(vals_rm)
    _heat_png(
        rmse_map,
        f"heatmap_mean_test_rmse_top{args.top_k}.png",
        "Mean test RMSE — lower is greener (reversed colormap)",
        "RMSE",
        "RdYlGn_r",
        vmin_rm,
        vmax_rm,
        lambda v: f"{float(v):.5g}",
        "held-out test",
    )

    mae_map = png_mean_maps["test/mae"]
    vals_mae = [float(v) for v in mae_map.values() if v is not None]
    vmin_mae, vmax_mae = (0.0, 1.0) if not vals_mae else auto_vmin_vmax(vals_mae)
    _heat_png(
        mae_map,
        f"heatmap_mean_test_mae_top{args.top_k}.png",
        "Mean test MAE — lower is greener (reversed colormap)",
        "MAE",
        "RdYlGn_r",
        vmin_mae,
        vmax_mae,
        lambda v: f"{float(v):.5g}",
        "held-out test",
    )

    vals_rt = [float(v) for v in png_runtime_map.values() if v is not None]
    vmin_rt, vmax_rt = (0.0, 3600.0) if not vals_rt else auto_vmin_vmax(vals_rt)
    _heat_png(
        png_runtime_map,
        f"heatmap_mean_runtime_s_top{args.top_k}.png",
        "Mean wall time (s), val-top-k — lower is greener",
        "seconds",
        "RdYlGn_r",
        vmin_rt,
        vmax_rt,
        lambda v: f"{float(v):.5g}",
        "wandb _runtime averaged over val-top-k",
    )

    for nk in HPARAM_HEATMAP_NUMERIC_KEYS:
        color_map: dict[tuple[str, str], float | None] = {}
        overlay: dict[tuple[str, str], str] = {}
        for mk in MODEL_ORDER:
            for sz in SIZE_ORDER:
                hk = f"{mk}/{sz}"
                cell = (mk, sz)
                s_hp = hp_by_cell.get(hk, {}).get(nk, {})
                color_map[cell] = numeric_value_for_hparam_heatmap(nk, s_hp)
                overlay[cell] = hyperparam_cell_display(nk, s_hp)

        masked, _ = build_matrix(color_map, fmt_cell=lambda _: "")
        txt_grid: list[list[str]] = []
        for mk in MODEL_ORDER:
            row_txt: list[str] = []
            for sz in SIZE_ORDER:
                row_txt.append(overlay[(mk, sz)])
            txt_grid.append(row_txt)

        vals_c = [float(v) for v in color_map.values() if v is not None]
        if not vals_c:
            continue
        vmin_c, vmax_c = auto_vmin_vmax(vals_c)
        safe_k = nk.replace("/", "_")
        lr_note = "; color=log10(lr)" if nk == "lr" else ""
        png_n = f"heatmap_repr_{safe_k}_top{args.top_k}.png"
        plot_heatmap(
            f"Representative {nk} (median over val-top-{args.top_k}){lr_note}",
            heat_subtitle,
            masked,
            txt_grid,
            png_dir / png_n,
            model_order=list(MODEL_ORDER),
            size_order=list(SIZE_ORDER),
            size_xticklabels=HEATMAP_SIZE_LABELS,
            vmin=vmin_c,
            vmax=vmax_c,
            cmap="viridis",
            colorbar_label="log10(lr)" if nk == "lr" else nk.replace("_", " "),
        )
        relative_pngs.append((f"png_mean_topk/{png_n}", f"hparam median {nk}"))

    methodology_txt = "\n".join(
        [
            REGRESSION_VAL_SELECTION_LINE,
            "",
            "Test-set columns and heatmaps show metrics only on held-out test data for the runs in each cell's "
            "val-top-k set (means/stds computed on test for those trials). This avoids selecting hyperparameters "
            "by test performance.",
            "",
            HPARAM_AGGREGATION_LINE,
            "",
            "LinOSS/JAX runs often omit val/MAE on wandb summary; ranking falls back to val/RMSE then val/R².",
            "",
            f"Hardware assumption: all jobs ran on {ASSUMED_GPU_LABEL}.",
            "",
            "Runtime is wandb `_runtime` averaged over the same top-k trials.",
        ]
    )
    (out_dir / "methodology_personal_sweeps.txt").write_text(methodology_txt + "\n", encoding="utf-8")

    compact_lines = [
        "# Regression — compact summary\n",
        "**Selection (per cell):** ",
        rank_caption,
        f"\nTop-k aggregates: **{args.top_k}**. Table values are mean test-set metrics among those trials.\n\n",
        "| model | size | n | mean val-top-k test MAE | mean val-top-k test R² | mean runtime (s) |\n",
        "| --- | --- | ---: | ---: | ---: | ---: |\n",
    ]
    for mk in MODEL_ORDER:
        for sz in SIZE_ORDER:
            cell = (mk, sz)
            group = buckets.get(cell, [])
            if not group:
                continue
            sorted_g = sorted(group, key=regression_val_sort_key)
            top = sorted_g[: max(1, args.top_k)]
            st = topk_metric_stats(top, metric_getters, higher_is_better=higher)
            mm = st["test/mae"]["mean"]
            tr = st["test/r2"]["mean"]
            rt = st["runtime_s"]["mean"]
            compact_lines.append(
                f"| {mk} | {sz} | {len(group)} | {mm if mm is not None else ''} | "
                f"{tr if tr is not None else ''} | {rt if rt is not None else ''} |\n"
            )
    (out_dir / "summary_compact.md").write_text("".join(compact_lines), encoding="utf-8")

    write_json(out_dir / "best_hyperparameters_by_cell.json", best_hp)

    (out_dir / "note_metrics.txt").write_text(
        "FLOPs are not logged; use param_total and optional runtime_s for rough compute.\n"
        "LinOSS regression often omits val/mae on summary; ranking uses val/rmse or val/r2 as fallback.\n"
        f"Hardware: summaries assume all jobs ran on {ASSUMED_GPU_LABEL}.\n"
        "See methodology_personal_sweeps.txt for validation vs test aggregation and hyperparameter summaries.\n"
        "Per-cell test-set scatter plots (eval_plots layout): top_model_test_plots/regression/ "
        "(use --no-save_winner_plots to skip).\n",
        encoding="utf-8",
    )

    cell_plot_lines: list[str] = []
    if args.save_winner_plots and winners_by_cell:
        from sweep_personal_top_plots import save_cell_winner_figures

        ckpt_r = args.ckpt_root or Path(os.environ.get("TESS_CKPT_DIR", "checkpoints/sweeps"))
        data_r = str(
            (args.data_dir or Path(os.environ.get("TESS_DATA_DIR", "data/TESS/.cache/TESS"))).resolve()
        )
        cell_plot_lines = save_cell_winner_figures(
            api,
            entity=entity,
            project=args.project,
            task_kind="regression",
            winners=winners_by_cell,
            ckpt_root=ckpt_r.resolve(),
            data_dir=data_r,
            seq_len=args.seq_len,
            num_workers=args.num_workers,
            inference_device=args.plot_device,
            out_dir=out_dir,
            split="test",
        )

    cov = [
        f"Merged sweep IDs ({len(sweep_ids)}): {', '.join(sweep_ids)}\n",
        f"Total runs kept: {len(rows)}\n",
        "Skipped counts:\n",
        *[f"  {k}: {v}\n" for k, v in sorted(skipped.items())],
    ]
    if cell_plot_lines:
        cov.append("\nCell-winner test figures:\n")
        cov.extend(f"  {ln}\n" for ln in cell_plot_lines)
    coverage_text = "".join(cov)

    meta_lines = [
        f"Sweep IDs ({len(sweep_ids)}): {', '.join(sweep_ids)}",
        f"Within each cell: sort by {rank_caption}. Top-k: {args.top_k}. Test metrics are held-out eval only.",
    ]
    write_minimal_sweep_report_html(
        out_dir / "report.html",
        page_title="TESS regression — personal sweep report",
        h1="TESS regression — merged sweeps (personal)",
        meta_paragraphs=meta_lines,
        coverage_pre=coverage_text,
        relative_pngs=relative_pngs,
        artifact_links=[
            ("all_runs.csv", "all_runs.csv"),
            ("by_cell_stats.csv", "by_cell_stats.csv"),
            ("by_cell_hyperparams.json", "by_cell_hyperparams.json"),
            ("methodology_personal_sweeps.txt", "methodology_personal_sweeps.txt"),
            ("summary_compact.md", "summary_compact.md"),
            ("best_hyperparameters_by_cell.json", "best_hyperparameters_by_cell.json"),
            ("note_metrics.txt", "note_metrics.txt"),
            ("coverage.txt", "coverage.txt"),
            ("grids_mean_topk/", "Markdown grids (mean top-k)"),
            ("grids_best_topk/", "Markdown grids (best in top-k)"),
            ("grids_hyperparams/", "Hyperparameter representative grids"),
            ("png_mean_topk/", "PNG heatmaps (metrics, runtime, hparams)"),
            ("top_model_test_plots/regression/", "Test scatter/residual — one PNG per populated cell winner"),
        ],
    )

    (out_dir / "coverage.txt").write_text(coverage_text, encoding="utf-8")

    print(f"Wrote personal regression report to {out_dir.resolve()}")


if __name__ == "__main__":
    main()
