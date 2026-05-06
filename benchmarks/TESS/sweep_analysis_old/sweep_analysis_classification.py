#!/usr/bin/env python3
"""Personal TESS *classification* sweep digest (not intended for the paper).

Within each ``(model_type, size)`` cell, runs are sorted by **val/accuracy**
(higher is better), **not** balanced accuracy, then the first ``--top_k`` runs
define aggregates. Summary tables and heatmaps report **held-out test** metrics
for those trials only.

**Macro precision / F1:** training now logs ``test/precision_macro`` (and weighted
variants). Old runs without them can be back-filled with ``--reeval_missing_metrics``
if checkpoints exist under ``--ckpt_root`` (see :mod:`sweep_digest.retest`).
Use that flag for P/R/F1 heatmaps whenever wandb lacks true macro recall (the digest
may otherwise substitute ``test/balanced_acc`` for recall).

**Compute:** parameter counts come from ``wandb.summary`` (see
``collect_benchmark_param_counters`` in training). FLOPs are not logged in this
repo.

**Artifacts:** trials use :func:`sweep_utils.tess_sweep_artifact_dir` under
``logs/tess_sweeps/classification/...``. Checkpoints default to
``TESS_CKPT_DIR``/``checkpoints/sweeps`` → ``classification/{model_type}/{run_id}/``.
This script writes reports under ``--out_dir`` including Markdown grids, **PNG
heatmaps** (``png_mean_topk/``; acc, balanced acc, macro precision/recall/F1,
runtime, hyperparameters), and ``report.html``.

Examples
--------
    uv run python benchmarks/TESS/sweep_personal_classification.py \\
        --sweep_id abc --sweep_id def --out_dir benchmarks/TESS/sweep_reports/personal/cls1

    uv run python benchmarks/TESS/sweep_personal_classification.py \\
        --from_file benchmarks/TESS/sweep_ids_classification.txt
"""

from __future__ import annotations

import argparse
import os
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any
_BENCH = Path(__file__).resolve().parent
if str(_BENCH) not in sys.path:
    sys.path.insert(0, str(_BENCH))
_TESS = Path(__file__).resolve().parent.parent
if str(_TESS) not in sys.path:
    sys.path.insert(0, str(_TESS))

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
    CLASSIFICATION_VAL_SELECTION_LINE,
    HPARAM_AGGREGATION_LINE,
    HPARAM_HEATMAP_NUMERIC_KEYS,
    aggregate_hyperparams_for_runs,
    bucket_by_model_size,
    classification_val_acc_desc_key,
    extract_runtime_s,
    first_summary_float,
    hyperparam_cell_display,
    macro_recall_from_class_accs,
    numeric_value_for_hparam_heatmap,
    topk_metric_stats,
    total_params_from_summary,
    write_csv,
    write_json,
    write_md_metric_grid,
    write_minimal_sweep_report_html,
)


MACRO_PREC_CANDIDATES = (
    "test/precision_macro",
    "test/macro_precision",
    "test/precision/macro",
    "val/precision_macro",
    "val/macro_precision",
)
MACRO_REC_CANDIDATES = (
    "test/recall_macro",
    "test/macro_recall",
    "test/recall/macro",
    "val/recall_macro",
)
MACRO_F1_CANDIDATES = (
    "test/f1_macro",
    "test/macro_f1",
    "test/f1/macro",
    "val/f1_macro",
)


def _fmt_metric_cell(metric_name: str, value: float) -> str:
    if metric_name in ("param_total", "runtime_s"):
        return f"{value:.6g}"
    if "acc" in metric_name or "macro" in metric_name or "weighted" in metric_name:
        return f"{value:.4g}"
    return f"{value:.6g}"


@dataclass
class RunRow:
    run_id: str
    source_sweep_id: str
    model_type: str
    size: str
    hyperparams: dict[str, Any]
    wandb_url: str | None
    val_acc: float | None
    val_balanced: float | None
    test_acc: float | None
    test_balanced: float | None
    test_macro_precision: float | None
    test_macro_recall: float | None
    test_macro_f1: float | None
    test_precision_weighted: float | None
    test_recall_weighted: float | None
    test_f1_weighted: float | None
    param_total: int | None
    runtime_s: float | None
    # True when ``test_macro_recall`` was imputed from balanced_acc / per-class accs, not wandb macro recall.
    macro_recall_is_fallback: bool


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
        val_acc = summary_get(summary, "val/acc")
        if val_acc is None:
            skipped["missing_val_acc"] += 1
            continue

        mp = first_summary_float(summary, MACRO_PREC_CANDIDATES)
        mr = first_summary_float(summary, MACRO_REC_CANDIDATES)
        macro_recall_is_fallback = False
        if mr is None:
            mr = summary_get(summary, "test/balanced_acc")
            if mr is not None:
                macro_recall_is_fallback = True
        if mr is None:
            mr_cls = macro_recall_from_class_accs(summary)
            if mr_cls is not None:
                mr = mr_cls
                macro_recall_is_fallback = True
        mf = first_summary_float(summary, MACRO_F1_CANDIDATES)
        p_w = summary_get(summary, "test/precision_weighted")
        r_w = summary_get(summary, "test/recall_weighted")
        f_w = summary_get(summary, "test/f1_weighted")

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
                val_acc=float(val_acc),
                val_balanced=summary_get(summary, "val/balanced_acc"),
                test_acc=summary_get(summary, "test/acc"),
                test_balanced=summary_get(summary, "test/balanced_acc"),
                test_macro_precision=mp,
                test_macro_recall=mr,
                test_macro_f1=mf,
                test_precision_weighted=p_w,
                test_recall_weighted=r_w,
                test_f1_weighted=f_w,
                param_total=total_params_from_summary(summary),
                runtime_s=extract_runtime_s(run),
                macro_recall_is_fallback=macro_recall_is_fallback,
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


def _apply_reeval_to_rows(
    api: wandb.Api,
    *,
    entity: str,
    project: str,
    rows: list[RunRow],
    ckpt_root: Path,
    data_dir: str,
    seq_len: int,
    num_workers: int,
    inference_device: str,
) -> list[str]:
    """Fill missing or imputed macro recall from local checkpoints via test inference.

    Wandb often lacks true ``test/recall_macro``; we then substitute ``test/balanced_acc``,
    which breaks P/R/F1 heatmaps unless checkpoints are re-run.
    """
    from sweep_digest.retest import reeval_classification_test_metrics

    lines: list[str] = []

    for r in tqdm(rows, desc="Re-eval (missing metrics)", unit="run", dynamic_ncols=True):
        need = (
            r.test_macro_precision is None
            or r.test_macro_recall is None
            or r.test_macro_f1 is None
            or r.test_precision_weighted is None
            or r.test_recall_weighted is None
            or r.test_f1_weighted is None
            or r.macro_recall_is_fallback
        )
        if not need:
            continue
        try:
            wr = api.run(f"{entity}/{project}/{r.run_id}")
            cfg = dict(wr.config)
        except Exception as exc:
            lines.append(f"{r.run_id}: wandb run fetch failed ({exc})")
            continue
        m = reeval_classification_test_metrics(
            wandb_cfg=cfg,
            run_id=r.run_id,
            ckpt_root=ckpt_root,
            data_dir=data_dir,
            seq_len=seq_len,
            num_workers=num_workers,
            device=inference_device,
        )
        if m is None:
            lines.append(f"{r.run_id}: no checkpoint or re-eval failed")
            continue
        lines.append(f"{r.run_id}: re-eval ok")
        # Always apply full extended block so recall/F1 stay consistent (not mixed with balanced_acc proxy).
        r.test_macro_precision = m.get("test/precision_macro")
        r.test_macro_recall = m.get("test/recall_macro")
        r.test_macro_f1 = m.get("test/f1_macro")
        r.test_precision_weighted = m.get("test/precision_weighted")
        r.test_recall_weighted = m.get("test/recall_weighted")
        r.test_f1_weighted = m.get("test/f1_weighted")
        r.macro_recall_is_fallback = False
    return lines


def _parse() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Personal TESS classification sweep benchmark tables")
    add_common_wandb_args(p)
    add_sweep_id_args(p)
    p.add_argument(
        "--out_dir",
        type=Path,
        default=Path("benchmarks/TESS/sweep_reports/personal/classification"),
        help="Where to write CSV/MD/JSON (default under benchmarks/TESS/sweep_reports/personal/)",
    )
    p.add_argument("--top_k", type=int, default=3, help="Runs per cell for mean/std (after val/acc sort)")
    p.add_argument(
        "--reeval_missing_metrics",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Run local test inference from checkpoints when macro/weighted P/R/F1 are missing in wandb, "
            "or when macro recall was imputed from balanced acc (recommended for P/R/F1 heatmaps)."
        ),
    )
    p.add_argument(
        "--ckpt_root",
        type=Path,
        default=None,
        help="Contains classification/{{model_type}}/{{run_id}}/ — defaults to TESS_CKPT_DIR or checkpoints/sweeps",
    )
    p.add_argument(
        "--data_dir",
        type=Path,
        default=None,
        help="TESS cache directory — defaults to TESS_DATA_DIR or data/TESS/.cache/TESS",
    )
    p.add_argument("--seq_len", type=int, default=1100)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument(
        "--inference_device",
        default="auto",
        choices=["auto", "cpu", "cuda"],
    )
    p.add_argument(
        "--save_winner_plots",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Save test-set confusion-matrix PNGs per cell winner (eval_plots layout; needs checkpoints)",
    )
    return p.parse_args()


def main() -> None:
    args = _parse()
    sweep_ids = load_sweep_id_list(args.sweep_id, args.from_file)
    api = wandb.Api()
    entity = resolve_wandb_entity(api, args.entity)

    rows, skipped = _merge_sweeps(api, entity=entity, project=args.project, sweep_ids=sweep_ids)
    ckpt_root = args.ckpt_root or Path(os.environ.get("TESS_CKPT_DIR", "checkpoints/sweeps"))
    data_dir_eval = args.data_dir or Path(os.environ.get("TESS_DATA_DIR", "data/TESS/.cache/TESS"))
    reeval_lines: list[str] = []
    if args.reeval_missing_metrics:
        reeval_lines = _apply_reeval_to_rows(
            api,
            entity=entity,
            project=args.project,
            rows=rows,
            ckpt_root=ckpt_root,
            data_dir=str(data_dir_eval.resolve()),
            seq_len=args.seq_len,
            num_workers=args.num_workers,
            inference_device=args.inference_device,
        )
    out_dir = args.out_dir

    rank_caption = "val/acc (higher is better; not balanced acc)"

    fields = (
        [
            "run_id",
            "source_sweep_id",
            "model_type",
            "size",
            "val/acc",
            "val/balanced_acc",
            "test/acc",
            "test/balanced_acc",
            "test/precision_macro",
            "test/recall_macro",
            "test/f1_macro",
            "test/precision_weighted",
            "test/recall_weighted",
            "test/f1_weighted",
            "param_total",
            "runtime_s",
        ]
        + list(SWEEP_HPARAM_KEYS)
        + ["wandb_url"]
    )
    all_csv_rows: list[dict[str, Any]] = []
    for r in sorted(
        rows,
        key=lambda x: (x.model_type, x.size, -(x.val_acc or 0.0)),
    ):
        all_csv_rows.append(
            {
                "run_id": r.run_id,
                "source_sweep_id": r.source_sweep_id,
                "model_type": r.model_type,
                "size": r.size,
                "val/acc": r.val_acc if r.val_acc is not None else "",
                "val/balanced_acc": r.val_balanced if r.val_balanced is not None else "",
                "test/acc": r.test_acc if r.test_acc is not None else "",
                "test/balanced_acc": r.test_balanced if r.test_balanced is not None else "",
                "test/precision_macro": r.test_macro_precision if r.test_macro_precision is not None else "",
                "test/recall_macro": r.test_macro_recall if r.test_macro_recall is not None else "",
                "test/f1_macro": r.test_macro_f1 if r.test_macro_f1 is not None else "",
                "test/precision_weighted": r.test_precision_weighted
                if r.test_precision_weighted is not None
                else "",
                "test/recall_weighted": r.test_recall_weighted if r.test_recall_weighted is not None else "",
                "test/f1_weighted": r.test_f1_weighted if r.test_f1_weighted is not None else "",
                "param_total": r.param_total if r.param_total is not None else "",
                "runtime_s": r.runtime_s if r.runtime_s is not None else "",
                **{k: r.hyperparams.get(k, "") for k in SWEEP_HPARAM_KEYS},
                "wandb_url": r.wandb_url or "",
            }
        )
    write_csv(out_dir / "all_runs.csv", fields, all_csv_rows)

    buckets = bucket_by_model_size(rows, model_key=lambda r: r.model_type, size_key=lambda r: r.size)

    metric_getters = {
        "val/acc": lambda r: r.val_acc,
        "val/balanced_acc": lambda r: r.val_balanced,
        "test/acc": lambda r: r.test_acc,
        "test/balanced_acc": lambda r: r.test_balanced,
        "test/precision_macro": lambda r: r.test_macro_precision,
        "test/recall_macro": lambda r: r.test_macro_recall,
        "test/f1_macro": lambda r: r.test_macro_f1,
        "test/precision_weighted": lambda r: r.test_precision_weighted,
        "test/recall_weighted": lambda r: r.test_recall_weighted,
        "test/f1_weighted": lambda r: r.test_f1_weighted,
        "param_total": lambda r: float(r.param_total) if r.param_total is not None else None,
        "runtime_s": lambda r: r.runtime_s,
    }
    higher = {k: True for k in metric_getters}
    higher["runtime_s"] = False

    metric_names = [
        "val/acc",
        "val/balanced_acc",
        "test/acc",
        "test/balanced_acc",
        "test/precision_macro",
        "test/recall_macro",
        "test/f1_macro",
        "test/precision_weighted",
        "test/recall_weighted",
        "test/f1_weighted",
        "param_total",
        "runtime_s",
    ]

    by_cell: list[dict[str, Any]] = []
    best_hp: dict[str, Any] = {}
    hp_by_cell: dict[str, Any] = {}
    grid_mean: dict[str, dict[tuple[str, str], str]] = {name: {} for name in metric_names}
    grid_best: dict[str, dict[tuple[str, str], str]] = {name: {} for name in metric_names}

    png_metrics = (
        "test/acc",
        "test/balanced_acc",
        "test/precision_macro",
        "test/recall_macro",
        "test/f1_macro",
    )
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
            n = len(group)
            sorted_g = sorted(group, key=classification_val_acc_desc_key, reverse=True)
            top = sorted_g[: max(1, args.top_k)]

            st = topk_metric_stats(top, metric_getters, higher_is_better=higher)
            row_out: dict[str, Any] = {
                "model_type": mk,
                "size": sz,
                "n_runs_in_cell": n,
                "top_k_used": min(len(top), args.top_k),
            }

            for mn in metric_names:
                s = st[mn]
                row_out[f"{mn}__best"] = s["best"]
                row_out[f"{mn}__mean_topk"] = s["mean"]
                row_out[f"{mn}__std_topk"] = s["std"]
                row_out[f"{mn}__n_scored_topk"] = s["n"]
                if s["mean"] is not None:
                    grid_mean[mn][cell] = _fmt_metric_cell(mn, float(s["mean"]))
                else:
                    grid_mean[mn][cell] = ""
                if s["best"] is not None:
                    grid_best[mn][cell] = _fmt_metric_cell(mn, float(s["best"]))
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
                    "val/acc": winner.val_acc,
                    "val/balanced_acc": winner.val_balanced,
                    "runtime_s": winner.runtime_s,
                    "selection_rule": rank_caption,
                }
                best_hp[hk]["topk_hyperparam_snapshot"] = [
                    {**r.hyperparams, "run_id": r.run_id, "val/acc": r.val_acc} for r in top
                ]

    stat_fields = ["model_type", "size", "n_runs_in_cell", "top_k_used"] + [
        f"{mn}__{suffix}"
        for mn in metric_names
        for suffix in ("best", "mean_topk", "std_topk", "n_scored_topk")
    ]
    write_csv(out_dir / "by_cell_stats.csv", stat_fields, by_cell)

    grids_dir = out_dir / "grids_mean_topk"
    grids_best = out_dir / "grids_best_topk"
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
            grids_best / f"{safe}.md",
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

    metric_png_specs: list[tuple[str, str, str, str, str]] = [
        ("test/acc", "test_acc", "Mean test/acc — higher is greener", "accuracy", "Mean test/acc"),
        (
            "test/balanced_acc",
            "test_balanced_acc",
            "Mean test/balanced_acc — higher is greener",
            "balanced acc",
            "Mean test/balanced_acc",
        ),
        (
            "test/precision_macro",
            "test_precision_macro",
            "Mean test/precision_macro — higher is greener",
            "precision (macro)",
            "Mean test/precision_macro",
        ),
        (
            "test/recall_macro",
            "test_recall_macro",
            "Mean test/recall_macro — higher is greener",
            "recall (macro)",
            "Mean test/recall_macro",
        ),
        (
            "test/f1_macro",
            "test_f1_macro",
            "Mean test/f1_macro — higher is greener",
            "F1 (macro)",
            "Mean test/f1_macro",
        ),
    ]
    for pm_key, fname_key, title, cbar_lbl, rel_caption in metric_png_specs:
        m = png_mean_maps[pm_key]
        vals_m = [float(v) for v in m.values() if v is not None]
        vmin_m, vmax_m = (0.0, 1.0) if not vals_m else auto_vmin_vmax(vals_m)
        masked_m, txt_m = build_matrix(m, fmt_cell=lambda v: f"{float(v):.4f}")
        out_png = f"heatmap_mean_{fname_key}_top{args.top_k}.png"
        plot_heatmap(
            title,
            heat_subtitle,
            masked_m,
            txt_m,
            png_dir / out_png,
            model_order=list(MODEL_ORDER),
            size_order=list(SIZE_ORDER),
            size_xticklabels=HEATMAP_SIZE_LABELS,
            vmin=vmin_m,
            vmax=vmax_m,
            cmap="RdYlGn",
            colorbar_label=cbar_lbl,
        )
        relative_pngs.append((f"png_mean_topk/{out_png}", rel_caption))

    vals_rt = [float(v) for v in png_runtime_map.values() if v is not None]
    vmin_rt, vmax_rt = (0.0, 3600.0) if not vals_rt else auto_vmin_vmax(vals_rt)
    masked_rt, txt_rt = build_matrix(png_runtime_map, fmt_cell=lambda v: f"{float(v):.5g}")
    rt_png = f"heatmap_mean_runtime_s_top{args.top_k}.png"
    plot_heatmap(
        "Mean wall time (s), val-top-k — lower is greener",
        heat_subtitle + "; wandb _runtime averaged",
        masked_rt,
        txt_rt,
        png_dir / rt_png,
        model_order=list(MODEL_ORDER),
        size_order=list(SIZE_ORDER),
        size_xticklabels=HEATMAP_SIZE_LABELS,
        vmin=vmin_rt,
        vmax=vmax_rt,
        cmap="RdYlGn_r",
        colorbar_label="seconds",
    )
    relative_pngs.append((f"png_mean_topk/{rt_png}", "Mean runtime (s)"))

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
            row_txt = [overlay[(mk, sz)] for sz in SIZE_ORDER]
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
            CLASSIFICATION_VAL_SELECTION_LINE,
            "",
            "Within each cell, the first top_k trials by val/acc are averaged; summary tables plot **test** "
            "metrics for those same trials only (no test-based selection).",
            "",
            HPARAM_AGGREGATION_LINE,
            "",
            f"Hardware assumption: all jobs ran on {ASSUMED_GPU_LABEL}.",
        ]
    )
    (out_dir / "methodology_personal_sweeps.txt").write_text(methodology_txt + "\n", encoding="utf-8")

    compact_lines = [
        "# Classification — compact summary\n",
        "**Selection (per cell):** ",
        rank_caption,
        f"\nTop-k aggregates: **{args.top_k}**. Metrics below are test-set means among val-top-k runs.\n\n",
        "| model | size | n | best val/acc in top-k | mean test/acc | mean test/bal-acc | mean runtime (s) |\n",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: |\n",
    ]
    for mk in MODEL_ORDER:
        for sz in SIZE_ORDER:
            cell = (mk, sz)
            group = buckets.get(cell, [])
            if not group:
                continue
            sorted_g = sorted(group, key=classification_val_acc_desc_key, reverse=True)
            top = sorted_g[: max(1, args.top_k)]
            st = topk_metric_stats(top, metric_getters, higher_is_better=higher)
            br = st["val/acc"]["best"]
            mt = st["test/acc"]["mean"]
            mb = st["test/balanced_acc"]["mean"]
            rt = st["runtime_s"]["mean"]
            compact_lines.append(
                f"| {mk} | {sz} | {len(group)} | {br if br is not None else ''} | "
                f"{mt if mt is not None else ''} | {mb if mb is not None else ''} | "
                f"{rt if rt is not None else ''} |\n"
            )
    (out_dir / "summary_compact.md").write_text("".join(compact_lines), encoding="utf-8")

    write_json(out_dir / "best_hyperparameters_by_cell.json", best_hp)

    note = (
        "Training now logs macro and support-weighted precision / recall / F1 on val and test "
        "(see :mod:`tasks.TESS.classification_metrics`).\n"
        "Macro recall in tables may still fall back to legacy ``test/balanced_acc`` / "
        "``test/acc_class_*`` when newer keys are absent.\n"
        "Use ``--reeval_missing_metrics`` to back-fill from ``best.ckpt`` / ``best.eqx`` "
        f"(default ckpt root: {ckpt_root}).\n"
        "FLOPs are not logged; use param_total.\n"
        f"Hardware: summaries assume all jobs ran on {ASSUMED_GPU_LABEL}.\n"
        "Selection uses val/acc only; see methodology_personal_sweeps.txt.\n"
        "Per-cell test-set plots (eval_plots confusion matrix): top_model_test_plots/classification/ "
        "(use --no-save_winner_plots to skip).\n"
    )
    (out_dir / "note_metrics.txt").write_text(note, encoding="utf-8")

    cell_plot_lines: list[str] = []
    if args.save_winner_plots and winners_by_cell:
        from sweep_personal_top_plots import save_cell_winner_figures

        cell_plot_lines = save_cell_winner_figures(
            api,
            entity=entity,
            project=args.project,
            task_kind="classification",
            winners=winners_by_cell,
            ckpt_root=ckpt_root.resolve(),
            data_dir=str(data_dir_eval.resolve()),
            seq_len=args.seq_len,
            num_workers=args.num_workers,
            inference_device=args.inference_device,
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
    if reeval_lines:
        cov.append("\nCheckpoint re-eval (missing metrics):\n")
        cov.extend(f"  {line}\n" for line in reeval_lines)
    coverage_text = "".join(cov)

    meta_lines = [
        f"Sweep IDs ({len(sweep_ids)}): {', '.join(sweep_ids)}",
        f"Within each cell: sort by {rank_caption}. Top-k: {args.top_k}. Test metrics are held-out eval.",
    ]
    write_minimal_sweep_report_html(
        out_dir / "report.html",
        page_title="TESS classification — personal sweep report",
        h1="TESS classification — merged sweeps (personal)",
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
            ("top_model_test_plots/classification/", "Test confusion matrix — one PNG per populated cell winner"),
        ],
    )

    (out_dir / "coverage.txt").write_text(coverage_text, encoding="utf-8")

    print(f"Wrote personal classification report to {out_dir.resolve()}")

if __name__ == "__main__":
    main()
