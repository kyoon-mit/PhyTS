"""Unified classification/regression sweep digest."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import matplotlib.pyplot as plt
import numpy as np
import wandb

from sweep_digest.collect import ClsRow, RegRow, bucket_model_size, merge_cls, merge_reg
from sweep_digest.core import (
    HEATMAP_SIZE_LABELS,
    HPARAM_HEATMAP_NUMERIC_KEYS,
    MODEL_ORDER,
    SIZE_ORDER,
    SWEEP_HPARAM_KEYS,
    add_wandb_cli,
    aggregate_hyperparams_mode,
    classification_val_sort_key,
    head_runs,
    hyperparam_cell_text,
    load_sweep_id_list,
    metrics_mean_std,
    numeric_value_for_hparam_heatmap,
    regression_val_sort_key,
    resolve_wandb_entity,
    sorted_runs,
    top_fraction_runs,
)
from sweep_digest.plots import build_column_matrix, build_matrix, plot_heatmap_2d
from sweep_digest.retest_metrics_store import resolve_retest_metrics_dir

# Default report root on ORCD pool (override with --out_dir).
DEFAULT_SWEEP_REPORTS_ROOT = Path("/home/allisone/orcd/pool/UROP_2025_Summer/TimeSeriesPhysics/sweep_reports")

PNG_SUBSET_MODEL_ORDER: tuple[str, ...] = ("s4d", "cnn", "transformer", "linoss_damped")
PNG_SUBSET_Y_LABELS: tuple[str, ...] = ("S4D", "CNN", "Transformer", "LinOSS")


def _task_long(task: str) -> str:
    return "Classification" if task == "cls" else "Regression"


def _model_ytick_full(m: str) -> str:
    return {
        "mlp": "MLP",
        "s4d": "S4D",
        "cnn": "CNN",
        "cnn_attn": "CNN–Attn",
        "transformer": "Transformer",
        "linoss_imex": "LinOSS (IMEX)",
        "linoss_damped": "LinOSS (damped)",
    }.get(m, m)


def _pretty_metric(mn: str) -> str:
    return {
        "test/acc": "Test accuracy",
        "test/balanced_acc": "Test balanced accuracy",
        "test/precision_macro": "Test macro precision",
        "test/recall_macro": "Test macro recall",
        "test/f1_macro": "Test macro F1",
        "test/precision_weighted": "Test weighted precision",
        "test/recall_weighted": "Test weighted recall",
        "test/f1_weighted": "Test weighted F1",
        "test/r2": "Test R²",
        "test/mae": "Test MAE",
        "test/rmse": "Test RMSE",
        "param_total": "Parameter count",
        "runtime_s": "Runtime (s)",
        "lr": "Learning rate (log10 of mode)",
        "dropout": "Dropout (mode)",
        "weight_decay": "Weight decay (log10 of mode)",
        "batch_size": "Batch size (mode)",
    }.get(mn, mn.replace("/", " ").replace("_", " "))


def _fmt_three(mean: Any, std: Any) -> str:
    if mean is None:
        return ""
    m = float(mean)
    if not np.isfinite(m):
        return ""
    base = f"{m:.3g}"
    if std is None:
        return base
    s = float(std)
    if not np.isfinite(s) or s <= 0:
        return base
    return f"{base} ± {s:.3g}"


def _fmt_cell_heatmap(v: float) -> str:
    if not np.isfinite(v):
        return ""
    return f"{v:.3g}"


def _write_heatmap_column_pair(
    *,
    out_dir: Path,
    rel_under_png: Path,
    task: str,
    agg_label: str,
    metric_key: str,
    by_model: dict[str, float | None],
    higher_is_better: bool,
) -> None:
    pretty = _pretty_metric(metric_key)
    cbar = _pretty_metric(metric_key)
    title = f"{_task_long(task)} — {agg_label} — {pretty}"
    pairs: tuple[tuple[Path, tuple[str, ...], tuple[str, ...]], ...] = (
        (
            out_dir / "png" / rel_under_png,
            MODEL_ORDER,
            tuple(_model_ytick_full(m) for m in MODEL_ORDER),
        ),
        (
            out_dir / "png_subset" / rel_under_png,
            PNG_SUBSET_MODEL_ORDER,
            PNG_SUBSET_Y_LABELS,
        ),
    )
    for path, mord, ylab in pairs:
        masked, txt = build_column_matrix(
            by_model, model_order=mord, fmt_cell=_fmt_cell_heatmap
        )
        plot_heatmap_2d(
            title=title,
            data=masked,
            cell_text=txt,
            out_path=path,
            yticklabels=ylab,
            xticklabels=[r"$\mu$ (mean)"],
            higher_is_better=higher_is_better,
            colorbar_label=cbar,
        )


def _write_heatmap_matrix_pair(
    *,
    out_dir: Path,
    rel_under_png: Path,
    task: str,
    agg_label: str,
    metric_key: str,
    pmap: dict[tuple[str, str], float | None],
    higher_is_better: bool,
) -> None:
    pretty = _pretty_metric(metric_key)
    cbar = _pretty_metric(metric_key)
    title = f"{_task_long(task)} — {agg_label} — {pretty}"
    pairs: tuple[tuple[Path, tuple[str, ...], tuple[str, ...]], ...] = (
        (
            out_dir / "png" / rel_under_png,
            MODEL_ORDER,
            tuple(_model_ytick_full(m) for m in MODEL_ORDER),
        ),
        (
            out_dir / "png_subset" / rel_under_png,
            PNG_SUBSET_MODEL_ORDER,
            PNG_SUBSET_Y_LABELS,
        ),
    )
    for path, mord, ylab in pairs:
        masked, txt = build_matrix(
            pmap, model_order=mord, size_order=SIZE_ORDER, fmt_cell=_fmt_cell_heatmap
        )
        plot_heatmap_2d(
            title=title,
            data=masked,
            cell_text=txt,
            out_path=path,
            yticklabels=ylab,
            xticklabels=list(HEATMAP_SIZE_LABELS),
            higher_is_better=higher_is_better,
            colorbar_label=cbar,
        )


def _safe_metric(s: str) -> str:
    return s.replace("/", "_")


def _coverage(path: Path, skipped: Counter[str], n: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    body = f"n_kept_runs: {n}\n" + "".join(f"  {k}: {skipped[k]}\n" for k in sorted(skipped))
    path.write_text(body, encoding="utf-8")


def _csv(path: Path, fieldnames: list[str], data: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fp:
        w = csv.DictWriter(fp, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        for row in data:
            w.writerow(row)


def _md_grid(path: Path, title: str, cells: dict[tuple[str, str], str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    hdr = "| model_type | " + " | ".join(HEATMAP_SIZE_LABELS) + " |\n"
    sep = "| " + " | ".join(["---"] * (1 + len(SIZE_ORDER))) + " |\n"
    body = "".join(
        "| " + mt + " | " + " | ".join(cells.get((mt, sz), "") for sz in SIZE_ORDER) + " |\n"
        for mt in MODEL_ORDER
    )
    path.write_text(f"## {title}\n\n" + hdr + sep + body + "\n", encoding="utf-8")


def _md_pooled(path: Path, title: str, rowdict: dict[str, dict[str, str]], cols: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    hdr = "| model_type | " + " | ".join(cols) + " |\n"
    sep = "| " + " | ".join(["---"] * (1 + len(cols))) + " |\n"
    body = ""
    for m in MODEL_ORDER:
        inner = rowdict.get(m, {})
        body += "| " + m + " | " + " | ".join(inner.get(c, "") for c in cols) + " |\n"
    path.write_text(f"## {title}\n\n" + hdr + sep + body + "\n", encoding="utf-8")


def _manifest(
    path: Path,
    banner: str,
    rule: str,
    subset: str,
    sections: list[tuple[str, list[Any]]],
    fmt_run: Callable[[Any, int], str],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    out = [banner, f"utc: {datetime.now(timezone.utc).isoformat()}", f"rule: {rule}", f"subset: {subset}", ""]
    for head, runs in sections:
        out.extend(["=" * 72, head, "=" * 72])
        for i, r in enumerate(runs, 1):
            out.append(fmt_run(r, i) + "\n" + "-" * 72)
    path.write_text("\n".join(out) + "\n", encoding="utf-8")


def _float_val(x: Any) -> float | None:
    if x is None or isinstance(x, bool):
        return None
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


def _hparam_md_out(path: Path, rows: list[Any]) -> None:
    parts: list[str] = ["# Hyperparameter distributions\n\n"]
    by_m: dict[str, list[Any]] = defaultdict(list)
    for r in rows:
        by_m[r.model_type].append(r)
    for mt in MODEL_ORDER:
        sub = by_m.get(mt, [])
        if not sub:
            continue
        parts.append(f"\n## {_model_ytick_full(mt)} (`{mt}`)\n\n")
        parts.append(f"n_runs={len(sub)}\n\n")
        n_by_sz = Counter(r.size for r in sub)
        sz_lab_of = dict(zip(SIZE_ORDER, HEATMAP_SIZE_LABELS, strict=True))
        line = ", ".join(
            f"{sz_lab_of.get(sz, sz)} (`{sz}`)={n_by_sz[sz]}" for sz in SIZE_ORDER if n_by_sz.get(sz)
        )
        parts.append(f"by size: {line}\n" if line else "by size: _none_\n")
        for k in SWEEP_HPARAM_KEYS:
            parts.append(f"\n### `{k}`\n\n")
            for sz, sz_lab in zip(SIZE_ORDER, HEATMAP_SIZE_LABELS, strict=True):
                vals = [r.hyperparams.get(k) for r in sub if r.size == sz]
                vals = [v for v in vals if v is not None and str(v).strip()]
                if not vals:
                    parts.append(f"- **{sz_lab}** (`{sz}`): _empty_\n")
                    continue
                nums = [_float_val(v) for v in vals]
                if nums and all(n is not None for n in nums):
                    a = np.asarray([float(x) for x in nums], dtype=float)
                    parts.append(
                        f"- **{sz_lab}** (`{sz}`): n={len(vals)}, "
                        f"min={a.min():.3g}, med={float(np.median(a)):.3g}, max={a.max():.3g}\n"
                    )
                else:
                    c = Counter(str(v) for v in vals)
                    top = ", ".join(f"`{val}`×{cnt}" for val, cnt in c.most_common(10))
                    parts.append(f"- **{sz_lab}** (`{sz}`): n={len(vals)}, {top}\n")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(parts), encoding="utf-8")


def _hparam_png_dir(
    out_dir: Path,
    rows: list[Any],
    *,
    task: str | None = None,
    model_types: tuple[str, ...] | None = None,
) -> None:
    """One PNG per (model_type, hyperparam): 2×2 subplots by size (xs…lg)."""
    out_dir.mkdir(parents=True, exist_ok=True)
    prefix = f"{_task_long(task)} — " if task else ""
    order = model_types if model_types is not None else MODEL_ORDER
    by_m: dict[str, list[Any]] = defaultdict(list)
    for r in rows:
        by_m[r.model_type].append(r)

    for mt in order:
        sub = by_m.get(mt, [])
        if not sub:
            continue
        mt_safe = _safe_metric(mt)
        mt_dir = out_dir / mt_safe
        mt_dir.mkdir(parents=True, exist_ok=True)
        mt_title = _model_ytick_full(mt)

        for k in SWEEP_HPARAM_KEYS:
            safe = k.replace("/", "_")
            fig, axes = plt.subplots(2, 2, figsize=(11.0, 9.0))
            ax_list = list(axes.ravel())
            any_data = False
            for idx, sz in enumerate(SIZE_ORDER):
                ax = ax_list[idx]
                sz_lab = HEATMAP_SIZE_LABELS[idx]
                cell_rows = [r for r in sub if r.size == sz]
                vals = [r.hyperparams.get(k) for r in cell_rows]
                vals = [v for v in vals if v is not None and str(v).strip()]
                if not vals:
                    ax.set_axis_off()
                    ax.text(0.5, 0.5, f"{sz_lab}\n(n=0)", ha="center", va="center", transform=ax.transAxes, fontsize=11)
                    continue
                any_data = True
                nums = [_float_val(v) for v in vals]
                if nums and all(n is not None for n in nums):
                    a = np.asarray([float(x) for x in nums], dtype=float)
                    nuniq = len(np.unique(a))
                    ax.hist(
                        a,
                        bins=min(20, max(3, nuniq)),
                        color="#2171b5",
                        edgecolor="white",
                    )
                else:
                    c = Counter(str(v) for v in vals)
                    mc = c.most_common(12)
                    labels = [t[0] for t in mc]
                    counts = [t[1] for t in mc]
                    ax.bar(range(len(labels)), counts, color="#2171b5")
                    ax.set_xticks(range(len(labels)), labels=list(labels), rotation=45, ha="right", fontsize=8)
                ax.set_title(f"{sz_lab} (`{sz}`), n={len(vals)}", fontsize=10, fontweight="600")
                ax.tick_params(labelsize=9)

            if not any_data:
                plt.close(fig)
                continue

            fig.suptitle(f"{prefix}{mt_title} — {k}", fontsize=13, fontweight="600", y=0.98)
            fig.tight_layout(rect=[0, 0, 1, 0.96])
            fig.savefig(mt_dir / f"{safe}_dist.png", dpi=175)
            plt.close(fig)


def _mf_cls(r: ClsRow, rank: int) -> str:
    return (
        f"rank={rank}\nrun_id={r.run_id}\nurl={r.wandb_url}\nsweep={r.source_sweep_id}\n"
        f"model={r.model_type} size={r.size}\n"
        f"val_bal={r.val_balanced} val_acc={r.val_acc}\n"
        f"test_acc={r.test_acc} test_bal={r.test_balanced}\n"
        f"macro_p={r.test_macro_precision} macro_r={r.test_macro_recall} macro_f1={r.test_macro_f1}\n"
        f"w_p={r.test_precision_weighted} w_r={r.test_recall_weighted} w_f1={r.test_f1_weighted}\n"
        f"params={r.param_total} runtime_s={r.runtime_s}\nhp={dict(r.hyperparams)}\n"
    )


def _mf_reg(r: RegRow, rank: int) -> str:
    return (
        f"rank={rank}\nrun_id={r.run_id}\nurl={r.wandb_url}\nsweep={r.source_sweep_id}\n"
        f"model={r.model_type} size={r.size}\n"
        f"val_r2={r.val_r2} val_mae={r.val_mae} val_rmse={r.val_rmse}\n"
        f"test_r2={r.test_r2} test_mae={r.test_mae} test_rmse={r.test_rmse}\n"
        f"params={r.param_total} runtime_s={r.runtime_s}\nhp={dict(r.hyperparams)}\n"
    )


def _digest_any(
    rows: list[Any],
    *,
    out_dir: Path,
    task: str,
    selection: str,
    val_key: Callable[[Any], tuple[float, float]],
    metrics: dict[str, Callable[[Any], float | None]],
    higher: dict[str, bool],
    mf: Callable[[Any, int], str],
    args: argparse.Namespace,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    by_t: dict[str, list[Any]] = defaultdict(list)
    for r in rows:
        by_t[r.model_type].append(r)
    buckets = bucket_model_size(rows)
    mnames = list(metrics.keys())

    def takes(tag: str):
        if tag == "topk":
            return lambda g: head_runs(g, args.top_k_mean)
        if tag == "frac":
            return lambda g: top_fraction_runs(g, args.top_fraction_mean)
        return lambda g: head_runs(g, 1)

    labels = {
        "topk": f"top-{args.top_k_mean} mean",
        "frac": f"top-{int(round(args.top_fraction_mean * 100))}pct mean",
        "top1": "top-1",
    }

    pooled_topk: list[tuple[str, list[Any]]] = []
    pooled_frac: list[tuple[str, list[Any]]] = []
    pooled_top1: list[tuple[str, list[Any]]] = []
    cell_topk: list[tuple[str, list[Any]]] = []
    cell_frac: list[tuple[str, list[Any]]] = []
    cell_top1: list[tuple[str, list[Any]]] = []

    pooled_specs = [
        ("topk_mean", "topk", pooled_topk),
        ("topfrac_mean", "frac", pooled_frac),
        ("top1_mean", "top1", pooled_top1),
    ]
    cell_specs = [
        ("topk_mean", "topk", cell_topk),
        ("topfrac_mean", "frac", cell_frac),
        ("top1_mean", "top1", cell_top1),
    ]

    for fname_tag, key_tag, mlist in pooled_specs:
        take = takes(key_tag)
        md_rows = {m: {} for m in MODEL_ORDER}
        png_by_m = {mn: {} for mn in mnames}
        for m in MODEL_ORDER:
            g = sorted_runs(by_t[m], val_key)
            ch = take(g)
            mlist.append((f"model_type={m} n={len(ch)}", ch))
            st = metrics_mean_std(ch, metrics)
            for mn in mnames:
                md_rows[m][mn] = _fmt_three(st[mn]["mean"], st[mn]["std"])
                png_by_m[mn][m] = st[mn]["mean"]
        _md_pooled(
            out_dir / "md" / f"pooled_by_type_{fname_tag}.md",
            f"{task} pooled by model_type ({labels[key_tag]})",
            md_rows,
            mnames,
        )
        for mn in mnames:
            _write_heatmap_column_pair(
                out_dir=out_dir,
                rel_under_png=Path("pooled_by_type") / fname_tag / f"{_safe_metric(mn)}.png",
                task=task,
                agg_label=labels[key_tag],
                metric_key=mn,
                by_model=png_by_m[mn],
                higher_is_better=higher[mn],
            )

    for fname_tag, key_tag, mlist_any in cell_specs:
        take = takes(key_tag)
        stats_bc: dict[tuple[str, str], Any] = {}
        for m in MODEL_ORDER:
            for s in SIZE_ORDER:
                g = sorted_runs(buckets.get((m, s), []), val_key)
                ch = take(g)
                mlist_any.append((f"{m}/{s} n={len(ch)}", ch))
                stats_bc[(m, s)] = metrics_mean_std(ch, metrics)
        for mn in mnames:
            ctext: dict[tuple[str, str], str] = {}
            pmap: dict[tuple[str, str], float | None] = {}
            for m in MODEL_ORDER:
                for s in SIZE_ORDER:
                    st = stats_bc[(m, s)][mn]
                    ctext[(m, s)] = _fmt_three(st["mean"], st["std"])
                    pmap[(m, s)] = st["mean"]
            _md_grid(out_dir / "md" / "by_cell" / fname_tag / f"{_safe_metric(mn)}.md", f"{mn} — {labels[key_tag]}", ctext)
            _write_heatmap_matrix_pair(
                out_dir=out_dir,
                rel_under_png=Path("by_cell") / fname_tag / f"{_safe_metric(mn)}.png",
                task=task,
                agg_label=labels[key_tag],
                metric_key=mn,
                pmap=pmap,
                higher_is_better=higher[mn],
            )

    md = out_dir / "manifests"
    _manifest(md / "manifest_topk_mean_by_model_type.txt", f"{task} top-k pooled", selection, f"k={args.top_k_mean}", pooled_topk, mf)
    _manifest(
        md / "manifest_top_fraction_mean_by_model_type.txt",
        f"{task} top-fraction pooled",
        selection,
        f"f={args.top_fraction_mean}",
        pooled_frac,
        mf,
    )
    _manifest(md / "manifest_topk_mean_by_cell.txt", f"{task} top-k by cell", selection, f"k={args.top_k_mean}", cell_topk, mf)
    _manifest(md / "manifest_top_fraction_mean_by_cell.txt", f"{task} top-frac by cell", selection, f"f={args.top_fraction_mean}", cell_frac, mf)
    _manifest(md / "manifest_top1_by_model_type.txt", f"{task} top-1 pooled", selection, "top1", pooled_top1, mf)
    _manifest(md / "manifest_top1_by_cell.txt", f"{task} top-1 cell", selection, "top1", cell_top1, mf)

    grids = {k: {} for k in SWEEP_HPARAM_KEYS}
    hp_json: dict[str, Any] = {}
    nums = {k: {} for k in HPARAM_HEATMAP_NUMERIC_KEYS}
    for m in MODEL_ORDER:
        for s in SIZE_ORDER:
            g = sorted_runs(buckets.get((m, s), []), val_key)
            ch = head_runs(g, args.top_k_hparam_mode)
            agg = aggregate_hyperparams_mode(ch, SWEEP_HPARAM_KEYS)
            hp_json[f"{m}/{s}"] = agg
            for k in SWEEP_HPARAM_KEYS:
                grids[k][(m, s)] = hyperparam_cell_text(k, agg.get(k, {}))
            for nk in HPARAM_HEATMAP_NUMERIC_KEYS:
                nums[nk][(m, s)] = numeric_value_for_hparam_heatmap(nk, agg.get(nk, {}))

    (out_dir / "json").mkdir(parents=True, exist_ok=True)
    (out_dir / "json" / "by_cell_hyperparams.json").write_text(json.dumps(hp_json, indent=2) + "\n", encoding="utf-8")
    for k in SWEEP_HPARAM_KEYS:
        _md_grid(out_dir / "md" / "hparam_mode" / f"{k}.md", f"{k} mode (top-{args.top_k_hparam_mode} val)", grids[k])
    for nk in HPARAM_HEATMAP_NUMERIC_KEYS:
        _write_heatmap_matrix_pair(
            out_dir=out_dir,
            rel_under_png=Path("hparam_mode") / f"{_safe_metric(nk)}.png",
            task=task,
            agg_label=f"HP mode (top-{args.top_k_hparam_mode} val)",
            metric_key=nk,
            pmap=nums[nk],
            higher_is_better=nk != "dropout",
        )

    _hparam_md_out(out_dir / "md" / "hyperparameter_distributions.md", rows)
    _hparam_png_dir(out_dir / "png" / "hparam_distributions", rows, task=task)
    rows_sub = [r for r in rows if r.model_type in PNG_SUBSET_MODEL_ORDER]
    _hparam_png_dir(
        out_dir / "png_subset" / "hparam_distributions",
        rows_sub,
        task=task,
        model_types=PNG_SUBSET_MODEL_ORDER,
    )


def _cls_metrics() -> tuple[dict[str, Any], dict[str, bool]]:
    m = {
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
    h = {k: True for k in m}
    h["runtime_s"] = False
    return m, h


def _reg_metrics() -> tuple[dict[str, Any], dict[str, bool]]:
    m = {
        "test/r2": lambda r: r.test_r2,
        "test/mae": lambda r: r.test_mae,
        "test/rmse": lambda r: r.test_rmse,
        "param_total": lambda r: float(r.param_total) if r.param_total is not None else None,
        "runtime_s": lambda r: r.runtime_s,
    }
    h = {"test/r2": True, "test/mae": False, "test/rmse": False, "param_total": True, "runtime_s": False}
    return m, h


def run_classification(args: argparse.Namespace) -> None:
    no = args.no_progress
    sids = load_sweep_id_list(*args.sweep_id, from_file=args.from_file)
    api = wandb.Api()
    ent = resolve_wandb_entity(api, args.entity)
    retest_dir = resolve_retest_metrics_dir(getattr(args, "retest_metrics_dir", None))
    rows, sk = merge_cls(
        api,
        entity=ent,
        project=args.project,
        sweep_ids=sids,
        no_prog=no,
        retest_metrics_dir=retest_dir,
    )
    out = Path(args.out_dir)
    _coverage(out / "coverage.txt", sk, len(rows))
    cols = [
        "run_id",
        "source_sweep_id",
        "model_type",
        "size",
        "val_acc",
        "val_balanced",
        "test_acc",
        "test_balanced",
        "test_macro_precision",
        "test_macro_recall",
        "test_macro_f1",
        "test_precision_weighted",
        "test_recall_weighted",
        "test_f1_weighted",
        "param_total",
        "runtime_s",
        *[f"hp_{k}" for k in SWEEP_HPARAM_KEYS],
    ]
    data = []
    for r in rows:
        d = {
            "run_id": r.run_id,
            "source_sweep_id": r.source_sweep_id,
            "model_type": r.model_type,
            "size": r.size,
            "val_acc": r.val_acc,
            "val_balanced": r.val_balanced,
            "test_acc": r.test_acc,
            "test_balanced": r.test_balanced,
            "test_macro_precision": r.test_macro_precision,
            "test_macro_recall": r.test_macro_recall,
            "test_macro_f1": r.test_macro_f1,
            "test_precision_weighted": r.test_precision_weighted,
            "test_recall_weighted": r.test_recall_weighted,
            "test_f1_weighted": r.test_f1_weighted,
            "param_total": r.param_total,
            "runtime_s": r.runtime_s,
        }
        for k in SWEEP_HPARAM_KEYS:
            d[f"hp_{k}"] = r.hyperparams.get(k)
        data.append(d)
    _csv(out / "all_runs.csv", cols, data)
    mg, hi = _cls_metrics()
    _digest_any(
        rows,
        out_dir=out,
        task="cls",
        selection="val/balanced_acc else val/acc",
        val_key=classification_val_sort_key,
        metrics=mg,
        higher=hi,
        mf=_mf_cls,
        args=args,
    )
    meta = {
        "task": "classification",
        "n_runs": len(rows),
        "sweeps": sids,
        "entity": ent,
        "project": args.project,
        "top_k_mean": args.top_k_mean,
        "top_fraction_mean": args.top_fraction_mean,
        "top_k_hparam_mode": args.top_k_hparam_mode,
        "retest_metrics_dir": str(retest_dir),
    }
    (out / "digest_meta.json").write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")


def run_regression(args: argparse.Namespace) -> None:
    no = args.no_progress
    sids = load_sweep_id_list(*args.sweep_id, from_file=args.from_file)
    api = wandb.Api()
    ent = resolve_wandb_entity(api, args.entity)
    retest_dir = resolve_retest_metrics_dir(getattr(args, "retest_metrics_dir", None))
    rows, sk = merge_reg(
        api,
        entity=ent,
        project=args.project,
        sweep_ids=sids,
        no_prog=no,
        retest_metrics_dir=retest_dir,
    )
    out = Path(args.out_dir)
    _coverage(out / "coverage.txt", sk, len(rows))
    cols = [
        "run_id",
        "source_sweep_id",
        "model_type",
        "size",
        "val_r2",
        "val_mae",
        "val_rmse",
        "test_r2",
        "test_mae",
        "test_rmse",
        "param_total",
        "runtime_s",
        *[f"hp_{k}" for k in SWEEP_HPARAM_KEYS],
    ]
    data = []
    for r in rows:
        d = {
            "run_id": r.run_id,
            "source_sweep_id": r.source_sweep_id,
            "model_type": r.model_type,
            "size": r.size,
            "val_r2": r.val_r2,
            "val_mae": r.val_mae,
            "val_rmse": r.val_rmse,
            "test_r2": r.test_r2,
            "test_mae": r.test_mae,
            "test_rmse": r.test_rmse,
            "param_total": r.param_total,
            "runtime_s": r.runtime_s,
        }
        for k in SWEEP_HPARAM_KEYS:
            d[f"hp_{k}"] = r.hyperparams.get(k)
        data.append(d)
    _csv(out / "all_runs.csv", cols, data)
    mg, hi = _reg_metrics()
    _digest_any(
        rows,
        out_dir=out,
        task="reg",
        selection="val/r2 else val/mae else val/rmse",
        val_key=regression_val_sort_key,
        metrics=mg,
        higher=hi,
        mf=_mf_reg,
        args=args,
    )
    meta = {
        "task": "regression",
        "n_runs": len(rows),
        "sweeps": sids,
        "entity": ent,
        "project": args.project,
        "top_k_mean": args.top_k_mean,
        "top_fraction_mean": args.top_fraction_mean,
        "top_k_hparam_mode": args.top_k_hparam_mode,
        "retest_metrics_dir": str(retest_dir),
    }
    (out / "digest_meta.json").write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")


def parser_cls() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="TESS classification sweep digest")
    add_wandb_cli(p)
    p.add_argument(
        "--out_dir",
        type=Path,
        default=DEFAULT_SWEEP_REPORTS_ROOT / "digest_classification",
    )
    p.add_argument("--top_k_mean", type=int, default=3)
    p.add_argument("--top_fraction_mean", type=float, default=0.75)
    p.add_argument("--top_k_hparam_mode", type=int, default=5)
    p.add_argument("--no_progress", action="store_true")
    p.add_argument(
        "--retest_metrics_dir",
        type=Path,
        default=None,
        help="Per-run JSON from sweep_digest.retest; fills missing test/* (default: sweep_digest/retest_metrics or env).",
    )
    return p


def parser_reg() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="TESS regression sweep digest")
    add_wandb_cli(p)
    p.add_argument(
        "--out_dir",
        type=Path,
        default=DEFAULT_SWEEP_REPORTS_ROOT / "digest_regression",
    )
    p.add_argument("--top_k_mean", type=int, default=3)
    p.add_argument("--top_fraction_mean", type=float, default=0.75)
    p.add_argument("--top_k_hparam_mode", type=int, default=5)
    p.add_argument("--no_progress", action="store_true")
    p.add_argument(
        "--retest_metrics_dir",
        type=Path,
        default=None,
        help="Per-run JSON from sweep_digest.retest; fills missing test/* (default: sweep_digest/retest_metrics or env).",
    )
    return p


def main_cls() -> None:
    run_classification(parser_cls().parse_args())


def main_reg() -> None:
    run_regression(parser_reg().parse_args())
