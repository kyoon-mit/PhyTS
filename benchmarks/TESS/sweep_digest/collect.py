"""W&B sweep → RunRow lists (classification / regression)."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import wandb
from tqdm.auto import tqdm

from sweep_digest.core import (
    MODEL_ORDER,
    SIZE_ORDER,
    SWEEP_HPARAM_KEYS,
    cls_has_val_ranking,
    iter_sweep_runs,
    reg_has_val_ranking,
    summary_get,
)
from sweep_digest.retest_metrics_store import summary_dict_with_retest_overlay

MACRO_PREC = (
    "test/precision_macro",
    "test/macro_precision",
    "test/precision/macro",
    "val/precision_macro",
)
MACRO_REC = ("test/recall_macro", "test/macro_recall", "test/recall/macro", "val/recall_macro")
MACRO_F1 = ("test/f1_macro", "test/macro_f1", "test/f1/macro", "val/f1_macro")


def _first_float(summary: Any, keys: tuple[str, ...]) -> float | None:
    for k in keys:
        v = summary_get(summary, k)
        if v is not None:
            return v
    return None


def _macro_recall(summary: Any) -> float | None:
    v = _first_float(summary, MACRO_REC)
    if v is not None:
        return v
    v = summary_get(summary, "test/balanced_acc")
    if v is not None:
        return v
    accs = [summary_get(summary, f"test/acc_class_{c}") for c in range(16)]
    accs = [float(x) for x in accs if x is not None]
    return sum(accs) / len(accs) if accs else None


def _runtime(run: Any) -> float | None:
    s = run.summary
    if isinstance(s, dict) and "_runtime" in s:
        try:
            return float(s["_runtime"])
        except (TypeError, ValueError):
            return None
    return None


def _params(summary: Any) -> int | None:
    if (t := summary_get(summary, "param_count_torch_nn")) is not None:
        return int(t)
    jm = summary_get(summary, "param_count_jax_model_float")
    if jm is None:
        return None
    tot = int(jm)
    js = summary_get(summary, "param_count_jax_state_arrays")
    if js is not None:
        tot += int(js)
    return tot


@dataclass
class ClsRow:
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


@dataclass
class RegRow:
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


def _url(run: Any, entity: str, project: str) -> str | None:
    if getattr(run, "url", None):
        return str(run.url)
    rid = getattr(run, "id", None)
    return f"https://wandb.ai/{entity}/{project}/runs/{rid}" if rid else None


def _collect_cls_sweep(
    sweep: Any,
    sid: str,
    project: str,
    entity: str,
    no_prog: bool,
    retest_metrics_dir: Path | None,
) -> tuple[list[ClsRow], Counter[str]]:
    sk: Counter[str] = Counter()
    rows: list[ClsRow] = []
    for run in iter_sweep_runs(sweep, desc=f"Sweep {sid}", disable_tqdm=no_prog):
        if run.state != "finished":
            sk[f"not_finished:{run.state}"] += 1
            continue
        cfg = dict(run.config) if isinstance(run.config, dict) else {}
        mt, sz = cfg.get("model_type"), str(cfg.get("size", ""))
        if mt not in MODEL_ORDER:
            sk["bad_model_type"] += 1
            continue
        if sz not in SIZE_ORDER:
            sk["bad_size"] += 1
            continue
        if retest_metrics_dir is not None:
            summary, used = summary_dict_with_retest_overlay(
                run, entity=entity, project=project, retest_root=retest_metrics_dir
            )
            if used:
                sk["retest_metrics_overlay"] += 1
        else:
            summary = run.summary
        va, vb = summary_get(summary, "val/acc"), summary_get(summary, "val/balanced_acc")
        stub = type("_", (), {})()
        stub.val_acc, stub.val_balanced = va, vb
        if not cls_has_val_ranking(stub):
            sk["missing_val"] += 1
            continue
        rows.append(
            ClsRow(
                run_id=str(run.id),
                source_sweep_id=sid,
                model_type=str(mt),
                size=sz,
                hyperparams={k: cfg.get(k) for k in SWEEP_HPARAM_KEYS},
                wandb_url=_url(run, entity, project),
                val_acc=va,
                val_balanced=vb,
                test_acc=summary_get(summary, "test/acc"),
                test_balanced=summary_get(summary, "test/balanced_acc"),
                test_macro_precision=_first_float(summary, MACRO_PREC),
                test_macro_recall=_macro_recall(summary),
                test_macro_f1=_first_float(summary, MACRO_F1),
                test_precision_weighted=summary_get(summary, "test/precision_weighted"),
                test_recall_weighted=summary_get(summary, "test/recall_weighted"),
                test_f1_weighted=summary_get(summary, "test/f1_weighted"),
                param_total=_params(summary),
                runtime_s=_runtime(run),
            )
        )
    return rows, sk


def _collect_reg_sweep(
    sweep: Any,
    sid: str,
    project: str,
    entity: str,
    no_prog: bool,
    retest_metrics_dir: Path | None,
) -> tuple[list[RegRow], Counter[str]]:
    sk: Counter[str] = Counter()
    rows: list[RegRow] = []
    for run in iter_sweep_runs(sweep, desc=f"Sweep {sid}", disable_tqdm=no_prog):
        if run.state != "finished":
            sk[f"not_finished:{run.state}"] += 1
            continue
        cfg = dict(run.config) if isinstance(run.config, dict) else {}
        mt, sz = cfg.get("model_type"), str(cfg.get("size", ""))
        if mt not in MODEL_ORDER:
            sk["bad_model_type"] += 1
            continue
        if sz not in SIZE_ORDER:
            sk["bad_size"] += 1
            continue
        if retest_metrics_dir is not None:
            summary, used = summary_dict_with_retest_overlay(
                run, entity=entity, project=project, retest_root=retest_metrics_dir
            )
            if used:
                sk["retest_metrics_overlay"] += 1
        else:
            summary = run.summary
        vr2, vmae, vrmse = (
            summary_get(summary, "val/r2"),
            summary_get(summary, "val/mae"),
            summary_get(summary, "val/rmse"),
        )
        stub = type("_", (), {})()
        stub.val_r2, stub.val_mae, stub.val_rmse = vr2, vmae, vrmse
        if not reg_has_val_ranking(stub):
            sk["missing_val"] += 1
            continue
        rows.append(
            RegRow(
                run_id=str(run.id),
                source_sweep_id=sid,
                model_type=str(mt),
                size=sz,
                hyperparams={k: cfg.get(k) for k in SWEEP_HPARAM_KEYS},
                wandb_url=_url(run, entity, project),
                val_r2=vr2,
                val_mae=vmae,
                val_rmse=vrmse,
                test_r2=summary_get(summary, "test/r2"),
                test_mae=summary_get(summary, "test/mae"),
                test_rmse=summary_get(summary, "test/rmse"),
                param_total=_params(summary),
                runtime_s=_runtime(run),
            )
        )
    return rows, sk


def merge_cls(
    api: wandb.Api,
    *,
    entity: str,
    project: str,
    sweep_ids: list[str],
    no_prog: bool,
    retest_metrics_dir: Path | None = None,
) -> tuple[list[ClsRow], Counter[str]]:
    all_sk: Counter[str] = Counter()
    merged: list[ClsRow] = []
    seen: set[str] = set()
    for sid in tqdm(sweep_ids, desc="W&B sweeps", unit="sweep", disable=no_prog, dynamic_ncols=True):
        rows, sk = _collect_cls_sweep(
            api.sweep(f"{entity}/{project}/{sid}"),
            sid,
            project,
            entity,
            no_prog,
            retest_metrics_dir,
        )
        all_sk.update(sk)
        for r in rows:
            if r.run_id in seen:
                all_sk["duplicate_run"] += 1
                continue
            seen.add(r.run_id)
            merged.append(r)
    return merged, all_sk


def merge_reg(
    api: wandb.Api,
    *,
    entity: str,
    project: str,
    sweep_ids: list[str],
    no_prog: bool,
    retest_metrics_dir: Path | None = None,
) -> tuple[list[RegRow], Counter[str]]:
    all_sk: Counter[str] = Counter()
    merged: list[RegRow] = []
    seen: set[str] = set()
    for sid in tqdm(sweep_ids, desc="W&B sweeps", unit="sweep", disable=no_prog, dynamic_ncols=True):
        rows, sk = _collect_reg_sweep(
            api.sweep(f"{entity}/{project}/{sid}"),
            sid,
            project,
            entity,
            no_prog,
            retest_metrics_dir,
        )
        all_sk.update(sk)
        for r in rows:
            if r.run_id in seen:
                all_sk["duplicate_run"] += 1
                continue
            seen.add(r.run_id)
            merged.append(r)
    return merged, all_sk


def bucket_model_size(rows: list[Any]) -> dict[tuple[str, str], list[Any]]:
    b: dict[tuple[str, str], list[Any]] = {}
    for r in rows:
        b.setdefault((r.model_type, r.size), []).append(r)
    return b
