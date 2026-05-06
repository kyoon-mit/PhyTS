"""Constants, W&B helpers, val-sort keys, and aggregation."""

from __future__ import annotations

import argparse
import math
import os
from collections import Counter
from collections.abc import Iterator
from pathlib import Path
from typing import Any, Callable, Sequence

import numpy as np
import wandb
from tqdm.auto import tqdm

MODEL_ORDER: tuple[str, ...] = (
    "mlp",
    "linoss_imex",
    "linoss_damped",
    "s4d",
    "cnn",
    "cnn_attn",
    "transformer",
)
SIZE_ORDER: tuple[str, ...] = ("xs", "sm", "md", "lg")
HEATMAP_SIZE_LABELS: tuple[str, ...] = ("~10k", "~100k", "~300k", "~700k")
SWEEP_HPARAM_KEYS: tuple[str, ...] = ("lr", "dropout", "weight_decay", "batch_size", "seed")
HPARAM_HEATMAP_NUMERIC_KEYS: tuple[str, ...] = ("lr", "dropout", "weight_decay", "batch_size")


def summary_get(summary: Any, key: str) -> float | None:
    if summary is None or not hasattr(summary, "get"):
        return None
    v = summary.get(key)
    if v is None:
        return None
    try:
        x = float(v)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(x):
        return None
    return x


def classification_val_sort_key(r: Any) -> tuple[float, float]:
    if r.val_balanced is not None and np.isfinite(float(r.val_balanced)):
        return (0.0, -float(r.val_balanced))
    if r.val_acc is not None and np.isfinite(float(r.val_acc)):
        return (1.0, -float(r.val_acc))
    return (9.0, 0.0)


def regression_val_sort_key(r: Any) -> tuple[float, float]:
    if r.val_r2 is not None and np.isfinite(float(r.val_r2)):
        return (0.0, -float(r.val_r2))
    if r.val_mae is not None and np.isfinite(float(r.val_mae)):
        return (1.0, float(r.val_mae))
    if r.val_rmse is not None and np.isfinite(float(r.val_rmse)):
        return (2.0, float(r.val_rmse))
    return (9.0, 0.0)


def cls_has_val_ranking(r: Any) -> bool:
    return r.val_balanced is not None or r.val_acc is not None


def reg_has_val_ranking(r: Any) -> bool:
    return r.val_r2 is not None or r.val_mae is not None or r.val_rmse is not None


def resolve_wandb_entity(api: wandb.Api, entity: str | None) -> str:
    if entity and str(entity).strip():
        return str(entity).strip()
    env_e = os.environ.get("WANDB_ENTITY", "").strip()
    if env_e:
        return env_e
    viewer = api.viewer
    if viewer is None:
        raise RuntimeError("W&B entity unknown: pass --entity or set WANDB_ENTITY")
    u = getattr(viewer, "username", None) or getattr(viewer, "entity", None)
    if not u:
        raise RuntimeError("W&B entity unknown: pass --entity or set WANDB_ENTITY")
    return str(u)


def load_sweep_id_list(*ids: str | None, from_file: Path | None = None) -> list[str]:
    out = [str(x).strip() for x in ids if x and str(x).strip()]
    if from_file is not None:
        for line in Path(from_file).read_text(encoding="utf-8").splitlines():
            s = line.split("#", 1)[0].strip()
            if s:
                out.append(s)
    dedup = list(dict.fromkeys(out))
    if not dedup:
        raise SystemExit("No sweep ids (--sweep_id or --from_file).")
    return dedup


def iter_sweep_runs(sweep: Any, *, desc: str, disable_tqdm: bool) -> Iterator[Any]:
    yield from tqdm(list(sweep.runs), desc=desc, unit="run", disable=disable_tqdm, dynamic_ncols=True)


def add_wandb_cli(p: argparse.ArgumentParser) -> None:
    p.add_argument("--entity", default=None, help="W&B entity (default: env or API)")
    p.add_argument("--project", default="TimeSeriesPhysics", help="W&B project")
    p.add_argument("--sweep_id", action="append", default=[], metavar="ID")
    p.add_argument("--from_file", type=Path, default=None)


def sorted_runs(runs: list[Any], key: Callable[[Any], tuple[float, float]]) -> list[Any]:
    return sorted(runs, key=key)


def head_runs(runs: list[Any], n: int) -> list[Any]:
    return runs[: min(n, len(runs))] if n > 0 else []


def top_fraction_runs(runs: list[Any], fraction: float) -> list[Any]:
    n = len(runs)
    if n == 0:
        return []
    return runs[: max(1, math.ceil(fraction * n))]


def metrics_mean_std(
    runs: Sequence[Any],
    getters: dict[str, Callable[[Any], float | None]],
) -> dict[str, dict[str, float | int | None]]:
    out: dict[str, dict[str, float | int | None]] = {}
    for name, getter in getters.items():
        vals = [
            float(v)
            for r in runs
            if (v := getter(r)) is not None and np.isfinite(float(v))
        ]
        if not vals:
            out[name] = {"mean": None, "std": None, "n": 0}
            continue
        a = np.asarray(vals, dtype=float)
        std = float(a.std(ddof=0)) if len(a) > 1 else 0.0
        out[name] = {"mean": float(a.mean()), "std": std, "n": len(vals)}
    return out


def _float_hp(x: Any) -> float | None:
    if x is None or isinstance(x, bool):
        return None
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


def _round_sig(x: float, sig: int = 4) -> str:
    from math import floor, log10

    if x == 0:
        return "0"
    nd = max(0, min(12, sig - 1 - floor(log10(abs(x)))))
    return f"{x:.{nd}f}"


def aggregate_hyperparams_mode(runs: Sequence[Any], hp_keys: Sequence[str]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for k in hp_keys:
        present = [
            v
            for r in runs
            if (v := r.hyperparams.get(k)) is not None and str(v).strip() != ""
        ]
        if not present:
            out[k] = {"rule": "empty", "value": None, "n": 0}
            continue
        if all(_float_hp(v) is not None for v in present):
            c = Counter(_round_sig(float(v)) for v in present)
            mode_s, sup = c.most_common(1)[0]
            out[k] = {"rule": "mode_numeric_rounded", "value": float(mode_s), "support": sup, "n": len(present)}
        else:
            c = Counter(str(v) for v in present)
            mode, sup = c.most_common(1)[0]
            out[k] = {"rule": "plurality_mode", "value": mode, "support": sup, "n": len(present)}
    return out


def hyperparam_cell_text(nk: str, cell: dict[str, Any]) -> str:
    if cell.get("rule") != "mode_numeric_rounded" or cell.get("value") is None:
        if cell.get("rule") == "plurality_mode" and cell.get("value") is not None:
            return str(cell["value"])[:12]
        return ""
    v = float(cell["value"])
    return f"{v:.0f}" if nk == "batch_size" else f"{v:.3g}"


def numeric_value_for_hparam_heatmap(hp_key: str, cell: dict[str, Any]) -> float | None:
    if cell.get("rule") != "mode_numeric_rounded" or cell.get("value") is None:
        return None
    x = float(cell["value"])
    if hp_key == "lr":
        return float(np.log10(max(x, 1e-12)))
    if hp_key == "weight_decay":
        return float(np.log10(max(x, 1e-20)))
    return x
