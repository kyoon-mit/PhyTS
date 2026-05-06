"""Local JSON backup of ``trainer.test`` metrics for digest overlay when W&B summary is stale.

Files are written by :mod:`sweep_digest.retest` (one JSON per run). The digest
loader merges these values when the Public API ``run.summary`` lacks a given
``test/*`` key.

Environment:

    TESS_RETEST_METRICS_DIR
        Override default directory (same as :option:`--retest_metrics_dir`).
"""

from __future__ import annotations

import json
import os
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

SCHEMA_VERSION = 1


def default_retest_metrics_dir() -> Path:
    """Package-local ``sweep_digest/retest_metrics`` (created on first save)."""
    return Path(__file__).resolve().parent / "retest_metrics"


def resolve_retest_metrics_dir(cli: Path | None) -> Path:
    """Directory for retest JSON files (CLI > env > default)."""
    if cli is not None:
        return cli.expanduser().resolve()
    env = os.environ.get("TESS_RETEST_METRICS_DIR", "").strip()
    if env:
        return Path(env).expanduser().resolve()
    return default_retest_metrics_dir()


def _safe_filename_part(s: str) -> str:
    return re.sub(r"[^a-zA-Z0-9._+-]+", "_", str(s)).strip("_") or "unknown"


def retest_metrics_path(entity: str, project: str, run_id: str, root: Path) -> Path:
    name = f"{_safe_filename_part(entity)}__{_safe_filename_part(project)}__{_safe_filename_part(run_id)}.json"
    return root / name


def save_retest_metrics(
    *,
    root: Path,
    entity: str,
    project: str,
    run_id: str,
    task: Literal["classification", "regression"],
    metrics: dict[str, float],
    wandb_push_error: str | None = None,
) -> Path:
    """Write or overwrite one run's retest metrics JSON."""
    root.mkdir(parents=True, exist_ok=True)
    path = retest_metrics_path(entity, project, run_id, root)
    rec: dict[str, Any] = {
        "schema": SCHEMA_VERSION,
        "entity": entity,
        "project": project,
        "run_id": str(run_id),
        "task": task,
        "written_at": datetime.now(timezone.utc).isoformat(),
        "metrics": {str(k): float(v) for k, v in metrics.items()},
        "wandb_push_error": wandb_push_error,
    }
    path.write_text(json.dumps(rec, indent=2, sort_keys=False) + "\n", encoding="utf-8")
    return path


def load_retest_metrics(
    entity: str,
    project: str,
    run_id: str,
    root: Path,
) -> dict[str, float] | None:
    """Return ``metrics`` dict if a JSON file exists; otherwise ``None``."""
    path = retest_metrics_path(entity, project, run_id, root)
    if not path.is_file():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    raw = data.get("metrics")
    if not isinstance(raw, dict):
        return None
    out: dict[str, float] = {}
    for k, v in raw.items():
        try:
            out[str(k)] = float(v)
        except (TypeError, ValueError):
            continue
    return out if out else None


def summary_dict_with_retest_overlay(
    run: Any,
    *,
    entity: str,
    project: str,
    retest_root: Path,
) -> tuple[dict[str, Any], bool]:
    """Build a dict like ``run.summary`` with missing ``test/*`` filled from local JSON.

    Returns
    -------
    merged, used_retest_file
        ``used_retest_file`` is True if any key came from the retest JSON.
    """
    s = run.summary
    base: dict[str, Any] = {}
    if hasattr(s, "keys"):
        for k in s.keys():
            base[str(k)] = s.get(k)
    elif isinstance(s, dict):
        base = dict(s)

    overlay = load_retest_metrics(entity, project, str(run.id), retest_root)
    if not overlay:
        return base, False

    used = False
    for k, v in overlay.items():
        cur = base.get(k)
        if cur is None:
            base[k] = v
            used = True
    return base, used
