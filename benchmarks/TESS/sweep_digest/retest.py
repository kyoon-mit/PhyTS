"""Backfill W&B ``test/*`` metrics by loading sweep checkpoints and running ``trainer.test``.

Uses the same Lightning path as :mod:`sweep_classification` / :mod:`sweep_regression`
(PyTorch: ``trainer.test(..., ckpt_path=best.ckpt)``; JAX: load ``best.eqx`` then
``trainer.test(..., ckpt_path=None)``), so all model families behave like a normal
training run’s test phase—not a hand-rolled inference loop.

CLI::

    cd benchmarks/TESS
    uv run python -m sweep_digest.retest classification \\
      --from_file /path/to/sweep_ids.txt --entity ENTITY

Environment: ``TESS_CKPT_DIR``, ``TESS_DATA_DIR``, ``WANDB_ENTITY``,
``TESS_RETEST_METRICS_DIR`` (optional; see ``--retest_metrics_dir``).

Each successful ``trainer.test`` writes **local JSON** under
``sweep_digest/retest_metrics/`` (override with ``--retest_metrics_dir``) so the
digest can fill missing ``test/*`` even when W&B summary updates fail.

On Engaging, ``run_sweep.sh`` defaults to a **pool** tree (``$TESS_POOL_ROOT`` or
``/home/<user>/orcd/pool/...``). Interactive ``uv run`` jobs often **do not**
inherit those env vars, so this module also probes common pool locations and
picks the first directory that already contains ``classification/`` or
``regression/``, then falls back to repo-local ``checkpoints/sweeps``.
"""

from __future__ import annotations

import argparse
import getpass
import os
import sys
from argparse import Namespace
from collections.abc import Iterator
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

import numpy as np
import torch
import wandb
from tqdm.auto import tqdm

_TESS = Path(__file__).resolve().parent.parent
if str(_TESS) not in sys.path:
    sys.path.insert(0, str(_TESS))

_REPO_ROOT = _TESS.parent.parent
if str(_REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT / "src"))

import lightning as L

from sweep_classification import build_classification_sweep_task_and_datamodule
from sweep_digest.collect import bucket_model_size
from sweep_digest.core import (
    MODEL_ORDER,
    SIZE_ORDER,
    classification_val_sort_key,
    head_runs,
    load_sweep_id_list,
    regression_val_sort_key,
    resolve_wandb_entity,
    sorted_runs,
)
from sweep_regression import build_regression_sweep_task_and_datamodule
from sweep_utils import JAX_MODELS, ThrottledTQDMProgressBar

from sweep_digest.retest_metrics_store import resolve_retest_metrics_dir, save_retest_metrics

from dataloader.tess_dataloader import TESSClassificationDataModule
from tasks.TESS.classification_metrics import multiclass_extended_metrics
from tasks.TESS.tess_classification import TESSClassificationCE


def _looks_like_sweep_ckpt_root(p: Path) -> bool:
    """True if ``p`` looks like the root that holds ``classification/`` / ``regression/``."""
    if not p.is_dir():
        return False
    return any((p / name).is_dir() for name in ("classification", "regression"))


def _ckpt_root_candidates(cli: Path | None) -> list[Path]:
    """Ordered probe list (``run_sweep.sh`` + repo + cwd). Used when choosing a default root."""
    out: list[Path] = []
    if cli is not None:
        return [cli.expanduser()]
    if env := os.environ.get("TESS_CKPT_DIR"):
        out.append(Path(env).expanduser())
    if pool := os.environ.get("TESS_POOL_ROOT"):
        out.append(Path(pool).expanduser() / "checkpoints" / "sweeps")
    user = getpass.getuser()
    out.append(
        Path(f"/home/{user}/orcd/pool/UROP_2025_Summer/TimeSeriesPhysics/checkpoints/sweeps")
    )
    out.append(
        Path.home()
        / "orcd"
        / "pool"
        / "UROP_2025_Summer"
        / "TimeSeriesPhysics"
        / "checkpoints"
        / "sweeps"
    )
    out.append(_REPO_ROOT / "checkpoints" / "sweeps")
    out.append(_TESS / "checkpoints" / "sweeps")
    out.append(Path("checkpoints/sweeps"))
    return out


def _resolved_ckpt_root(cli: Path | None) -> Path:
    """Resolve sweep checkpoint root (env, pool heuristics, then first plausible existing dir)."""
    if cli is not None:
        return cli.expanduser().resolve()
    candidates = _ckpt_root_candidates(None)
    for cand in candidates:
        resolved = cand.resolve()
        if _looks_like_sweep_ckpt_root(resolved):
            return resolved
    for cand in candidates:
        resolved = cand.resolve()
        if resolved.is_dir():
            return resolved
    return candidates[-1].resolve()


def _pool_data_dir_candidates() -> list[Path]:
    out: list[Path] = []
    if pool := os.environ.get("TESS_POOL_ROOT"):
        out.append(
            Path(pool).expanduser() / "data_engaging" / "TESS" / ".cache" / "TESS"
        )
    user = getpass.getuser()
    out.append(
        Path(f"/home/{user}/orcd/pool/UROP_2025_Summer/TimeSeriesPhysics")
        / "data_engaging"
        / "TESS"
        / ".cache"
        / "TESS"
    )
    out.append(
        Path.home()
        / "orcd"
        / "pool"
        / "UROP_2025_Summer"
        / "TimeSeriesPhysics"
        / "data_engaging"
        / "TESS"
        / ".cache"
        / "TESS"
    )
    return out


def _resolved_data_dir(cli: str | None) -> str:
    """Prefer CLI, then ``TESS_DATA_DIR``, then pool heuristics like ``run_sweep.sh``."""
    if cli is not None:
        return cli
    env = os.environ.get("TESS_DATA_DIR")
    if env:
        return env
    for cand in _pool_data_dir_candidates():
        if cand.is_dir():
            return str(cand)
    return "data/TESS/.cache/TESS"


CLASSIFICATION_RETEST_SUMMARY_KEYS = (
    "test/acc",
    "test/balanced_acc",
    "test/precision_macro",
    "test/recall_macro",
    "test/f1_macro",
    "test/precision_weighted",
    "test/recall_weighted",
    "test/f1_weighted",
)


def classification_sweep_ckpt_dir(ckpt_root: Path | str, model_type: str, run_id: str) -> Path:
    return Path(ckpt_root) / "classification" / model_type / run_id


def regression_sweep_ckpt_dir(ckpt_root: Path | str, model_type: str, run_id: str) -> Path:
    return Path(ckpt_root) / "regression" / model_type / run_id


def _trainer_test_metrics(trainer: L.Trainer) -> dict[str, float]:
    """Collect final ``test/*`` scalars from Lightning (epoch-aggregated)."""
    out: dict[str, float] = {}
    for src in (trainer.callback_metrics, getattr(trainer, "logged_metrics", {}) or {}):
        for k, v in src.items():
            if not str(k).startswith("test/"):
                continue
            if v is None:
                continue
            try:
                if hasattr(v, "detach"):
                    v = v.detach()
                if hasattr(v, "item"):
                    v = v.item()
                fv = float(v)
            except (TypeError, ValueError):
                continue
            if np.isfinite(fv):
                out[str(k)] = fv
    return out


def lightning_retest_classification_metrics(
    *,
    wandb_cfg: dict[str, Any],
    run_id: str,
    ckpt_root: Path | str,
    data_dir: str,
    seq_len: int,
    num_workers: int,
) -> dict[str, float] | None:
    """Run ``trainer.test`` on ``best.ckpt`` / ``best.eqx``; return ``test/*`` metrics."""
    mt = wandb_cfg.get("model_type")
    if not isinstance(mt, str):
        return None
    cdir = classification_sweep_ckpt_dir(ckpt_root, mt, run_id)
    is_jax = mt in JAX_MODELS

    args_ns = Namespace(
        data_dir=data_dir,
        seq_len=seq_len,
        num_workers=num_workers,
        num_classes=int(wandb_cfg.get("num_classes", 8)),
        ckpt_dir=str(ckpt_root),
        max_epochs=1,
        patience=1,
    )

    if is_jax:
        eqx = cdir / "best.eqx"
        if not eqx.exists():
            return None
    else:
        ckpt_path = cdir / "best.ckpt"
        if not ckpt_path.exists():
            return None

    seed = int(wandb_cfg.get("seed", 0))
    L.seed_everything(seed, workers=not is_jax)
    task, dm, _ = build_classification_sweep_task_and_datamodule(wandb_cfg, args_ns)

    if is_jax:
        from models.utils.jax.load_model import load_model as jax_load_model

        task.jax_model, task.jax_model_state = jax_load_model(
            path=str(eqx),
            model=task.jax_model,
            model_state=task.jax_model_state,
        )
        ckpt_arg: str | None = None
    else:
        ckpt_arg = str(cdir / "best.ckpt")

    accel = "gpu" if torch.cuda.is_available() else "cpu"
    trainer = L.Trainer(
        accelerator=accel,
        devices=1,
        logger=False,
        enable_progress_bar=True,
        callbacks=[ThrottledTQDMProgressBar()],
    )
    trainer.test(task, datamodule=dm, ckpt_path=ckpt_arg)
    return _trainer_test_metrics(trainer) or None


def lightning_retest_regression_metrics(
    *,
    wandb_cfg: dict[str, Any],
    run_id: str,
    ckpt_root: Path | str,
    data_dir: str,
    seq_len: int,
    num_workers: int,
) -> dict[str, float] | None:
    mt = wandb_cfg.get("model_type")
    if not isinstance(mt, str):
        return None
    cdir = regression_sweep_ckpt_dir(ckpt_root, mt, run_id)
    is_jax = mt in JAX_MODELS

    args_ns = Namespace(
        data_dir=data_dir,
        seq_len=seq_len,
        num_workers=num_workers,
        ckpt_dir=str(ckpt_root),
        max_epochs=1,
        patience=1,
    )

    if is_jax:
        eqx = cdir / "best.eqx"
        if not eqx.exists():
            return None
    else:
        ckpt_path = cdir / "best.ckpt"
        if not ckpt_path.exists():
            return None

    seed = int(wandb_cfg.get("seed", 0))
    L.seed_everything(seed, workers=not is_jax)
    task, dm, _ = build_regression_sweep_task_and_datamodule(wandb_cfg, args_ns)

    if is_jax:
        from models.utils.jax.load_model import load_model as jax_load_model

        task.jax_model, task.jax_model_state = jax_load_model(
            path=str(eqx),
            model=task.jax_model,
            model_state=task.jax_model_state,
        )
        ckpt_arg = None
    else:
        ckpt_arg = str(cdir / "best.ckpt")

    accel = "gpu" if torch.cuda.is_available() else "cpu"
    trainer = L.Trainer(
        accelerator=accel,
        devices=1,
        logger=False,
        enable_progress_bar=True,
        callbacks=[ThrottledTQDMProgressBar()],
    )
    trainer.test(task, datamodule=dm, ckpt_path=ckpt_arg)
    m = _trainer_test_metrics(trainer)
    return m or None


# --- Backward-compatible names for analysis scripts ---------------------------------


def test_metrics_numpy(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    *,
    num_classes: int,
) -> dict[str, float]:
    """Flat ``test/*`` keys aligned with Lightning logging (numpy-only, no checkpoint)."""
    y_true = np.asarray(y_true, dtype=np.int64).ravel()
    y_pred = np.asarray(y_pred, dtype=np.int64).ravel()
    acc = float(np.mean(y_true == y_pred))
    per_class = [
        float(np.mean(y_pred[y_true == c] == c))
        for c in range(num_classes)
        if np.any(y_true == c)
    ]
    balanced = float(sum(per_class) / len(per_class)) if per_class else 0.0
    ext = multiclass_extended_metrics(y_true, y_pred, num_classes=num_classes)
    out: dict[str, float] = {
        "test/acc": acc,
        "test/balanced_acc": balanced,
    }
    for k, v in ext.items():
        out[f"test/{k}"] = float(v)
    for c in range(num_classes):
        mask = y_true == c
        if mask.sum() > 0:
            out[f"test/acc_class_{c}"] = float(np.mean(y_pred[mask] == c))
    return out


def reeval_classification_test_metrics(
    *,
    wandb_cfg: dict[str, Any],
    run_id: str,
    ckpt_root: Path | str,
    data_dir: str,
    seq_len: int,
    num_workers: int,
    device: str = "auto",
) -> dict[str, float] | None:
    del device  # Lightning picks GPU when available (matches sweep scripts)
    return lightning_retest_classification_metrics(
        wandb_cfg=wandb_cfg,
        run_id=run_id,
        ckpt_root=ckpt_root,
        data_dir=data_dir,
        seq_len=seq_len,
        num_workers=num_workers,
    )


def reeval_regression_test_metrics(
    *,
    wandb_cfg: dict[str, Any],
    run_id: str,
    ckpt_root: Path | str,
    data_dir: str,
    seq_len: int,
    num_workers: int,
    device: str = "auto",
) -> dict[str, float] | None:
    del device
    return lightning_retest_regression_metrics(
        wandb_cfg=wandb_cfg,
        run_id=run_id,
        ckpt_root=ckpt_root,
        data_dir=data_dir,
        seq_len=seq_len,
        num_workers=num_workers,
    )


def gather_classification_preds_for_plots(
    *,
    wandb_cfg: dict[str, Any],
    run_id: str,
    ckpt_root: Path | str,
    data_dir: str,
    seq_len: int,
    num_workers: int,
    device: str = "auto",
    split: Literal["val", "test"] = "test",
) -> tuple[np.ndarray, np.ndarray, list[str]] | None:
    """Load checkpoint and run inference (same builders as training)."""
    mt = wandb_cfg.get("model_type")
    if not isinstance(mt, str):
        return None
    cdir = classification_sweep_ckpt_dir(ckpt_root, mt, run_id)
    args_ns = Namespace(
        data_dir=data_dir,
        seq_len=seq_len,
        num_workers=num_workers,
        num_classes=int(wandb_cfg.get("num_classes", 8)),
        ckpt_dir=str(ckpt_root),
        max_epochs=1,
        patience=1,
    )
    is_jax = mt in JAX_MODELS

    if is_jax:
        eqx = cdir / "best.eqx"
        if not eqx.exists():
            return None
        L.seed_everything(int(wandb_cfg.get("seed", 0)), workers=False)
        task, dm, _ = build_classification_sweep_task_and_datamodule(wandb_cfg, args_ns)
        try:
            from models.utils.jax.load_model import load_model as jax_load_model

            task.jax_model, task.jax_model_state = jax_load_model(
                path=str(eqx),
                model=task.jax_model,
                model_state=task.jax_model_state,
            )
        except Exception:
            return None
        y_hat, y_true, names = _jax_cls_preds(task, dm, split=split)
    else:
        ckpt = cdir / "best.ckpt"
        if not ckpt.exists():
            return None
        L.seed_everything(int(wandb_cfg.get("seed", 0)), workers=True)
        try:
            task = TESSClassificationCE.load_from_checkpoint(str(ckpt), map_location="cpu")
        except Exception:
            return None
        dm = _cls_dm_only(wandb_cfg, args_ns)
        if device == "cuda":
            dev = torch.device("cuda")
        elif device == "cpu":
            dev = torch.device("cpu")
        else:
            dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        y_hat, y_true, names = _torch_cls_preds(task, dm, dev, split=split)

    return y_true, y_hat, names


def _cls_dm_only(wandb_cfg: Any, args_ns: Namespace) -> TESSClassificationDataModule:
    if isinstance(wandb_cfg, dict):
        cfg = Namespace(**wandb_cfg)
    else:
        cfg = wandb_cfg
    is_jax = cfg.model_type in JAX_MODELS
    return TESSClassificationDataModule(
        data_dir=args_ns.data_dir,
        batch_size=cfg.batch_size,
        num_workers=0 if is_jax else args_ns.num_workers,
        seq_len=args_ns.seq_len,
    )


def _torch_cls_preds(
    task: TESSClassificationCE,
    dm: TESSClassificationDataModule,
    device: torch.device,
    *,
    split: Literal["val", "test"],
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    task.eval()
    task.to(device)
    if split == "val":
        dm.setup("validate")
        loader = dm.val_dataloader()
        label_names = list(dm.val.label_names)
    else:
        dm.setup("test")
        loader = dm.test_dataloader()
        label_names = list(dm.test.label_names)
    preds_all: list[np.ndarray] = []
    labels_all: list[np.ndarray] = []
    with torch.no_grad():
        for batch in loader:
            flux, mask, label = batch
            flux = flux.to(device)
            mask = mask.to(device) if mask is not None else None
            logits = task(flux, mask)
            pr = logits.argmax(-1)
            preds_all.append(pr.cpu().numpy())
            labels_all.append(label.cpu().numpy())
    return np.concatenate(preds_all), np.concatenate(labels_all), label_names


def _jax_cls_preds(
    task: L.LightningModule,
    dm: TESSClassificationDataModule,
    *,
    split: Literal["val", "test"],
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    import jax
    import jax.numpy as jnp

    from models.utils.jax.training import jax_inference
    from models.utils.jax.utils import tensor_to_jax

    task.eval()
    if split == "val":
        dm.setup("validate")
        loader = dm.val_dataloader()
        label_names = list(dm.val.label_names)
    else:
        dm.setup("test")
        loader = dm.test_dataloader()
        label_names = list(dm.test.label_names)
    preds_all: list[np.ndarray] = []
    labels_all: list[np.ndarray] = []
    for batch in loader:
        batch_jax = tensor_to_jax(batch)
        x, y = task._prepare_batch(batch_jax)
        keys = task._batched_keys(task.key, jax.tree.leaves(x)[0].shape[0])
        outputs = jax_inference(task.jax_model, x, task.jax_model_state, keys)
        preds = jnp.argmax(outputs, axis=-1)
        preds_all.append(np.asarray(preds))
        labels_all.append(np.asarray(y))
    return np.concatenate(preds_all), np.concatenate(labels_all), label_names


# --- CLI ---------------------------------------------------------------------------


def load_run_id_list(*ids: str | None, from_file: Path | None = None) -> list[str]:
    out = [str(x).strip() for x in ids if x and str(x).strip()]
    if from_file is not None:
        for line in Path(from_file).read_text(encoding="utf-8").splitlines():
            s = line.split("#", 1)[0].strip()
            if s:
                out.append(s)
    return list(dict.fromkeys(out))


def _summary_get_finite(run: Any, key: str) -> float | None:
    s = run.summary
    if s is None or not hasattr(s, "get"):
        return None
    v = s.get(key)
    if v is None:
        return None
    try:
        x = float(v)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(x):
        return None
    return x


def _mini_cls_stub(run: Any) -> Any:
    c = dict(run.config)

    class R:
        pass

    r = R()
    r.run_id = str(run.id)
    r.model_type = c.get("model_type")
    r.size = str(c.get("size", ""))
    r.val_balanced = _summary_get_finite(run, "val/balanced_acc")
    r.val_acc = _summary_get_finite(run, "val/acc")
    return r


def _mini_reg_stub(run: Any) -> Any:
    c = dict(run.config)

    class R:
        pass

    r = R()
    r.run_id = str(run.id)
    r.model_type = c.get("model_type")
    r.size = str(c.get("size", ""))
    r.val_r2 = _summary_get_finite(run, "val/r2")
    r.val_mae = _summary_get_finite(run, "val/mae")
    r.val_rmse = _summary_get_finite(run, "val/rmse")
    return r


def topk_run_ids_per_cell(
    runs: list[Any],
    *,
    task: Literal["classification", "regression"],
    top_k: int,
) -> set[str]:
    val_key = classification_val_sort_key if task == "classification" else regression_val_sort_key
    stubs: list[Any] = []
    for run in runs:
        if getattr(run, "state", None) != "finished":
            continue
        c = dict(run.config)
        mt, sz = c.get("model_type"), str(c.get("size", ""))
        if mt not in MODEL_ORDER or sz not in SIZE_ORDER:
            continue
        stubs.append(_mini_cls_stub(run) if task == "classification" else _mini_reg_stub(run))
    buckets = bucket_model_size(stubs)
    out: set[str] = set()
    k = max(1, int(top_k))
    for m in MODEL_ORDER:
        for s in SIZE_ORDER:
            g = sorted_runs(buckets.get((m, s), []), val_key)
            for row in head_runs(g, k):
                out.add(row.run_id)
    return out


def needs_retest(
    run: Any,
    task: Literal["classification", "regression"],
    *,
    force: bool,
) -> bool:
    if force:
        return True
    if task == "regression":
        return _summary_get_finite(run, "test/r2") is None
    return any(_summary_get_finite(run, k) is None for k in CLASSIFICATION_RETEST_SUMMARY_KEYS)


def iter_sweep_runs(sweep: Any, *, disable_tqdm: bool) -> Iterator[Any]:
    yield from tqdm(list(sweep.runs), desc="runs", unit="run", disable=disable_tqdm, dynamic_ncols=True)


def collect_runs(
    api: wandb.Api,
    *,
    entity: str,
    project: str,
    sweep_ids: list[str],
    run_ids: list[str],
    no_progress: bool,
) -> list[Any]:
    seen: set[str] = set()
    out: list[Any] = []
    for sid in sweep_ids:
        sw = api.sweep(f"{entity}/{project}/{sid}")
        for run in iter_sweep_runs(sw, disable_tqdm=no_progress):
            if run.id in seen:
                continue
            seen.add(run.id)
            out.append(run)
    for rid in run_ids:
        if rid in seen:
            continue
        seen.add(rid)
        out.append(api.run(f"{entity}/{project}/{rid}"))
    return out


def push_summary_metrics(
    api: wandb.Api,
    entity: str,
    project: str,
    run_id: str,
    metrics: dict[str, float],
    *,
    audit: bool,
) -> None:
    if not metrics:
        return
    path = f"{entity}/{project}/{run_id}"
    r = api.run(path)
    try:
        r.load(force=True)
    except Exception:
        pass
    # Cached HTTPSummary may pre-date full ``summaryMetrics``; force rebuild.
    setattr(r, "_summary", None)
    payload = {k: float(v) for k, v in metrics.items()}
    if audit:
        payload["retest_missing_completed_at"] = datetime.now(timezone.utc).isoformat()
    # ``update()`` ends with ``_write(commit=True)`` (required for persistence).
    r.summary.update(payload)


def main() -> None:
    p = argparse.ArgumentParser(
        description=(
            "Backfill missing test/* via Lightning trainer.test on sweep checkpoints "
            "(all Torch + JAX families). Default: top-K per cell only; use --retest_all_missing for every run."
        )
    )
    p.add_argument("task", choices=["classification", "regression"])
    p.add_argument("--entity", default=None)
    p.add_argument("--project", default="TimeSeriesPhysics")
    p.add_argument("--sweep_id", action="append", default=[], metavar="ID")
    p.add_argument("--from_file", type=Path, default=None)
    p.add_argument("--run_id", action="append", default=[], metavar="RUN_ID")
    p.add_argument("--run_id_file", type=Path, default=None)
    p.add_argument("--ckpt_dir", type=Path, default=None)
    p.add_argument("--data_dir", default=None)
    p.add_argument("--seq_len", type=int, default=1100)
    p.add_argument("--num_workers", type=int, default=8)
    p.add_argument("--dry_run", action="store_true")
    p.add_argument("--force", action="store_true")
    p.add_argument("--include_non_finished", action="store_true")
    p.add_argument("--no_progress", action="store_true")
    p.add_argument("--audit", action="store_true")
    p.add_argument("--retest_all_missing", action="store_true")
    p.add_argument("--top_k_per_cell", type=int, default=3, metavar="K")
    p.add_argument("--verbose", action="store_true")
    p.add_argument(
        "--retest_metrics_dir",
        type=Path,
        default=None,
        help="Local JSON backup directory (default: sweep_digest/retest_metrics or TESS_RETEST_METRICS_DIR).",
    )
    p.add_argument(
        "--no_wandb_push",
        action="store_true",
        help="Only write local JSON; do not call the W&B API to update run summary.",
    )
    p.add_argument(
        "--no_local_save",
        action="store_true",
        help="Skip writing per-run JSON (not recommended).",
    )
    args = p.parse_args()

    ckpt = _resolved_ckpt_root(args.ckpt_dir)
    data_dir = _resolved_data_dir(args.data_dir)
    print(f"[retest] ckpt_root={ckpt}", file=sys.stderr)
    print(f"[retest] data_dir={data_dir}", file=sys.stderr)
    if not _looks_like_sweep_ckpt_root(ckpt):
        print(
            "[retest] warning: ckpt_root has no classification/ or regression/ subdirs — "
            "set TESS_CKPT_DIR or TESS_POOL_ROOT, or pass --ckpt_dir",
            file=sys.stderr,
        )

    sids = (
        load_sweep_id_list(*args.sweep_id, from_file=args.from_file)
        if (args.sweep_id or args.from_file)
        else []
    )
    extra_runs = load_run_id_list(*args.run_id, from_file=args.run_id_file)
    if not sids and not extra_runs:
        p.error("Provide --sweep_id / --from_file / --run_id / --run_id_file")

    metrics_dir = resolve_retest_metrics_dir(args.retest_metrics_dir)
    print(f"[retest] retest_metrics_dir={metrics_dir}", file=sys.stderr)

    api = wandb.Api()
    ent = resolve_wandb_entity(api, args.entity)
    runs = collect_runs(
        api,
        entity=ent,
        project=args.project,
        sweep_ids=sids,
        run_ids=extra_runs,
        no_progress=args.no_progress,
    )

    allowed: set[str] | None = None
    if not args.retest_all_missing:
        allowed = topk_run_ids_per_cell(
            runs,
            task=args.task,
            top_k=args.top_k_per_cell,
        )

    task_lm = args.task
    n_need = n_computed = n_pushed = n_dry = n_skip = n_fail = n_skip_not_topk = n_saved_local = 0
    for run in runs:
        st = getattr(run, "state", None)
        if st != "finished" and not args.include_non_finished:
            continue
        cfg = dict(run.config)
        mt = cfg.get("model_type")
        if not isinstance(mt, str):
            n_skip += 1
            continue
        if allowed is not None and str(run.id) not in allowed:
            n_skip_not_topk += 1
            continue
        if not needs_retest(run, task_lm, force=args.force):
            continue
        n_need += 1
        fn = (
            lightning_retest_classification_metrics
            if task_lm == "classification"
            else lightning_retest_regression_metrics
        )
        try:
            metrics = fn(
                wandb_cfg=cfg,
                run_id=run.id,
                ckpt_root=ckpt,
                data_dir=data_dir,
                seq_len=args.seq_len,
                num_workers=args.num_workers,
            )
        except Exception as e:
            n_fail += 1
            print(f"[fail] {run.id} ({mt}): {e}", file=sys.stderr)
            if args.verbose:
                import traceback

                traceback.print_exc()
            continue
        if not metrics:
            n_fail += 1
            sub = "classification" if task_lm == "classification" else "regression"
            print(
                f"[fail] {run.id} ({mt}): no checkpoint or test produced no metrics "
                f"({ckpt}/{sub}/{mt}/{run.id}/best.ckpt or best.eqx)",
                file=sys.stderr,
            )
            continue
        n_computed += 1
        if args.dry_run:
            n_dry += 1
            print(f"[dry_run] would update {run.id} ({mt}): {sorted(metrics.keys())}")
            continue

        wandb_err: str | None = None
        if args.no_wandb_push:
            wandb_err = "skipped_no_wandb_push"
        else:
            try:
                push_summary_metrics(api, ent, args.project, run.id, metrics, audit=args.audit)
            except Exception as e:
                wandb_err = str(e)
                n_fail += 1
                print(f"[fail] summary update {run.id}: {e}", file=sys.stderr)
            else:
                n_pushed += 1

        if not args.no_local_save:
            try:
                save_retest_metrics(
                    root=metrics_dir,
                    entity=ent,
                    project=args.project,
                    run_id=str(run.id),
                    task="classification" if task_lm == "classification" else "regression",
                    metrics=metrics,
                    wandb_push_error=wandb_err,
                )
                n_saved_local += 1
            except Exception as e:
                print(f"[warn] local metrics save {run.id}: {e}", file=sys.stderr)

        if wandb_err is None:
            print(f"[ok] {run.id} ({mt})")
        elif wandb_err == "skipped_no_wandb_push":
            print(f"[ok] {run.id} ({mt}) local_json_only", file=sys.stderr)
        else:
            print(
                f"[warn] {run.id} ({mt}): W&B push failed; metrics saved under {metrics_dir}",
                file=sys.stderr,
            )

    print(
        f"done: need={n_need} test_metrics_ok={n_computed} wandb_updated={n_pushed} "
        f"local_json_saved={n_saved_local} dry_run_skipped_push={n_dry} failed={n_fail} "
        f"skipped_no_model_type={n_skip} skipped_not_in_top_k={n_skip_not_topk}"
    )


if __name__ == "__main__":
    main()
