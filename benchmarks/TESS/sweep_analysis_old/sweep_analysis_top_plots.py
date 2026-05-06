"""Per-cell sweep winners: figures matching :mod:`tasks.TESS.eval_plots`.

Loads ``best.ckpt`` / ``best.eqx`` and runs inference on **val** or **test** (default:
**test**, aligned with held-out metrics in the digest).

Outputs under ``out_dir/top_model_<split>_plots/<regression|classification>/``.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path
from typing import Any, Literal

import lightning as L
import matplotlib.pyplot as plt
import numpy as np

_BENCH = Path(__file__).resolve().parent
if str(_BENCH) not in sys.path:
    sys.path.insert(0, str(_BENCH))
_TESS = Path(__file__).resolve().parent.parent
if str(_TESS) not in sys.path:
    sys.path.insert(0, str(_TESS))

from sweep_regression import build_regression_sweep_task_and_datamodule
from sweep_digest.retest import gather_classification_preds_for_plots
from sweep_utils import JAX_MODELS, _REPO_ROOT

if str(_REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT / "src"))

import torch

from dataloader.tess_dataloader import TESSRegressionDataModule
from tasks.TESS.eval_plots import (
    classification_results_dict,
    make_classification_figure,
    make_regression_figure,
    regression_results_dict,
)
from tasks.TESS.tess_regression import TESSRegressionMSE


def regression_sweep_ckpt_dir(ckpt_root: Path | str, model_type: str, run_id: str) -> Path:
    return Path(ckpt_root) / "regression" / model_type / run_id


def _safe_segment(s: str) -> str:
    return re.sub(r"[^\w\-]+", "_", s)


def _regression_datamodule(wandb_cfg: dict[str, Any], args_ns: argparse.Namespace) -> TESSRegressionDataModule:
    mt = wandb_cfg.get("model_type")
    jax = mt in JAX_MODELS
    return TESSRegressionDataModule(
        data_dir=args_ns.data_dir,
        batch_size=int(wandb_cfg.get("batch_size", 32)),
        num_workers=0 if jax else args_ns.num_workers,
        seq_len=args_ns.seq_len,
    )


def _torch_regression_y_yhat(
    task: TESSRegressionMSE,
    dm: TESSRegressionDataModule,
    device: torch.device,
    *,
    split: Literal["val", "test"],
) -> tuple[np.ndarray, np.ndarray]:
    task.eval()
    task.to(device)
    if split == "val":
        dm.setup("validate")
        loader = dm.val_dataloader()
    else:
        dm.setup("test")
        loader = dm.test_dataloader()
    ys: list[np.ndarray] = []
    yhs: list[np.ndarray] = []
    with torch.no_grad():
        for batch in loader:
            flux, mask, frot = batch
            flux = flux.to(device)
            mask = mask.to(device)
            pred = task(flux, mask)
            yhs.append(pred.cpu().numpy().astype(np.float64))
            ys.append(frot.cpu().numpy().astype(np.float64))
    return np.concatenate(ys), np.concatenate(yhs)


def _jax_regression_y_yhat(
    task: Any, dm: TESSRegressionDataModule, *, split: Literal["val", "test"]
) -> tuple[np.ndarray, np.ndarray]:
    import jax

    from models.utils.jax.training import jax_inference
    from models.utils.jax.utils import tensor_to_jax

    task.eval()
    if split == "val":
        dm.setup("validate")
        loader = dm.val_dataloader()
    else:
        dm.setup("test")
        loader = dm.test_dataloader()
    ys: list[np.ndarray] = []
    yhs: list[np.ndarray] = []
    for batch in loader:
        batch_jax = tensor_to_jax(batch)
        x, y = task._prepare_batch(batch_jax)
        keys = task._batched_keys(task.key, jax.tree.leaves(x)[0].shape[0])
        out = jax_inference(task.jax_model, x, task.jax_model_state, keys)
        yhs.append(np.asarray(out.squeeze(-1), dtype=np.float64))
        ys.append(np.asarray(y, dtype=np.float64))
    return np.concatenate(ys), np.concatenate(yhs)


def gather_regression_y_yhat_for_plots(
    *,
    wandb_cfg: dict[str, Any],
    run_id: str,
    ckpt_root: Path | str,
    data_dir: str,
    seq_len: int,
    num_workers: int,
    device: str = "auto",
    split: Literal["val", "test"] = "test",
) -> tuple[np.ndarray, np.ndarray] | None:
    """Return ``(y_true, y_hat)`` on **val** or **test** for one sweep winner."""
    mt = wandb_cfg.get("model_type")
    if not isinstance(mt, str):
        return None
    cdir = regression_sweep_ckpt_dir(ckpt_root, mt, run_id)
    args_ns = argparse.Namespace(
        data_dir=data_dir,
        seq_len=seq_len,
        num_workers=num_workers,
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
        task, dm, _ = build_regression_sweep_task_and_datamodule(wandb_cfg, args_ns)
        try:
            from models.utils.jax.load_model import load_model as jax_load_model

            task.jax_model, task.jax_model_state = jax_load_model(
                path=str(eqx),
                model=task.jax_model,
                model_state=task.jax_model_state,
            )
        except Exception:
            return None
        return _jax_regression_y_yhat(task, dm, split=split)

    ckpt = cdir / "best.ckpt"
    if not ckpt.exists():
        return None
    try:
        task = TESSRegressionMSE.load_from_checkpoint(str(ckpt), map_location="cpu")
    except Exception:
        return None
    L.seed_everything(int(wandb_cfg.get("seed", 0)), workers=True)
    dm = _regression_datamodule(wandb_cfg, args_ns)
    if device == "auto":
        dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    elif device == "cuda":
        dev = torch.device("cuda")
    else:
        dev = torch.device("cpu")
    return _torch_regression_y_yhat(task, dm, dev, split=split)


def save_cell_winner_figures(
    api: Any,
    *,
    entity: str,
    project: str,
    task_kind: Literal["regression", "classification"],
    winners: dict[tuple[str, str], str],
    ckpt_root: Path,
    data_dir: str,
    seq_len: int,
    num_workers: int,
    inference_device: str,
    out_dir: Path,
    split: Literal["val", "test"] = "test",
) -> list[str]:
    """Fetch each winner's wandb config, load checkpoint, save figure aligned with ``eval_plots``.

    winners
        Maps ``(model_type, size_code)`` → wandb ``run_id`` (same cell as sweep digest).
    """
    split_dir = f"top_model_{split}_plots"
    sub = out_dir / split_dir / task_kind
    sub.mkdir(parents=True, exist_ok=True)
    lines: list[str] = []

    for (mk, sz), rid in sorted(winners.items()):
        outp = sub / mk / f"{_safe_segment(sz)}_{rid}_{split}.png"
        outp.parent.mkdir(parents=True, exist_ok=True)

        title = f"{mk} ({sz}); run={rid} ({split})"

        try:
            run = api.run(f"{entity}/{project}/{rid}")
            cfg_dict = dict(run.config) if isinstance(run.config, dict) else {}
        except Exception as exc:
            lines.append(f"{mk}/{sz} {rid}: wandb fetch failed ({exc})")
            continue

        try:
            if task_kind == "regression":
                arr = gather_regression_y_yhat_for_plots(
                    wandb_cfg=cfg_dict,
                    run_id=rid,
                    ckpt_root=ckpt_root,
                    data_dir=data_dir,
                    seq_len=seq_len,
                    num_workers=num_workers,
                    device=inference_device,
                    split=split,
                )
                if arr is None:
                    lines.append(f"{mk}/{sz} {rid}: regression checkpoint or inference failed")
                    continue
                y_true, y_hat = arr
                res = regression_results_dict(y_true, y_hat)
                fig = make_regression_figure(res, title)
            else:
                arr = gather_classification_preds_for_plots(
                    wandb_cfg=cfg_dict,
                    run_id=rid,
                    ckpt_root=ckpt_root,
                    data_dir=data_dir,
                    seq_len=seq_len,
                    num_workers=num_workers,
                    device=inference_device,
                    split=split,
                )
                if arr is None:
                    lines.append(f"{mk}/{sz} {rid}: classification checkpoint or inference failed")
                    continue
                y_true, y_hat, names = arr
                res = classification_results_dict(y_true, y_hat, names)
                fig = make_classification_figure(res, title, names)
            fig.savefig(outp, dpi=150)
            plt.close(fig)
            lines.append(f"{mk}/{sz} {rid}: wrote {outp}")
        except Exception as exc:
            lines.append(f"{mk}/{sz} {rid}: figure error ({exc})")

    return lines
