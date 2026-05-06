"""
Evaluate a LIGO regression model on the test set: RMSE and R² per target variable.

Architecture and backend (PyTorch vs JAX) are auto-detected from the YAML
training config task class:

    python benchmarks/LIGO/eval_pipeline.py \\
        --config     configs/LIGO/train_ligo_linoss_regression.yaml \\
        --checkpoint checkpoints/ligo_linoss_regression/best.eqx

Detection rules:
  * task class_path module contains '_jax'  -> JAX/equinox  (.eqx checkpoint)
  * otherwise                               -> PyTorch Lightning (.ckpt)

The display name (CNN / S4D / LinOSS / ...) is taken from the inner model
``class_path``. Metrics (RMSE, R²) are reported per target variable in the
physical units produced by the model (no z-score inversion is applied, since
the LIGO dataloader does not z-score targets).
"""

import argparse
import csv
import importlib
import os
import sys
from pathlib import Path

import numpy as np
import torch
import yaml

_REPO_ROOT = Path(__file__).resolve().parents[2]
_SRC = _REPO_ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from dataloader.LIGO_dataloader import LIGODataModule  # noqa: E402


_ARCH_NAMES = {
    "models.s4d.S4Model":                       "S4D",
    "models.conv_regressor.Conv1DRegressor":    "CNN",
    "models.linoss.LinOSS":                     "LinOSS",
    "models.transformer.TransformerClassifier": "Transformer",
}


# ─── helpers ────────────────────────────────────────────────────────────────

def _load_cfg(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def _import_class(class_path: str):
    module_name, class_name = class_path.rsplit(".", 1)
    return getattr(importlib.import_module(module_name), class_name)


def _instantiate(spec: dict):
    """Instantiate ``{class_path, init_args}`` from a YAML config block."""
    return _import_class(spec["class_path"])(**(spec.get("init_args") or {}))


def detect_kind(cfg: dict) -> tuple[str, str, str]:
    """Return (kind, arch_name, inner_class_path).

    ``kind`` is ``'pt'`` or ``'jax'`` determined by the task class_path:
    task modules ending in ``_jax`` are JAX/equinox; everything else is PyTorch.
    The inner model lives at ``model.init_args.model`` for both backends.
    """
    task_class_path: str = cfg["model"]["class_path"]
    task_module = task_class_path.rsplit(".", 1)[0]
    kind = "jax" if task_module.endswith("_jax") else "pt"

    inner = (cfg["model"].get("init_args") or {}).get("model")
    if inner is None:
        raise ValueError(
            "Could not find model.init_args.model in config. "
            "Expected a nested 'model' block with the inner architecture."
        )
    inner_path = inner["class_path"]
    arch_name = _ARCH_NAMES.get(inner_path, inner_path.rsplit(".", 1)[1])
    return kind, arch_name, inner_path


def build_datamodule(cfg: dict) -> LIGODataModule:
    dm = LIGODataModule(**cfg["data"]["init_args"])
    dm.setup("test")
    return dm


# ─── PyTorch loading ─────────────────────────────────────────────────────────

def load_pt_model(cfg: dict, ckpt_path: str, device: torch.device) -> torch.nn.Module:
    """Load inner PyTorch model from a Lightning checkpoint.

    RegressionMSE stores weights under the 'model.*' key prefix in the state dict.
    We instantiate the inner nn.Module and load only those weights.
    """
    model_cfg = cfg["model"]["init_args"]["model"]
    model = _instantiate(model_cfg)

    ckpt = torch.load(ckpt_path, map_location=device, weights_only=True)
    state_dict = {
        k[len("model."):]: v
        for k, v in ckpt["state_dict"].items()
        if k.startswith("model.")
    }
    model.load_state_dict(state_dict)
    return model.to(device).eval()


@torch.no_grad()
def predict_pt(
    model: torch.nn.Module,
    dm: LIGODataModule,
    device: torch.device,
) -> np.ndarray:
    """Run PT model over the test set.

    Batch X is (B, n_ifos, L); the model expects (B, L, n_ifos), so we transpose.
    Returns array of shape (N, n_targets).
    """
    preds = []
    for X, _y, _z in dm.test_dataloader():
        out = model(X.transpose(1, 2).to(device))  # (B, L, n_ifos) -> (B, n_targets)
        preds.append(out.cpu().numpy())
    return np.concatenate(preds, axis=0)


# ─── JAX/equinox loading ─────────────────────────────────────────────────────

def load_jax_task(cfg: dict, ckpt_path: str):
    """Instantiate the JAX Lightning task and restore weights from .eqx checkpoint.

    We pass load_from_checkpoint to the task constructor so the task's own
    __init__ handles loading via the shared utility (same path as plot scripts).
    """
    init_args = dict(cfg["model"]["init_args"])
    init_args["model"] = _instantiate(init_args["model"])
    init_args["load_from_checkpoint"] = ckpt_path
    task_cls = _import_class(cfg["model"]["class_path"])
    return task_cls(**init_args)


def predict_jax(task, dm: LIGODataModule) -> np.ndarray:
    """Run JAX task over the test set.

    JAXLightningModule.forward does not call _prepare_batch, so we transpose
    X from (B, n_ifos, L) to (B, L, n_ifos) before forwarding.
    Returns array of shape (N, n_targets).
    """
    preds = []
    for X, _y, _z in dm.test_dataloader():
        # transpose to (B, L, n_ifos) — matches _prepare_batch in the task
        X_t = X.transpose(1, 2)
        out = task.forward(X_t)         # JAX array (B, n_targets)
        preds.append(np.asarray(out))
    return np.concatenate(preds, axis=0)


# ─── ground truth + metrics ──────────────────────────────────────────────────

def collect_true(dm: LIGODataModule) -> np.ndarray:
    """Collect target labels (B, n_targets) over the full test set."""
    ys = []
    for _X, y, _z in dm.test_dataloader():
        ys.append(y.numpy())
    return np.concatenate(ys, axis=0)


def rmse_r2(y_true: np.ndarray, y_pred: np.ndarray) -> tuple[float, float]:
    diff = y_true - y_pred
    rmse = float(np.sqrt(np.mean(diff ** 2)))
    ss_res = float(np.sum(diff ** 2))
    ss_tot = float(np.sum((y_true - y_true.mean()) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    return rmse, r2


# ─── driver ──────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--config",     required=True, help="Training YAML config.")
    parser.add_argument("--checkpoint", required=True, help="Checkpoint file (.ckpt or .eqx).")
    parser.add_argument("--out_dir",    default="benchmarks/LIGO")
    parser.add_argument("--device",     default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument(
        "--jax_platforms",
        default=None,
        help="Override JAX_PLATFORMS (e.g. 'cpu', 'cuda'). "
             "Defaults to follow --device; useful when JAX's CUDA plugin "
             "fails to initialise on a misconfigured node.",
    )
    args = parser.parse_args()

    # Set JAX platform before any JAX import.
    jax_plat = args.jax_platforms or ("cuda" if args.device.startswith("cuda") else "cpu")
    os.environ.setdefault("JAX_PLATFORMS", jax_plat)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)

    cfg = _load_cfg(args.config)
    kind, arch_name, inner_path = detect_kind(cfg)
    target_vars: list[str] = cfg["data"]["init_args"]["target_variables"]

    print(f"=== {arch_name} [{kind}]  ({inner_path}) ===")
    print(f"  cfg      : {args.config}")
    print(f"  ckpt     : {args.checkpoint}")
    print(f"  targets  : {target_vars}")

    dm = build_datamodule(cfg)

    if kind == "pt":
        model = load_pt_model(cfg, args.checkpoint, device)
        y_pred = predict_pt(model, dm, device)
    else:
        task = load_jax_task(cfg, args.checkpoint)
        y_pred = predict_jax(task, dm)

    y_true = collect_true(dm)   # (N, n_targets)

    # Ensure 2-D for consistent indexing when n_targets == 1
    if y_true.ndim == 1:
        y_true = y_true[:, None]
    if y_pred.ndim == 1:
        y_pred = y_pred[:, None]

    n_test = int(y_pred.shape[0])
    rows = []
    print()
    for i, var in enumerate(target_vars):
        rmse, r2 = rmse_r2(y_true[:, i], y_pred[:, i])
        print(f"  {var:30s}  RMSE = {rmse:.6g}   R² = {r2:.6f}   (N={n_test})")
        rows.append({
            "model":   arch_name,
            "backend": kind,
            "target":  var,
            "n_test":  n_test,
            "rmse":    rmse,
            "r2":      r2,
        })

    csv_path = out_dir / f"regression_metrics_{arch_name.lower()}.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["model", "backend", "target", "n_test", "rmse", "r2"])
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nWrote {csv_path}")


if __name__ == "__main__":
    main()
