"""
Evaluate one Project 8 energy regressor on the test set: RMSE [eV] and R^2.

Architecture and backend (PyTorch vs JAX) are auto-detected from the YAML
training config:

    python benchmarks/Project8/eval_pipeline.py \\
        --config     configs/Project8/train_project8_s4d_regression_energy_gaussiannll.yaml \\
        --checkpoint checkpoints/project8_s4d_regression_energy_gaussiannll/best.ckpt

Detection rules:

  * task with ``init_args.encoder``     -> PyTorch (Lightning ``.ckpt``)
  * task with ``init_args.model``       -> JAX/equinox  (``.eqx``)

The display name (CNN / S4D / LinOSS / ...) is taken from the encoder /
inner-model ``class_path``.  We run ``test_dataloader``, undo the
``energy_eV`` z-score using the DataModule's ``mu``/``stds``, and report
RMSE in eV and R^2 = 1 - SS_res / SS_tot.
"""

import argparse
import csv
import importlib
import sys
from pathlib import Path

import numpy as np
import torch
import yaml

# Make ``src/`` importable when running this script directly from the repo root.
_REPO_ROOT = Path(__file__).resolve().parents[2]
_SRC = _REPO_ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from dataloader.project8_dataloader import Project8DataModule  # noqa: E402


# Friendly architecture names for known model classes; falls back to class name.
_ARCH_NAMES = {
    "models.s4d.S4Model":                          "S4D",
    "models.conv_regressor.Conv1DRegressor":       "CNN",
    "models.linoss.LinOSS":                        "LinOSS",
    "models.transformer.TransformerClassifier":    "Transformer",
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
    """Return (kind, arch_name, encoder_class_path).

    ``kind`` is ``'pt'`` or ``'jax'``; the inner model lives at
    ``init_args.encoder`` for PyTorch tasks and ``init_args.model`` for the
    JAX wrapper task.
    """
    init_args = cfg["model"].get("init_args") or {}
    if "encoder" in init_args:
        kind = "pt"
        inner = init_args["encoder"]
    elif "model" in init_args:
        kind = "jax"
        inner = init_args["model"]
    else:
        raise ValueError(
            "Could not detect architecture: model.init_args has neither "
            "'encoder' (PyTorch) nor 'model' (JAX)."
        )
    inner_path = inner["class_path"]
    arch_name = _ARCH_NAMES.get(inner_path, inner_path.rsplit(".", 1)[1])
    return kind, arch_name, inner_path


def build_datamodule(cfg: dict) -> Project8DataModule:
    """Build the Project 8 DataModule from a training config and run setup('test')."""
    dm = Project8DataModule(**cfg["data"]["init_args"])
    dm.setup("test")
    return dm


# ─── PyTorch tasks (e.g. CNN, S4D) ──────────────────────────────────────────

def load_pt_task(cfg: dict, ckpt_path: str, device: torch.device):
    init_args = cfg["model"]["init_args"]
    encoder = _instantiate(init_args["encoder"])
    task_cls = _import_class(cfg["model"]["class_path"])
    task = task_cls.load_from_checkpoint(
        ckpt_path,
        encoder=encoder,
        map_location=device,
    )
    return task.to(device).eval()


@torch.no_grad()
def predict_pt(task, dm: Project8DataModule, device: torch.device) -> np.ndarray:
    preds = []
    for x, _var in dm.test_dataloader():
        out = task(x.to(device)).cpu().numpy()  # (B, 2) = [mean, raw_var]
        preds.append(out[:, 0])
    return np.concatenate(preds)


# ─── JAX/equinox tasks (e.g. LinOSS) ────────────────────────────────────────

def load_jax_task(cfg: dict, ckpt_path: str):
    init_args = dict(cfg["model"]["init_args"])
    init_args["model"] = _instantiate(init_args["model"])
    init_args["load_from_checkpoint"] = ckpt_path
    task_cls = _import_class(cfg["model"]["class_path"])
    return task_cls(**init_args)


def predict_jax(task, dm: Project8DataModule) -> np.ndarray:
    import jax
    import jax.numpy as jnp
    from models.utils.jax.training import jax_inference
    from models.utils.jax.utils import tensor_to_jax

    preds = []
    for x, _var in dm.test_dataloader():
        x_jax = tensor_to_jax(x)
        batch_size = jax.tree.leaves(x_jax)[0].shape[0]
        keys = jax.random.split(task.key, batch_size)
        out = jax_inference(
            model=task.jax_model,
            x=x_jax,
            state=task.jax_model_state,
            key=keys,
        )
        mu = jnp.split(out, 2, axis=-1)[0].squeeze(-1)
        preds.append(np.asarray(mu))
    return np.concatenate(preds)


# ─── ground-truth + metrics ─────────────────────────────────────────────────

def collect_true_z(dm: Project8DataModule) -> np.ndarray:
    ys = []
    for _x, var in dm.test_dataloader():
        v = var.numpy()
        if v.ndim > 1 and v.shape[-1] == 1:
            v = v[:, 0]
        ys.append(v)
    return np.concatenate(ys)


def rmse_r2(y_true: np.ndarray, y_pred: np.ndarray) -> tuple[float, float]:
    diff = y_true - y_pred
    rmse = float(np.sqrt(np.mean(diff ** 2)))
    ss_res = float(np.sum(diff ** 2))
    ss_tot = float(np.sum((y_true - y_true.mean()) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    return rmse, r2


# ─── driver ─────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--config",     required=True, help="Training YAML config.")
    parser.add_argument("--checkpoint", required=True, help="Checkpoint file (.ckpt or .eqx).")
    parser.add_argument("--out_dir",    default="benchmarks/Project8")
    parser.add_argument("--device",     default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)

    cfg = _load_cfg(args.config)
    kind, arch_name, inner_path = detect_kind(cfg)
    print(f"=== {arch_name} [{kind}]  ({inner_path}) ===")
    print(f"  cfg : {args.config}")
    print(f"  ckpt: {args.checkpoint}")

    dm = build_datamodule(cfg)
    target_name = dm.hparams.variables[0]
    mu, std = float(dm.mu[0]), float(dm.stds[0])
    print(f"  target: {target_name}  (mu={mu:.4f}, std={std:.4f})")

    y_true_eV = collect_true_z(dm) * std + mu

    if kind == "pt":
        task = load_pt_task(cfg, args.checkpoint, device)
        y_pred_z = predict_pt(task, dm, device)
    else:
        task = load_jax_task(cfg, args.checkpoint)
        y_pred_z = predict_jax(task, dm)

    y_pred_eV = y_pred_z * std + mu
    rmse_eV, r2 = rmse_r2(y_true_eV, y_pred_eV)
    n_test = int(y_pred_eV.shape[0])

    print(f"\n{arch_name}: RMSE = {rmse_eV:.4f} eV   R^2 = {r2:.6f}   (N={n_test})")

    csv_path = out_dir / f"energy_metrics_{arch_name.lower()}.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["model", "backend", "n_test", "rmse_eV", "r2"])
        writer.writeheader()
        writer.writerow({
            "model": arch_name,
            "backend": kind,
            "n_test": n_test,
            "rmse_eV": rmse_eV,
            "r2": r2,
        })
    print(f"Wrote {csv_path}")


if __name__ == "__main__":
    main()
