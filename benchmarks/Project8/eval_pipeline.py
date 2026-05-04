"""
Evaluate Project 8 energy regressors on the test set: RMSE [eV] and R^2.

Supports the three architectures with configs under ``configs/Project8/``:

  * CNN     -> ``models.conv_regressor.Conv1DRegressor``  (PyTorch)
  * S4D     -> ``models.s4d.S4Model``                     (PyTorch)
  * LinOSS  -> ``models.linoss.LinOSS``                   (JAX/equinox)

PyTorch checkpoints are Lightning ``.ckpt`` files; LinOSS checkpoints are
equinox ``.eqx`` files written by :class:`models.utils.jax.save_model.JAXCheckpointManager`.

For each model we run ``test_dataloader`` once, undo the z-score on the
``energy_eV`` target using the DataModule's ``mu``/``stds`` and report:

  * RMSE in eV
  * coefficient of determination R^2 = 1 - SS_res / SS_tot

Usage:

    python benchmarks/Project8/eval_pipeline.py \\
        --cnn_cfg     configs/Project8/train_project8_conv_regression_energy_gaussiannll.yaml \\
        --cnn_ckpt    checkpoints/project8_conv_regression_energy_gaussiannll/best.ckpt \\
        --s4d_cfg     configs/Project8/train_project8_s4d_regression_energy_gaussiannll.yaml \\
        --s4d_ckpt    checkpoints/project8_s4d_regression_energy_gaussiannll/best.ckpt \\
        --linoss_cfg  configs/Project8/train_project8_linoss_regression_energy_gaussiannll.yaml \\
        --linoss_ckpt checkpoints/project8_linoss_regression_energy_gaussiannll/best.eqx \\
        --out_dir     benchmarks/Project8

Any of the three (cfg, ckpt) pairs may be omitted to skip that architecture.
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

# Make ``src/`` importable when running this script directly from the repo root.
_REPO_ROOT = Path(__file__).resolve().parents[2]
_SRC = _REPO_ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from dataloader.project8_dataloader import Project8DataModule  # noqa: E402


# ─── helpers ────────────────────────────────────────────────────────────────

def _instantiate(spec: dict):
    """Instantiate ``{class_path, init_args}`` from a YAML config block."""
    class_path = spec["class_path"]
    init_args = spec.get("init_args", {}) or {}
    module_name, class_name = class_path.rsplit(".", 1)
    cls = getattr(importlib.import_module(module_name), class_name)
    return cls(**init_args)


def _load_cfg(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def build_datamodule(cfg: dict) -> Project8DataModule:
    """Build the Project 8 DataModule from a training config and run setup('test')."""
    data_cfg = cfg["data"]["init_args"]
    dm = Project8DataModule(**data_cfg)
    dm.setup("test")
    return dm


# ─── PyTorch (CNN, S4D) ─────────────────────────────────────────────────────

def load_pt_task(cfg: dict, ckpt_path: str, device: torch.device):
    """Load a PyTorch ``Project8Regression`` task from a Lightning checkpoint."""
    model_cfg = cfg["model"]
    init_args = model_cfg.get("init_args", {})
    encoder = _instantiate(init_args["encoder"])

    task_class_path = model_cfg["class_path"]
    module_name, class_name = task_class_path.rsplit(".", 1)
    task_cls = getattr(importlib.import_module(module_name), class_name)

    task = task_cls.load_from_checkpoint(
        ckpt_path,
        encoder=encoder,
        map_location=device,
    )
    task = task.to(device).eval()
    return task


@torch.no_grad()
def predict_pt(task, dm: Project8DataModule, device: torch.device) -> np.ndarray:
    """Run inference and return the predicted z-scored mean (N,)."""
    preds = []
    for x, _var in dm.test_dataloader():
        out = task(x.to(device)).cpu().numpy()  # (B, 2) = [mean, raw_var]
        preds.append(out[:, 0])
    return np.concatenate(preds)


# ─── JAX (LinOSS) ───────────────────────────────────────────────────────────

def load_jax_task(cfg: dict, ckpt_path: str):
    """Instantiate the JAX task and deserialise its weights from ``ckpt_path``."""
    model_cfg = cfg["model"]
    init_args = dict(model_cfg.get("init_args", {}))

    inner_model = _instantiate(init_args["model"])
    init_args["model"] = inner_model

    task_class_path = model_cfg["class_path"]
    module_name, class_name = task_class_path.rsplit(".", 1)
    task_cls = getattr(importlib.import_module(module_name), class_name)

    # Load weights via the wrapper's built-in deserialiser.
    init_args["load_from_checkpoint"] = ckpt_path
    return task_cls(**init_args)


def predict_jax(task, dm: Project8DataModule) -> np.ndarray:
    """Run JAX inference and return the predicted z-scored mean (N,)."""
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
    """Concatenate the z-scored ground-truth target across the test loader."""
    ys = []
    for _x, var in dm.test_dataloader():
        v = var.numpy()
        if v.ndim > 1 and v.shape[-1] == 1:
            v = v[:, 0]
        ys.append(v)
    return np.concatenate(ys)


def rmse_r2(y_true: np.ndarray, y_pred: np.ndarray) -> tuple[float, float]:
    """RMSE and coefficient of determination (R^2) in the units of the inputs."""
    diff = y_true - y_pred
    rmse = float(np.sqrt(np.mean(diff ** 2)))
    ss_res = float(np.sum(diff ** 2))
    ss_tot = float(np.sum((y_true - y_true.mean()) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    return rmse, r2


# ─── driver ─────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--cnn_cfg")
    parser.add_argument("--cnn_ckpt")
    parser.add_argument("--s4d_cfg")
    parser.add_argument("--s4d_ckpt")
    parser.add_argument("--linoss_cfg")
    parser.add_argument("--linoss_ckpt")
    parser.add_argument("--out_dir", default="benchmarks/Project8")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    jobs = [
        ("CNN",    args.cnn_cfg,    args.cnn_ckpt,    "pt"),
        ("S4D",    args.s4d_cfg,    args.s4d_ckpt,    "pt"),
        ("LinOSS", args.linoss_cfg, args.linoss_ckpt, "jax"),
    ]
    jobs = [j for j in jobs if j[1] and j[2]]
    if not jobs:
        parser.error("Provide at least one of --{cnn,s4d,linoss}_{cfg,ckpt}.")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)

    # Use the first available config to build the shared DataModule. All
    # Project 8 configs use the same data block, so any choice is fine.
    dm = build_datamodule(_load_cfg(jobs[0][1]))
    target_name = dm.hparams.variables[0]
    mu = float(dm.mu[0])
    std = float(dm.stds[0])
    print(f"Target: {target_name}  (mu={mu:.4f}, std={std:.4f})")

    y_true_z = collect_true_z(dm)
    y_true_eV = y_true_z * std + mu

    results: list[dict] = []
    for name, cfg_path, ckpt_path, kind in jobs:
        print(f"\n=== {name} ===")
        print(f"  cfg : {cfg_path}")
        print(f"  ckpt: {ckpt_path}")
        cfg = _load_cfg(cfg_path)

        if kind == "pt":
            task = load_pt_task(cfg, ckpt_path, device)
            y_pred_z = predict_pt(task, dm, device)
        else:
            task = load_jax_task(cfg, ckpt_path)
            y_pred_z = predict_jax(task, dm)

        y_pred_eV = y_pred_z * std + mu
        rmse_eV, r2 = rmse_r2(y_true_eV, y_pred_eV)

        print(f"  RMSE = {rmse_eV:.4f} eV")
        print(f"  R^2  = {r2:.6f}")

        results.append({
            "model": name,
            "n_test": int(y_pred_eV.shape[0]),
            "rmse_eV": rmse_eV,
            "r2": r2,
        })

    csv_path = out_dir / "energy_metrics.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["model", "n_test", "rmse_eV", "r2"])
        writer.writeheader()
        for row in results:
            writer.writerow(row)

    print(f"\nSummary ({target_name}):")
    print(f"{'model':<8} {'N':>7} {'RMSE [eV]':>14} {'R^2':>10}")
    for r in results:
        print(f"{r['model']:<8} {r['n_test']:>7d} {r['rmse_eV']:>14.4f} {r['r2']:>10.6f}")
    print(f"\nWrote {csv_path}")


if __name__ == "__main__":
    main()
