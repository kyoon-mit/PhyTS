"""
Evaluate Project 8 energy regressors on the test set: RMSE [eV] and R^2.

Architecture and backend (PyTorch vs JAX) are auto-detected from the YAML
training config — pass any number of ``(cfg, ckpt)`` pairs:

    python benchmarks/Project8/eval_pipeline.py \\
        configs/Project8/train_project8_conv_regression_energy_gaussiannll.yaml \\
        checkpoints/project8_conv_regression_energy_gaussiannll/best.ckpt \\
        configs/Project8/train_project8_s4d_regression_energy_gaussiannll.yaml \\
        checkpoints/project8_s4d_regression_energy_gaussiannll/best.ckpt \\
        configs/Project8/train_project8_linoss_regression_energy_gaussiannll.yaml \\
        checkpoints/project8_linoss_regression_energy_gaussiannll/best.eqx

Detection rules:

  * task with ``init_args.encoder``     -> PyTorch (Lightning ``.ckpt``)
  * task with ``init_args.model``       -> JAX/equinox  (``.eqx``)

The display name (CNN / S4D / LinOSS / ...) is taken from the encoder /
inner-model ``class_path``.  For each model we run ``test_dataloader``,
undo the ``energy_eV`` z-score using the DataModule's ``mu``/``stds``,
and report RMSE in eV and R^2 = 1 - SS_res / SS_tot.
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
    "models.s4d.S4Model":                 "S4D",
    "models.conv_regressor.Conv1DRegressor": "CNN",
    "models.linoss.LinOSS":               "LinOSS",
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

def _evaluate_one(cfg_path: str, ckpt_path: str, dm: Project8DataModule,
                  y_true_eV: np.ndarray, mu: float, std: float,
                  device: torch.device) -> dict:
    cfg = _load_cfg(cfg_path)
    kind, arch_name, inner_path = detect_kind(cfg)
    print(f"\n=== {arch_name} [{kind}]  ({inner_path}) ===")
    print(f"  cfg : {cfg_path}")
    print(f"  ckpt: {ckpt_path}")

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
    return {
        "model": arch_name,
        "backend": kind,
        "n_test": int(y_pred_eV.shape[0]),
        "rmse_eV": rmse_eV,
        "r2": r2,
    }


def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "pairs", nargs="+", metavar="CFG CKPT",
        help="One or more (config.yaml, checkpoint) pairs. Architecture is auto-detected.",
    )
    parser.add_argument("--out_dir", default="benchmarks/Project8")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    if len(args.pairs) % 2 != 0:
        parser.error("Positional arguments must come in (cfg, ckpt) pairs.")
    jobs = list(zip(args.pairs[0::2], args.pairs[1::2]))

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)

    # All Project 8 training configs share the same data block, so the first
    # one suffices to build the shared test loader and norm stats.
    dm = build_datamodule(_load_cfg(jobs[0][0]))
    target_name = dm.hparams.variables[0]
    mu, std = float(dm.mu[0]), float(dm.stds[0])
    print(f"Target: {target_name}  (mu={mu:.4f}, std={std:.4f})")

    y_true_eV = collect_true_z(dm) * std + mu

    results = [
        _evaluate_one(cfg, ckpt, dm, y_true_eV, mu, std, device)
        for cfg, ckpt in jobs
    ]

    csv_path = out_dir / "energy_metrics.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["model", "backend", "n_test", "rmse_eV", "r2"])
        writer.writeheader()
        writer.writerows(results)

    print(f"\nSummary ({target_name}):")
    print(f"{'model':<10} {'backend':<8} {'N':>7} {'RMSE [eV]':>14} {'R^2':>10}")
    for r in results:
        print(f"{r['model']:<10} {r['backend']:<8} {r['n_test']:>7d} "
              f"{r['rmse_eV']:>14.4f} {r['r2']:>10.6f}")
    print(f"\nWrote {csv_path}")


if __name__ == "__main__":
    main()
