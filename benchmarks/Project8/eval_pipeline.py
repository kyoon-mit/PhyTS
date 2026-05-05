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

def _walk(obj, depth: int = 0, max_depth: int = 6):
    """Recursively yield all values inside dicts/lists/tuples."""
    if depth > max_depth:
        return
    yield obj
    if isinstance(obj, dict):
        for v in obj.values():
            yield from _walk(v, depth + 1, max_depth)
    elif isinstance(obj, (list, tuple)):
        for v in obj:
            yield from _walk(v, depth + 1, max_depth)


def _find_in_pickle(payload, want_cls):
    """First object in the torch-pickle payload that is an instance of ``want_cls``."""
    for value in _walk(payload):
        if isinstance(value, want_cls):
            return value
    return None


def _load_jax_weights_from_ckpt(
    ckpt_path: str,
    fresh_model,
    fresh_state,
):
    """Restore (model, state) for a JAX task from either a .eqx or a .ckpt file.

    Saving paths supported:

      * ``.eqx`` written by :class:`models.utils.jax.save_model.JAXCheckpointManager`
        — uses equinox's tree deserialisation (single model or
        ``(model, state, opt_state)`` tuple).
      * ``.ckpt`` torch pickle written by Lightning ``ModelCheckpoint`` —
        provided the LightningModule actually pickled the eqx model
        somewhere reachable (e.g. via ``save_hyperparameters`` or by
        attribute on the module). Walks the pickle dict for an
        ``eqx.Module`` and an ``eqx.nn.State``.
    """
    import equinox as eqx

    p = Path(ckpt_path)
    if not p.exists():
        raise FileNotFoundError(f"Checkpoint not found: {p}")

    # 1. Equinox-native format.
    if p.suffix == ".eqx":
        from models.utils.jax.load_model import load_model
        return load_model(path=p, model=fresh_model, model_state=fresh_state)

    # 2. Lightning torch-pickle format. Two storage layouts supported:
    #    (a) JAXLightningModule.on_save_checkpoint -> bytes under JAX_CKPT_KEY
    #    (b) legacy: an eqx.Module pickled somewhere in the dict (e.g. via
    #        save_hyperparameters)
    import torch
    payload = torch.load(p, map_location="cpu", weights_only=False)

    # (a) JAXLightningModule on_save_checkpoint format.
    if isinstance(payload, dict):
        from models.utils.jax.wrapper import JAXLightningModule
        blob = payload.get(JAXLightningModule.JAX_CKPT_KEY)
        if blob is not None:
            import io
            buf = io.BytesIO(blob)
            return eqx.tree_deserialise_leaves(buf, (fresh_model, fresh_state))

    # (b) Walk the pickle for a stray eqx.Module / eqx.nn.State.
    found_model = _find_in_pickle(payload, type(fresh_model))
    if found_model is None:
        # Fall back to any eqx.Module — there should be exactly one.
        found_model = _find_in_pickle(payload, eqx.Module)
    found_state = _find_in_pickle(payload, eqx.nn.State)

    if found_model is None:
        keys = list(payload.keys()) if isinstance(payload, dict) else type(payload).__name__
        raise ValueError(
            f"Could not find any JAX weights inside the torch checkpoint at {p}.\n"
            f"Top-level keys: {keys}\n"
            f"This means the JAX model was never persisted to disk for this run: "
            f"Lightning's ModelCheckpoint only writes state_dict() (which is "
            f"empty for JAXLightningModule) and Python hyperparameters.\n"
            f"Fix: re-run training with the updated JAXLightningModule that "
            f"defines on_save_checkpoint, or use a .eqx written by "
            f"JAXCheckpointManager. Existing .ckpt files from before that fix "
            f"do not contain trained weights and cannot be evaluated."
        )

    if found_state is None:
        print(
            "  WARN: no eqx.nn.State found in checkpoint; using fresh state. "
            "Stateful layers (e.g. BatchNorm) may use uninitialised running stats."
        )
        found_state = fresh_state

    return found_model, found_state


def load_jax_task(cfg: dict, ckpt_path: str):
    """Instantiate the JAX task and restore weights from ``ckpt_path``.

    Supports both equinox ``.eqx`` and Lightning torch-pickle ``.ckpt``
    formats — see :func:`_load_jax_weights_from_ckpt` for details.
    """
    init_args = dict(cfg["model"]["init_args"])
    init_args["model"] = _instantiate(init_args["model"])
    # Defer weight loading: build the task with fresh weights, then overwrite
    # jax_model/jax_model_state from whatever format the checkpoint is in.
    init_args.pop("load_from_checkpoint", None)
    task_cls = _import_class(cfg["model"]["class_path"])
    task = task_cls(**init_args)

    model, state = _load_jax_weights_from_ckpt(
        ckpt_path,
        fresh_model=task.jax_model,
        fresh_state=task.jax_model_state,
    )
    task.jax_model = model
    task.jax_model_state = state
    return task


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
    parser.add_argument(
        "--jax_platforms",
        default=None,
        help="Override JAX_PLATFORMS (e.g. 'cpu', 'cuda'). "
             "Defaults to follow --device; useful when JAX's CUDA plugin "
             "fails to initialise on a misconfigured node.",
    )
    args = parser.parse_args()

    # JAX must be told the platform before its first import. Otherwise the
    # CUDA plugin will try to initialise even when we only want CPU. Set the
    # env var here, before we import any JAX module (lazily, via load_jax_task).
    jax_plat = args.jax_platforms or ("cuda" if args.device.startswith("cuda") else "cpu")
    os.environ.setdefault("JAX_PLATFORMS", jax_plat)

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
