"""Shared model builders and argument helpers for TESS sweep scripts.

Imported by sweep_classification.py and sweep_regression.py.  Both scripts
support the same Torch + JAX model families at the same four size tiers; only the output
dimension (num_classes vs 1), task class, and dataloader differ.

Width per tier (xs/sm/md/lg), including transformer ``(d_model, num_layers)``, comes from
configs/TESS/sweep/all_model_sweep_dims.yaml.
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from argparse import Namespace
from pathlib import Path
from typing import Any

import torch.nn as nn
import yaml
from lightning.pytorch.callbacks import TQDMProgressBar

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
_SWEEP_DIMS = _REPO_ROOT / "configs" / "TESS" / "sweep" / "all_model_sweep_dims.yaml"
if str(_REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT / "src"))

from models.mlp import MLPRegressor
from models.conv_ae import ConvAE
from models.conv_attn_ae import ConvAttnAE
from models.transformer import TransformerClassifier

TORCH_MODELS = {"mlp", "s4d", "cnn", "cnn_attn", "transformer"}
JAX_MODELS   = {"linoss_imex", "linoss_damped"}
ALL_MODELS   = TORCH_MODELS | JAX_MODELS


class ThrottledTQDMProgressBar(TQDMProgressBar):
    """``tqdm`` progress bar with coarser updates than every step.

    Updates at most every ``refresh_rate`` batches (default 40). If ``min_interval_s``
    is set (default 30), also refreshes when that many seconds have passed since the
    last refresh so very slow steps still show movement.

    Parameters
    ----------
    refresh_rate : int
        Lightning batch counter; refresh when ``current % refresh_rate == 0``.
    min_interval_s : float or None
        Minimum seconds between refreshes when the batch counter has not hit a
        multiple of ``refresh_rate``. Use ``None`` to disable time-based updates.
    """

    def __init__(
        self,
        refresh_rate: int = 40,
        min_interval_s: float | None = 30.0,
        **kwargs: Any,
    ) -> None:
        super().__init__(refresh_rate=refresh_rate, **kwargs)
        self._min_interval_s = min_interval_s
        self._last_refresh_m = 0.0

    def _reset_refresh_clock(self) -> None:
        self._last_refresh_m = time.monotonic()

    def on_sanity_check_start(self, *args: Any) -> None:
        self._reset_refresh_clock()
        super().on_sanity_check_start(*args)

    def on_train_epoch_start(self, trainer: Any, *args: Any) -> None:
        self._reset_refresh_clock()
        super().on_train_epoch_start(trainer, *args)

    def on_validation_start(self, trainer: Any, pl_module: Any) -> None:
        self._reset_refresh_clock()
        super().on_validation_start(trainer, pl_module)

    def on_test_start(self, trainer: Any, pl_module: Any) -> None:
        self._reset_refresh_clock()
        super().on_test_start(trainer, pl_module)

    def on_predict_start(self, trainer: Any, pl_module: Any) -> None:
        self._reset_refresh_clock()
        super().on_predict_start(trainer, pl_module)

    def _should_update(self, current: int, total: int) -> bool:
        if not self.is_enabled:
            return False
        if current == total:
            return True
        now = time.monotonic()
        if current % self.refresh_rate == 0:
            self._last_refresh_m = now
            return True
        if (
            self._min_interval_s is not None
            and (now - self._last_refresh_m) >= self._min_interval_s
        ):
            self._last_refresh_m = now
            return True
        return False


_TIERS_FROZEN = frozenset({"xs", "sm", "md", "lg"})


def _tier_int_map(path: Path, mapping: dict, label: str) -> dict[str, int]:
    """Require exactly xs/sm/md/lg keys and coerce values to ``int``."""
    if set(mapping.keys()) != _TIERS_FROZEN:
        raise ValueError(
            f"{path}: section {label!r} must define exactly tiers {sorted(_TIERS_FROZEN)}, "
            f"got {sorted(mapping)}"
        )
    return {t: int(mapping[t]) for t in mapping}


def _load_model_dim_tiers() -> tuple[tuple[dict[str, int], ...], dict[str, tuple[int, int]]]:
    """Load per-architecture tiers from configs/TESS/sweep/all_model_sweep_dims.yaml."""
    path = _SWEEP_DIMS
    with path.open(encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    keys = ("mlp", "s4d", "cnn", "cnn_attn", "linoss")
    out = [_tier_int_map(path, raw[k], k) for k in keys]

    xf = raw.get("transformer")
    if xf is None:
        raise ValueError(f"{path}: missing required section 'transformer'")
    d_model = _tier_int_map(path, xf["d_model"], "transformer.d_model")
    n_layers = _tier_int_map(path, xf["num_layers"], "transformer.num_layers")
    xf_pairs = {t: (d_model[t], n_layers[t]) for t in _TIERS_FROZEN}
    return tuple(out), xf_pairs


(
    (_MLP_DIMS, _S4D_DIMS, _CNN_DIMS, _CATTN_DIMS, _LIN_DIMS),
    _TRANSFORMER_DIMS_AND_LAYERS,
) = _load_model_dim_tiers()


# ── Model builders ────────────────────────────────────────────────────────────
# d_output = num_classes for classification, 1 for regression.

def build_mlp(size: str, d_output: int, seq_len: int, dropout: float) -> nn.Module:
    """MLPRegressor with two equal hidden layers of width h.

    params ≈ h² + (seq_len + 2 + d_output)*h + d_output
    With seq_len=1100, d_output=8:  h² + 1114*h + 8
    With seq_len=1100, d_output=1:  h² + 1103*h + 1
    Inverse (d_output=8): h = round((-1114 + sqrt(1114² + 4*(target − 8))) / 2)

    Tiers (hidden_dims=[h, h]):
        xs (~10K):  h=9   →  10,115 (cls) / 10,008 (reg)
        sm (~100K): h=83  →  99,359 (cls) / 98,689 (reg)
        md (~300K): h=224 → 299,720 (cls) / 298,673 (reg)
        lg (~700K): h=448 → 699,784 (cls) / 697,697 (reg)
    """
    h = _MLP_DIMS[size]
    return MLPRegressor(seq_len=seq_len, d_output=d_output, hidden_dims=[h, h], dropout=dropout)


def build_s4d(size: str, d_output: int, dropout: float) -> nn.Module:
    """S4Model with n_layers=4, d_state=64.

    params ≈ 8*d² + (538 + d_output)*d + d_output   (n_layers=4, d_state=64, d_input=1)
    Inverse (d_output=8): d = round((-546 + sqrt(546² + 32*(target − 8))) / 16)

    Tiers (d_model=d):
        xs (~10K):  d=16  →  10,792 (cls) / 10,433 (reg)
        sm (~100K): d=80  →  94,888 (cls) / 93,400 (reg)
        md (~300K): d=160 → 292,168 (cls) / 288,329 (reg)
        lg (~700K): d=264 → 701,720 (cls) / 695,497 (reg)
    """
    from models.s4d import S4Model

    d = _S4D_DIMS[size]
    return S4Model(d_input=1, d_output=d_output, d_model=d, d_state=64,
                   n_layers=4, dropout=dropout)


def build_cnn(size: str, d_output: int, dropout: float = 0.0) -> nn.Module:
    """ConvAE classification mode: encoder + global avg pool + Linear(C, d_output).

    Dropout is applied after each encoder LeakyReLU (training only). The sweep passes
    ``cfg.dropout`` from W&B.

    params ≈ 15*C² + (17 + d_output)*C + d_output   (k=5, n_layers=4)
    With d_output=8:  15*C² + 25*C + 8
    Inverse (d_output=8): C = round((-25 + sqrt(625 + 60*(target − 8))) / 30)

    Tiers (latent_channels=C):
        xs (~10K):  C=25  →  10,008 (cls) /  9,826 (reg)
        sm (~100K): C=80  →  98,008 (cls) / 95,761 (reg)
        md (~300K): C=140 → 297,508 (cls) / 290,621 (reg)
        lg (~700K): C=216 → 705,248 (cls) / 688,625 (reg)
    """
    c = _CNN_DIMS[size]
    return ConvAE(
        n_layers=4,
        latent_channels=c,
        kernel_size=5,
        pool_stride=2,
        num_classes=d_output,
        dropout=dropout,
    )


def build_cnn_attn(
    size: str,
    d_output: int,
    dropout: float = 0.0,
) -> nn.Module:
    """ConvAttnAE classification mode: encoder + bottleneck MHA + pool + Linear head.

    ``dropout`` is applied after each conv LeakyReLU and is also used as
    ``attn_dropout`` on :class:`torch.nn.MultiheadAttention`, so the sweep's
    single dropout hyperparameter controls both.

    params ≈ 19*C² + (23 + d_output)*C + d_output   (k=5, n_layers=4, num_heads=4)
    With d_output=8:  19*C² + 31*C + 8   (C must be divisible by 4)
    Inverse (d_output=8): C = round((-31 + sqrt(961 + 76*(target − 8))) / 38) → mult of 4

    Tiers (latent_channels=C, divisible by 4):
        xs (~10K):  C=24  →  11,696 (cls) / 11,521 (reg)
        sm (~100K): C=72  → 100,736 (cls) / 98,161 (reg)
        md (~300K): C=124 → 295,996 (cls) / 288,749 (reg)
        lg (~700K): C=192 → 706,376 (cls) / 690,569 (reg)
    """
    c = _CATTN_DIMS[size]
    return ConvAttnAE(
        n_layers=4,
        latent_channels=c,
        kernel_size=5,
        pool_stride=2,
        num_heads=4,
        dropout=dropout,
        attn_dropout=dropout,
        num_classes=d_output,
    )

def build_transformer(
    size: str,
    d_output: int,
    seq_len: int,
    dropout: float,
) -> nn.Module:
    """Transformer encoder + masked mean pool + head; ``TransformerClassifier``.

    Tiers bundle ``(d_model, num_layers)``.  We fix ``nhead=4`` and ``dim_feedforward = 2 * d_model``
    (matching ``configs/TESS/other/train_tess_transformer_classification.yaml`` defaults
    scaled per tier).

    Approximate ``nn.Parameter`` totals (positional encodings use buffers excluded from counts),
    classification ``d_output=8``: xs ~10.6k, sm ~107k, md ~310k, lg ~693k.

    Parameters
    ----------
    size : str
        Tier key ``xs`` / ``sm`` / ``md`` / ``lg``.
    d_output : int
        Number of classes or 1 for regression.
    seq_len : int
        Maximum sequence length (positional table size).
    dropout : float
        Dropout on encoder layers and classifier head submodules.
    """
    d_model, num_layers = _TRANSFORMER_DIMS_AND_LAYERS[size]
    ff = 2 * d_model
    return TransformerClassifier(
        seq_len=seq_len,
        d_output=d_output,
        d_model=d_model,
        nhead=4,
        num_layers=num_layers,
        dim_feedforward=ff,
        dropout=dropout,
    )


def build_linoss(
    size: str,
    d_output: int,
    discretization: str,
    seed: int,
    dropout: float = 0.05,
):
    """LinOSS with num_blocks=4, ssm_size == H (tied).

    IMEX:   params ≈ 24*H² + 30*H + H*d_output + d_output.  The coefficient 30 on
            ``H`` folds in four affine=False BatchNorm blocks (each stores ``2H``
            momentum placeholders alongside the residual stack).
    Damped: params ≈ 24*H² + 34*H + H*d_output + d_output (``+4H`` vs IMEX for the
            per-block ``G_diag`` vectors).
    Inverse (IMEX, given target total ``T`` and integer ``d_output``)::

        H = round((-(30 + d_output) + sqrt((30 + d_output)**2 + 96*(T - d_output))) / 48)

    Tiers (``H = ssm_size``; shown as **cls** ``d_output=8`` / **reg** ``d_output=1``):
        xs (~10K):   H=20  →   10,368 / 10,221 IMEX · 10,448 / 10,301 Damped
        sm (~100K): H=64  →  100,744 / 100,289 IMEX · 101,000 / 100,545 Damped
        md (~300K): H=112 →  305,320 / 304,529 IMEX · 305,768 / 304,977 Damped
        lg (~700K): H=170 →  700,068 / 698,871 IMEX · 700,748 / 699,551 Damped
    """
    from models.linoss import LinOSS
    h = _LIN_DIMS[size]
    return LinOSS(
        num_blocks=4,
        N=1,
        ssm_size=h,
        H=h,
        output_dim=d_output,
        task="regression",   # raw outputs; loss fn applies CE or MSE externally
        output_step=1,
        discretization=discretization,
        drop_rate=dropout,
        seed=seed,
    )


# ── Common argument helpers ───────────────────────────────────────────────────

def collect_benchmark_param_counters(model_type: str, task) -> dict[str, int]:
    """Return deterministic parameter sizes for wandb/logging (TESS sweep entrypoints).

    PyTorch totals follow ``torch.nn.Module.parameters()`` (no buffers).

    JAX tasks report floats stored on ``jax_model`` plus all array leaves in an
    initial ``jax_model_state`` shard (Stateful layers such as BatchNorm).
    """
    if model_type in JAX_MODELS:
        from models.utils.jax.print_params import (
            count_array_elements,
            count_inexact_array_elements,
        )

        return {
            "param_count_jax_model_float": count_inexact_array_elements(task.jax_model),
            "param_count_jax_state_arrays": count_array_elements(task.jax_model_state),
        }

    m = task.model
    n = sum(p.numel() for p in m.parameters())
    out = {"param_count_torch_nn": int(n)}
    if model_type == "s4d":
        from models.s4d import count_s4model_nn_parameters

        layer0 = m.s4_layers[0]
        out["param_count_s4_analytic_nn"] = int(
            count_s4model_nn_parameters(
                d_input=m.encoder.weight.shape[1],
                d_output=m.decoder.bias.shape[0],
                d_model=m.encoder.weight.shape[0],
                d_state=layer0.n,
                n_layers=len(m.s4_layers),
            )
        )
    return out


def add_infra_args(parser) -> None:
    """Add infrastructure args shared by cls and reg sweep scripts."""
    parser.add_argument("--model_type", choices=sorted(ALL_MODELS), required=True)
    parser.add_argument("--ckpt_dir",
                        default=os.environ.get("TESS_CKPT_DIR", "checkpoints/sweeps"))
    parser.add_argument("--seq_len",     type=int, default=1100)
    parser.add_argument("--max_epochs",  type=int, default=200)
    parser.add_argument("--patience",    type=int, default=20)
    parser.add_argument("--num_workers", type=int, default=8)
    # --data_dir is task-specific; each script adds it with its own default.


def add_sweep_args(parser) -> None:
    """Add sweep hyperparameter args (overridden by wandb sweep controller)."""
    parser.add_argument("--size",         choices=["xs", "sm", "md", "lg"], default="sm")
    parser.add_argument("--lr",           type=float, default=1e-3)
    parser.add_argument("--dropout",      type=float, default=0.1)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--batch_size",   type=int,   default=128)
    parser.add_argument("--seed",         type=int,   default=42)


def tess_sweep_artifact_dir(task: str, model_type: str, run_id: str) -> Path:
    """Directory for ``metrics.csv`` (CSVLogger) and ``run_config.yaml`` for one sweep trial."""
    root = Path(os.environ.get("TSP_LOCAL_LOG_DIR", "logs"))
    return root / "tess_sweeps" / task / model_type / run_id


def _namespace_to_plain(obj: Any) -> Any:
    """Make argparse/jsonargparse namespaces round-trippable to YAML."""
    if isinstance(obj, Namespace):
        return {k: _namespace_to_plain(v) for k, v in vars(obj).items()}
    if isinstance(obj, dict):
        return {k: _namespace_to_plain(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_namespace_to_plain(v) for v in obj]
    return obj


def dump_sweep_run_config(path: Path, args: argparse.Namespace, wandb_run) -> None:
    """Write CLI args and wandb config for offline reproducibility."""
    payload = {
        "cli_args": _namespace_to_plain(args),
        "wandb": dict(wandb_run.config),
        "wandb_run_id": getattr(wandb_run, "id", None),
        "wandb_mode": os.environ.get("WANDB_MODE"),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(payload, f, sort_keys=False, default_flow_style=False)
