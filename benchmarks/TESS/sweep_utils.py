"""Shared model builders and argument helpers for TESS sweep scripts.

Imported by sweep_classification.py and sweep_regression.py.  Both scripts
support the same six model types at the same four size tiers; only the output
dimension (num_classes vs 1), task class, and dataloader differ.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import torch.nn as nn

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT / "src"))

from models.mlp import MLPRegressor
from models.s4d import S4Model
from models.conv_ae import ConvAE
from models.conv_attn_ae import ConvAttnAE

TORCH_MODELS = {"mlp", "s4d", "cnn", "cnn_attn"}
JAX_MODELS   = {"linoss_imex", "linoss_damped"}
ALL_MODELS   = TORCH_MODELS | JAX_MODELS

# ── Size tiers ────────────────────────────────────────────────────────────────
# All tiers target ~10K / ~100K / ~300K / ~700K parameters.
# Changing d_output from 8 (classification) to 1 (regression) shifts counts by
# less than 2% at every tier, so the same dims serve both tasks.

_MLP_DIMS   = {"xs":   9, "sm":  83, "md": 224, "lg": 448}
_S4D_DIMS   = {"xs":  16, "sm":  80, "md": 160, "lg": 264}
_CNN_DIMS   = {"xs":  25, "sm":  80, "md": 140, "lg": 216}
_CATTN_DIMS = {"xs":  24, "sm":  72, "md": 124, "lg": 192}  # must be div by num_heads=4
_LIN_DIMS   = {"xs":  20, "sm":  64, "md": 112, "lg": 170}


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
    d = _S4D_DIMS[size]
    return S4Model(d_input=1, d_output=d_output, d_model=d, d_state=64,
                   n_layers=4, dropout=dropout)


def build_cnn(size: str, d_output: int) -> nn.Module:
    """ConvAE classification mode: encoder + global avg pool + Linear(C, d_output).

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
    return ConvAE(n_layers=4, latent_channels=c, kernel_size=5, pool_stride=2,
                  num_classes=d_output)


def build_cnn_attn(size: str, d_output: int) -> nn.Module:
    """ConvAttnAE classification mode: encoder + bottleneck MHA + pool + Linear head.

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
    return ConvAttnAE(n_layers=4, latent_channels=c, kernel_size=5, pool_stride=2,
                      num_heads=4, attn_dropout=0.1, num_classes=d_output)


def build_linoss(size: str, d_output: int, discretization: str, seed: int):
    """LinOSS with num_blocks=4, ssm_size == H (tied).

    IMEX:   params ≈ 24*H² + 30*H + d_output
    Damped: params ≈ 24*H² + 34*H + d_output
    Inverse (IMEX, d_output=8): H = round((-30 + sqrt(900 + 96*(target − 8))) / 48)

    Tiers (H = ssm_size):
        xs (~10K):  H=20  →  10,208 / 10,288 (IMEX/Damped)
        sm (~100K): H=64  → 100,232 / 100,488
        md (~300K): H=112 → 304,424 / 304,872
        lg (~700K): H=170 → 698,708 / 699,388
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
        seed=seed,
    )


# ── Common argument helpers ───────────────────────────────────────────────────

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
    parser.add_argument("--batch_size",   type=int,   default=64)
    parser.add_argument("--seed",         type=int,   default=42)
