#!/bin/bash
# TESS LinOSS benchmark pipeline — submits chained SLURM jobs on MIT Engaging.
#
# Engaging: ``module load cuda`` resolves to CUDA toolkit 13.1.0.  This script assumes
# that everywhere (JAX extra ``cu13``, comments, and venv sync).  Other sites: override
# with ``LINOSS_JAX_CUDA_EXTRA=cu12`` (or ``none``) if your default ``cuda`` module is
# not the CUDA 13 stack.
#
# Stage 1 (parallel):  Train LinOSS regression and classification end-to-end.
#                      Both start immediately (no pretraining dependency).
# Stage 2 (eval):      Evaluate all models (PyTorch + LinOSS) and write CSVs + plots.
#                      The eval job depends on both LinOSS training jobs completing.
#
# LinOSS uses JAX/Equinox internally.  Checkpoints are saved as .eqx files
# (not .ckpt) via the JAXModelCheckpoint callback.  The eval pipeline handles
# both formats automatically.
#
# Before running:
#   1. Download data (login node only — compute nodes have no internet):
#        bash benchmarks/TESS/setup_data.sh
#   2. On the login node: ``module load cuda miniforge && uv sync --extra jax --extra cu13``
#      (same extras this script uses by default).  CPU-only JAX: ``LINOSS_JAX_CUDA_EXTRA=none``.
#      Batch steps call .venv/bin/python only — not `uv run` on compute nodes.
#   3. Run from the repo root, or from anywhere — WORKDIR is derived from the
#      script location.
#
# Usage:
#   bash benchmarks/TESS/run_linoss.sh
#
# Monitor:
#   squeue -u $USER
#   tail -f benchmarks/TESS/logs/tess_linoss_reg_<jobid>.out

set -e

# Resolve repo root from this script so WORKDIR is correct no matter where sbatch was invoked from.
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
WORKDIR=$(cd "$SCRIPT_DIR/../.." && pwd)
POOL=/home/allisone/orcd/pool/UROP_2025_Summer/TimeSeriesPhysics
DATA_DIR=$POOL/data_engaging/TESS/.cache/TESS
CKPT_DIR=$POOL/checkpoints
RESULTS_DIR=$POOL/results
WANDB_ROOT=$POOL/wandb
LOGS=$WORKDIR/benchmarks/TESS/logs
mkdir -p "$LOGS" "$CKPT_DIR" "$RESULTS_DIR" "$WANDB_ROOT"

# Abort early if data hasn't been downloaded yet
if [ ! -f "$DATA_DIR/tess_regression.parquet" ] || [ ! -f "$DATA_DIR/tess_classification.parquet" ]; then
    echo "ERROR: TESS data not found at $DATA_DIR"
    echo "Run from the login node first: bash benchmarks/TESS/setup_data.sh"
    exit 1
fi

if [ ! -d "$WORKDIR/.venv" ]; then
    echo "ERROR: $WORKDIR/.venv not found."
    echo "On the login node: cd $WORKDIR && module load cuda miniforge && uv sync --extra jax --extra cu13"
    exit 1
fi
# Sync on the submit host: LinOSS + CUDA 13 JAX (``cu13``; Engaging default ``cuda`` = 13.1.0).
# Set LINOSS_JAX_CUDA_EXTRA=cu12 or none for other stacks.  Compute jobs use .venv only.
LINOSS_JAX_CUDA_EXTRA="cu13"
if command -v uv >/dev/null 2>&1; then
    if [ "$LINOSS_JAX_CUDA_EXTRA" = "none" ] || [ "$LINOSS_JAX_CUDA_EXTRA" = "cpu" ]; then
        (cd "$WORKDIR" && uv sync --extra jax) || { echo "ERROR: uv sync --extra jax failed"; exit 1; }
    else
        (cd "$WORKDIR" && uv sync --extra jax --extra "$LINOSS_JAX_CUDA_EXTRA") || {
            echo "ERROR: uv sync --extra jax --extra $LINOSS_JAX_CUDA_EXTRA failed"
            exit 1
        }
    fi
fi
PYTHON_EXE="$WORKDIR/.venv/bin/python"
if [ ! -x "$PYTHON_EXE" ]; then
    echo "ERROR: $PYTHON_EXE is missing or not executable after uv sync."
    exit 1
fi

SBATCH_COMMON="
  --gpus=1
  --nodes=1
  --ntasks-per-node=1
  --cpus-per-task=8
  --mem=32G
  --time=6:00:00
  --chdir=$WORKDIR
"

# ``module load cuda`` → toolkit 13.1.0 on Engaging.  JAX wheels still bundle their
# own CUDA; node driver must satisfy JAX’s CUDA-13 requirements.  XLA mem cap vs dataloader:
# WANDB_DIR keeps wandb/lightning artifacts on the pool (avoids nested ./TimeSeriesPhysics trees on NFS).
ACTIVATE="module load cuda miniforge &&
  export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 NUMEXPR_NUM_THREADS=8
  OPENBLAS_NUM_THREADS=8 PANDAS_USE_PYARROW=1
  XLA_PYTHON_CLIENT_MEM_FRACTION=0.7
  export WANDB_DIR=$WANDB_ROOT"

RUN="srun --cpu-bind=none $PYTHON_EXE"

# # ── Stage 1: LinOSS regression (no dependency) ───────────────────────────────
# JOB_LINOSS_REG=$(sbatch --parsable \
#   $SBATCH_COMMON \
#   --job-name=tess_linoss_reg \
#   --output="$LOGS/tess_linoss_reg_%j.out" \
#   --error="$LOGS/tess_linoss_reg_%j.err" \
#   --wrap="$ACTIVATE && export TESS_LINOSS_CKPT_DIR=$CKPT_DIR/tess_linoss_regression && $RUN main.py fit \
#     --config configs/TESS/train_tess_linoss_regression.yaml \
#     --data.init_args.data_dir $DATA_DIR")
# echo "[1/3] LinOSS regression submitted: job $JOB_LINOSS_REG"

# # ── Stage 1: LinOSS classification (no dependency) ───────────────────────────
# JOB_LINOSS_CLS=$(sbatch --parsable \
#   $SBATCH_COMMON \
#   --job-name=tess_linoss_cls \
#   --output="$LOGS/tess_linoss_cls_%j.out" \
#   --error="$LOGS/tess_linoss_cls_%j.err" \
#   --wrap="$ACTIVATE && export TESS_LINOSS_CKPT_DIR=$CKPT_DIR/tess_linoss_classification && $RUN main.py fit \
#     --config configs/TESS/train_tess_linoss_classification.yaml \
#     --data.init_args.data_dir $DATA_DIR")
# echo "[2/3] LinOSS classification submitted: job $JOB_LINOSS_CLS"

# ── Stage 2: Eval (waits for both LinOSS training jobs) ──────────────────────
# The eval pipeline skips any model whose checkpoint is not found, so it is
# safe to run it against the full model list — completed PyTorch checkpoints
# (mlp, s4d, s4d_head) will be evaluated alongside the new LinOSS ones.


# --dependency=afterok:$JOB_LINOSS_REG:$JOB_LINOSS_CLS \

JOB_EVAL=$(sbatch --parsable \
  $SBATCH_COMMON \
  --job-name=tess_eval \
  --output="$LOGS/tess_eval_%j.out" \
  --error="$LOGS/tess_eval_%j.err" \
  --wrap="$ACTIVATE && $RUN benchmarks/TESS/eval_pipeline.py \
    --data_dir $DATA_DIR \
    --ckpt_dir $CKPT_DIR \
    --out_dir  $RESULTS_DIR")
# echo "[3/3] Eval submitted: job $JOB_EVAL (depends on $JOB_LINOSS_REG, $JOB_LINOSS_CLS)"

# echo ""
# echo "Pipeline:"
# echo "  [$JOB_LINOSS_REG linoss_reg, $JOB_LINOSS_CLS linoss_cls] → $JOB_EVAL eval"
# echo "Monitor: squeue -u \$USER"
