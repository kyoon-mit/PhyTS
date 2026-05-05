#!/bin/bash
# TESS benchmark pipeline — submits chained SLURM jobs on MIT Engaging.
#
# Stage 1 (reconstruction):  Train S4D seq2seq backbone on flux reconstruction.
# Stage 2 (parallel):        Train all end-to-end models (MLP reg, MLP cls,
#                            S4D reg, S4D cls) + frozen-backbone heads.
#                            End-to-end jobs start immediately; frozen-backbone
#                            head jobs depend on stage 1.
# Stage 3 (eval):            Evaluate all models and write CSVs + plots.
#
# Before running:
#   1. Download data (login node only — compute nodes have no internet):
#        bash benchmarks/TESS/setup_data.sh
#   2. Materialize the project venv on the login node (module load cuda miniforge, etc.):
#        cd <repo> && uv sync
#      Batch steps call .venv/bin/python only — not `uv run` on compute nodes — so many
#      parallel jobs never mutate .venv on NFS (avoids errno 116 Stale file handle).
#   3. Run from the repo root, or from anywhere — WORKDIR is derived from the script location.
#   4. (Already done) Classification YAMLs have num_classes=8 hardcoded.
#      To verify: uv run python -c "
#        from dataloader.tess_dataloader import TESSClassificationDataset
#        ds = TESSClassificationDataset('data/TESS/.cache/TESS', 'train')
#        print('num_classes:', ds.num_classes, ds.label_names)"
#
# Usage:
#   bash benchmarks/TESS/run.sh
#
# Monitor:
#   squeue -u $USER
#   tail -f benchmarks/TESS/logs/tess_s4d_recon_<jobid>.out

set -e

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
WORKDIR=$(cd "$SCRIPT_DIR/../.." && pwd)
POOL="${TESS_POOL_ROOT:-/home/allisone/orcd/pool/UROP_2025_Summer/TimeSeriesPhysics}"
DATA_DIR=$POOL/data_engaging/TESS/.cache/TESS
CKPT_DIR=$POOL/checkpoints
RESULTS_DIR=$POOL/results
WANDB_ROOT=$POOL
LOGS=$WORKDIR/logs/engaging_logs/train
mkdir -p "$LOGS" "$CKPT_DIR" "$RESULTS_DIR" "$WANDB_ROOT"

# Abort early if data hasn't been downloaded yet
if [ ! -f "$DATA_DIR/tess_regression_train.parquet" ] || [ ! -f "$DATA_DIR/tess_classification_train.parquet" ]; then
    echo "ERROR: TESS data not found at $DATA_DIR"
    echo "Run from the login node first: bash benchmarks/TESS/setup_data.sh"
    exit 1
fi

if [ ! -d "$WORKDIR/.venv" ]; then
    echo "ERROR: $WORKDIR/.venv not found."
    echo "On the login node: cd $WORKDIR && module load cuda miniforge && uv sync"
    exit 1
fi
# One sync on the submit host; compute jobs only execute the venv interpreter (no uv on NFS).
if command -v uv >/dev/null 2>&1; then
    (cd "$WORKDIR" && uv sync) || { echo "ERROR: uv sync failed"; exit 1; }
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

ACTIVATE="module load cuda miniforge &&
  export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 NUMEXPR_NUM_THREADS=8 \
    OPENBLAS_NUM_THREADS=8 PANDAS_USE_PYARROW=1 WANDB_DIR=$WANDB_ROOT"

# Log visible GPU/driver in SLURM .out files (helps verify allocation on Engaging).
NVIDIA_SMI_PROBE="echo '=== nvidia-smi (job start) ===' && nvidia-smi && echo ''"

RUN="srun --cpu-bind=none $PYTHON_EXE"

# ── Stage 1: S4D reconstruction backbone ──────────────────────────────────────
JOB_RECON=$(sbatch --parsable \
  $SBATCH_COMMON \
  --job-name=tess_s4d_recon \
  --output="$LOGS/tess_s4d_recon_%j.out" \
  --error="$LOGS/tess_s4d_recon_%j.err" \
  --wrap="$ACTIVATE && $NVIDIA_SMI_PROBE && export TESS_PYTORCH_CKPT_DIR=$CKPT_DIR/tess_s4d_reconstruction && $RUN main.py fit \
    --config configs/TESS/other/train_tess_s4d_reconstruction.yaml \
    --data.init_args.data_dir $DATA_DIR")
echo "[1/8] S4D reconstruction submitted: job $JOB_RECON"

# ── Stage 2a: End-to-end models (no dependency) ───────────────────────────────
JOB_MLP_REG=$(sbatch --parsable \
  $SBATCH_COMMON \
  --job-name=tess_mlp_reg \
  --output="$LOGS/tess_mlp_reg_%j.out" \
  --error="$LOGS/tess_mlp_reg_%j.err" \
  --wrap="$ACTIVATE && $NVIDIA_SMI_PROBE && export TESS_PYTORCH_CKPT_DIR=$CKPT_DIR/tess_mlp_regression && $RUN main.py fit \
    --config configs/TESS/other/train_tess_mlp_regression.yaml \
    --data.init_args.data_dir $DATA_DIR")
echo "[2/8] MLP regression submitted: job $JOB_MLP_REG"

JOB_MLP_CLS=$(sbatch --parsable \
  $SBATCH_COMMON \
  --job-name=tess_mlp_cls \
  --output="$LOGS/tess_mlp_cls_%j.out" \
  --error="$LOGS/tess_mlp_cls_%j.err" \
  --wrap="$ACTIVATE && $NVIDIA_SMI_PROBE && export TESS_PYTORCH_CKPT_DIR=$CKPT_DIR/tess_mlp_classification && $RUN main.py fit \
    --config configs/TESS/other/train_tess_mlp_classification.yaml \
    --data.init_args.data_dir $DATA_DIR")
echo "[3/8] MLP classification submitted: job $JOB_MLP_CLS"

JOB_S4D_REG=$(sbatch --parsable \
  $SBATCH_COMMON \
  --job-name=tess_s4d_reg \
  --output="$LOGS/tess_s4d_reg_%j.out" \
  --error="$LOGS/tess_s4d_reg_%j.err" \
  --wrap="$ACTIVATE && $NVIDIA_SMI_PROBE && export TESS_PYTORCH_CKPT_DIR=$CKPT_DIR/tess_s4d_regression && $RUN main.py fit \
    --config configs/TESS/other/train_tess_s4d_regression.yaml \
    --data.init_args.data_dir $DATA_DIR")
echo "[4/8] S4D regression submitted: job $JOB_S4D_REG"

JOB_S4D_CLS=$(sbatch --parsable \
  $SBATCH_COMMON \
  --job-name=tess_s4d_cls \
  --output="$LOGS/tess_s4d_cls_%j.out" \
  --error="$LOGS/tess_s4d_cls_%j.err" \
  --wrap="$ACTIVATE && $NVIDIA_SMI_PROBE && export TESS_PYTORCH_CKPT_DIR=$CKPT_DIR/tess_s4d_classification && $RUN main.py fit \
    --config configs/TESS/other/train_tess_s4d_classification.yaml \
    --data.init_args.data_dir $DATA_DIR")
echo "[5/8] S4D classification submitted: job $JOB_S4D_CLS"

# ── Stage 2b: Frozen-backbone heads (depend on reconstruction) ────────────────
JOB_HEAD_REG=$(sbatch --parsable \
  $SBATCH_COMMON \
  --dependency=afterok:$JOB_RECON \
  --job-name=tess_s4d_head_reg \
  --output="$LOGS/tess_s4d_head_reg_%j.out" \
  --error="$LOGS/tess_s4d_head_reg_%j.err" \
  --wrap="$ACTIVATE && $NVIDIA_SMI_PROBE && export TESS_PYTORCH_CKPT_DIR=$CKPT_DIR/tess_s4d_head_regression && $RUN main.py fit \
    --config configs/TESS/other/train_tess_s4d_head_regression.yaml \
    --data.init_args.data_dir $DATA_DIR \
    --model.init_args.backbone_ckpt $CKPT_DIR/tess_s4d_reconstruction/best.ckpt")
echo "[6/8] S4D head regression submitted: job $JOB_HEAD_REG (depends on $JOB_RECON)"

JOB_HEAD_CLS=$(sbatch --parsable \
  $SBATCH_COMMON \
  --dependency=afterok:$JOB_RECON \
  --job-name=tess_s4d_head_cls \
  --output="$LOGS/tess_s4d_head_cls_%j.out" \
  --error="$LOGS/tess_s4d_head_cls_%j.err" \
  --wrap="$ACTIVATE && $NVIDIA_SMI_PROBE && export TESS_PYTORCH_CKPT_DIR=$CKPT_DIR/tess_s4d_head_classification && $RUN main.py fit \
    --config configs/TESS/other/train_tess_s4d_head_classification.yaml \
    --data.init_args.data_dir $DATA_DIR \
    --model.init_args.backbone_ckpt $CKPT_DIR/tess_s4d_reconstruction/best.ckpt")
echo "[7/8] S4D head classification submitted: job $JOB_HEAD_CLS (depends on $JOB_RECON)"

# ── Stage 3: Eval (waits for all training jobs) ───────────────────────────────
ALL_TRAIN="$JOB_RECON:$JOB_MLP_REG:$JOB_MLP_CLS:$JOB_S4D_REG:$JOB_S4D_CLS:$JOB_HEAD_REG:$JOB_HEAD_CLS"

JOB_EVAL=$(sbatch --parsable \
  $SBATCH_COMMON \
  --dependency=afterok:$ALL_TRAIN \
  --job-name=tess_eval \
  --output="$LOGS/tess_eval_%j.out" \
  --error="$LOGS/tess_eval_%j.err" \
  --wrap="$ACTIVATE && $NVIDIA_SMI_PROBE && $RUN benchmarks/TESS/eval_pipeline.py \
    --data_dir $DATA_DIR \
    --ckpt_dir $CKPT_DIR \
    --out_dir  $RESULTS_DIR")
echo "[8/8] Eval submitted: job $JOB_EVAL (depends on all training jobs)"

# echo ""
# echo "Pipeline:"
# echo "  $JOB_RECON → [$JOB_HEAD_REG, $JOB_HEAD_CLS]"
# echo "  [$JOB_MLP_REG, $JOB_MLP_CLS, $JOB_S4D_REG, $JOB_S4D_CLS, $JOB_HEAD_REG, $JOB_HEAD_CLS] → $JOB_EVAL"
# echo "Monitor: squeue -u \$USER"
