#!/bin/bash
# TESS benchmark: vanilla Transformer classification on Engaging.
#
# Usage:
#   bash benchmarks/TESS/run.sh
#
# Monitor:
#   squeue -u $USER
#   tail -f benchmarks/TESS/logs/tess_transformer_classification_<jobid>.out

set -e

WORKDIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
LOGS=$WORKDIR/benchmarks/TESS/logs

mkdir -p "$LOGS"

SBATCH_COMMON="
    --gres=gpu:1
    --cpus-per-task=4
    --mem=48G
    --time=2:00:00
    --chdir=$WORKDIR
"

JOB1=$(sbatch --parsable \
    $SBATCH_COMMON \
    --job-name=tess_transformer_classification \
    --output=$LOGS/tess_transformer_classification_%j.out \
    --error=$LOGS/tess_transformer_classification_%j.err \
    --wrap="srun --cpu-bind=none uv run python main.py fit \
        --config configs/TESS/train_tess_transformer_classification.yaml")

echo "[1/1] Transformer classifier submitted: job $JOB1"
echo "Monitor:  squeue -u $USER"