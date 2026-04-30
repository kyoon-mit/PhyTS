#!/bin/bash
# TESS classification benchmark 
#
# Usage:
#   bash benchmarks/TESS/run.sh
#
# Monitor:
#   squeue -u $USER

set -e

WORKDIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
LOGS=$WORKDIR/benchmarks/TESS/logs
mkdir -p "$LOGS"

ACTIVATE="source ~/.bashrc && conda activate ts_cuda312"

SBATCH_COMMON="--partition=gpu_requeue --nodes=1 --ntasks=1 --ntasks-per-node=1 \
    --gres=gpu:1 --cpus-per-task=4 --mem=40G --time=1-00:00:00 --chdir=$WORKDIR"

JOB=$(sbatch --parsable $SBATCH_COMMON \
    --job-name=tess_transformer_classification \
    --output=$LOGS/tess_transformer_classification_%j.out \
    --error=$LOGS/tess_transformer_classification_%j.err \
    --wrap="$ACTIVATE && srun --cpu-bind=none python main.py fit \
        --config configs/TESS/train_tess_transformer_classification.yaml")

echo "[1/1] Transformer classifier submitted: job $JOB"
echo "Monitor: squeue -u $USER"