#!/bin/bash
#SBATCH --job-name=tidmad_chronos_h5
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --partition=rtx6k-shared
#SBATCH --qos=72h_8g
#SBATCH -t 24:00:00

# Export Chronos-Tiny denoised H5 files in official TIDMAD pre-chunked format.
# Output is compatible with tidmad_denoising.py (official scoring script).
#
# Chronos is slow (~781 forward passes per 100k sub-window).
# With stride=50: 4 chunks/file × 100 sub-windows × ~13 Chronos API calls = ~5K calls total.
# Expected runtime: 2-4 hours for all 8 files.
# Override: STRIDE=100 for faster run (2 chunks/file, ~2.5K calls total).
#
# Score afterward with:
#   python src/tasks/TIDMAD/tidmad_denoising.py \
#       --data_dir benchmarks/TIDMAD/results/h5_export/chronos_tiny --coarse
#
# Optional env vars:
#   MODEL_SIZE  — Chronos model size: tiny/small/base/large (default: tiny)
#   STRIDE      — export every Nth chunk (default: 10, matches --coarse)
#   BATCH_SIZE  — sub-window batch size (default: 8)

set -euo pipefail

export APPTAINER_BIND="$PRJ_WORKSPACE"

REPO=/rg/perets_prj/ilay.kamai/PhyTS/TimeSeriesPhysics
LOGDIR=/home/ilay.kamai/athena/logs

mkdir -p "${LOGDIR}"

MODEL_SIZE="${MODEL_SIZE:-tiny}"

srun --ntasks=1 \
  --output="${LOGDIR}/%x_%j_%N_%t.out" \
  --error="${LOGDIR}/%x_%j_%N_%t.err" \
  --container=/apps/apptainer/images/pytorch/latest/torch-tf-cu130.sif \
  bash -lc "
    pip install -q transformers accelerate optimum chronos-forecasting &&
    pip install -q jsonargparse[signatures] torchmetrics h5py pandas matplotlib &&
    cd ${REPO} &&
    PYTHONPATH=src python3 -u benchmarks/TIDMAD/export_denoised_h5.py \
      --model_type  chronos \
      --model_size  ${MODEL_SIZE} \
      --data_dir    data/TIDMAD/original \
      --scale_path  data/TIDMAD/preprocessed/scale.npy \
      --out_dir     benchmarks/TIDMAD/results/h5_export/chronos_${MODEL_SIZE} \
      --batch_size  ${BATCH_SIZE:-8} \
      --stride      ${STRIDE:-50}
  "
