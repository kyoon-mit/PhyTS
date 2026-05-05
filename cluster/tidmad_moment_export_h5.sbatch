#!/bin/bash
#SBATCH --job-name=tidmad_moment_h5
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --partition=rtx6k-shared
#SBATCH --qos=24h_4g
#SBATCH -t 2:00:00

# Export MOMENT denoised H5 files in official TIDMAD pre-chunked format.
# Output is compatible with tidmad_denoising.py (official scoring script).
#
# Each 10M-sample chunk is denoised in 100k sub-windows, each split further
# into 512-sample pieces for MOMENT's reconstruction head (~20k forward passes/file).
# With stride=10: ~20 chunks/file × 100 sub-windows = ~2k MOMENT calls/file.
# Expected runtime: ~30 min for all 8 files.
#
# Score afterward with:
#   python src/tasks/TIDMAD/tidmad_denoising.py \
#       --data_dir benchmarks/TIDMAD/results/h5_export/moment_base --coarse
#
# Optional env vars:
#   MODEL_SIZE  — MOMENT model size: small/base/large (default: base)
#   STRIDE      — export every Nth chunk (default: 10, matches --coarse)
#   BATCH_SIZE  — sub-window batch size (default: 8)

set -euo pipefail

export APPTAINER_BIND="$PRJ_WORKSPACE"

REPO=/rg/perets_prj/ilay.kamai/PhyTS/TimeSeriesPhysics
LOGDIR=/home/ilay.kamai/athena/logs

mkdir -p "${LOGDIR}"

MODEL_SIZE="${MODEL_SIZE:-base}"

srun --ntasks=1 \
  --output="${LOGDIR}/%x_${MODEL_SIZE}_%j_%N_%t.out" \
  --error="${LOGDIR}/%x_${MODEL_SIZE}_%j_%N_%t.err" \
  --container=/apps/apptainer/images/pytorch/latest/torch-tf-cu130.sif \
  bash -lc "
    pip install -q momentfm --no-deps &&
    pip install -q jsonargparse[signatures] torchmetrics h5py pandas matplotlib &&
    cd ${REPO} &&
    PYTHONPATH=src python3 -u benchmarks/TIDMAD/export_denoised_h5.py \
      --model_type  moment \
      --model_size  ${MODEL_SIZE} \
      --data_dir    data/TIDMAD/original \
      --scale_path  data/TIDMAD/preprocessed/scale.npy \
      --out_dir     benchmarks/TIDMAD/results/h5_export/moment_${MODEL_SIZE} \
      --batch_size  ${BATCH_SIZE:-8} \
      --stride      ${STRIDE:-10}
  "
