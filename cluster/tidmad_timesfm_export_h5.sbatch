#!/bin/bash
#SBATCH --job-name=tidmad_timesfm_h5
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --partition=l40s-shared
#SBATCH --qos=12h_4g
#SBATCH -t 8:00:00

# Export TimesFM denoised H5 files in official TIDMAD pre-chunked format.
#
# TimesFM is a patch-based decoder-only model (200M), non-autoregressive.
# 100k window -> 195 chunks of 512 -> 2 fwd passes per chunk (bidirectional forecast).
# With stride=20: 10 chunks/file x 100 sub-windows x 195 TimesFM calls each.
# Expected runtime: ~4-6 hours for all 8 files.
#
# Score afterward with:
#   python src/tasks/TIDMAD/tidmad_denoising.py \
#       --data_dir benchmarks/TIDMAD/results/h5_export/timesfm_base --coarse
#
# Optional env vars:
#   MODEL_SIZE  — base (200M, default) or large (500M)
#   STRIDE      — export every Nth chunk (default: 20)
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
    pip install -q timesfm==1.3.0 &&
    pip install -q jsonargparse[signatures] torchmetrics h5py pandas matplotlib &&
    cd ${REPO} &&
    PYTHONPATH=src python3 -u benchmarks/TIDMAD/export_denoised_h5.py \
      --model_type  timesfm \
      --model_size  ${MODEL_SIZE} \
      --data_dir    data/TIDMAD/original \
      --scale_path  data/TIDMAD/preprocessed/scale.npy \
      --out_dir     benchmarks/TIDMAD/results/h5_export/timesfm_${MODEL_SIZE} \
      --batch_size  ${BATCH_SIZE:-8} \
      --stride      ${STRIDE:-20}
  "
