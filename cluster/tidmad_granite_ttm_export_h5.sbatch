#!/bin/bash
#SBATCH --job-name=tidmad_granite_ttm_h5
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --partition=l40s-shared
#SBATCH --qos=12h_4g
#SBATCH -t 4:00:00

# Export Granite TTM denoised H5 files in official TIDMAD pre-chunked format.
#
# Granite TTM is a fast MLP-mixer (patch-based, single forward pass per chunk).
# 100k window -> 195 chunks of 512 -> 2 fwd passes per chunk (bidirectional forecast).
# With stride=10: 20 chunks/file x 100 sub-windows x 195 TTM calls each.
# Expected runtime: ~2-3 hours for all 8 files.
#
# Score afterward with:
#   python src/tasks/TIDMAD/tidmad_denoising.py \
#       --data_dir benchmarks/TIDMAD/results/h5_export/granite_ttm_base --coarse
#
# Optional env vars:
#   MODEL_SIZE  — base (default) or small/large
#   STRIDE      — export every Nth chunk (default: 10)
#   BATCH_SIZE  — sub-window batch size passed to export script (default: 8)

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
    pip install -q transformers==4.44.2 accelerate==1.13.0 &&
    pip install -q granite-tsfm==0.2.23 &&
    pip install -q jsonargparse[signatures] torchmetrics h5py pandas matplotlib &&
    cd ${REPO} &&
    PYTHONPATH=src python3 -u benchmarks/TIDMAD/export_denoised_h5.py \
      --model_type  granite_ttm \
      --model_size  ${MODEL_SIZE} \
      --data_dir    data/TIDMAD/original \
      --scale_path  data/TIDMAD/preprocessed/scale.npy \
      --out_dir     benchmarks/TIDMAD/results/h5_export/granite_ttm_${MODEL_SIZE} \
      --batch_size  ${BATCH_SIZE:-8} \
      --stride      ${STRIDE:-10}
  "
