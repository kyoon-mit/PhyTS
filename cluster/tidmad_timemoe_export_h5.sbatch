#!/bin/bash
#SBATCH --job-name=tidmad_timemoe_h5
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --partition=l40s-shared
#SBATCH --qos=24h_4g
#SBATCH -t 12:00:00

# Export Time-MoE denoised H5 files in official TIDMAD pre-chunked format.
#
# Time-MoE is an autoregressive sparse-MoE model (50M for base).
# 100k window -> 195 chunks of 512 -> 2 autoregressive passes (generating 256 tokens each).
# With stride=20: 10 chunks/file x 100 sub-windows x 195 TimeMoE calls each.
# Expected runtime: ~6-10 hours for all 8 files.
#
# Score afterward with:
#   python src/tasks/TIDMAD/tidmad_denoising.py \
#       --data_dir benchmarks/TIDMAD/results/h5_export/timemoe_base --coarse
#
# Optional env vars:
#   MODEL_SIZE  — base (default) or large (200M)
#   STRIDE      — export every Nth chunk (default: 20)
#   BATCH_SIZE  — sub-window batch size passed to export loop (default: 2)
#   SUB_BATCH   — inner chunk batch size for model inference (default: 16)

set -euo pipefail

export APPTAINER_BIND="${PRJ_WORKSPACE:-/rg/perets_prj/ilay.kamai}"

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
    pip install -q jsonargparse[signatures] torchmetrics h5py pandas matplotlib &&
    cd ${REPO} &&
    PYTHONPATH=src python3 -u benchmarks/TIDMAD/export_denoised_h5.py \
      --model_type  timemoe \
      --model_size  ${MODEL_SIZE} \
      --data_dir    data/TIDMAD/original \
      --scale_path  data/TIDMAD/preprocessed/scale.npy \
      --out_dir     benchmarks/TIDMAD/results/h5_export/timemoe_${MODEL_SIZE} \
      --batch_size  ${BATCH_SIZE:-2} \
      --sub_batch   ${SUB_BATCH:-2} \
      --stride      ${STRIDE:-20}
  "
