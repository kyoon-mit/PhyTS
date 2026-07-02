#!/bin/bash
#SBATCH --job-name=tidmad_moirai_h5
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --partition=l40s-shared
#SBATCH --qos=12h_4g
#SBATCH -t 8:00:00

# Export MOIRAI denoised H5 files in official TIDMAD pre-chunked format.
#
# MOIRAI is a patch-based probabilistic encoder (91M base), non-autoregressive.
# Uses GluonTS ListDataset interface; 2 forecast passes per 512-sample chunk.
# With stride=20: 10 chunks/file x 100 sub-windows x 195 MOIRAI calls each.
# Expected runtime: ~4-6 hours for all 8 files.
#
# Score afterward with:
#   python src/tasks/TIDMAD/tidmad_denoising.py \
#       --data_dir benchmarks/TIDMAD/results/h5_export/moirai_base --coarse
#
# Optional env vars:
#   MODEL_SIZE  — base (default) or small/large
#   STRIDE      — export every Nth chunk (default: 20)
#   BATCH_SIZE  — sub-window batch size (default: 8)

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
    pip install -q --no-deps uni2ts && pip install -r ${REPO}/cluster/requirements_moirai.txt && pip show jaxtyping &&
    pip install -q jsonargparse[signatures] torchmetrics h5py pandas matplotlib &&
    cd ${REPO} &&
    PYTHONPATH=src python3 -u benchmarks/TIDMAD/export_denoised_h5.py \
      --model_type  moirai \
      --model_size  ${MODEL_SIZE} \
      --data_dir    data/TIDMAD/original \
      --scale_path  data/TIDMAD/preprocessed/scale.npy \
      --out_dir     benchmarks/TIDMAD/results/h5_export/moirai_${MODEL_SIZE} \
      --batch_size  ${BATCH_SIZE:-8} \
      --stride      ${STRIDE:-20}
  "
