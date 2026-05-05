#!/bin/bash
#SBATCH --job-name=tidmad_train
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:nvidia_rtx6k:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --partition=rtx6k-shared
#SBATCH --qos=24h_4g
#SBATCH -t 24:00:00

set -euo pipefail

export APPTAINER_BIND="$PRJ_WORKSPACE"

# CONFIG must be set via --export=CONFIG=...
REPO=/rg/perets_prj/ilay.kamai/PhyTS/TimeSeriesPhysics
LOGDIR=/home/ilay.kamai/athena/logs

mkdir -p "${LOGDIR}"

srun --ntasks=1 \
  --output="${LOGDIR}/%x_%j_%N_%t.out" \
  --error="${LOGDIR}/%x_%j_%N_%t.err" \
  --container=/apps/apptainer/images/pytorch/latest/torch-tf-cu130.sif \
  bash -lc "pip install -q jsonargparse[signatures] torchmetrics h5py && cd ${REPO} && PYTHONPATH=src python3 main.py fit --config ${CONFIG}"
