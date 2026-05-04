#!/bin/bash
#SBATCH --job-name=tidmad_conv_xl
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --partition=l40s-shared
#SBATCH --qos=24h_4g
#SBATCH -t 8:00:00

# Runs ConvAE xl variant (n_layers=5, channels=128, kernel=9, ~759K params).

set -euo pipefail

export APPTAINER_BIND="$PRJ_WORKSPACE"

REPO=/rg/perets_prj/ilay.kamai/PhyTS/TimeSeriesPhysics
LOGDIR=/home/ilay.kamai/athena/logs

mkdir -p "${LOGDIR}"

srun --ntasks=1 \
  --output="${LOGDIR}/%x_%j_%N_%t.out" \
  --error="${LOGDIR}/%x_%j_%N_%t.err" \
  --container=/apps/apptainer/images/pytorch/latest/torch-tf-cu130.sif \
  bash -lc "
    pip install -q jsonargparse[signatures] torchmetrics h5py
    cd ${REPO}
    echo \"=== ConvAE ablation: xl ===\"
    PYTHONPATH=src python3 main.py fit \
      --config configs/TIDMAD/ablation/conv_abl_xl.yaml
  "
