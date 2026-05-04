#!/bin/bash
#SBATCH --job-name=tidmad_conv_abl
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --partition=l40s-shared
#SBATCH --qos=24h_4g
#SBATCH -t 4:00:00

# Runs all ConvAE ablation variants (s, m, l, w) sequentially — 5 epochs each.

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
    for TAG in s m l w; do
      echo \"=== ConvAE ablation: \${TAG} ===\"
      PYTHONPATH=src python3 main.py fit \
        --config configs/TIDMAD/ablation/conv_abl_\${TAG}.yaml
    done
  "
