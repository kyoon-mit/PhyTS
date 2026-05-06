#!/bin/bash
#SBATCH --job-name=tidmad_conv_l
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --partition=l40s-shared
#SBATCH --qos=12h_4g
#SBATCH -t 12:00:00

# Full training of ConvAE-L (n_layers=5, latent_channels=64, kernel_size=9, ~190K params).
# Best conv ablation variant. Logs to logs/tidmad_conv_l/, ckpt to checkpoints/tidmad_conv_l/.

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
    pip install -q jsonargparse[signatures] torchmetrics h5py &&
    cd ${REPO} &&
    PYTHONPATH=src python3 main.py fit \
      --config configs/TIDMAD/train_tidmad_conv_l_denoising.yaml
  "
