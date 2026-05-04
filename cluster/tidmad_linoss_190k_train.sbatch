#!/bin/bash
#SBATCH --job-name=tidmad_linoss_190k
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --partition=l40s-shared
#SBATCH --qos=24h_4g
#SBATCH -t 24:00:00

# Full training of LinOSS ~190K params (blocks=4, ssm_size=32, H=128, damped_IMEX).
# Matched to ConvAE-L param count for fair benchmark comparison.
# lr=1e-4 to avoid gradient explosion seen with lr=1e-3.

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
    pip install -q -r ${REPO}/cluster/requirements_jax.txt &&
    cd ${REPO} &&
    PYTHONPATH=src python3 main.py fit \
      --config configs/TIDMAD/train_tidmad_linoss_190k_denoising.yaml
  "
