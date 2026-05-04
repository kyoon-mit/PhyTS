#!/bin/bash
#SBATCH --job-name=tidmad_linoss_m
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --partition=l40s-shared
#SBATCH --qos=24h_4g
#SBATCH -t 8:00:00

# LinOSS-M retry with lr=1e-4 (was 1e-3, caused NaN at epoch 1 step 101451).
# blocks=2, ssm_size=32, H=64, damped_IMEX, ~34K params.

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
      --config configs/TIDMAD/ablation/linoss_abl_m_lr1e4.yaml
  "
