#!/bin/bash
#SBATCH --job-name=tidmad_eval
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:nvidia_rtx6k:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --partition=rtx6k-shared
#SBATCH --qos=24h_4g
#SBATCH -t 4:00:00

set -euo pipefail

export APPTAINER_BIND="$PRJ_WORKSPACE"

REPO=/rg/perets_prj/ilay.kamai/PhyTS/TimeSeriesPhysics
LOGDIR=/home/ilay.kamai/athena/logs

mkdir -p "${LOGDIR}"

srun --ntasks=1 \
  --output="${LOGDIR}/%x_%j_%N_%t.out" \
  --error="${LOGDIR}/%x_%j_%N_%t.err" \
  --container=/apps/apptainer/images/pytorch/latest/torch-tf-cu130.sif \
  bash -lc "pip install -q jsonargparse[signatures] torchmetrics h5py matplotlib && cd ${REPO} && PYTHONPATH=src python3 benchmarks/TIDMAD/eval_pipeline.py \
    --denoiser_ckpt        checkpoints/tidmad_s4d_denoising/best.ckpt \
    --denoiser_cfg         configs/TIDMAD/train_tidmad_s4d_denoising.yaml \
    --regressor_raw_ckpt   checkpoints/tidmad_mlp_regression_raw/best.ckpt \
    --regressor_raw_cfg    configs/TIDMAD/train_tidmad_mlp_regression_raw.yaml \
    --regressor_clean_ckpt checkpoints/tidmad_mlp_regression_clean/best.ckpt \
    --regressor_clean_cfg  configs/TIDMAD/train_tidmad_mlp_regression_clean.yaml \
    --data_dir             data/TIDMAD/preprocessed \
    --out_dir              benchmarks/TIDMAD"
