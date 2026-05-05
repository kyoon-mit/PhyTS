#!/bin/bash
#SBATCH --job-name=tidmad_linoss_inf
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --partition=l40s-shared
#SBATCH --qos=72h_8g
#SBATCH -t 24:00:00

# Inference-only evaluation for LinOSS-190K (MSE and PSD) on TIDMAD test set.
#
# Uses stride=10 (every 10th window) → 2000 windows/file, ~16k total.
# Covers all injection frequencies in the sweep.  batch_size=4 ensures both
# 20000 and 20100-window files divide evenly → single JAX JIT compilation.
#
# Outputs per variant:
#   results/{variant}/test_scores.csv   — per-window SNR values
#   results/{variant}/summary.json      — score + metadata
#   results/master_results.csv          — merged aggregate
#
# Optional env vars:
#   VARIANTS   — override subset, e.g. "linoss_190k_mse"
#   STRIDE     — window stride (default: 10)
#   BATCH_SIZE — inference batch size (default: 4)

set -euo pipefail

export APPTAINER_BIND="$PRJ_WORKSPACE"

REPO=/rg/perets_prj/ilay.kamai/PhyTS/TimeSeriesPhysics
LOGDIR=/home/ilay.kamai/athena/logs

mkdir -p "${LOGDIR}"

DEFAULT_VARIANTS="linoss_190k_mse linoss_190k_full"
VARIANTS_ARG="--variants ${VARIANTS:-${DEFAULT_VARIANTS}}"
STRIDE_ARG="--stride ${STRIDE:-10}"

srun --ntasks=1 \
  --output="${LOGDIR}/%x_%j_%N_%t.out" \
  --error="${LOGDIR}/%x_%j_%N_%t.err" \
  --container=/apps/apptainer/images/pytorch/latest/torch-tf-cu130.sif \
  bash -lc "
    pip install -q jsonargparse[signatures] torchmetrics h5py pandas matplotlib &&
    pip install -q -r ${REPO}/cluster/requirements_jax.txt &&
    cd ${REPO} &&
    PYTHONPATH=src python3 -u benchmarks/TIDMAD/evaluate_all.py \
      --data_dir   data/TIDMAD/original \
      --scale_path data/TIDMAD/preprocessed/scale.npy \
      --out_dir    benchmarks/TIDMAD/results \
      --batch_size ${BATCH_SIZE:-4} \
      ${VARIANTS_ARG} \
      ${STRIDE_ARG}
  "
