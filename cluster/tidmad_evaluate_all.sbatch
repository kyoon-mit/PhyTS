#!/bin/bash
#SBATCH --job-name=tidmad_eval
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --partition=l40s-shared
#SBATCH --qos=72h_8g
#SBATCH -t 24:00:00

# Evaluate all trained TIDMAD denoising variants (conv + linoss).
# Chronos zero-shot baselines are handled separately by tidmad_evaluate_chronos.sbatch.
#   - Loss curve plots (results/{variant}/loss_curve.png)
#   - Test-set benchmark scores (results/{variant}/test_scores.csv)
#   - Per-variant summary.json with n_params + all metrics
#   - Aggregated results/master_results.csv
#
# Optional env vars:
#   VARIANTS    — space-separated subset, e.g. "conv_l_full linoss_190k_full"
#   PLOT_ONLY   — set to 1 to skip inference and only generate loss curves
#   BATCH_SIZE  — inference batch size (default 64; use 8 for conv-only runs)

set -euo pipefail

export APPTAINER_BIND="$PRJ_WORKSPACE"

REPO=/rg/perets_prj/ilay.kamai/PhyTS/TimeSeriesPhysics
LOGDIR=/home/ilay.kamai/athena/logs

mkdir -p "${LOGDIR}"

# Default: full training runs only (ablations are plot_only; chronos uses its own sbatch)
DEFAULT_VARIANTS="conv_abl_s conv_abl_m conv_abl_l conv_abl_w conv_abl_xl conv_l_mse conv_l_full linoss_abl_s linoss_190k_mse linoss_190k_full"

VARIANTS_ARG="--variants ${VARIANTS:-${DEFAULT_VARIANTS}}"

PLOT_ONLY_ARG=""
if [[ "${PLOT_ONLY:-0}" == "1" ]]; then
    PLOT_ONLY_ARG="--plot_only"
fi

STRIDE_ARG="--stride ${STRIDE:-1}"

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
      --batch_size ${BATCH_SIZE:-64} \
      ${VARIANTS_ARG} ${PLOT_ONLY_ARG} ${STRIDE_ARG}
  "
