#!/bin/bash
#SBATCH --job-name=tidmad_chronos_eval
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --partition=l40s-shared
#SBATCH --qos=24h_4g
#SBATCH -t 24:00:00

# Evaluate Chronos zero-shot baselines on the TIDMAD test set.
# Runs chronos_tiny and chronos_base through the same pipeline as trained models:
#   - results/chronos_{tiny,base}/test_scores.csv
#   - results/chronos_{tiny,base}/summary.json
#   - updates results/master_results.csv
#
# Uses stride=50 (every 50th window) → 400 windows/file, ~3200 total.
# Covers all injection frequencies in the sweep without the full 2h+ per-file cost.
#
# Optional env vars:
#   CHRONOS_SIZES        — space-separated, default "tiny"
#   CHRONOS_MAX_WINDOWS  — integer cap on windows per file (default: all)
#   STRIDE               — window stride (default: 50)

set -euo pipefail

export APPTAINER_BIND="$PRJ_WORKSPACE"

REPO=/rg/perets_prj/ilay.kamai/PhyTS/TimeSeriesPhysics
LOGDIR=/home/ilay.kamai/athena/logs

mkdir -p "${LOGDIR}"

SIZES="${CHRONOS_SIZES:-tiny}"
VARIANTS_ARG=$(echo $SIZES | tr ' ' '\n' | sed 's/^/chronos_/' | tr '\n' ' ')

MAX_WIN_ARG=""
if [[ -n "${CHRONOS_MAX_WINDOWS:-}" ]]; then
    MAX_WIN_ARG="--chronos_max_windows ${CHRONOS_MAX_WINDOWS}"
fi

STRIDE_ARG="--stride ${STRIDE:-50}"

srun --ntasks=1 \
  --output="${LOGDIR}/%x_%j_%N_%t.out" \
  --error="${LOGDIR}/%x_%j_%N_%t.err" \
  --container=/apps/apptainer/images/pytorch/latest/torch-tf-cu130.sif \
  bash -lc "
    pip install -q transformers accelerate optimum chronos-forecasting &&
    pip install -q jsonargparse[signatures] torchmetrics h5py pandas matplotlib &&
    cd ${REPO} &&
    PYTHONPATH=src python3 -u benchmarks/TIDMAD/evaluate_all.py \
      --variants ${VARIANTS_ARG} \
      --data_dir   data/TIDMAD/original \
      --scale_path data/TIDMAD/preprocessed/scale.npy \
      --out_dir    benchmarks/TIDMAD/results \
      --batch_size 8 \
      ${MAX_WIN_ARG} ${STRIDE_ARG}
  "
