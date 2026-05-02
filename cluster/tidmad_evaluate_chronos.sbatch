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
# Chronos is very slow (~10-20 min/file for tiny). Set CHRONOS_MAX_WINDOWS to
# cap windows per file for a quick sanity check (e.g. CHRONOS_MAX_WINDOWS=50).
# Leave unset for the full evaluation (all windows, may take the full 24h).
#
# Optional env vars:
#   CHRONOS_SIZES        — space-separated, default "tiny base"
#   CHRONOS_MAX_WINDOWS  — integer cap on windows per file (default: all)

set -euo pipefail

export APPTAINER_BIND="$PRJ_WORKSPACE"

REPO=/rg/perets_prj/ilay.kamai/PhyTS/TimeSeriesPhysics
LOGDIR=/home/ilay.kamai/athena/logs

mkdir -p "${LOGDIR}"

SIZES="${CHRONOS_SIZES:-tiny base}"
VARIANTS_ARG=$(echo $SIZES | tr ' ' '\n' | sed 's/^/chronos_/' | tr '\n' ' ')

MAX_WIN_ARG=""
if [[ -n "${CHRONOS_MAX_WINDOWS:-}" ]]; then
    MAX_WIN_ARG="--chronos_max_windows ${CHRONOS_MAX_WINDOWS}"
fi

srun --ntasks=1 \
  --output="${LOGDIR}/%x_%j_%N_%t.out" \
  --error="${LOGDIR}/%x_%j_%N_%t.err" \
  --container=/apps/apptainer/images/pytorch/latest/torch-tf-cu130.sif \
  bash -lc "
    pip install -q transformers accelerate optimum &&
    pip install -q jsonargparse[signatures] torchmetrics h5py pandas matplotlib &&
    cd ${REPO} &&
    PYTHONPATH=src python3 benchmarks/TIDMAD/evaluate_all.py \
      --variants ${VARIANTS_ARG} \
      --data_dir   data/TIDMAD/original \
      --scale_path data/TIDMAD/preprocessed/scale.npy \
      --out_dir    benchmarks/TIDMAD/results \
      --batch_size 8 \
      ${MAX_WIN_ARG}
  "
