#!/bin/bash
#SBATCH --job-name=tidmad_moment_eval
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --partition=rtx6k-shared
#SBATCH --qos=24h_4g
#SBATCH -t 6:00:00

# Evaluate MOMENT zero-shot baselines on the TIDMAD test set.
# Uses the native reconstruction head (no fine-tuning, no checkpoint).
# Each 100k-sample window is chunked into 512-sample pieces (MOMENT seq_len),
# reconstructed in a single forward pass, and reassembled.
#
# Uses stride=10 (every 10th window) → ~2000 windows/file, ~16k total.
# Estimated runtime: ~20 min.
#
# Outputs:
#   results/moment_{small,base}/test_scores.csv
#   results/moment_{small,base}/summary.json
#   updates results/master_results.csv
#
# Optional env vars:
#   MOMENT_SIZES  — space-separated sizes, default "base"
#   STRIDE        — window stride (default: 10)

set -euo pipefail

export APPTAINER_BIND="$PRJ_WORKSPACE"

REPO=/rg/perets_prj/ilay.kamai/PhyTS/TimeSeriesPhysics
LOGDIR=/home/ilay.kamai/athena/logs

mkdir -p "${LOGDIR}"

SIZES="${MOMENT_SIZES:-base}"
VARIANTS_ARG=$(echo $SIZES | tr ' ' '\n' | sed 's/^/moment_/' | tr '\n' ' ')

STRIDE_ARG="--stride ${STRIDE:-10}"

srun --ntasks=1 \
  --output="${LOGDIR}/%x_%j_%N_%t.out" \
  --error="${LOGDIR}/%x_%j_%N_%t.err" \
  --container=/apps/apptainer/images/pytorch/latest/torch-tf-cu130.sif \
  bash -lc "
    pip install -q momentfm --no-deps &&
    pip install -q jsonargparse[signatures] torchmetrics h5py pandas matplotlib &&
    cd ${REPO} &&
    PYTHONPATH=src python3 -u benchmarks/TIDMAD/evaluate_all.py \
      --variants ${VARIANTS_ARG} \
      --data_dir   data/TIDMAD/original \
      --scale_path data/TIDMAD/preprocessed/scale.npy \
      --out_dir    benchmarks/TIDMAD/results \
      --batch_size 8 \
      ${STRIDE_ARG}
  "
