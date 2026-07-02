#!/bin/bash
#SBATCH --job-name=tidmad_official_score
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --partition=l40s-shared
#SBATCH --qos=24h_4g
#SBATCH -t 4:00:00

# Run the official TIDMAD coarse scorer on pre-exported H5 files.
# Scores each variant in MODELS (space-separated) sequentially.
# Reads from: benchmarks/TIDMAD/results/h5_export/<model>/
# Prints score to stdout (log captured by srun --output).
#
# Usage:
#   sbatch cluster/tidmad_official_score.sbatch
#   MODELS="granite_ttm_base moirai_base" sbatch cluster/tidmad_official_score.sbatch

set -euo pipefail

export APPTAINER_BIND="$PRJ_WORKSPACE"

REPO=/rg/perets_prj/ilay.kamai/PhyTS/TimeSeriesPhysics
LOGDIR=/home/ilay.kamai/athena/logs

mkdir -p "${LOGDIR}"

MODELS="${MODELS:-granite_ttm_base moirai_base}"

srun --ntasks=1 \
  --output="${LOGDIR}/%x_%j_%N_%t.out" \
  --error="${LOGDIR}/%x_%j_%N_%t.err" \
  --container=/apps/apptainer/images/pytorch/latest/torch-tf-cu130.sif \
  bash -lc "
    pip install -q h5py &&
    cd ${REPO} &&
    for MODEL in ${MODELS}; do
      echo \"====== Scoring \${MODEL} ======\" &&
      PYTHONPATH=src python3 -u src/tasks/TIDMAD/tidmad_denoising.py \
        --data_dir benchmarks/TIDMAD/results/h5_export/\${MODEL} \
        --coarse --parallel --num_workers 8
    done
  "
