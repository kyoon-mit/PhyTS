#!/bin/bash
#SBATCH --job-name=abl_compare
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --partition=l40s-shared
#SBATCH --qos=24h_4g

set -euo pipefail

REPO=/rg/perets_prj/ilay.kamai/PhyTS/TimeSeriesPhysics
LOGDIR=/home/ilay.kamai/athena/logs

mkdir -p "${LOGDIR}"

srun --ntasks=1 \
  --output="${LOGDIR}/%x_%j_%N_%t.out" \
  --error="${LOGDIR}/%x_%j_%N_%t.err" \
  bash -lc "cd ${REPO} && python3 benchmarks/TIDMAD/compare_ablations.py \
              | tee benchmarks/TIDMAD/ablation_results.txt"
