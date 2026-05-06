#!/bin/bash
#SBATCH --job-name=tidmad_preprocess
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
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
  bash -lc "pip install -q h5py && cd ${REPO} && python3 data/TIDMAD/preprocess_tidmad.py \
    --data_dir  data/TIDMAD/original \
    --out_dir   data/TIDMAD/preprocessed \
    --n_per_file 1666 \
    --seed 42"
