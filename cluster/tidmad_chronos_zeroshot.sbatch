#!/bin/bash
#SBATCH --job-name=tidmad_chronos_zs
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --partition=l40s-shared
#SBATCH --qos=24h_4g
#SBATCH -t 4:00:00

# Chronos zero-shot denoising benchmark on TIDMAD test set.
# MODEL_SIZE: tiny / small / base / large  (default: base)
# CHUNK_SIZE: Chronos context window in samples (default: 512)

set -euo pipefail

export APPTAINER_BIND="$PRJ_WORKSPACE"

MODEL_SIZE=${MODEL_SIZE:-base}
CHUNK_SIZE=${CHUNK_SIZE:-512}
REPO=/rg/perets_prj/ilay.kamai/PhyTS/TimeSeriesPhysics
LOGDIR=/home/ilay.kamai/athena/logs

mkdir -p "${LOGDIR}"

srun --ntasks=1 \
  --output="${LOGDIR}/%x_%j_%N_%t.out" \
  --error="${LOGDIR}/%x_%j_%N_%t.err" \
  --container=/apps/apptainer/images/pytorch/latest/torch-tf-cu130.sif \
  bash -lc "
    pip install -q chronos-forecasting h5py
    cd ${REPO}
    echo \"=== Chronos zero-shot: size=${MODEL_SIZE} chunk=${CHUNK_SIZE} ===\"
    PYTHONPATH=src python3 benchmarks/TIDMAD/run_inference.py \
      --model_type chronos \
      --model_size \${MODEL_SIZE} \
      --chunk_size \${CHUNK_SIZE} \
      --data_dir   data/TIDMAD/original \
      --scale_path data/TIDMAD/preprocessed/scale.npy \
      --out_dir    benchmarks/TIDMAD/results/chronos_\${MODEL_SIZE}
  "
