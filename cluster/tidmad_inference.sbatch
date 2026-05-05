#!/bin/bash
#SBATCH --job-name=tidmad_inference
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --partition=l40s-shared
#SBATCH --qos=24h_4g
#SBATCH -t 4:00:00

# Required exports (set via --export):
#   MODEL_TYPE  — conv | linoss | s4d
#   CFG         — path to training config YAML
#   CKPT        — checkpoint file path      (conv/s4d: .ckpt)
#   CKPT_DIR    — checkpoint directory      (linoss:   directory of .eqx files)
# Provide either CKPT or CKPT_DIR (not both).

CKPT="${CKPT:-}"
CKPT_DIR="${CKPT_DIR:-}"

set -euo pipefail

export APPTAINER_BIND="$PRJ_WORKSPACE"

REPO=/rg/perets_prj/ilay.kamai/PhyTS/TimeSeriesPhysics
LOGDIR=/home/ilay.kamai/athena/logs

mkdir -p "${LOGDIR}"

# Build the checkpoint argument
if [ -n "${CKPT}" ]; then
  CKPT_ARG="--ckpt ${CKPT}"
else
  CKPT_ARG="--ckpt_dir ${CKPT_DIR}"
fi

srun --ntasks=1 \
  --output="${LOGDIR}/%x_%j_%N_%t.out" \
  --error="${LOGDIR}/%x_%j_%N_%t.err" \
  --container=/apps/apptainer/images/pytorch/latest/torch-tf-cu130.sif \
  bash -lc "pip install -q h5py && \
            pip install -q -r ${REPO}/cluster/requirements_jax.txt && \
            cd ${REPO} && PYTHONPATH=src python3 benchmarks/TIDMAD/run_inference.py \
    --model_type ${MODEL_TYPE} \
    ${CKPT_ARG} \
    --cfg        ${CFG} \
    --data_dir   data/TIDMAD/original \
    --scale_path data/TIDMAD/preprocessed/scale.npy \
    --out_dir    benchmarks/TIDMAD/results/${MODEL_TYPE} \
    --batch_size 8"
