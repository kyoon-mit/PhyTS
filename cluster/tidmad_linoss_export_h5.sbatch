#!/bin/bash
#SBATCH --job-name=tidmad_linoss_h5
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --partition=rtx6k-shared
#SBATCH --qos=24h_4g
#SBATCH -t 24:00:00

# Export denoised H5 files for LinOSS models in the official TIDMAD pre-chunked format.
# Output is compatible with tidmad_denoising.py (official scoring script).
#
# Each output file contains:
#   time_series_ch1  : (n_chunks, 10_000_000) float32 — denoised ch1
#   time_series_ch2  : (n_chunks, 10_000_000) float32 — original ch2
#   signal_frequency : (n_chunks, 1)          float32 — injection freq at 1 Hz resolution
#
# Score afterward with:
#   python src/tasks/TIDMAD/tidmad_denoising.py \
#       --data_dir benchmarks/TIDMAD/results/h5_export/linoss_190k_psd --coarse
#
# Optional env vars:
#   VARIANTS   — "mse", "psd", or "mse psd" (default: "mse psd")
#   STRIDE     — export every Nth chunk (default: 10, matches --coarse)
#   BATCH_SIZE — sub-window inference batch size (default: 4)

set -euo pipefail

export APPTAINER_BIND="$PRJ_WORKSPACE"

REPO=/rg/perets_prj/ilay.kamai/PhyTS/TimeSeriesPhysics
LOGDIR=/home/ilay.kamai/athena/logs

mkdir -p "${LOGDIR}"

VARIANTS="${VARIANTS:-psd}"
STRIDE="${STRIDE:-10}"
BATCH_SIZE="${BATCH_SIZE:-4}"

INSTALL_CMD="pip install -q jsonargparse[signatures] torchmetrics h5py pandas matplotlib && pip install -q -r ${REPO}/cluster/requirements_jax.txt && cd ${REPO}"

for VARIANT in ${VARIANTS}; do
    if [[ "${VARIANT}" == "mse" ]]; then
        CKPT="${REPO}/logs/tidmad_linoss_190k/version_0/checkpoints/checkpoint_step_53312.eqx"
        CFG="${REPO}/configs/TIDMAD/train_tidmad_linoss_190k_denoising.yaml"
        OUT="${REPO}/benchmarks/TIDMAD/results/h5_export/linoss_190k_mse"
    elif [[ "${VARIANT}" == "psd" ]]; then
        CKPT="${REPO}/logs/tidmad_linoss_190k_psd/version_0/checkpoints/checkpoint_step_213248.eqx"
        CFG="${REPO}/configs/TIDMAD/train_tidmad_linoss_190k_denoising.yaml"
        OUT="${REPO}/benchmarks/TIDMAD/results/h5_export/linoss_190k_psd"
    else
        echo "Unknown variant: ${VARIANT}"; exit 1
    fi

    echo "=== Exporting linoss_190k_${VARIANT} ==="
    srun --ntasks=1 \
      --output="${LOGDIR}/%x_${VARIANT}_%j_%N_%t.out" \
      --error="${LOGDIR}/%x_${VARIANT}_%j_%N_%t.err" \
      --container=/apps/apptainer/images/pytorch/latest/torch-tf-cu130.sif \
      bash -lc "${INSTALL_CMD} && PYTHONPATH=src python3 -u benchmarks/TIDMAD/export_denoised_h5.py \
        --model_type linoss \
        --ckpt  ${CKPT} \
        --cfg   ${CFG} \
        --out_dir ${OUT} \
        --batch_size ${BATCH_SIZE} \
        --stride ${STRIDE}"
done
