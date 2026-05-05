#!/usr/bin/env bash
# MOMENT LoRA on Project 8 (cavity noise, full 24576 cutoff) for krakas
# (24 GiB RTX 4500 Ada -- enough headroom for bs=8 at full length).
# Mirrors the Condor submission used elsewhere; written as a direct-run
# script because the existing krakas pattern runs locally on the node
# rather than via the Condor scheduler.
#
# Usage (on krakas):
#   bash benchmarks/foundation/condor/run_project8_moment_lora_krakas.sh
#
# Re-runs the moment LoRA cell that died on chokai (job 385) with an HDF5
# I/O fault.  Same hyperparams as the original Condor submission so the
# resulting cell is comparable to the rest of the LoRA column.
set -u

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "${REPO}"

PY="${FM_PYTHON:-/esat/smcdata/users/kkontras/Image_Dataset/no_backup/envs/tsenv_fm/bin/python}"
LOG_DIR="${REPO}/benchmarks/foundation/condor/logs"
mkdir -p "${LOG_DIR}"
TS="$(date +%Y%m%d_%H%M%S)"
LOG="${LOG_DIR}/krakas_p8_moment_lora_${TS}.log"

echo "=========================================================================="
echo "[p8-lora] moment  ->  ${LOG}"
echo "[p8-lora] $(date)  start"
echo "=========================================================================="

"${PY}" -u benchmarks/foundation/run_project8_finetune.py \
    --data_root data/Project8 \
    --models moment \
    --model_size base \
    --out_dir plots/Project8/lora_cav \
    --noise_type cav \
    --cutoff 24576 \
    --batch_size 8 \
    --num_workers 8 \
    --prefetch_factor 4 \
    --pin_memory \
    --persistent_workers \
    --adapter lora \
    --epochs 10 \
    --head_lr 1e-3 \
    --backbone_lr 1e-4 \
    --lora_r 8 \
    --lora_alpha 16 \
    --patience 3 \
    >"${LOG}" 2>&1
rc=$?

echo "[p8-lora] $(date)  end  rc=${rc}"
exit "${rc}"
