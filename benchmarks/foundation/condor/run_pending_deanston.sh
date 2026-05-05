#!/usr/bin/env bash
# Direct-run script for deanston (24 GiB).
# Cells: Project 8 Full-FT for granite_ttm, moirai, timemoe — sequential.
set -u

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "${REPO}"

export FM_PYTHON="${FM_PYTHON:-/esat/smcdata/users/kkontras/Image_Dataset/no_backup/envs/tsenv_fm/bin/python}"
export HF_HOME="${HF_HOME:-/esat/smcdata/users/kkontras/Image_Dataset/no_backup/huggingface_cache}"
export LAGLLAMA_CKPT="${LAGLLAMA_CKPT:-/esat/smcdata/users/kkontras/Image_Dataset/no_backup/checkpoints/lag-llama/lag-llama.ckpt}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

LOG_DIR="${REPO}/benchmarks/foundation/condor/logs"
mkdir -p "${LOG_DIR}"
TS="$(date +%Y%m%d_%H%M%S)"
HOST="$(hostname -s)"

run_cell () {
  local name="$1"; shift
  local log="${LOG_DIR}/${HOST}_${TS}_${name}.log"
  echo "==========================================================================" | tee -a "${log}"
  echo "[${HOST}/${name}] start  $(date)" | tee -a "${log}"
  echo "[${HOST}/${name}] log    ${log}"
  "${FM_PYTHON}" -u "$@" >>"${log}" 2>&1
  local rc=$?
  echo "[${HOST}/${name}] end    $(date)  rc=${rc}" | tee -a "${log}"
}

P8_COMMON=(
  --data_root data/Project8
  --model_size base
  --out_dir plots/Project8/full_cav
  --noise_type cav --cutoff 24576
  --batch_size 4 --num_workers 8 --prefetch_factor 4
  --pin_memory --persistent_workers
  --adapter full
  --epochs 10 --head_lr 1e-3 --backbone_lr 1e-5
  --lora_r 8 --lora_alpha 16 --patience 3
)

for m in granite_ttm moirai timemoe; do
  run_cell "p8_full_${m}" benchmarks/foundation/run_project8_finetune.py \
    --models "${m}" "${P8_COMMON[@]}"
done

echo "[${HOST}] all cells done."
