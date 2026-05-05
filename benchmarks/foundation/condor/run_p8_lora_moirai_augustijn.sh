#!/usr/bin/env bash
# Move from brigand: P8 LoRA moirai. Brigand stalled on this cell for 11+h
# with 0 epochs printed; re-launching on augustijn (8 GiB RTX 3070, free).
# Reduced num_workers to 2 in case the brigand stall was data-loader contention.
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
LOG="${LOG_DIR}/${HOST}_${TS}_p8_lora_moirai.log"
echo "[${HOST}/p8_lora_moirai] start $(date)  log=${LOG}"

"${FM_PYTHON}" -u benchmarks/foundation/run_project8_finetune.py \
  --data_root data/Project8 --models moirai --model_size base \
  --out_dir plots/Project8/lora_cav \
  --noise_type cav --cutoff 24576 \
  --batch_size 4 --num_workers 2 --prefetch_factor 2 \
  --pin_memory --persistent_workers \
  --adapter lora --epochs 10 --head_lr 1e-3 --backbone_lr 1e-4 \
  --lora_r 8 --lora_alpha 16 --patience 3 \
  > "${LOG}" 2>&1
echo "[${HOST}/p8_lora_moirai] end $(date) rc=$?"
