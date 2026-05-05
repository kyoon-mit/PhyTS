#!/usr/bin/env bash
# Retry P8 Full-FT chronos at bs=2 on krakas, after the main run_pending_krakas.sh
# script finishes. Polls every 90s for the main script to exit.
set -u

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "${REPO}"

export FM_PYTHON="${FM_PYTHON:-/esat/smcdata/users/kkontras/Image_Dataset/no_backup/envs/tsenv_fm/bin/python}"
export HF_HOME="${HF_HOME:-/esat/smcdata/users/kkontras/Image_Dataset/no_backup/huggingface_cache}"
export LAGLLAMA_CKPT="${LAGLLAMA_CKPT:-/esat/smcdata/users/kkontras/Image_Dataset/no_backup/checkpoints/lag-llama/lag-llama.ckpt}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

LOG_DIR="${REPO}/benchmarks/foundation/condor/logs"
mkdir -p "${LOG_DIR}"
HOST="$(hostname -s)"

# Wait for the main script to finish.
echo "[${HOST}/retry] waiting for run_pending_krakas.sh to finish..."
while pgrep -u "$USER" -f 'run_pending_krakas\.sh' > /dev/null; do
  sleep 90
done
echo "[${HOST}/retry] main script done; chronos full-FT bs=2 launching $(date)"

TS="$(date +%Y%m%d_%H%M%S)"
LOG="${LOG_DIR}/${HOST}_${TS}_p8_full_chronos_bs2.log"

"${FM_PYTHON}" -u benchmarks/foundation/run_project8_finetune.py \
  --data_root data/Project8 --models chronos --model_size base \
  --out_dir plots/Project8/full_cav --noise_type cav --cutoff 24576 \
  --batch_size 2 --num_workers 8 --prefetch_factor 4 \
  --pin_memory --persistent_workers \
  --adapter full --epochs 10 --head_lr 1e-3 --backbone_lr 1e-5 \
  --lora_r 8 --lora_alpha 16 --patience 3 \
  > "${LOG}" 2>&1
echo "[${HOST}/retry] chronos full-FT bs=2 finished rc=$? $(date)  log=${LOG}"
