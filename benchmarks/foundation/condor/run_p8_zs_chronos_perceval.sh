#!/usr/bin/env bash
# perceval (RTX 4070 Ti SUPER, 16 GiB) — P8 zero-shot chronos.
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
LOG="${LOG_DIR}/${HOST}_${TS}_p8_zs_chronos.log"
echo "[${HOST}/p8_zs_chronos] start $(date)  log=${LOG}"

"${FM_PYTHON}" -u benchmarks/foundation/run_project8.py \
  --data_root data/Project8 --models chronos --model_size base \
  --out_dir plots/Project8/linear_probe_cav \
  --noise_type cav --cutoff 24576 \
  --batch_size 16 --num_workers 8 --prefetch_factor 4 \
  --pin_memory --persistent_workers > "${LOG}" 2>&1
echo "[${HOST}/p8_zs_chronos] end $(date) rc=$?"
