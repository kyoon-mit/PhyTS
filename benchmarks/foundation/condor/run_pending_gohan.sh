#!/usr/bin/env bash
# Direct-run script for gohan.
# Cells:
#   1) P8 zero-shot — chronos
#   2) TESS-cls Full-FT — granite_ttm, timemoe, timesfm
# Sequential.
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

# ─── P8 zero-shot — chronos ──────────────────────────────────────────────
run_cell "p8_zs_chronos" benchmarks/foundation/run_project8.py \
  --data_root data/Project8 --models chronos --model_size base \
  --out_dir plots/Project8/linear_probe_cav \
  --noise_type cav --cutoff 24576 \
  --batch_size 8 --num_workers 8 --prefetch_factor 4 \
  --pin_memory --persistent_workers

# ─── TESS-cls Full-FT — granite_ttm, timemoe, timesfm ────────────────────
TESS_FULL_COMMON=(
  --model_size base
  --batch_size 2 --seed 42
  --adapter full --epochs 30 --patience 5 --backbone_lr 1e-5
  --out_dir plots/tess/full_condor
)
for m in granite_ttm timemoe timesfm; do
  run_cell "tess_full_${m}" benchmarks/foundation/run_tess_finetune.py \
    --models "${m}" "${TESS_FULL_COMMON[@]}"
done

echo "[${HOST}] all cells done."
