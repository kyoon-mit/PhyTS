#!/usr/bin/env bash
# TimesFM-large (2.0-500m) sweep on Kepler Q9v3 for krakas (24 GiB).
# Sequence: LoRA r=8 → LoRA r=16 → full FT.
# bs=8 for LoRA (light memory), bs=4 for full FT (heavier optim state).
# 100 ep / patience 20 / head_lr=1e-3 / backbone_lr=1e-4 (LoRA) or 1e-5 (full).
set -u

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "${REPO}"

PY="${FM_PYTHON:-/esat/smcdata/users/kkontras/Image_Dataset/no_backup/envs/tsenv_fm/bin/python}"
LOG_DIR="${REPO}/benchmarks/foundation/condor/logs"
mkdir -p "${LOG_DIR}"
TS="$(date +%Y%m%d_%H%M%S)"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

run_variant() {
  local name="$1"; shift
  local log="${LOG_DIR}/krakas_${name}_${TS}.log"
  echo "=========================================================================="
  echo "[sweep] starting ${name}  -> ${log}"
  echo "[sweep] $(date)  start  ${name}"
  "${PY}" -u benchmarks/foundation/run_kepler_finetune.py "$@" >"${log}" 2>&1
  local rc=$?
  echo "[sweep] $(date)  end    ${name}  rc=${rc}"
}

COMMON_LORA="--zip data/keplerq9v3.zip --models timesfm --model_size large \
  --batch_size 8 --seed 42 \
  --adapter lora --epochs 100 --patience 20 \
  --head_lr 1e-3 --backbone_lr 1e-4"

run_variant "lora_timesfm_large_r8" ${COMMON_LORA} \
  --lora_r 8  --lora_alpha 16 \
  --out_dir plots/kepler_q9v3/lora_timesfm_large_r8_krakas

run_variant "lora_timesfm_large_r16" ${COMMON_LORA} \
  --lora_r 16 --lora_alpha 32 \
  --out_dir plots/kepler_q9v3/lora_timesfm_large_r16_krakas

run_variant "full_ft_timesfm_large" \
  --zip data/keplerq9v3.zip --models timesfm --model_size large \
  --batch_size 4 --seed 42 \
  --adapter full --epochs 100 --patience 20 \
  --head_lr 1e-3 --backbone_lr 1e-5 \
  --out_dir plots/kepler_q9v3/full_ft_timesfm_large_krakas

echo "[sweep] all variants done at $(date)"
