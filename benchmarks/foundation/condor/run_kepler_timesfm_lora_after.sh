#!/usr/bin/env bash
# TimesFM LoRA sweep on Kepler Q9v3, queued behind any in-flight finetune.
# bs=4 to stay within ramee's 12 GiB GPU (TimesFM-200M ~= chronos-base
# in size; chronos full FT used 8.0 GiB at bs=4, LoRA saves ~1 GiB).
# r=8 then r=16, 100 ep, patience 20, head_lr=1e-3, backbone_lr=1e-4.
set -u

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "${REPO}"

PY="${FM_PYTHON:-/esat/smcdata/users/kkontras/Image_Dataset/no_backup/envs/tsenv_fm/bin/python}"
LOG_DIR="${REPO}/benchmarks/foundation/condor/logs"
mkdir -p "${LOG_DIR}"

echo "[queue] $(date)  waiting for any running run_kepler_finetune.py..."
while pgrep -f "run_kepler_finetune.py" > /dev/null; do
  sleep 60
done
echo "[queue] $(date)  GPU free; starting timesfm LoRA sweep."

TS="$(date +%Y%m%d_%H%M%S)"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

COMMON="--zip data/keplerq9v3.zip --models timesfm --model_size base \
  --batch_size 4 --seed 42 \
  --adapter lora --epochs 100 --patience 20 \
  --head_lr 1e-3 --backbone_lr 1e-4"

run_variant() {
  local name="$1"; shift
  local log="${LOG_DIR}/ramee_${name}_${TS}.log"
  echo "=========================================================================="
  echo "[sweep] starting ${name}  -> ${log}"
  echo "=========================================================================="
  echo "[sweep] $(date)  start  ${name}"
  "${PY}" -u benchmarks/foundation/run_kepler_finetune.py ${COMMON} "$@" >"${log}" 2>&1
  local rc=$?
  echo "[sweep] $(date)  end    ${name}  rc=${rc}"
}

for r in 8 16; do
  alpha=$((2 * r))
  run_variant "lora_timesfm_r${r}_bs4" \
    --lora_r ${r} --lora_alpha ${alpha} \
    --out_dir "plots/kepler_q9v3/lora_timesfm_r${r}_bs4"
done

echo "[sweep] all variants done at $(date)"
