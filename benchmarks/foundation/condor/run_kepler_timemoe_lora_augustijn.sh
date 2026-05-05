#!/usr/bin/env bash
# TimeMoE LoRA sweep on Kepler Q9v3 for augustijn (RTX 3070, 8 GiB).
# Uses bs=4: empirically bs=2 only used 3.95 GiB so we have plenty of
# headroom; bs=4 halves the per-epoch wall time (RTX 3070 is the
# bottleneck, not VRAM here).
# r=8 then r=16, 100 ep, patience 20, head_lr=1e-3, backbone_lr=1e-4.
set -u

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "${REPO}"

PY="${FM_PYTHON:-/esat/smcdata/users/kkontras/Image_Dataset/no_backup/envs/tsenv_fm/bin/python}"
LOG_DIR="${REPO}/benchmarks/foundation/condor/logs"
mkdir -p "${LOG_DIR}"
TS="$(date +%Y%m%d_%H%M%S)"

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

COMMON="--zip data/keplerq9v3.zip --models timemoe --model_size small \
  --batch_size 4 --seed 42 \
  --adapter lora --epochs 100 --patience 20 \
  --head_lr 1e-3 --backbone_lr 1e-4"

run_variant() {
  local name="$1"; shift
  local log="${LOG_DIR}/augustijn_${name}_${TS}.log"
  echo "=========================================================================="
  echo "[sweep] starting ${name}  -> ${log}"
  echo "=========================================================================="
  echo "[sweep] $(date)  start  ${name}"
  "${PY}" -u benchmarks/foundation/run_kepler_finetune.py ${COMMON} "$@" >"${log}" 2>&1
  local rc=$?
  echo "[sweep] $(date)  end    ${name}  rc=${rc}"
}

for r in 8; do
  alpha=$((2 * r))
  run_variant "lora_timemoe_r${r}" \
    --lora_r ${r} --lora_alpha ${alpha} \
    --out_dir "plots/kepler_q9v3/lora_timemoe_r${r}_augustijn"
done

echo "[sweep] all variants done at $(date)"
