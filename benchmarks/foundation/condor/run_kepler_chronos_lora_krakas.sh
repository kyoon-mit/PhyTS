#!/usr/bin/env bash
# Chronos LoRA sweep on Kepler Q9v3 for krakas (24 GiB RTX 4500 Ada).
# Uses the original overnight-sweep recipe (bs=16) since krakas has plenty
# of VRAM headroom -- the ramee bs=4 cut was only needed on the 12 GiB card.
# r=8 then r=16, 100 ep, patience 20, head_lr=1e-3, backbone_lr=1e-4.
set -u

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "${REPO}"

PY="${FM_PYTHON:-/esat/smcdata/users/kkontras/Image_Dataset/no_backup/envs/tsenv_fm/bin/python}"
LOG_DIR="${REPO}/benchmarks/foundation/condor/logs"
mkdir -p "${LOG_DIR}"
TS="$(date +%Y%m%d_%H%M%S)"

COMMON="--zip data/keplerq9v3.zip --models chronos --model_size base \
  --batch_size 16 --seed 42 \
  --adapter lora --epochs 100 --patience 20 \
  --head_lr 1e-3 --backbone_lr 1e-4"

run_variant() {
  local name="$1"; shift
  local log="${LOG_DIR}/krakas_${name}_${TS}.log"
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
  run_variant "lora_chronos_r${r}" \
    --lora_r ${r} --lora_alpha ${alpha} \
    --out_dir "plots/kepler_q9v3/lora_chronos_r${r}_krakas"
done

echo "[sweep] all variants done at $(date)"
