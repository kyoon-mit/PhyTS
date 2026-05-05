#!/usr/bin/env bash
# Sequential full fine-tuning of chronos and timemoe on Kepler Q9v3.
# Same recipe as MOMENT/granite full-FT (100 ep, patience 20,
# head_lr=1e-3, backbone_lr=1e-5) but batch_size=4: bs=8 also OOMed on
# ramee's 11.6 GiB GPU (PyTorch held ~10.5 GiB at the wall). bs=4 should
# halve activation memory and fit comfortably.
set -u

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "${REPO}"

PY="${FM_PYTHON:-/esat/smcdata/users/kkontras/Image_Dataset/no_backup/envs/tsenv_fm/bin/python}"
LOG_DIR="${REPO}/benchmarks/foundation/condor/logs"
mkdir -p "${LOG_DIR}"
TS="$(date +%Y%m%d_%H%M%S)"

# Reduce fragmentation so peak allocations don't trip OOM at the edge.
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

COMMON="--zip data/keplerq9v3.zip --batch_size 4 --seed 42 \
  --adapter full --epochs 100 --patience 20 \
  --head_lr 1e-3 --backbone_lr 1e-5"

run_variant() {
  local name="$1"; shift
  local log="${LOG_DIR}/ramee_full_ft_${name}_${TS}.log"
  echo "=========================================================================="
  echo "[sweep] starting ${name}  -> ${log}"
  echo "=========================================================================="
  echo "[sweep] $(date)  start  ${name}"
  "${PY}" -u benchmarks/foundation/run_kepler_finetune.py ${COMMON} "$@" >"${log}" 2>&1
  local rc=$?
  echo "[sweep] $(date)  end    ${name}  rc=${rc}"
}

run_variant "chronos" \
  --models chronos --model_size base \
  --out_dir plots/kepler_q9v3/full_ft_chronos

run_variant "timemoe" \
  --models timemoe --model_size small \
  --out_dir plots/kepler_q9v3/full_ft_timemoe

echo "[sweep] all variants done at $(date)"
