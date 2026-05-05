#!/usr/bin/env bash
# MOIRAI-large (1.1-R-large, 311M) full SFT on Kepler Q9v3 for deanston (24 GiB).
# bs=2 to leave headroom for the Adam optimizer state of all 311M params.
set -u

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "${REPO}"

PY="${FM_PYTHON:-/esat/smcdata/users/kkontras/Image_Dataset/no_backup/envs/tsenv_fm/bin/python}"
LOG_DIR="${REPO}/benchmarks/foundation/condor/logs"
mkdir -p "${LOG_DIR}"
TS="$(date +%Y%m%d_%H%M%S)"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

LOG="${LOG_DIR}/deanston_full_ft_moirai_large_${TS}.log"
echo "=========================================================================="
echo "[sweep] starting full_ft_moirai_large  -> ${LOG}"
echo "[sweep] $(date)  start"
"${PY}" -u benchmarks/foundation/run_kepler_finetune.py \
  --zip data/keplerq9v3.zip --models moirai --model_size large \
  --batch_size 2 --seed 42 \
  --adapter full --epochs 100 --patience 20 \
  --head_lr 1e-3 --backbone_lr 1e-5 \
  --out_dir plots/kepler_q9v3/full_ft_moirai_large_deanston \
  >"${LOG}" 2>&1
rc=$?
echo "[sweep] $(date)  end  rc=${rc}"
