#!/usr/bin/env bash
# Condor executable for PhyTS-bench TESS regression fine-tuning runs.
# Forwards all arguments to run_tess_regression_finetune.py.
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "${REPO}"

PYTHON_BIN="${FM_PYTHON:-/esat/smcdata/users/kkontras/Image_Dataset/no_backup/envs/tsenv_fm/bin/python}"

echo "[reg-tess] host=$(hostname)  pid=$$"
echo "[reg-tess] python=${PYTHON_BIN}"
echo "[reg-tess] args: $*"
echo "[reg-tess] $(date)  start"

"${PYTHON_BIN}" -u benchmarks/foundation/run_tess_regression_finetune.py "$@"

echo "[reg-tess] $(date)  done"
