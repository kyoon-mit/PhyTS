#!/usr/bin/env bash
# Condor executable for PhyTS-bench TESS classification fine-tuning runs.
# Forwards all arguments to run_tess_finetune.py.
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "${REPO}"

PYTHON_BIN="${FM_PYTHON:-/esat/smcdata/users/kkontras/Image_Dataset/no_backup/envs/tsenv_fm/bin/python}"

echo "[ft-tess] host=$(hostname)  pid=$$"
echo "[ft-tess] python=${PYTHON_BIN}"
echo "[ft-tess] args: $*"
echo "[ft-tess] $(date)  start"

"${PYTHON_BIN}" -u benchmarks/foundation/run_tess_finetune.py "$@"

echo "[ft-tess] $(date)  done"
