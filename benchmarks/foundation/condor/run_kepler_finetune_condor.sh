#!/usr/bin/env bash
# Condor executable for Kepler Q9v3 fine-tuning runs.
# Forwards all arguments to run_kepler_finetune.py.
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "${REPO}"

PYTHON_BIN="${FM_PYTHON:-/esat/smcdata/users/kkontras/Image_Dataset/no_backup/envs/tsenv_fm/bin/python}"

echo "[ft] host=$(hostname)  pid=$$"
echo "[ft] python=${PYTHON_BIN}"
echo "[ft] args: $*"
echo "[ft] $(date)  start"

"${PYTHON_BIN}" -u benchmarks/foundation/run_kepler_finetune.py "$@"

echo "[ft] $(date)  done"
