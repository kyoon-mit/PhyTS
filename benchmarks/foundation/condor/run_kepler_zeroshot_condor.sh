#!/usr/bin/env bash
# Condor executable for Kepler Q9v3 zero-shot linear probe.
# Forwards all arguments to run_kepler.py.
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "${REPO}"

PYTHON_BIN="${FM_PYTHON:-/esat/smcdata/users/kkontras/Image_Dataset/no_backup/envs/tsenv_fm/bin/python}"

echo "[zs] host=$(hostname)  pid=$$"
echo "[zs] python=${PYTHON_BIN}"
echo "[zs] args: $*"
echo "[zs] $(date)  start"

# run_kepler.py picks up LAGLLAMA_CKPT from the environment (set by the
# Condor job file). No CLI arg needed.
"${PYTHON_BIN}" -u benchmarks/foundation/run_kepler.py "$@"

echo "[zs] $(date)  done"
