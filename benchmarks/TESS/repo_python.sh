#!/usr/bin/env bash
# Resolve the repo virtualenv interpreter without relying on PATH (wandb agent
# subprocesses sometimes pick up conda/base ``python`` and miss jax/equinox).
#
# Sweep YAMLs should invoke this script instead of bare ``python``. Override:
#   export TIMESERIES_PHYSICS_PYTHON=/path/to/python
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PY="${TIMESERIES_PHYSICS_PYTHON:-$ROOT/.venv/bin/python}"
if [[ ! -x "$PY" ]]; then
  echo "repo_python.sh: interpreter not found or not executable: $PY" >&2
  echo "Set TIMESERIES_PHYSICS_PYTHON or create $ROOT/.venv (uv sync ...)." >&2
  exit 127
fi
exec "$PY" "$@"
