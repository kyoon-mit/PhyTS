#!/bin/bash
# Ridge linear probe on cached LIGO v2 FM embeddings (benchmarks/LIGO/fm_linear_probe.py), all 12 model/input pairs.
# Usage (via condor, initialdir ~/TimeSeriesPhysics): fm_linear_probe.sh [extra args]
set -uo pipefail
export OMP_NUM_THREADS=8
cd /home/barmstrong/TimeSeriesPhysics
benchmarks/foundation/.venv/bin/python benchmarks/LIGO/fm_linear_probe.py "$@"
RC=$?
/home/barmstrong/cluster/htcondor/notify.sh "📐 ${CONDOR_JOB_ID:-local} FM ridge probe finished (rc=$RC)"
exit $RC
