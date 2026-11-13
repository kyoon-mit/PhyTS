#!/bin/bash
# Zero-shot FM eval on LIGO 2048 Hz v2 (benchmarks/LIGO/fm_zeroshot_ligo.py) for one model/input.
# Resumable: finished embedding shards are skipped; exits 85 after the 4 h time slice so condor
# requeues it (on_exit_remove), then trains the MLP head and scores the test split.
# Usage (via condor, initialdir ~/TimeSeriesPhysics): fm_zeroshot_ligo.sh <model> <256hz|2048hz> [extra args]
set -uo pipefail
MODEL=${1:?model}; INPUT=${2:?input}; shift 2
NOTIFY=/home/barmstrong/cluster/htcondor/notify.sh
JOB=${CONDOR_JOB_ID:-local}
export OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 MKL_NUM_THREADS=8 TOKENIZERS_PARALLELISM=false
export HF_HOME=/fast/barmstrong/hf/
cd /home/barmstrong/TimeSeriesPhysics
START=$(mktemp); trap 'rm -f "$START"' EXIT
benchmarks/foundation/.venv/bin/python benchmarks/LIGO/fm_zeroshot_ligo.py --model "$MODEL" --input "$INPUT" "$@"
RC=$?
[[ $RC -eq 85 ]] && { echo "time slice done, requeueing $(date)"; exit 85; }
[[ $RC -ne 0 ]] && exit $RC
M=$(ls -t results/LIGO/v2_fm_${MODEL}_${INPUT}*/metrics.json 2>/dev/null | head -1)  # also _c<len> / _chunks variants
if [[ -n $M && $M -nt $START ]]; then   # only the worker that trained the head reports
  RES=$(benchmarks/foundation/.venv/bin/python -c "import json,sys; r=json.load(open(sys.argv[1]))['test']; print(f\"test RMSE {r['rmse']:.4f}, R2 {r['r2']:.3f}, n={r['n']}\")" "$M")
  $NOTIFY "📊 ${JOB} FM zero-shot $(basename $(dirname $M)): ${RES}"
fi
