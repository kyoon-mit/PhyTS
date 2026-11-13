#!/bin/bash
# Ridge(alpha=1) probe (--paper_protocol) on the existing mean-pooled LIGO FM embeddings of the
# models whose paper pooling is mean: MOMENT, Chronos, Granite-TTM, both inputs.
set -uo pipefail
export OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 MKL_NUM_THREADS=8 HF_HOME=/fast/barmstrong/hf/
cd /home/barmstrong/TimeSeriesPhysics
RES=()
for inp in 256hz 2048hz; do for m in moment chronos granite_ttm; do
  echo "================ $m $inp $(date)"
  benchmarks/foundation/.venv/bin/python benchmarks/LIGO/fm_zeroshot_ligo.py --model $m --input $inp \
    --paper_protocol --stage head
  RC=$?
  M=results/LIGO/v2_fm_${m}_${inp}_probe/metrics.json
  if [[ $RC -eq 0 && -f $M ]]; then
    RES+=("$(benchmarks/foundation/.venv/bin/python -c "import json,sys; r=json.load(open(sys.argv[1]))['test']; print(f\"$m $inp {r['rmse']:.4f}/{r['r2']:.3f}\")" $M)")
  else
    RES+=("$m $inp FAILED rc=$RC")
  fi
done; done
printf '%s\n' "${RES[@]}"
/home/barmstrong/cluster/htcondor/notify.sh "📊 LIGO FM ridge probe (reused emb): $(printf '%s; ' "${RES[@]}")"
