#!/bin/bash
# Diagnostic: --paper_protocol ridge probe fitted on the validation split (same O3a / low-SNR
# distribution as test) instead of train, scored on test. Runs every paper-protocol embedding set
# that is complete (95 shards); prints the ones it skipped. Usage: fm_paper_probe_valfit.sh [model_input ...]
set -uo pipefail
export OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 MKL_NUM_THREADS=8 HF_HOME=/fast/barmstrong/hf/
cd /home/barmstrong/TimeSeriesPhysics
PY=benchmarks/foundation/.venv/bin/python
EMB=/fast/barmstrong/outputs/ligo_fm
PAIRS=${*:-"moment_256hz moment_2048hz chronos_256hz chronos_2048hz granite_ttm_256hz granite_ttm_2048hz timesfm_256hz timesfm_2048hz moirai_256hz moirai_2048hz timemoe_2048hz"}
RES=()
for pair in $PAIRS; do
  m=${pair%_*}; inp=${pair##*_}
  case $m in timesfm|timemoe) dir=${m}_${inp}_last;; moirai) dir=${m}_${inp}_h128;; *) dir=${m}_${inp};; esac
  n=$(ls $EMB/$dir/emb 2>/dev/null | grep -c '^[a-z]*_[0-9]\{4\}\.npy$')
  if [[ $n -ne 95 ]]; then RES+=("$pair SKIPPED ($n/95 shards)"); continue; fi
  echo "================ $pair $(date)"
  $PY benchmarks/LIGO/fm_zeroshot_ligo.py --model $m --input $inp --paper_protocol --probe_fit val --stage head
  RC=$?
  M=results/LIGO/v2_fm_${dir}_probe_valfit/metrics.json
  if [[ $RC -eq 0 && -f $M ]]; then
    RES+=("$($PY -c "import json,sys; r=json.load(open(sys.argv[1]))['test']; print(f\"$pair {r['rmse']:.4f}/{r['r2']:.3f}\")" $M)")
  else
    RES+=("$pair FAILED rc=$RC")
  fi
done
printf '%s\n' "${RES[@]}"
/home/barmstrong/cluster/htcondor/notify.sh "📊 LIGO FM ridge probe fitted on VAL: $(printf '%s; ' "${RES[@]}")"
