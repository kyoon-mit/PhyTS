#!/bin/bash
# Smoke test: every FM x input on 1024 samples/split (2 shards each), prints throughput + metrics.
set -uo pipefail
export OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 MKL_NUM_THREADS=8 TOKENIZERS_PARALLELISM=false HF_HOME=/fast/barmstrong/hf/
cd /home/barmstrong/TimeSeriesPhysics
OUT=/fast/barmstrong/outputs/ligo_fm_smoke; rm -rf "$OUT"
SUMMARY=()
for inp in 256hz 2048hz; do for m in moment chronos timesfm timemoe moirai granite_ttm; do
  echo "================ $m $inp $(date)"
  benchmarks/foundation/.venv/bin/python benchmarks/LIGO/fm_zeroshot_ligo.py --model $m --input $inp \
    --max_samples 1024 --shard_size 512 --head_epochs 5 --out_root $OUT
  RC=$?; SUMMARY+=("$m $inp rc=$RC")
done; done
printf '%s\n' "${SUMMARY[@]}"
/home/barmstrong/cluster/htcondor/notify.sh "🧪 FM zero-shot smoke done: $(printf '%s; ' "${SUMMARY[@]}")"
