#!/bin/bash
# Smoke test for --keep_chunks (per-chunk embeddings + order-aware head) on 1024 samples/split.
set -uo pipefail
export OMP_NUM_THREADS=8 TOKENIZERS_PARALLELISM=false HF_HOME=/fast/barmstrong/hf/
cd /home/barmstrong/TimeSeriesPhysics
OUT=/fast/barmstrong/outputs/ligo_fm_smoke_chunks; rm -rf "$OUT"
for a in "granite_ttm 256hz" "granite_ttm 2048hz" "timesfm 2048hz"; do set -- $a
  echo "================ $1 $2 $(date)"
  benchmarks/foundation/.venv/bin/python benchmarks/LIGO/fm_zeroshot_ligo.py --model $1 --input $2 --keep_chunks \
    --max_samples 1024 --shard_size 512 --head_epochs 3 --out_root $OUT; echo "rc=$?"
done
