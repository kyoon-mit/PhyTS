#!/bin/bash
# Throughput / memory smoke test for Time-MoE on longer chunks (full window): 256hz c1024, 2048hz c4096, 2048hz c8192.
set -uo pipefail
export OMP_NUM_THREADS=8 TOKENIZERS_PARALLELISM=false HF_HOME=/fast/barmstrong/hf/
cd /home/barmstrong/TimeSeriesPhysics
OUT=/fast/barmstrong/outputs/ligo_fm_smoke_fullwin; rm -rf "$OUT"
run() { echo "================ $* $(date)"; benchmarks/foundation/.venv/bin/python benchmarks/LIGO/fm_zeroshot_ligo.py --model timemoe \
  --max_samples 512 --shard_size 256 --stage embed --out_root $OUT "$@"; echo "rc=$?"; nvidia-smi --query-gpu=memory.used --format=csv; }
run --input 256hz --chunk_len 1024 --batch_size 128
run --input 2048hz --chunk_len 4096 --batch_size 16
run --input 2048hz --chunk_len 8192 --batch_size 8
