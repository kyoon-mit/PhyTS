#!/bin/bash
export OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 HF_HOME=/fast/barmstrong/hf/
cd /home/barmstrong/TimeSeriesPhysics
benchmarks/foundation/.venv/bin/python /lustre/home/barmstrong/phyts/cluster/diag/bench_fm_bf16.py chronos timemoe moment
