#!/bin/bash
# Run one diag_linoss.py variant. Usage: run_diag.sh <tag> [diag args...]
cd /home/barmstrong/TimeSeriesPhysics
uv run --no-sync python /lustre/home/barmstrong/phyts/cluster/diag/diag_linoss.py --tag "$@"
