#!/bin/bash
# TIDMAD ablation sweep — submits all ConvAE and LinOSS variants (10 epochs each).
# All variants run in parallel. A comparison job runs after all finish.
#
# Workflow:
#   1. bash cluster/run_tidmad_ablation.sh      # submit ablations
#   2. Wait for completion, then inspect results:
#      python benchmarks/TIDMAD/compare_ablations.py
#   3. Update the full-training configs with the best hyperparams:
#      configs/TIDMAD/train_tidmad_conv_denoising.yaml
#      configs/TIDMAD/train_tidmad_linoss_denoising.yaml
#   4. bash cluster/run_tidmad_pipeline.sh      # full training + inference
#
# Monitor:
#   squeue -u $USER
#   tail -f /home/ilay.kamai/athena/logs/abl_*.out

set -e

REPO=/rg/perets_prj/ilay.kamai/PhyTS/TimeSeriesPhysics

# ── One job per model (variants run sequentially inside) ───────────────────
JOB_CONV=$(sbatch --parsable ${REPO}/cluster/tidmad_conv_ablation.sbatch)
echo "ConvAE ablations (s,m,l,w): job ${JOB_CONV}"

JOB_LINOSS=$(sbatch --parsable ${REPO}/cluster/tidmad_linoss_ablation.sbatch)
echo "LinOSS ablations (s,m,l,d): job ${JOB_LINOSS}"

# ── Comparison report (waits for both) ────────────────────────────────────
COMPARE_JID=$(sbatch --parsable \
  --dependency=afterany:${JOB_CONV}:${JOB_LINOSS} \
  ${REPO}/cluster/tidmad_compare.sbatch)
echo "Compare:                     job ${COMPARE_JID}"

echo ""
echo "Monitor:  squeue -u \$USER"
echo "Results:  benchmarks/TIDMAD/ablation_results.txt"
