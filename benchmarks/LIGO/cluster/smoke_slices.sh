#!/bin/bash
# Smoke test for time-sliced training: resume an old-format last.ckpt with the new callbacks,
# run two 20 s slices (each should exit 85 and advance last.ckpt by one epoch).
# Usage (via condor): smoke_slices.sh <run>
R=${1:?run}
export WANDB_MODE=offline
CFG=configs/LIGO/v2/smoke/train_ligo_smoke_${R}.yaml
D=checkpoints/smoke_slices_${R}
ep() { uv run --no-sync python -c "import torch,sys; c=torch.load(sys.argv[1],map_location='cpu',weights_only=False); print(c['epoch'], c['global_step'], [k[:40] for k in c['callbacks']])" $D/last.ckpt; }
echo "start: $(ep)"
for i in 1 2; do
  /lustre/home/barmstrong/phyts/cluster/train_ligo_cfg.sh $CFG; echo "slice $i rc=$? last: $(ep)"; ls $D
done
