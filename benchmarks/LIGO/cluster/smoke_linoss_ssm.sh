#!/bin/bash
# Smoke test: LinOSS with ssm param groups + skip_nonfinite; batch 0 contains a NaN sample.
cd /home/barmstrong/TimeSeriesPhysics
/lustre/home/barmstrong/phyts/cluster/train_ligo_cfg.sh configs/LIGO/v2/_smoke_linoss_ssm.yaml && echo "PASS run1" || echo "FAIL run1"
uv run --no-sync python -c "
import torch, io, equinox as eqx, jax, jax.numpy as jnp
c = torch.load('/fast/barmstrong/LIGO_2048Hz_v2_smoke/ckpt_ssm/last.ckpt', map_location='cpu', weights_only=False)
print('epoch', c['epoch'], 'step', c['global_step'], 'has opt state', 'jax_opt_state' in c)
" && echo "PASS ckpt-load"
sed -i 's/max_epochs: 1$/max_epochs: 2/' configs/LIGO/v2/_smoke_linoss_ssm.yaml
/lustre/home/barmstrong/phyts/cluster/train_ligo_cfg.sh configs/LIGO/v2/_smoke_linoss_ssm.yaml && echo "PASS run2" || echo "FAIL run2"
