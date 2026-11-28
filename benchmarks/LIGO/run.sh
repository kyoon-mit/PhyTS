#!/bin/bash
# LIGO benchmark (paper Table 2): train S4D / LinOSS / CNN, evaluate the six zero-shot
# foundation models, and score everything on the test split.
#
# Prerequisites:
#   python data/download.py --domain ligo          # ~4 TB
#   python data/LIGO/extract_window.py --src data/LIGO --dest data/LIGO
#   make env-jax                                   # PyTorch + JAX env (S4D, CNN, LinOSS)
#   make env-fm                                    # foundation-model env
#
# Every model trains for 60 epochs (seed 42); test uses the best-validation checkpoint.
# Approximate cost on one A100: S4D / LinOSS ~35 h, CNN ~6 h.
#
# Usage: bash benchmarks/LIGO/run.sh

set -e

for model in s4d conv1d linoss; do
    cfg=configs/LIGO/train_ligo_${model}_gaussnll_regression.yaml
    python main.py fit  --config $cfg
    python main.py test --config $cfg --ckpt_path checkpoints/ligo_${model}_gaussnll_regression/best.ckpt
done

# Zero-shot foundation models: frozen embeddings + ridge probe fitted on val, scored on test.
# Chronos / Time-MoE take ~40 GPU-hours each; split them with --worker i --num_workers k.
for model in moment chronos timesfm timemoe moirai granite_ttm; do
    benchmarks/foundation/.venv/bin/python benchmarks/LIGO/fm_zeroshot_ligo.py --model $model
done

python benchmarks/LIGO/score.py
