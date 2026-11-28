#!/bin/bash
# Project 8 benchmark (paper Table 2): train S4D / LinOSS / CNN to regress the electron energy
# with a Gaussian NLL, evaluate the six zero-shot foundation models, and score everything on
# the test split.
#
# Prerequisites:
#   python data/download.py --domain project8      # ~44 GB
#   make env-jax                                   # PyTorch + JAX env (S4D, CNN, LinOSS)
#   make env-fm                                    # foundation-model env
#
# Each model trains for up to 100 epochs (early stopping on val loss, patience 20); test uses
# the best-validation checkpoint. At batch 64 S4D needs ~64 GB and LinOSS ~83 GB of GPU memory.
#
# Usage: bash benchmarks/Project8/run.sh

set -e

for model in s4d linoss conv; do
    cfg=configs/Project8/train_project8_${model}_gaussnll_regression.yaml
    python main.py fit  --config $cfg
    python main.py test --config $cfg --ckpt_path checkpoints/project8_${model}_gaussnll_regression/best.ckpt
done

# Zero-shot foundation models: frozen embeddings + ridge probe fitted on train, scored on test.
for model in moment chronos timesfm timemoe moirai granite_ttm; do
    benchmarks/foundation/.venv/bin/python benchmarks/Project8/fm_zeroshot_project8.py --model $model
done

python benchmarks/Project8/score.py
