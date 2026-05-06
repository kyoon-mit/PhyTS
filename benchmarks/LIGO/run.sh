#!/bin/bash
# LIGO benchmark — train all models and run evaluation.
#
# Prerequisites:
#   source .venv/bin/activate          # make env (PyTorch) or make env-jax (LinOSS)
#   python data/download.py --domain ligo --sample   # or full dataset
#
# Usage:
#   bash benchmarks/LIGO/run.sh

set -e

# S4D — GaussNLL chirp-mass regression
python main.py fit  --config configs/LIGO/train_ligo_s4d_gaussnll_regression.yaml
python main.py test --config configs/LIGO/train_ligo_s4d_gaussnll_regression.yaml \
    --ckpt_path checkpoints/ligo_s4d_gaussnll_regression/best.ckpt

# 1D CNN — GaussNLL chirp-mass regression
python main.py fit  --config configs/LIGO/train_ligo_conv1d_gaussnll_regression.yaml
python main.py test --config configs/LIGO/train_ligo_conv1d_gaussnll_regression.yaml \
    --ckpt_path checkpoints/ligo_conv1d_gaussnll_regression/best.ckpt

# LinOSS (JAX) — GaussNLL chirp-mass regression
python main.py fit  --config configs/LIGO/train_ligo_linoss_gaussnll_regression.yaml
python main.py test --config configs/LIGO/train_ligo_linoss_gaussnll_regression.yaml \
    --ckpt_path checkpoints/ligo_linoss_gaussnll_regression/best.ckpt
