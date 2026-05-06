#!/bin/bash
# TESS benchmark — train all models and run evaluation.
#
# Prerequisites:
#   source .venv/bin/activate                  # make env (PyTorch models)
#   # LinOSS also needs JAX: make env-jax && source .venv/bin/activate
#   python data/TESS/download_tess.py           # downloads data/TESS/tess_classification.parquet
#
# Usage:
#   bash benchmarks/TESS/run.sh

set -e

# S4D — 8-class stellar variability classification
python main.py fit  --config configs/TESS/train_tess_s4d_classification.yaml
python main.py test --config configs/TESS/train_tess_s4d_classification.yaml \
    --ckpt_path checkpoints/tess_s4d_classification/best.ckpt

# 1D CNN — classification
python main.py fit  --config configs/TESS/train_tess_conv_classification.yaml
python main.py test --config configs/TESS/train_tess_conv_classification.yaml \
    --ckpt_path checkpoints/tess_conv_classification/best.ckpt

# RNN — classification
python main.py fit  --config configs/TESS/train_tess_rnn_classification.yaml
python main.py test --config configs/TESS/train_tess_rnn_classification.yaml \
    --ckpt_path checkpoints/tess_rnn_classification/best.ckpt

# LinOSS (JAX) — classification
python main.py fit  --config configs/TESS/train_tess_linoss_classification.yaml
python main.py test --config configs/TESS/train_tess_linoss_classification.yaml \
    --ckpt_path checkpoints/tess_linoss_classification/best.eqx

# Evaluate all models
python benchmarks/TESS/eval_pipeline.py \
    --data_dir data/TESS \
    --ckpt_dir checkpoints \
    --out_dir  results/TESS
