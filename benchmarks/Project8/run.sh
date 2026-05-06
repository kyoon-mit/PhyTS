#!/bin/bash
# Project 8 benchmark — train all models and run evaluation.
#
# Prerequisites:
#   source .venv/bin/activate          # make env (PyTorch models)
#   # LinOSS also needs JAX: make env-jax && source .venv/bin/activate
#   # Data in data/Project8/{train,valid,test}/ (from PhyTS-team/PhyTS-bench on HuggingFace)
#
# Usage:
#   bash benchmarks/Project8/run.sh

set -e

# S4D — GaussNLL energy regression
python main.py fit --config configs/Project8/train_project8_s4d_regression_energy_gaussiannll.yaml
python benchmarks/Project8/eval_pipeline.py \
    --config     configs/Project8/train_project8_s4d_regression_energy_gaussiannll.yaml \
    --checkpoint checkpoints/project8_s4d_regression_energy_gaussiannll/best.ckpt

# LinOSS (JAX) — GaussNLL energy regression
python main.py fit --config configs/Project8/train_project8_linoss_regression_energy_gaussiannll.yaml
python benchmarks/Project8/eval_pipeline.py \
    --config     configs/Project8/train_project8_linoss_regression_energy_gaussiannll.yaml \
    --checkpoint checkpoints/project8_linoss_regression_energy_gaussiannll/best.eqx

# 1D CNN — GaussNLL energy regression
python main.py fit --config configs/Project8/train_project8_conv_regression_energy_gaussiannll.yaml
python benchmarks/Project8/eval_pipeline.py \
    --config     configs/Project8/train_project8_conv_regression_energy_gaussiannll.yaml \
    --checkpoint checkpoints/project8_conv_regression_energy_gaussiannll/best.ckpt
