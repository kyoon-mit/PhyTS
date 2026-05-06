#!/bin/bash
# TIDMAD (ABRACADABRA) benchmark — train all denoising models and evaluate.
#
# Prerequisites:
#   source .venv/bin/activate          # make env (PyTorch) or make env-jax (LinOSS)
#   python data/download.py --domain tidmad --sample   # or full dataset
#   python data/TIDMAD/preprocess_tidmad.py \
#       --data_dir data/TIDMAD/original --out_dir data/TIDMAD/preprocessed
#
# Usage:
#   bash benchmarks/TIDMAD/run.sh

set -e

# LinOSS (JAX) — denoising
python main.py fit --config configs/TIDMAD/train_tidmad_linoss_denoising.yaml

# CNN (ConvAE-L) — denoising
python main.py fit --config configs/TIDMAD/train_tidmad_conv_l_denoising.yaml

# Compute TIDMAD denoising score across all trained variants
PYTHONPATH=src python benchmarks/TIDMAD/evaluate_all.py
