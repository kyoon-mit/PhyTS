#!/bin/bash
# TIDMAD (ABRACADABRA) benchmark — train all denoising models and evaluate.
#
# Prerequisites:
#   source .venv/bin/activate          # make env (PyTorch models)
#   # LinOSS also needs JAX: make env-jax && source .venv/bin/activate
#   # Preprocess raw HDF5 data first: python data/TIDMAD/preprocess_tidmad.py
#   #   (see data/TIDMAD/README.md for HuggingFace download instructions)
#
# Usage:
#   bash benchmarks/TIDMAD/run.sh

set -e

# LinOSS (JAX) — denoising
python main.py fit --config configs/TIDMAD/train_tidmad_linoss_denoising.yaml

# Conv-AE — MSE denoising
python main.py fit --config configs/TIDMAD/train_tidmad_conv_ae_denoising_mse.yaml

# Conv-Attn-AE — MSE denoising
python main.py fit --config configs/TIDMAD/train_tidmad_conv_attn_ae_denoising_mse.yaml

# RNN Seq2Seq — MSE denoising
python main.py fit --config configs/TIDMAD/train_tidmad_rnn_denoising_mse.yaml

# MLP denoiser — MSE
python main.py fit --config configs/TIDMAD/train_tidmad_mlp_denoiser_denoising_mse.yaml

# Compute TIDMAD denoising score across all variants
PYTHONPATH=src python benchmarks/TIDMAD/evaluate_all.py
