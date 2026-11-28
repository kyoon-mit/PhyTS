#!/bin/bash
# TIDMAD benchmark (paper Table 2): train ConvAE / S4D / LinOSS, denoise the test file with them
# and with the four zero-shot foundation models, and score everything with the official scorer.
#
# Prerequisites:
#   python data/download.py --domain tidmad                                 # ~163 GB
#   python data/TIDMAD/preprocess_tidmad.py --data_dir data/TIDMAD/original --out_dir data/TIDMAD/preprocessed
#   make env-jax   # PyTorch + JAX env (all three supervised models)
#   make env-fm    # foundation-model env
#
# Each model trains for up to 100 epochs (early stopping, patience 10); the best-validation
# checkpoint denoises every 10th one-second chunk of the test file.
#
# Usage: bash benchmarks/TIDMAD/run.sh

set -e

python benchmarks/TIDMAD/denoise.py --model raw    # baseline: the undenoised test file

for model in conv s4d linoss; do
    cfg=configs/TIDMAD/train_tidmad_${model}_denoising.yaml
    python main.py fit  --config $cfg
    python main.py test --config $cfg --ckpt_path checkpoints/tidmad_${model}_denoising/best.ckpt
    python benchmarks/TIDMAD/denoise.py --model $model
done

for model in moment chronos granite_ttm moirai; do
    benchmarks/foundation/.venv/bin/python benchmarks/TIDMAD/denoise.py --model $model
done

python benchmarks/TIDMAD/score.py
