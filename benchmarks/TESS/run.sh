#!/bin/bash
# TESS benchmark (paper Table 2): train CNN / S4D / LinOSS on variability classification and on
# rotation (frot) regression, evaluate the six zero-shot foundation models, and score everything
# on the test split.
#
# Prerequisites:
#   python data/download.py --domain tess          # ~250 MB
#   make env-jax                                   # PyTorch + JAX env (S4D, CNN, LinOSS)
#   make env-fm                                    # foundation-model env
#
# Each model trains for up to 200 epochs (early stopping on val loss, patience 20); test uses the
# best-validation checkpoint. Table 2 reports the ~300K-parameter models; the configs use that size.
#
# Usage: bash benchmarks/TESS/run.sh

set -e

for task in classification regression; do
    for model in conv s4d linoss; do
        cfg=configs/TESS/train_tess_${model}_${task}.yaml
        python main.py fit  --config $cfg
        python main.py test --config $cfg --ckpt_path checkpoints/tess_${model}_${task}/best.ckpt
    done
done

# Zero-shot foundation models: frozen embeddings + linear probe fitted on train, scored on test.
for model in moment chronos timesfm timemoe moirai granite_ttm; do
    benchmarks/foundation/.venv/bin/python benchmarks/TESS/fm_zeroshot_tess.py --model $model
done

python benchmarks/TESS/score.py
