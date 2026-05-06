# Running LIGO Experiments

All commands run from the repo root. Activate the environment first:
```bash
source .venv/bin/activate        # PyTorch models
# or for LinOSS:
source .venv/bin/activate        # env-jax build (make env-jax)
```

Data must be downloaded from HuggingFace ([`PhyTS-team/PhyTS-bench`](https://huggingface.co/datasets/PhyTS-team/PhyTS-bench)) and placed under `data/LIGO/{train,val,test}/`.

---

## Conv1D — GaussNLL regression

### Train + test (interactive)
```bash
python main.py fit \
    --config configs/LIGO/train_ligo_conv1d_gaussnll_regression.yaml

python main.py test \
    --config configs/LIGO/train_ligo_conv1d_gaussnll_regression.yaml \
    --ckpt_path checkpoints/ligo_conv1d_gaussnll_regression/best.ckpt
```

### Submit to SLURM
```bash
sbatch benchmarks/LIGO/slurm/ligo_conv1d_gaussnll.sh
```

### Key settings
| Setting | Value |
|---|---|
| Model | ResNet1D, layers=[1,1,1,1], kernel=7, ~3.85M params |
| Loss | GaussNLL + tanh soft-clamp (±5) |
| Optimizer | AdamW lr=1e-3, decay=0.99/epoch |
| Input | (B, 2, 1024) whitened strain, 59–63 s window |
| Target | chirp_mass (M☉) |

---

## LinOSS (JAX) — GaussNLL regression

### Train + test (interactive)
```bash
python main.py fit \
    --config configs/LIGO/train_ligo_linoss_gaussnll_regression.yaml

python main.py test \
    --config configs/LIGO/train_ligo_linoss_gaussnll_regression.yaml \
    --ckpt_path checkpoints/ligo_linoss_gaussnll_regression/best.ckpt
```

### Submit to SLURM
```bash
sbatch benchmarks/LIGO/slurm/ligo_linoss_gaussnll.sh
```

### Key settings
| Setting | Value |
|---|---|
| Model | LinOSS, H=512, ssm_size=32, num_blocks=4, ~2.37M params |
| Loss | GaussNLL + tanh soft-clamp (±5) |
| Optimizer | AdamW lr=1e-3 (optax) |
| accelerator | cpu (JAX manages GPU via XLA) |

> `accelerator: cpu` is intentional — JAX bypasses Lightning's GPU management.

---

## Chronos (Foundation Model)

### Zero-shot
```bash
sbatch benchmarks/LIGO/slurm/ligo_chronos_zeroshot.sh
# or interactively (foundation env):
python benchmarks/LIGO/chronos_ligo.py \
    --mode zero_shot --model_size small \
    --out_dir results/LIGO/chronos_zeroshot
```

### Fine-tune (LoRA)
```bash
sbatch benchmarks/LIGO/slurm/ligo_chronos_finetune.sh
# or interactively:
python benchmarks/LIGO/chronos_ligo.py \
    --mode finetune --model_size small \
    --finetune_strategy lora --lora_rank 4 --finetune_n_blocks 2 \
    --out_dir results/LIGO/chronos_finetune
```

Results: `results/LIGO/chronos_{zeroshot,finetune}/metrics.json`

---

## Monitor jobs
```bash
squeue -u $USER --format="%.10i %.25j %.8T %.10M %R"
```

## Results locations
| Model | File |
|---|---|
| Conv1D | `results/LIGO/conv1d_gaussnll_regression/test_predictions.csv` |
| LinOSS | `results/LIGO/linoss_gaussnll_regression/test_predictions.csv` |
| Chronos ZS | `results/LIGO/chronos_zeroshot/metrics.json` |
| Chronos FT | `results/LIGO/chronos_finetune/metrics.json` |
