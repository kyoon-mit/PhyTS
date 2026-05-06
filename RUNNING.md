# Running LIGO Regression Experiments

## Quick start

All commands run from the repo root:
```
cd /n/holystore01/LABS/iaifi_lab/Lab/kyoon/TimeSeriesPhysics
```

---

## Conv1D (ResNet1D) — GaussNLL regression

### Train + test (interactive / local GPU)
```bash
uv run python main.py fit \
    --config configs/LIGO/train_ligo_conv1d_gaussnll_regression.yaml \
    --trainer.logger.init_args.name="ligo_conv1d_$(date +%Y%m%d_%H%M%S)"

uv run python main.py test \
    --config configs/LIGO/train_ligo_conv1d_gaussnll_regression.yaml \
    --ckpt_path checkpoints/ligo_conv1d_gaussnll_regression/best.ckpt \
    --trainer.logger.init_args.name="ligo_conv1d_$(date +%Y%m%d_%H%M%S)_test"
```

### Submit to SLURM (gpu_test, unique WandB run per job)
```bash
sbatch benchmarks/LIGO/slurm/ligo_conv1d_gaussnll.sh
```
WandB run name: `ligo_conv1d_gaussnll_regression_<SLURM_JOB_ID>`

### Key config: `configs/LIGO/train_ligo_conv1d_gaussnll_regression.yaml`
| Setting | Value | Notes |
|---|---|---|
| Model | ResNet1DRegressor | layers=[1,1,1,1], kernel=7, ~3.85M params |
| Loss | GaussNLL + tanh soft-clamp (±5) | outputs mean + log-var |
| Optimizer | AdamW lr=1e-3, decay=0.99/epoch | |
| Gradient clip | 1.0 | via trainer.gradient_clip_val |
| Early stopping | patience=200 | monitors val/loss |
| Input | (B, 2, 1024) whitened strain, 59–63 s window | H1+L1 |
| Target | chirp_mass (M☉) | range ≈ [0.87, 2.17] |

### Delete checkpoint to start fresh
```bash
rm -f checkpoints/ligo_conv1d_gaussnll_regression/best.ckpt
```

---

## LinOSS (JAX/Equinox) — GaussNLL regression

### Train + test (interactive)
```bash
module load cudnn/9.10.2.21_cuda12-fasrc01

uv run python main.py fit \
    --config configs/LIGO/train_ligo_linoss_gaussnll_regression.yaml \
    --trainer.logger.init_args.name="ligo_linoss_$(date +%Y%m%d_%H%M%S)"

uv run python main.py test \
    --config configs/LIGO/train_ligo_linoss_gaussnll_regression.yaml \
    --ckpt_path checkpoints/ligo_linoss_gaussnll_regression/best.ckpt \
    --trainer.logger.init_args.name="ligo_linoss_$(date +%Y%m%d_%H%M%S)_test"
```

### Submit to SLURM
```bash
sbatch benchmarks/LIGO/slurm/ligo_linoss_gaussnll.sh
```
WandB run name: `ligo_linoss_gaussnll_regression_<SLURM_JOB_ID>`

### Key config: `configs/LIGO/train_ligo_linoss_gaussnll_regression.yaml`
| Setting | Value | Notes |
|---|---|---|
| Model | LinOSS | H=512, ssm_size=32, num_blocks=4, ~2.37M params |
| Loss | GaussNLL + tanh soft-clamp (±5) | |
| Optimizer | AdamW lr=1e-3 | optax, no LR schedule |
| Gradient clip | 1.0 | clip_grad_norm in task |
| Early stopping | patience=200 | |
| accelerator | cpu | JAX manages its own GPU via XLA |

> **Note:** The `accelerator: cpu` in the config is intentional — JAX bypasses Lightning's GPU management. The cuDNN module load is required.

### Delete checkpoint to start fresh
```bash
rm -f checkpoints/ligo_linoss_gaussnll_regression/best.ckpt
```

---

## Chronos (Foundation Model)

### Zero-shot
```bash
sbatch benchmarks/LIGO/slurm/ligo_chronos_zeroshot.sh
# or interactively:
uv run --project benchmarks/foundation \
    python benchmarks/LIGO/chronos_ligo.py \
        --mode zero_shot --model_size small \
        --out_dir results/LIGO/chronos_zeroshot
```

### Fine-tune (LoRA)
```bash
sbatch benchmarks/LIGO/slurm/ligo_chronos_finetune.sh
# or interactively:
uv run --project benchmarks/foundation \
    python benchmarks/LIGO/chronos_ligo.py \
        --mode finetune --model_size small \
        --finetune_strategy lora --lora_rank 4 --finetune_n_blocks 2 \
        --out_dir results/LIGO/chronos_finetune
```

Results written to `results/LIGO/chronos_{zeroshot,finetune}/metrics.json`.

---

## Monitor jobs
```bash
squeue -u kyoon --format="%.10i %.25j %.8T %.10M %R"
```

## Check training logs
```bash
# Live tail
tail -f slurm_logs/LIGO/ligo_conv1d_gaussnll_<JOB_ID>.out

# Extract val/loss per epoch
grep -oP "Epoch \d+: 100%.*val/loss=\S+" slurm_logs/LIGO/ligo_conv1d_gaussnll_<JOB_ID>.out
```

## WandB
Project: `TimeSeriesPhysics`  
Each SLURM run creates a unique run named `ligo_<model>_gaussnll_regression_<JOB_ID>`.  
URL: https://wandb.ai/kyoon-mit-massachusetts-institute-of-technology/TimeSeriesPhysics

## Results CSV locations
| Model | CSV |
|---|---|
| Conv1D | `results/LIGO/conv1d_gaussnll_regression/test_predictions.csv` |
| LinOSS | `results/LIGO/linoss_gaussnll_regression/test_predictions.csv` |
| Chronos ZS | `results/LIGO/chronos_zeroshot/metrics.json` |
| Chronos FT | `results/LIGO/chronos_finetune/metrics.json` |
