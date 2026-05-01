# TESS Classification Sweep

End-to-end hyperparameter sweep for TESS variability classification (8 classes).
No pretraining — all models are trained from scratch.

## What was changed / added

### New data loading (`src/dataloader/tess_dataloader.py`)
`TESSClassificationDataset` and `TESSClassificationDataModule` read
Hub split shards (`tess_classification_{train,val,test}.parquet`, typically
mirrored into `data/TESS/.cache/TESS/`) instead of performing a random
70/15/15 split in code.  Preprocessing is identical to the previous classes
(z-score normalization, pad/crop to `seq_len=1100`, bool mask).

### Model changes (`src/models/`)
- **`conv_ae.py`** — `ConvAE` gained a `num_classes` parameter.  When `> 0`, the
  decoder is replaced by a `Linear(latent_channels, num_classes)` head after
  global average pooling (classification mode).  Default behaviour (`num_classes=0`)
  is unchanged.
- **`conv_attn_ae.py`** — same pattern for `ConvAttnAE`; `latent_channels` must
  remain divisible by `num_heads`.

### New files
| File | Purpose |
|------|---------|
| `benchmarks/TESS/sweep_classification.py` | Unified sweep training script |
| `benchmarks/TESS/sweep_configs/mlp.yaml` | wandb sweep config for MLP |
| `benchmarks/TESS/sweep_configs/s4d.yaml` | wandb sweep config for S4D |
| `benchmarks/TESS/sweep_configs/cnn.yaml` | wandb sweep config for CNN |
| `benchmarks/TESS/sweep_configs/cnn_attn.yaml` | wandb sweep config for CNN+Attn |
| `benchmarks/TESS/sweep_configs/linoss_imex.yaml` | wandb sweep config for LinOSS-IMEX |
| `benchmarks/TESS/sweep_configs/linoss_damped.yaml` | wandb sweep config for LinOSS-Damped |
| `benchmarks/TESS/run_sweep.sh` | SLURM agent submission script |

---

## Data setup

Download from HuggingFace into `data/TESS/.cache/TESS/` (login node or any
machine with Hub access). `data/TESS/download_tess.py` runs `snapshot_download`
for `TESS/split/*` and **copies** shards into `TESS/*.parquet` so classification
and regression share one `data_dir`.

```bash
export HF_TOKEN=hf_...   # optional but recommended to avoid rate limits
uv run --extra jax python data/TESS/download_tess.py
# or: uv run --extra jax python data/TESS/download_tess.py --cache-dir /path/to/cache
```

The datamodule expects `--data_dir` to be that TESS root
(default: `data/TESS/.cache/TESS`, or `$TESS_DATA_DIR` on Engaging — set by
`benchmarks/TESS/run_sweep.sh`).

---

## Running a sweep

### Step 1 — Create the sweep (once, on a machine with internet)

```bash
# From the repo root:
wandb sweep benchmarks/TESS/sweep_configs/mlp.yaml --project TimeSeriesPhysics
# → prints: sweep ID, e.g. abc123def
```

### Step 2 — Launch agents on Engaging

```bash
# PyTorch models (mlp, s4d, cnn, cnn_attn):
bash benchmarks/TESS/run_sweep.sh \
    --model_type mlp \
    --sweep_id abc123def \
    --n_agents 4

# JAX models (linoss_imex, linoss_damped) — adds --jax automatically:
bash benchmarks/TESS/run_sweep.sh \
    --model_type linoss_imex \
    --sweep_id xyz789 \
    --n_agents 4
```

Each SLURM job runs one wandb agent, which executes trials until the sweep
is complete or the 6-hour time limit is reached.  Submit more agents to
parallelize.

### Step 3 — Monitor

```bash
squeue -u $USER
# or visit wandb.ai → TimeSeriesPhysics → Sweeps → <sweep_id>
```

---

## Local / debugging run (no SLURM)

```bash
# Disable wandb to avoid writing to the cloud:
WANDB_MODE=disabled \
uv run python benchmarks/TESS/sweep_classification.py \
    --model_type s4d \
    --data_dir data/TESS/.cache/TESS \
    --size xs --lr 1e-3 --batch_size 32 --dropout 0.1 --weight_decay 1e-4

# LinOSS requires the jax extra:
WANDB_MODE=disabled \
uv run --extra jax python benchmarks/TESS/sweep_classification.py \
    --model_type linoss_imex \
    --data_dir data/TESS/.cache/TESS \
    --size xs --lr 1e-3 --batch_size 32
```

---

## Model sizes and parameter count formulas

All size tiers (`--size xs/sm/md/lg`) target ~10K / ~100K / ~300K / ~700K
parameters.  The controlling dimension and formula are printed as comments
in `sweep_classification.py`.  Summary:

### MLP (`MLPRegressor`, `hidden_dims=[h, h]`)
```
params = h² + (seq_len + 6 + num_classes)*h + num_classes
       = h² + 1114*h + 8    (seq_len=1100, num_classes=8)
inverse: h = round((-1114 + sqrt(1114² + 4*(target − 8))) / 2)
```
| size | h | params |
|------|---|--------|
| xs   |   9 |  10,115 |
| sm   |  83 |  99,359 |
| md   | 224 | 299,720 |
| lg   | 448 | 699,784 |

### S4D (`S4Model`, `n_layers=4`, `d_state=64`)
```
params = 8*d² + 546*d + 8
inverse: d = round((-546 + sqrt(546² + 32*(target − 8))) / 16)
```
| size | d_model | params |
|------|---------|--------|
| xs   |  16 |  10,792 |
| sm   |  80 |  94,888 |
| md   | 160 | 292,168 |
| lg   | 264 | 701,720 |

### CNN (`ConvAE` classify mode, `n_layers=4`, `k=5`)
```
params = 15*C² + 25*C + 8
inverse: C = round((-25 + sqrt(625 + 60*(target − 8))) / 30)
```
| size | C | params |
|------|---|--------|
| xs   |  25 |  10,008 |
| sm   |  80 |  98,008 |
| md   | 140 | 297,508 |
| lg   | 216 | 705,248 |

### CNN+Attn (`ConvAttnAE` classify mode, `n_layers=4`, `k=5`, `num_heads=4`)
```
params = 19*C² + 31*C + 8    (C must be divisible by num_heads=4)
inverse: C = round((-31 + sqrt(961 + 76*(target − 8))) / 38) → nearest multiple of 4
```
| size | C | params |
|------|---|--------|
| xs   |  24 |  11,696 |
| sm   |  72 | 100,736 |
| md   | 124 | 295,996 |
| lg   | 192 | 706,376 |

### LinOSS-IMEX (`LinOSS`, `num_blocks=4`, `ssm_size=H`)
```
params = 24*H² + 30*H + 8    (ssm_size = H, tied)
inverse: H = round((-30 + sqrt(900 + 96*(target − 8))) / 48)
```
| size | H | params |
|------|---|--------|
| xs   |  20 |  10,208 |
| sm   |  64 | 100,232 |
| md   | 112 | 304,424 |
| lg   | 170 | 698,708 |

### LinOSS-Damped (`LinOSS`, `num_blocks=4`, `ssm_size=H`, `discretization=damped_IMEX`)
```
params = 24*H² + 34*H + 8    (adds G_diag per block)
```
Identical tier values of H; parameter counts ~4H higher than IMEX.

---

## Sweep hyperparameters

| Parameter | Type | Range |
|-----------|------|-------|
| `size` | categorical | xs, sm, md, lg |
| `lr` | log-uniform | 5×10⁻⁵ – 10⁻² |
| `dropout` | uniform | 0.0 – 0.5 (MLP/S4D); fixed 0.0 (CNN); fixed 0.05 (LinOSS) |
| `weight_decay` | log-uniform | 10⁻⁶ – 10⁻² |
| `batch_size` | categorical | 32, 64, 128 |
| `seed` | categorical | 0, 1, 2 |

Search method: **Bayesian optimization** (`method: bayes`), optimising `val/acc`.
Early termination: **Hyperband** (`min_iter=10`, `eta=3`) — underperforming runs
are stopped after epoch 10, 30, or 90 so compute is focused on promising trials.

---

## Checkpoints

Each trial saves its best checkpoint to:
```
$TESS_CKPT_DIR/<model_type>/<wandb_run_id>/
  best.ckpt   (PyTorch models)
  best.eqx    (LinOSS)
```

`TESS_CKPT_DIR` defaults to `checkpoints/sweeps/classification` relative to the
repo root, but is overridden on Engaging by `run_sweep.sh`.
