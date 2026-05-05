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

- `**conv_ae.py**` — `ConvAE` gained a `num_classes` parameter.  When `> 0`, the
decoder is replaced by a `Linear(latent_channels, num_classes)` head after
global average pooling (classification mode).  Default behaviour (`num_classes=0`)
is unchanged.
- `**conv_attn_ae.py**` — same pattern for `ConvAttnAE`; `latent_channels` must
remain divisible by `num_heads`.

### New files


| File                                              | Purpose                                                                                                                                 |
| ------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------- |
| `benchmarks/TESS/sweep_classification.py`         | Unified sweep training script (classification)                                                                                          |
| `benchmarks/TESS/sweep_regression.py`             | Unified sweep training script (regression)                                                                                              |
| `configs/TESS/sweep/sweep_cls_*.yaml`             | wandb sweep configs for classification (one file per architecture)                                                                      |
| `configs/TESS/sweep/sweep_reg_*.yaml`             | wandb sweep configs for regression                                                                                                      |
| `configs/TESS/sweep/all_model_sweep_dims.yaml`    | Hidden widths per tier (xs/sm/md/lg) for MLP/S4D/CNN/ConvAttn/LinOSS; transformers use inlined tiers in `sweep_utils.build_transformer` |
| `benchmarks/TESS/sweep_aggregate_utils.py`        | Helpers for aggregated sweep summaries                                                                                                  |
| `benchmarks/TESS/analyze_classification_sweep.py` | Merge N classification sweeps → heatmaps + CSV/JSON/HTML                                                                                |
| `benchmarks/TESS/analyze_regression_sweep.py`     | Merge N regression sweeps → heatmaps + CSV/JSON/HTML                                                                                    |
| `benchmarks/TESS/run_sweep.sh`                    | SLURM agent submission script                                                                                                           |


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
# From the repo root (classification example; use sweep_reg_*.yaml for regression):
wandb sweep configs/TESS/sweep/sweep_cls_mlp.yaml --project TimeSeriesPhysics
# → prints: sweep ID, e.g. abc123def
```

Sweep YAMLs call **`benchmarks/TESS/repo_python.sh`** instead of bare ``python`` so each trial runs under the repo **``.venv``** (with jax/equinox when synced). Agents started via ``run_sweep.sh`` already prepend ``.venv/bin`` to ``PATH``, but the launcher avoids conda/base Python entirely.

If you change the sweep ``command`` in YAML after creating a sweep, **wandb keeps the old command** until you run ``wandb sweep ...`` again and point agents at the **new** sweep id.

Each sweep YAML sets `**run_cap: 150**`, so a **single** sweep schedules at most **150** trials (Hyperband may still prune many of those runs early).

**Important:** wandb budgets are **per sweep**, not pooled across architectures. Running every classifier `**sweep_cls_*`** plus every `**sweep_reg_***` (seven architectures each × two tasks → **fourteen** YAMLs) would allow **up to 14 × 150** trials unless you pause sweeps sooner. To approximate **N** trials shared evenly across `**M`** sweep files, divide manually (`**run_cap: floor(N/M)**`) **before** calling `wandb sweep`, or shorten the list of architectures you tune.

### Step 2 — Launch agents on Engaging

```bash
# PyTorch models (mlp, s4d, cnn, cnn_attn, transformer):
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

### Step 4 — Merge all sweep IDs (classification or regression)

After every architecture sweep you care about reports **Finished** with test metrics logged, pull them in one shot (from a machine that can reach the wandb API). Pass **every** sweep id printed by `wandb sweep` for that task (classification vs regression separately)—there are seven classification and seven regression sweep YAMLs (`mlp`, `s4d`, `cnn`, `cnn_attn`, `transformer`, `linoss_imex`, `linoss_damped`):

```bash
# Classification — repeat --sweep_id or use --from_file sweep_ids_cls.txt:
uv run python benchmarks/TESS/analyze_classification_sweep.py \
    --sweep_id aaa111 --sweep_id bbb222 --sweep_id ccc333

# Regression:
uv run python benchmarks/TESS/analyze_regression_sweep.py \
    --from_file sweep_ids_reg.txt \
    --out_dir benchmarks/TESS/sweep_reports/my_reg_run
```

Default output dirs (overridable with `--out_dir`):

- `benchmarks/TESS/sweep_reports/classification_all_sweeps/`
- `benchmarks/TESS/sweep_reports/regression_all_sweeps/`

Each run writes `report.html`, CSVs, `best_hyperparameters.json`, and heatmap PNGs. See script docstrings for metric definitions and caveats.

---

## LinOSS / JAX troubleshooting

| Symptom | Likely cause | What to do |
| ------- | ------------ | ---------- |
| ``ModuleNotFoundError: No module named 'equinox'`` | Trial subprocess used the wrong interpreter (conda/base ``python``). | Ensure agents run YAMLs that invoke ``benchmarks/TESS/repo_python.sh``; **recreate the sweep** after updating YAMLs. Confirm ``.venv`` has jax: ``uv sync --extra jax --extra cu12`` or ``cu13``. |
| ``ImportError: cannot import name 'multihost_utils' from 'jax.experimental'`` | Incomplete or mismatched JAX install while loading ``.eqx`` checkpoints (often after partial ``pip``/conda mixing). | From repo root: ``uv sync --extra jax --extra cu12`` (or ``cu13``). Verify ``python -c "from jax.experimental import multihost_utils"``. |
| ``RuntimeWarning: os.fork() was called... JAX is multithreaded`` | Forking worker processes after JAX has started threads. | Sweep scripts force ``num_workers=0`` for LinOSS and disable Lightning worker RNG seeding; upgrade to current sweep scripts if you still see this on old checkouts. |

Override the launcher explicitly if needed:

```bash
export TIMESERIES_PHYSICS_PYTHON=/path/to/your/.venv/bin/python
```

---

## Local / debugging run (no SLURM)

```bash
# Disable wandb to avoid writing to the cloud:
WANDB_MODE=disabled \
uv run python benchmarks/TESS/sweep_classification.py \
    --model_type s4d \
    --data_dir data/TESS/.cache/TESS \
    --size xs --lr 1e-3 --batch_size 128 --dropout 0.1 --weight_decay 1e-4

# LinOSS requires the jax extra:
WANDB_MODE=disabled \
uv run --extra jax python benchmarks/TESS/sweep_classification.py \
    --model_type linoss_imex \
    --data_dir data/TESS/.cache/TESS \
    --size xs --lr 1e-3 --batch_size 128
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


| size | h   | params  |
| ---- | --- | ------- |
| xs   | 9   | 10,115  |
| sm   | 83  | 99,359  |
| md   | 224 | 299,720 |
| lg   | 448 | 699,784 |


### S4D (`S4Model`, `n_layers=4`, `d_state=64`)

```
params = 8*d² + 546*d + 8
inverse: d = round((-546 + sqrt(546² + 32*(target − 8))) / 16)
```


| size | d_model | params  |
| ---- | ------- | ------- |
| xs   | 16      | 10,792  |
| sm   | 80      | 94,888  |
| md   | 160     | 292,168 |
| lg   | 264     | 701,720 |


### CNN (`ConvAE` classify mode, `n_layers=4`, `k=5`)

```
params = 15*C² + 25*C + 8
inverse: C = round((-25 + sqrt(625 + 60*(target − 8))) / 30)
```


| size | C   | params  |
| ---- | --- | ------- |
| xs   | 25  | 10,008  |
| sm   | 80  | 98,008  |
| md   | 140 | 297,508 |
| lg   | 216 | 705,248 |


### CNN+Attn (`ConvAttnAE` classify mode, `n_layers=4`, `k=5`, `num_heads=4`)

```
params = 19*C² + 31*C + 8    (C must be divisible by num_heads=4)
inverse: C = round((-31 + sqrt(961 + 76*(target − 8))) / 38) → nearest multiple of 4
```


| size | C   | params  |
| ---- | --- | ------- |
| xs   | 24  | 11,696  |
| sm   | 72  | 100,736 |
| md   | 124 | 295,996 |
| lg   | 192 | 706,376 |


### Transformer (`TransformerClassifier`, `benchmarks/TESS/sweep_utils.build_transformer`)

Sweep tiers bundle `(d_model, num_layers)`; `nhead=4`; `dim_feedforward=2*d_model`; dropout is tuned on encoder layers and the classifier/regressor head (`dropout` in configs).

Sinusoidal positional encodings register non-persistent buffers, so they are omitted from `.parameters()`; `param_count_torch_nn` logged by sweep scripts matches the table below.


| size | `d_model` | `num_layers` | params (cls `d_output=8`) |
| ---- | --------- | ------------ | ------------------------- |
| xs   | 24        | 2            | **10,640**                |
| sm   | 56        | 4            | **106,688**               |
| md   | 96        | 4            | **309,608**               |
| lg   | 144       | 4            | **692,504**               |


Regression (`d_output=1`) at the same architecture: **10,465 · 106,289 · 308,929 · 691,489**.

The **xs** tier uses two encoder layers; with `**num_layers=4`**, widths small enough for the usual ~10K budget would skew much larger (often near ~79K PyTorch-only parameters alone), so depth is shortened for that tier.

### LinOSS-IMEX (`LinOSS`, `num_blocks=4`, `ssm_size=H`)

```
params = 24*H² + 30*H + H*d_output + d_output    (ssm_size = H; sweeps use tied H)
inverse: H = round((-(30 + d_output) + sqrt((30 + d_output)² + 96*(target − d_output))) / 48)

Classification (``d_output=8``) and regression (``d_output=1``) differ by ``7*(H+1)`` parameters.
```


| size | H   | params (cls `d_out=8`) |
| ---- | --- | ---------------------- |
| xs   | 20  | 10,368                 |
| sm   | 64  | 100,744                |
| md   | 112 | 305,320                |
| lg   | 170 | 700,068                |


Regression (`d_output=1`) at the same H: 10,221 · 100,289 · 304,529 · 698,871.

### LinOSS-Damped (`LinOSS`, `num_blocks=4`, `ssm_size=H`, `discretization=damped_IMEX`)

```
params = 24*H² + 34*H + H*d_output + d_output    (adds trainable G_diag per block: +4H vs IMEX)
```

Same H column as IMEX; damped adds `+4H` (e.g. cls: 10,448 · 101,000 · 305,768 · 700,748 at the tiers above).

---

## Sweep hyperparameters


| Parameter      | Type        | Range                                                                                  |
| -------------- | ----------- | -------------------------------------------------------------------------------------- |
| `size`         | categorical | xs, sm, md, lg                                                                         |
| `lr`           | log-uniform | 5×10⁻⁵ – 2×10⁻² (MLP, S4D, CNN, CNN+Attn, Transformer); 5×10⁻⁵ – 10⁻² (LinOSS)         |
| `dropout`      | uniform     | 0 – 0.5 (`sweep_cls_*.yaml` / `sweep_reg_*.yaml` for PyTorch architectures and LinOSS) |
| `weight_decay` | log-uniform | 10⁻⁶ – 10⁻²                                                                            |
| `batch_size`   | categorical | 128, 256, 512                                                                          |
| `seed`         | categorical | 0, 1, 2                                                                                |


Search method: **Bayesian optimization** (`method: bayes`), optimising `val/balanced_acc` (classification) or `val/r2` (regression).
Early termination: **Hyperband** with `**min_iter: 20`** and `**eta: 3**` (`early_terminate` blocks in each `configs/TESS/sweep/sweep_*_*.yaml`).

---

## Checkpoints

Each trial saves its best checkpoint under the task subdirectory (matches `benchmarks/TESS/sweep_{classification|regression}.py`):

```
<$TESS_CKPT_DIR-or-default>/<classification|regression>/<model_type>/<wandb_run_id>/
  best.ckpt   (PyTorch models)
  best.eqx    (LinOSS)
```

CLI default for the base directory is `checkpoints/sweeps` (when `TESS_CKPT_DIR` is unset): e.g.
`checkpoints/sweeps/classification/mlp/<run_id>/best.ckpt`. On Engaging, `benchmarks/TESS/run_sweep.sh` sets `TESS_CKPT_DIR` under the shared pool (`$TESS_POOL_ROOT/checkpoints/sweeps` by convention).