# TESS Benchmarking Pipeline

This pipeline benchmarks several sequence architectures—see `README.md` for the config table—and includes a self-supervised reconstruction pretraining stage that allows frozen **S4D** backbone + MLP-head transfer learning. Hyperparameter wandb sweeps over **all** Torch/JAX runners are documented separately in **`README_Sweep.md`**.

---

## 1. Data

### Source

Data comes from the [PhyTS-bench](https://huggingface.co/datasets/PhyTS-team/PhyTS-bench) HuggingFace dataset and is stored locally in `data/TESS/.cache/TESS/` as two Parquet files:


| File                          | Rows   | Target           | Notes                                    |
| ----------------------------- | ------ | ---------------- | ---------------------------------------- |
| `tess_regression.parquet`     | 4,183  | `frot` (float)   | Stellar rotation frequency in cycles/day |
| `tess_classification.parquet` | 25,935 | `label` (string) | Variability class; 8 unique labels       |


Each row has a `flux` column containing a variable-length 1D array of photometric flux measurements (542–3,917 samples per lightcurve). There is no separate "clean" vs. "noisy" channel — TESS flux is a single observed stream.

The 8 classification labels are: `APERIODIC`, `CONTACT_ROT`, `DSCT_BCEP`, `ECLIPSE`, `GDOR_SPB`, `INSTRUMENT/JUNK`, `RRLYR_CEPH`, `SOLARLIKE`.

### Preprocessing (applied in `src/dataloader/tess_dataloader.py`)

All preprocessing is done in `__init__` at dataset construction time; `__getitem__` only performs tensor conversion (plus noise injection for reconstruction).

**Normalization.** Each lightcurve is z-score normalized independently:

```
normed = (flux − nanmedian(flux)) / nanstd(flux)
```

Non-finite values (NaN, Inf) are replaced with 0.0 after normalization. If `std < 1e-8` (effectively flat lightcurve), the entire array is set to zero. This produces flux values roughly in the range [−3, 3] with unit variance.

**Padding and masking.** All sequences are cropped or zero-padded to a fixed length of `seq_len = 1100` time steps. A boolean mask of the same length is produced alongside: `True` where a real cadence exists, `False` in the zero-padded region. This pair `(flux, mask)` is what every dataset returns for its input.

**Why 1100?** The longest lightcurve in the regression dataset is ~3,917 steps, but 1100 covers the majority of sequences; cropping from the start is used for the rest. The crop discards the tail, which in TESS data tends to be the most gap-affected region.

### Train / Val / Test Splits

Splits are 70 / 15 / 15 with fixed `seed=42`, stratified to ensure each split has the same distribution of targets:

- **Classification:** stratified by integer class label directly.
- **Regression:** `frot` is continuous, so it is first binned into 10 quantile deciles and stratification is performed on the bin index. This ensures each split covers the full range of rotation frequencies.

The same stratification logic is reused in `TESSReconstructionDataset` so that reconstruction and downstream datasets draw from identical sets of rows (no leakage across folds).

### Reconstruction: Synthetic Noise

Because TESS has only one signal stream (no ground truth "clean" counterpart), the reconstruction pretraining task is built by adding synthetic Gaussian noise during `__getitem__`:

```python
noisy_flux = clean_flux + Normal(0, noise_std=0.3) * mask
```

Noise is only applied to valid (non-padded) positions. The original normalized flux is the reconstruction target. A `noise_std=0.3` means roughly 30% of the signal's standard deviation is added as noise.

---

## 2. Models

### MLPRegressor (`src/models/mlp.py`)

A simple feed-forward network. The input sequence is flattened and passed through fully-connected layers with LayerNorm, GELU activation, and dropout between each layer.

**Architecture for TESS (from configs):**

```
Input: (B, 1100, 1)  →  flatten to (B, 1100)
Linear(1100 → 512) + LayerNorm + GELU + Dropout(0.1)
Linear(512  → 256) + LayerNorm + GELU + Dropout(0.1)
Linear(256  → 128) + LayerNorm + GELU + Dropout(0.1)
Linear(128  → d_output)         ← d_output=1 for regression, 8 for classification
Output: (B, d_output)
```

Parameter count: ~850K for regression, ~851K for classification.

The MLP has no notion of sequence order or temporal structure. It treats the 1100-step lightcurve as a flat feature vector. It serves as the baseline.

### S4Model (`src/models/s4d.py`) — for end-to-end tasks

A stack of S4D (Structured State Space Diagonal) layers followed by mean-pooling over the sequence dimension, used for classification and regression tasks that need a single output vector per sample.

**Architecture (from configs: `d_model=256, d_state=64, n_layers=4`):**

```
Input: (B, L=1100, d_input=1)
Linear encoder: (B, L, 1) → (B, L, 256)
Transpose:                  → (B, 256, L)

For each of 4 layers:
  S4D kernel (diagonal SSM, FFT convolution) → residual + LayerNorm
  Output:  (B, 256, L)

Transpose back:  (B, L, 256)
Mean-pool over L: (B, 256)
Linear decoder:  (B, 256) → (B, d_output)
```

The S4D kernel learns a diagonal complex state-space model. It computes a convolution kernel from learnable SSM parameters `(A, C, dt)` and applies it via FFT, giving O(L log L) sequence complexity. This allows the model to capture long-range temporal dependencies across the full 1100-step lightcurve.

### S4ModelSeq2Seq (`src/models/s4d_seq2seq.py`) — for reconstruction pretraining

Identical to `S4Model` except the mean-pooling step is removed, so the full sequence shape is preserved end-to-end.

```
Input:  (B, L=1100, 1)
...same S4D stack as above...
Linear decoder: (B, L, 256) → (B, L, 1)
Output: (B, L, 1)   ← reconstructed flux, same shape as input
```

Used only in the reconstruction pretraining stage.

---

## 3. Tasks and Training

All training uses PyTorch Lightning + LightningCLI (launched via `python main.py fit --config <yaml>`). The task classes wrap a model with a loss function, logging, and optimizer configuration.

### Stage 1 — Reconstruction Pretraining (`tess_reconstruction.py`)

**Class:** `TESSReconstructionMSE`  
**Config:** `configs/TESS/other/train_tess_s4d_reconstruction.yaml`

Trains `S4ModelSeq2Seq` to map noisy flux back to clean flux. Loss is MSE computed only over the valid (non-padded) positions:

```
loss = sum((reconstructed − clean)² × mask) / sum(mask)
```

An SNR-like monitoring metric is also logged:

```
SNR_dB = 10 × log10(signal_power / loss)
```

where `signal_power` is the mean squared value of the clean flux over valid positions. This gives a human-interpretable quality metric in decibels alongside the raw loss. The checkpoint from this stage (`checkpoints/tess_s4d_reconstruction/best.ckpt`) is required by the frozen-backbone runs.

### Stage 2 — End-to-End Tasks

These run independently of reconstruction and in parallel with each other.

#### Regression (`tess_regression.py` → `TESSRegressionMSE`)

```
Input:  flux (B, 1100)
        → unsqueeze(-1) → (B, 1100, 1)
        → model (MLP or S4Model)
        → (B, 1)
        → squeeze(-1)
        → prediction (B,)
Loss:   MSE(prediction, frot)
```

#### Classification (`tess_classification.py` → `TESSClassificationCE`)

```
Input:  flux (B, 1100)
        → unsqueeze(-1) → (B, 1100, 1)
        → model (MLP or S4Model)
        → logits (B, 8)
Loss:   CrossEntropy(logits, label)
```

### Stage 3 — Frozen-Backbone Tasks

Requires the reconstruction checkpoint from Stage 1.

**Classes:** `TESSFrozenBackboneRegressionMSE`, `TESSFrozenBackboneClassificationCE`  
**Configs:** `configs/TESS/other/train_tess_s4d_head_regression.yaml`, `configs/TESS/other/train_tess_s4d_head_classification.yaml`

At construction time, `S4ModelSeq2Seq` is loaded from the checkpoint, all its parameters are frozen (`requires_grad=False`), and it is permanently kept in `eval()` mode. Only the MLP head is trained.

```
Input:  flux (B, 1100)
        → unsqueeze(-1) → (B, 1100, 1)
        → [frozen] S4ModelSeq2Seq → (B, 1100, 1)   ← denoised/reconstructed flux
        → MLPRegressor or MLPClassifier head
        → output (B, 1) or (B, 8)
```

The key subtlety: Lightning calls `model.train()` at the start of every epoch, which recursively enables dropout in all submodules including the frozen backbone. A `train()` override in both frozen-backbone classes ensures the backbone stays in eval mode regardless:

```python
def train(self, mode=True):
    super().train(mode)
    self.backbone.eval()   # always keep frozen backbone in eval mode
    return self
```

**Optimizer:** Only the head parameters are passed to AdamW. The backbone parameters carry `requires_grad=False` and are never touched by the optimizer.

### Optimizer and Scheduler (all tasks)

AdamW with `lr=1e-3`, `weight_decay=1e-2` (PyTorch default), with an exponential LR decay of `gamma=0.99` applied each epoch.

### Training hyperparameters (all configs)


| Setting                | Value                              |
| ---------------------- | ---------------------------------- |
| `max_epochs`           | 200                                |
| `batch_size`           | 64                                 |
| `seq_len`              | 1100                               |
| `num_workers`          | 8                                  |
| EarlyStopping patience | 20 epochs on `val/loss`            |
| Checkpoint metric      | `val/loss` (min)                   |
| Logger                 | WandB, project `TimeSeriesPhysics` |


---

## 4. Evaluation (`benchmarks/TESS/eval_pipeline.py`)

Run after training to collect metrics and produce plots for all models at once:

```bash
uv run python benchmarks/TESS/eval_pipeline.py \
    --data_dir data/TESS/.cache/TESS \
    --out_dir  benchmarks/TESS
```

Models whose checkpoint file does not exist are silently skipped, so you can run the eval script after any subset of training jobs.

### Regression metrics (per model)


| Metric | Description                                       |
| ------ | ------------------------------------------------- |
| RMSE   | Root mean squared error on the test set           |
| MAE    | Mean absolute error                               |
| R²     | Coefficient of determination: 1 − SS_res / SS_tot |


**Plots** (saved as `<model_name>_regression.png`):

- Scatter plot: predicted frot vs. true frot, with identity line
- Residual histogram: distribution of (predicted − true)

**CSV** (`<model_name>_regression_results.csv`): one row per test sample with `frot_true`, `frot_hat`, `residual`.

### Classification metrics (per model)


| Metric             | Description                                             |
| ------------------ | ------------------------------------------------------- |
| Overall accuracy   | Fraction of correct predictions across all test samples |
| Per-class accuracy | Accuracy within each of the 8 variability classes       |


**Plots** (saved as `<model_name>_classification.png`):

- Confusion matrix heatmap
- Per-class accuracy bar chart

**CSV** (`<model_name>_classification_results.csv`): one row per test sample with `label_true_idx`, `label_hat_idx`, `label_true`, `label_hat`, `correct`.

### Model loading

The eval script uses two loaders depending on the model type:

- **`load_model`**: for end-to-end models (MLP, S4D). Parses the LightningCLI YAML, instantiates the bare `nn.Module`, and strips the `"model."` prefix from the checkpoint's state dict.
- **`load_task`**: for frozen-backbone models. Instantiates the full LightningModule (which in turn loads and freezes the backbone from its own checkpoint path embedded in the YAML).

---

## 5. Running the Pipeline

### Prerequisites

The two Parquet files (~225 MB total) must be present at `data/TESS/.cache/TESS/` relative to the repo root before any training or evaluation is run. Download them with:

```bash
bash benchmarks/TESS/setup_data.sh
```

This calls `data/TESS/download_tess.py` via `uv run --extra jax` (`huggingface_hub` lives in the `jax` extras; `pyarrow` is a core dep). The script fetches both files from the [PhyTS-bench HuggingFace dataset](https://huggingface.co/datasets/PhyTS-team/PhyTS-bench), prints a schema summary and sample row for each file, and is idempotent (HuggingFace Hub skips files already cached). **On Engaging, run this from the login node** — compute nodes do not have outbound internet access.

Verify afterwards:
```bash
ls data/TESS/.cache/TESS/
# tess_regression.parquet  tess_classification.parquet
```

### Local smoke test (CPU, 1 epoch)

Verifies data loading, model construction, and the training loop all wire together correctly:

```bash
uv run python main.py fit \
    --config configs/TESS/other/train_tess_mlp_regression.yaml \
    --trainer.max_epochs 1 \
    --trainer.accelerator cpu \
    --trainer.logger false
```

### Manual full run (local GPU)

The three stages have the following dependency structure — reconstruction must finish first:

```
[reconstruction]           (required by frozen-backbone runs)
[mlp_regression]  ─────┐
[mlp_classification] ──┤
[s4d_regression]  ─────┤──> [eval_pipeline]
[s4d_classification] ──┤
[s4d_head_regression]  ┤  (waits for reconstruction)
[s4d_head_classification]┘ (waits for reconstruction)
```

```bash
# Stage 1: Reconstruction pretraining
uv run python main.py fit --config configs/TESS/other/train_tess_s4d_reconstruction.yaml

# Stage 2: End-to-end (run in any order, or in parallel in separate terminals)
uv run python main.py fit --config configs/TESS/other/train_tess_mlp_regression.yaml
uv run python main.py fit --config configs/TESS/other/train_tess_s4d_regression.yaml
uv run python main.py fit --config configs/TESS/other/train_tess_mlp_classification.yaml
uv run python main.py fit --config configs/TESS/other/train_tess_s4d_classification.yaml

# Stage 3: Frozen backbone (after Stage 1 completes)
uv run python main.py fit --config configs/TESS/other/train_tess_s4d_head_regression.yaml
uv run python main.py fit --config configs/TESS/other/train_tess_s4d_head_classification.yaml

# Evaluation
uv run python benchmarks/TESS/eval_pipeline.py \
    --data_dir data/TESS/.cache/TESS \
    --out_dir  benchmarks/TESS
```

### MIT Engaging cluster (SLURM)

**Step 1 — Sync the repository** (from your local machine):
```bash
rsync -r -av --progress \
    --exclude .git --exclude '.venv' --exclude '.claude' \
    --exclude checkpoints --exclude '*.parquet' --exclude '*.csv' \
    TimeSeriesPhysics allisone@orcd-login001.mit.edu:/home/allisone/documents/UROP_2025_Summer/
```

**Step 2 — Download data** (from the Engaging login node; compute nodes have no internet):
```bash
bash benchmarks/TESS/setup_data.sh
```
This is idempotent — re-running it skips files that already exist.

**Step 3 — Submit the pipeline**:
```bash
bash benchmarks/TESS/run.sh
```

`run.sh` verifies data is present before submitting anything, then submits 8 SLURM jobs with automatic dependency wiring via `--dependency=afterok:<job_id>`. All jobs run under:

```
--gpus=1 --nodes=1 --ntasks-per-node=1 --cpus-per-task=8 --mem=100G --time=6:00:00
```

Both scripts derive the repo root from their own location (`dirname "$0"/../..`) and use it for all paths, so they work regardless of where you call them from.

Monitor with `squeue -u $USER`. Per-job stdout/stderr logs go to `benchmarks/TESS/logs/`.

### Eval only (skip training)

```bash
# Regression only
uv run python benchmarks/TESS/eval_pipeline.py --skip_classification

# Classification only
uv run python benchmarks/TESS/eval_pipeline.py --skip_regression
```

---

## 6. What is Not Included

**LinOSS.** The LinOSS model is implemented in JAX/Equinox and uses a separate training infrastructure. It cannot share weights with PyTorch MLP heads, has no seq2seq mode, and is not wired into this pipeline. End-to-end LinOSS benchmarking would require adapting the existing JAX training loop (`toy_regression_jax.py` pattern) to the TESS data format.