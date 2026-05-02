# TESS Benchmarking Pipeline

Benchmarks sequence models end-to-end on two TESS variable-star tasks (MLP, S4D,
CNN, Conv+Attention, Transformer, and JAX LinOSS; see configs and
`benchmarks/TESS/README_Sweep.md` for multi-architecture wandb sweeps):

- **Regression**: predict stellar rotation frequency (`frot`) from flux
- **Classification**: predict variability class (8 labels) from flux

---

## What was built


| File                                    | Purpose                                                                                                   |
| --------------------------------------- | --------------------------------------------------------------------------------------------------------- |
| `src/dataloader/tess_dataloader.py`     | Datasets + DataModules for regression, classification, and reconstruction pretraining                     |
| `src/tasks/TESS/tess_regression.py`     | `TESSRegressionMSE` (end-to-end) and `TESSFrozenBackboneRegressionMSE` (frozen S4D + MLP head)            |
| `src/tasks/TESS/tess_classification.py` | `TESSClassificationCE` and `TESSFrozenBackboneClassificationCE`                                           |
| `src/tasks/TESS/tess_reconstruction.py` | `TESSReconstructionMSE` — self-supervised S4D pretraining via denoising                                   |
| `configs/TESS/other/`                   | LightningCLI training configs (see table below)                                                           |
| `configs/TESS/sweep/`                   | wandb sweep YAMLs + `all_model_sweep_dims.yaml`                                                           |
| `data/TESS/download_tess.py`            | Download PhyTS-bench TESS shards from Hugging Face; `--where local` / `--where engaging` or `--cache-dir` |
| `benchmarks/TESS/setup_data.sh`         | Downloads TESS Parquet files from HuggingFace (run once from login node)                                  |
| `benchmarks/TESS/run.sh`                | SLURM submission script for MIT Engaging                                                                  |
| `benchmarks/TESS/eval_pipeline.py`      | Standalone evaluation: metrics, scatter/confusion plots, CSVs                                             |


### Training configs

All files below are under `configs/TESS/other/`.


| Config                                       | Model                 | Task                                                |
| -------------------------------------------- | --------------------- | --------------------------------------------------- |
| `train_tess_s4d_reconstruction.yaml`         | S4ModelSeq2Seq        | Denoising pretraining (run first)                   |
| `train_tess_mlp_regression.yaml`             | MLP                   | End-to-end regression                               |
| `train_tess_s4d_regression.yaml`             | S4D                   | End-to-end regression                               |
| `train_tess_linoss_regression.yaml`          | LinOSS (JAX)          | End-to-end regression (`uv sync --extra jax`)       |
| `train_tess_transformer_classification.yaml` | Transformer           | End-to-end classification                           |
| `train_tess_s4d_head_regression.yaml`        | Frozen S4D + MLP head | Regression (requires reconstruction checkpoint)     |
| `train_tess_mlp_classification.yaml`         | MLP                   | End-to-end classification                           |
| `train_tess_s4d_classification.yaml`         | S4D                   | End-to-end classification                           |
| `train_tess_linoss_classification.yaml`      | LinOSS (JAX)          | End-to-end classification (`uv sync --extra jax`)   |
| `train_tess_s4d_head_classification.yaml`    | Frozen S4D + MLP head | Classification (requires reconstruction checkpoint) |


CNN/ConvAttn and additional LinOSS/transformer widths are exercised via `**configs/TESS/sweep/sweep_{cls|reg}_*.yaml`** and `**benchmarks/TESS/sweep_{classification|regression}.py**` (wandb sweep), not standalone `train_*` YAMLs here.

### Data preprocessing

- Sequences cropped or zero-padded to `seq_len=1100`; bool mask marks valid positions
- Per-sample z-score normalization: `(flux − median) / std`
- Splits: 70 / 15 / 15 train / val / test, stratified by label (classification) or frot decile (regression), fixed `seed=42`

### Downloading TESS Parquet shards

Training expects Parquet files directly under `<cache_parent>/TESS/`. The Hugging Face dataset hosts shards under `TESS/split/`; `download_tess.py` moves them to `TESS/*.parquet` and removes the local `split` folder. Use `<cache_parent>/TESS` as `data_dir` in configs and evaluation.

Use `data/TESS/download_tess.py` (requires `uv sync --extra jax`):


| CLI                       | Cache parent                                                                                        |
| ------------------------- | --------------------------------------------------------------------------------------------------- |
| `--where local` (default) | `<repo>/data/TESS/.cache` → Parquets in `data/TESS/.cache/TESS/`                                    |
| `--where engaging`        | `/home/allisone/orcd/pool/UROP_2025_Summer/TimeSeriesPhysics/data_engaging/TESS/.cache` → `…/TESS/` |
| `--cache-dir PATH`        | Any directory (overrides `--where`); Parquets end up in `PATH/TESS/`                                |


```bash
uv run --extra jax python data/TESS/download_tess.py
uv run --extra jax python data/TESS/download_tess.py --where engaging
```

On Engaging, `bash benchmarks/TESS/setup_data.sh` calls the same script with `--cache-dir` under the pool tree (`$TESS_POOL_ROOT/data_engaging/TESS/.cache`). The default `TESS_POOL_ROOT` matches `--where engaging`. If you change `TESS_POOL_ROOT`, keep using `setup_data.sh` or pass `--cache-dir` yourself rather than `--where engaging`.

---

## Running locally

**Smoke test** (no GPU, 1 epoch):

```bash
cd /path/to/TimeSeriesPhysics
uv run python main.py fit \
    --config configs/TESS/other/train_tess_mlp_regression.yaml \
    --trainer.max_epochs 1 \
    --trainer.accelerator cpu \
    --trainer.logger false
```

**Full training order** (if running manually):

1. Reconstruction pretraining (required before frozen-backbone runs):

```bash
uv run python main.py fit --config configs/TESS/other/train_tess_s4d_reconstruction.yaml
```

1. End-to-end models (independent, can run in any order):

```bash
uv run python main.py fit --config configs/TESS/other/train_tess_mlp_regression.yaml
uv run python main.py fit --config configs/TESS/other/train_tess_s4d_regression.yaml
uv run python main.py fit --config configs/TESS/other/train_tess_mlp_classification.yaml
uv run python main.py fit --config configs/TESS/other/train_tess_s4d_classification.yaml
```

1. Frozen-backbone models (after step 1 completes):

```bash
uv run python main.py fit --config configs/TESS/other/train_tess_s4d_head_regression.yaml
uv run python main.py fit --config configs/TESS/other/train_tess_s4d_head_classification.yaml
```

**Evaluation** (after any subset of models are trained):

```bash
uv run python benchmarks/TESS/eval_pipeline.py \
    --data_dir data/TESS/.cache/TESS \
    --out_dir  benchmarks/TESS
```

Writes `*_regression_results.csv`, `*_classification_results.csv`, and `.png` plots into `--out_dir`. Pass `--skip_regression` or `--skip_classification` to run only one task. Models with missing checkpoints are skipped automatically.

---

## Running on MIT Engaging

First, sync the repository to Engaging (login node):

```bash
rsync -r -av --progress \
    --exclude .git --exclude '.venv' \
    --exclude checkpoints --exclude '*.parquet' --exclude '*.csv' --exclude '*.npz' \
    --exclude '*.wandb' --exclude '*.err' --exclude '*.out' --exclude '*/logs/' --exclude wandb --exclude '*/.cache' \
    TimeSeriesPhysics allisone@orcd-login001.mit.edu:/home/allisone/documents/UROP_2025_Summer/
```

Sync results or code back from Engaging to your local machine (run this on your local machine):

```bash
rsync -r -av --progress \
    --exclude .git --exclude '.venv' \
    --exclude checkpoints --exclude '*.parquet' --exclude '*.csv' --exclude '*.npz' \
    --exclude '*.wandb' --exclude '*.err' --exclude '*.out' --exclude '*/logs/' --exclude wandb \
    allisone@orcd-login001.mit.edu:/home/allisone/documents/UROP_2025_Summer/TimeSeriesPhysics ./
```

Then, from the Engaging login node, download the data (only needed once; login nodes have internet, compute nodes do not):

```bash
bash benchmarks/TESS/setup_data.sh
```

Then submit the full pipeline:

```bash
bash benchmarks/TESS/run.sh
```

This submits 8 SLURM jobs with the correct dependency chain:

- `JOB_RECON` runs first
- `JOB_HEAD_REG` and `JOB_HEAD_CLS` wait for `JOB_RECON`
- All end-to-end jobs run in parallel
- `JOB_EVAL` waits for all training jobs

Monitor progress with `squeue -u $USER`. Logs are written to `benchmarks/TESS/logs/`.

Both scripts derive the repo root from their own location (`dirname "$0"/../..`), so they work regardless of where you call them from.

---

## Notes

- **LinOSS** uses the JAX/Equinox path (`train_tess_linoss_*.yaml` or sweeps): it does not share the Lightning S4 seq2seq object, so it cannot use the **frozen S4 backbone + head** setup based on reconstruction pretraining (`train_tess_s4d_*_head_`*).
- The reconstruction pretraining adds synthetic Gaussian noise (`noise_std=0.3`) to the clean flux — the original flux serves as the reconstruction target.
- Frozen backbone: the S4D backbone is loaded and its weights are locked; only the MLP head is trained. Lightning's `model.train()` call is overridden to keep the backbone in eval mode (preserving dropout-off behavior) throughout head training.

