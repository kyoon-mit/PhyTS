# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## New Setup:

Use `uv` as the package manager

### Old  Setup

Two Python environments are used:

- **`tsenv`** (Python 3.12) — main training/eval env. Created from `env.yaml`, installs this package via `pip install -e .`. Used for everything in `src/`, `configs/`, and the TIDMAD/toy/LIGO benchmarks. SLURM scripts activate it as `conda activate ts_cuda312` on the cluster.
- **`fm` env** (Python 3.10) — foundation-model wrappers in `benchmarks/foundation/`. Dependencies (torch 2.4.1, momentfm, chronos-forecasting, timesfm, uni2ts, lag-llama, etc.) are pinned in `benchmarks/foundation/requirements_fm.txt`. Lag-Llama is installed separately from its GitHub repo and requires `setuptools<72`.

The foundation benchmark is kept on a separate, older Python/torch stack because several of the pretrained models have conflicting pinned deps with the main env — do not try to merge them into `env.yaml`.

## Common commands

```bash
# Train (any task) via LightningCLI
python main.py fit --config configs/<domain>/<file>.yaml

# CLI overrides (jsonargparse syntax)
python main.py fit --config configs/toy/train_toy_mlp_regression_raw.yaml \
  --model.init_args.lr 5e-4 \
  --data.init_args.batch_size 128

# Save a "trained" checkpoint for a parameter-free model (e.g. classical filters)
python tools/save_checkpoint.py --cfg <yaml> --out checkpoints/<name>/best.ckpt

# Toy end-to-end benchmark (interactive SLURM menu: denoisers / regressors / eval / full)
bash benchmarks/toy/run.sh

# TIDMAD end-to-end benchmark (chained SLURM: denoiser → reg_raw+reg_clean → eval)
bash benchmarks/TIDMAD/run.sh

# TIDMAD one-time preprocessing (raw H5 → memory-mappable .npy)
python data/TIDMAD/preprocess_tidmad.py --data_dir data/TIDMAD/original --out_dir data/TIDMAD/preprocessed

# Foundation-model benchmark (zero-shot or fine-tuned)
python benchmarks/foundation/run_benchmark.py \
    --models moment --tasks forecasting denoising embedding --mode zero_shot \
    --data_dir data/toy/sinusoidal_signal_white_noise --out_dir plots/toy/foundation
# Add --smoke_test for a 2-batch end-to-end sanity check.
```

There is no lint or test command wired up; `ruff` is configured (`line-length = 100`) but not invoked automatically.

## Architecture

### LightningCLI is the single entry point

`main.py` is just `LightningCLI(subclass_mode_model=True)`. All training runs are driven by a YAML in `configs/<domain>/` that wires together three subclassed objects:

```
model  → tasks.<domain>.<task>.<TaskClass>          (LightningModule)
  └── model: models.<arch>.<ModelClass>             (plain nn.Module)
data   → dataloader.<domain>_dataloader.<DataModule>
```

The task class accepts `model: nn.Module` as a constructor arg, so swapping architectures only means editing the nested `model.init_args.model` block in the YAML. This is the key pattern to preserve when adding new domains/models.

### Directory layout (the "why")

- `src/tasks/<domain>/` — LightningModules organized **by domain, not by model**. Same model class is reused across domains; the task class encodes the dataset-specific loss/batch layout (e.g. `ToyDataset` returns `(sig_bkg, sig, params)`, `TIDMAD` returns something else).
- `src/models/` — plain `nn.Module`s. S4D is split into `s4d.py` (pooled classifier) and `s4d_seq2seq.py` (per-step output for denoising). New architectures go here and should not import from `tasks/`.
- `src/dataloader/` — one `LightningDataModule` per domain. `Param` IntEnums (e.g. `dataloader.toy_dataloader.Param`) name columns of the per-sample `params` tensor.
- `configs/<domain>/` — one YAML per (model, task, variant). Naming convention: `train_<domain>_<model>_<task>[_<losstag>].yaml`.
- `benchmarks/<domain>/run.sh` — SLURM submission wrappers; chain jobs via `--dependency=afterok:<id>`.
- `checkpoints/<run_name>/best.ckpt` — written by each config's `ModelCheckpoint` callback; eval pipelines reload these by walking YAML `model.init_args.model` + `torch.load(..., weights_only=True)` and stripping the `model.` prefix (see `benchmarks/toy/eval_pipeline.py::load_model`).

### The denoising-vs-regression benchmark shape

Both `benchmarks/toy/` and `benchmarks/TIDMAD/` compare three pipelines:

| Pipeline | Eval input to regressor |
|---|---|
| **raw** | noisy `sig_bkg` → `reg_raw` (trained on noisy) |
| **denoised** | `denoiser(sig_bkg)` → `reg_clean` (trained on clean) |
| **oracle** | clean `sig` → `reg_clean` |

`reg_clean` is deliberately trained on clean signals and used at eval time on the denoiser's output — the gap between the three rows tells you how much of the error is from the denoiser vs. inherent in the regression task. Keep this invariant when adding new denoisers: they plug into the `denoised` pipeline without retraining `reg_clean`.

### Foundation-model benchmark

`benchmarks/foundation/` is a separate, self-contained subsystem with its own `__init__.py` and is importable by running `run_benchmark.py` from the repo root (it mutates `sys.path` to add `src/` and the repo root). The architecture:

- `wrappers/base.py::BaseFoundationModel` — abstract class exposing `forecast`, `denoise`, `embed`. Defaults: `denoise_via_forecast` (bidirectional context-split forecasting) and `embed_via_hidden_states` (forward hook + mean-pool) so wrappers only need to implement what's natively supported.
- `wrappers/<name>_wrapper.py` — one per model (moment, chronos, timesfm, timemoe, moirai, lagllama). Registered lazily in `run_benchmark._get_wrapper` so a missing optional dep doesn't break unrelated runs.
- `evaluators/` — task-specific evaluation harnesses (`forecasting.py`, `denoising.py`, `embedding_regression.py`). The denoising evaluator loads the **same** `reg_raw` / `reg_clean` checkpoints from the toy benchmark and writes a `results.csv` to `plots/toy/<model>_<mode>/` so `benchmarks/toy/compare_pipelines.py` can aggregate foundation-model results alongside trained denoisers.
- `finetuning/finetune.py` — narrowly scoped to the embedding-regression path: `frozen`, `last_n` (unfreeze last N transformer blocks), and `lora` (inject LoRA adapters into attention linears). Forecasting/denoising fine-tuning is intentionally **not** implemented — each model's task head is too library-specific to unify.
- `results/{writer,plotting}.py` — `append_summary_row` appends to `summary.csv` so concurrent SLURM jobs don't clobber each other. Delete `summary.csv` manually to start a fresh sweep.

## Conventions

- Raw data directories (`data/TIDMAD/original/`, `data/TIDMAD/preprocessed/`, `*.h5`, `*.npz`) are gitignored — they must be symlinked or regenerated. See the per-directory READMEs in `data/`.
- Plots committed to git live under `plots/`; the `.gitignore` explicitly allow-lists `plots/**/*.png` against the global `*.png` ignore.
- Wandb is the default logger (`project: TimeSeriesPhysics`). Configs set `logger.class_path: lightning.pytorch.loggers.WandbLogger`.
- S4D code in `src/models/s4d*.py` is derived from [state-spaces/s4](https://github.com/ML4GW) under Apache 2.0. The `NOTICE` file is the full attribution — preserve its header on derived files.
