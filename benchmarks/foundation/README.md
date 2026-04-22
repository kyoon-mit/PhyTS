# Foundation-model benchmark

Zero-shot and fine-tuned evaluation of pretrained time-series foundation models
(MOMENT, Chronos, TimesFM, Time-MoE, MOIRAI, Lag-Llama) against the same
denoising / forecasting / embedding-regression tasks used by the toy benchmark.

This directory is a **standalone `uv` project** — separate from the repo root
because these models pin conflicting deps (old `torch`, `transformers`,
`numpy`) that cannot coexist with the main Python-3.12 env. It has its own
`pyproject.toml`, its own `uv.lock`, and its own `.venv/`.

## Setup

From the repo root:

```bash
uv --project benchmarks/foundation sync
```

This creates `benchmarks/foundation/.venv`, installs torch 2.4.1 from the cu121
wheel index, pulls Lag-Llama from GitHub, and applies the `transformers` /
`numpy` overrides needed to satisfy MOMENT's old pins (see
`[tool.uv] override-dependencies` in `pyproject.toml` — add entries here if a
new foundation lib introduces another pin conflict).

One-time Lag-Llama checkpoint download:

```bash
huggingface-cli download time-series-foundation-models/Lag-Llama lag-llama.ckpt \
    --local-dir <your_checkpoints_dir>/lag-llama/
```

## Running

Always invoke through this project's env so the right torch / transformers
versions are used:

```bash
uv --project benchmarks/foundation run \
    python benchmarks/foundation/run_benchmark.py \
    --models moment --tasks forecasting denoising embedding \
    --mode zero_shot \
    --data_dir data/toy/sinusoidal_signal_white_noise \
    --out_dir  plots/toy/foundation
```

Add `--smoke_test` for a 2-batch end-to-end sanity check.

<!-- The denoising evaluator reuses the **same** `reg_raw` / `reg_clean` checkpoints
trained by the toy benchmark, so results land in `plots/toy/<model>_<mode>/`
and can be aggregated alongside trained denoisers via
`benchmarks/toy/compare_pipelines.py`. -->

## Structure

```
wrappers/      # One per model — subclass BaseFoundationModel (forecast/denoise/embed)
evaluators/    # Task harnesses: forecasting.py, denoising.py, embedding_regression.py
finetuning/    # frozen / last_n / lora adapters for embedding-regression only
results/       # summary.csv writer + plotting helpers
run_benchmark.py
```
