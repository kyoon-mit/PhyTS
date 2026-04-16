"""
Unified foundation-model benchmarking pipeline.

Usage
-----
  # Zero-shot: MOMENT, all 3 tasks
  python benchmarks/foundation/run_benchmark.py \
      --data_dir data/toy/sinusoidal_signal_white_noise \
      --out_dir  plots/toy/foundation \
      --models   moment \
      --tasks    forecasting denoising embedding \
      --mode     zero_shot \
      --baseline_regressor_raw_ckpt   checkpoints/toy_mlp_regression_raw/best.ckpt \
      --baseline_regressor_raw_cfg    configs/toy/train_toy_mlp_regression_raw.yaml \
      --baseline_regressor_clean_ckpt checkpoints/toy_mlp_regression_clean/best.ckpt \
      --baseline_regressor_clean_cfg  configs/toy/train_toy_mlp_regression_clean.yaml

  # Smoke test: only 2 batches
  python benchmarks/foundation/run_benchmark.py \
      --models moment --tasks forecasting --mode zero_shot \
      --smoke_test
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
import yaml

# ── Make sibling packages importable whether the script is run as a module
# or directly (common for benchmark scripts in this repo).
_HERE = Path(__file__).resolve().parent
_REPO = _HERE.parent.parent
for p in (str(_REPO), str(_HERE.parent), str(_REPO / "src")):
    if p not in sys.path:
        sys.path.insert(0, p)

from dataloader.toy_dataloader import ToyDataModule, Param
from foundation.evaluators.forecasting import evaluate_forecasting
from foundation.evaluators.denoising import evaluate_denoising
from foundation.evaluators.embedding_regression import evaluate_embedding_regression
from foundation.results.writer import (
    append_summary_row,
    save_denoise_csv,
    save_embedding_csv,
    save_forecast_csv,
)


# ────────────────────────────────────────────────────────────────────────────
# Registry: model name → wrapper class.  Imports are lazy so that missing
# optional deps don't break unrelated model runs.
# ────────────────────────────────────────────────────────────────────────────

def _get_wrapper(name: str):
    name = name.lower()
    if name == "moment":
        from foundation.wrappers.moment_wrapper import MomentWrapper
        return MomentWrapper()
    if name == "chronos":
        from foundation.wrappers.chronos_wrapper import ChronosWrapper
        return ChronosWrapper()
    if name == "timesfm":
        from foundation.wrappers.timesfm_wrapper import TimesFMWrapper
        return TimesFMWrapper()
    if name == "timemoe":
        from foundation.wrappers.timemoe_wrapper import TimeMoEWrapper
        return TimeMoEWrapper()
    if name == "moirai":
        from foundation.wrappers.moirai_wrapper import MoiraiWrapper
        return MoiraiWrapper()
    if name in ("lagllama", "lag_llama", "lag-llama"):
        from foundation.wrappers.lagllama_wrapper import LagLlamaWrapper
        return LagLlamaWrapper()
    raise ValueError(f"Unknown model '{name}'")


# ────────────────────────────────────────────────────────────────────────────
# Baseline regressor loading (reuses existing pattern from eval_pipeline.py)
# ────────────────────────────────────────────────────────────────────────────

def _load_trained_model(ckpt_path: str, cfg_path: str, device: torch.device):
    """Mirror of `benchmarks/toy/eval_pipeline.py::load_model`."""
    import importlib
    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)
    model_cfg = cfg["model"]["init_args"]["model"]
    class_path = model_cfg["class_path"]
    init_args = model_cfg.get("init_args", {})
    module_name, class_name = class_path.rsplit(".", 1)
    cls = getattr(importlib.import_module(module_name), class_name)
    model = cls(**init_args)
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=True)
    sd = {k.replace("model.", ""): v for k, v in ckpt["state_dict"].items()
          if k.startswith("model.")}
    model.load_state_dict(sd)
    model.eval()
    return model.to(device), cfg


# ────────────────────────────────────────────────────────────────────────────
# Main
# ────────────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir", default="data/toy/sinusoidal_signal_white_noise")
    p.add_argument("--out_dir",  default="plots/toy/foundation")
    p.add_argument("--models",   nargs="+", required=True,
                   help="One or more of: moment chronos timesfm timemoe moirai lagllama")
    p.add_argument("--tasks",    nargs="+", default=["forecasting"],
                   choices=["forecasting", "denoising", "embedding"])
    p.add_argument("--mode",     choices=["zero_shot", "finetuned", "both"],
                   default="zero_shot")
    p.add_argument("--model_size", default="base")
    p.add_argument("--device",   default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--num_workers", type=int, default=0)

    # Forecasting-specific
    p.add_argument("--context_len", type=int, default=512)
    p.add_argument("--horizon",     type=int, default=128)
    p.add_argument("--sample_rate", type=float, default=64.0)

    # Embedding-specific
    p.add_argument("--head_epochs", type=int, default=50,
                   help="Epochs to train the MLP head on frozen embeddings")
    p.add_argument("--target_params", nargs="+",
                   default=["amplitude", "frequency_hz", "phase_rad"])

    # Fine-tuning
    p.add_argument("--finetune_epochs", type=int, default=20)
    p.add_argument("--finetune_lr",     type=float, default=1e-4)
    p.add_argument("--finetune_adapter",
                   choices=["frozen", "last_n", "lora"],
                   default="last_n",
                   help="Backbone adapter strategy for fine-tuning")
    p.add_argument("--finetune_unfreeze_last_n", type=int, default=2,
                   help="For adapter=last_n: number of final blocks to unfreeze")
    p.add_argument("--lora_r",       type=int, default=8)
    p.add_argument("--lora_alpha",   type=int, default=16)
    p.add_argument("--lora_dropout", type=float, default=0.0)
    p.add_argument("--lora_last_n_blocks", type=int, default=None,
                   help="For adapter=lora: restrict injection to last N blocks; "
                        "default None = whole backbone")

    # Denoising baseline regressors (required for the denoising task so the
    # output CSV is schema-compatible with compare_pipelines.py).
    p.add_argument("--baseline_regressor_raw_ckpt",   default=None)
    p.add_argument("--baseline_regressor_raw_cfg",    default=None)
    p.add_argument("--baseline_regressor_clean_ckpt", default=None)
    p.add_argument("--baseline_regressor_clean_cfg",  default=None)

    # Model-specific overrides
    p.add_argument("--lagllama_ckpt", default=None,
                   help="Path to lag-llama.ckpt (auto-discovered if not set)")

    # Smoke-test knobs
    p.add_argument("--smoke_test", action="store_true",
                   help="Limit each split to a few batches for a quick end-to-end run")
    p.add_argument("--max_train_batches", type=int, default=None)
    p.add_argument("--max_test_batches",  type=int, default=None)

    args = p.parse_args()

    if args.smoke_test:
        args.max_train_batches = args.max_train_batches or 2
        args.max_test_batches  = args.max_test_batches  or 2

    device = torch.device(args.device)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    summary_path = out_dir / "summary.csv"
    # Append to existing summary.csv so concurrent Condor jobs don't clobber
    # each other.  If you want a fresh run, delete the file manually first.

    dm = ToyDataModule(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )
    dm.setup("fit")
    dm.setup("test")
    train_loader = dm.train_dataloader()
    val_loader = dm.val_dataloader()
    test_loader = dm.test_dataloader()

    # ── Dispatch per model × task × mode ─────────────────────────────────
    modes = ["zero_shot", "finetuned"] if args.mode == "both" else [args.mode]

    for model_name in args.models:
        print(f"\n{'=' * 70}\n[{model_name}] loading (size={args.model_size}, device={device})\n{'=' * 70}")
        wrapper = _get_wrapper(model_name)
        # MOMENT needs forecast_horizon at load time; pass via kwargs if supported.
        # Include context_len/horizon so GluonTS-based models (MOIRAI, LagLlama)
        # pick up the right values; other wrappers absorb them via **_unused.
        load_kwargs = {
            "device": args.device,
            "model_size": args.model_size,
            "context_len": args.context_len,
            "horizon": args.horizon,
        }
        # Model-specific extras (absorbed by **_unused if not relevant)
        if args.lagllama_ckpt:
            load_kwargs["ckpt_path"] = args.lagllama_ckpt
        try:
            wrapper.load(forecast_horizon=args.horizon, seq_len=args.context_len, **load_kwargs)
        except TypeError:
            wrapper.load(**load_kwargs)

        for mode in modes:
            print(f"\n[{model_name}] mode={mode}")
            if mode == "finetuned":
                # Fine-tuning is implemented per task in foundation.finetuning.
                # For now we error out cleanly so the zero-shot path remains usable.
                try:
                    from foundation.finetuning.finetune import finetune
                except ImportError:
                    print(f"  ! fine-tuning module not yet implemented; skipping {model_name}/{mode}")
                    continue
                finetune(
                    wrapper,
                    train_loader=train_loader,
                    val_loader=val_loader,
                    tasks=args.tasks,
                    epochs=args.finetune_epochs,
                    lr=args.finetune_lr,
                    device=args.device,
                    context_len=args.context_len,
                    horizon=args.horizon,
                    adapter=args.finetune_adapter,
                    unfreeze_last_n=args.finetune_unfreeze_last_n,
                    lora_r=args.lora_r,
                    lora_alpha=args.lora_alpha,
                    lora_dropout=args.lora_dropout,
                    lora_last_n_blocks=args.lora_last_n_blocks,
                )

            # ── FORECASTING ──────────────────────────────────────────────
            if "forecasting" in args.tasks:
                print(f"  - forecasting (C={args.context_len}, H={args.horizon})")
                res = evaluate_forecasting(
                    wrapper, test_loader,
                    context_len=args.context_len,
                    horizon=args.horizon,
                    sample_rate=args.sample_rate,
                    max_batches=args.max_test_batches,
                )
                model_out_dir = out_dir / f"{model_name}_{mode}"
                model_out_dir.mkdir(parents=True, exist_ok=True)
                save_forecast_csv(
                    model_out_dir,
                    model=model_name,
                    mode=mode,
                    snr_gt=res["snr_gt"],
                    freq_true=res["freq_true"],
                    per_sample_metrics=res["per_sample"][wrapper.name],
                )
                for metric, agg in res["aggregate"][wrapper.name].items():
                    append_summary_row(
                        summary_path, model=model_name, task="forecasting", mode=mode,
                        metric=metric, value=agg["mean"],
                        ci_lower=agg["ci_lower"], ci_upper=agg["ci_upper"],
                    )
                # Also log each naive baseline once per run (same for zero_shot/finetuned;
                # only log during the first mode).
                if mode == modes[0]:
                    for baseline_name, metrics in res["aggregate"].items():
                        if baseline_name == wrapper.name:
                            continue
                        for metric, agg in metrics.items():
                            append_summary_row(
                                summary_path, model=baseline_name, task="forecasting",
                                mode="zero_shot", metric=metric, value=agg["mean"],
                                ci_lower=agg["ci_lower"], ci_upper=agg["ci_upper"],
                            )

            # ── DENOISING ────────────────────────────────────────────────
            if "denoising" in args.tasks:
                if not (args.baseline_regressor_raw_ckpt
                        and args.baseline_regressor_raw_cfg
                        and args.baseline_regressor_clean_ckpt
                        and args.baseline_regressor_clean_cfg):
                    print(
                        "  ! denoising task skipped: missing "
                        "--baseline_regressor_{raw,clean}_{ckpt,cfg}"
                    )
                else:
                    print("  - denoising (→ regressors → compare-compatible results.csv)")
                    reg_raw,   _       = _load_trained_model(
                        args.baseline_regressor_raw_ckpt,
                        args.baseline_regressor_raw_cfg, device,
                    )
                    reg_clean, reg_cfg = _load_trained_model(
                        args.baseline_regressor_clean_ckpt,
                        args.baseline_regressor_clean_cfg, device,
                    )
                    target_params = reg_cfg["model"]["init_args"]["target_params"]
                    target_idx = [int(Param[p]) for p in target_params]
                    res = evaluate_denoising(
                        wrapper, test_loader,
                        regressor_raw=reg_raw, regressor_clean=reg_clean,
                        target_idx=target_idx, device=args.device,
                        max_batches=args.max_test_batches,
                    )
                    # Write to plots/toy/{model}_{mode}/results.csv so
                    # compare_pipelines.py picks it up.
                    compare_dir = Path("plots/toy") / f"{model_name}_{mode}"
                    save_denoise_csv(
                        compare_dir,
                        y_true=res["y_true"],
                        y_raw=res["y_raw"],
                        y_den=res["y_den"],
                        y_signal=res["y_signal"],
                        snr_gt=res["snr_gt"],
                        snr_raw=res["snr_raw"],
                        snr_denoised=res["snr_denoised"],
                        param_names=target_params,
                    )
                    import numpy as np
                    append_summary_row(
                        summary_path, model=model_name, task="denoising", mode=mode,
                        metric="mse", value=float(np.mean(res["mse_per_sample"])),
                    )
                    append_summary_row(
                        summary_path, model=model_name, task="denoising", mode=mode,
                        metric="psd_mse", value=float(np.mean(res["psd_mse_per_sample"])),
                    )
                    append_summary_row(
                        summary_path, model=model_name, task="denoising", mode=mode,
                        metric="snr_denoised_mean",
                        value=float(np.mean(res["snr_denoised"])),
                    )

            # ── EMBEDDING + REGRESSION ───────────────────────────────────
            if "embedding" in args.tasks:
                if not wrapper.supports_embed:
                    print(f"  ! {model_name} does not support embedding; skipping")
                else:
                    print(f"  - embedding + regression (targets={args.target_params})")
                    res = evaluate_embedding_regression(
                        wrapper,
                        train_loader=train_loader,
                        val_loader=val_loader,
                        test_loader=test_loader,
                        target_params=args.target_params,
                        device=args.device,
                        epochs=args.head_epochs,
                        max_train_batches=args.max_train_batches,
                        max_test_batches=args.max_test_batches,
                    )
                    model_out_dir = out_dir / f"{model_name}_{mode}"
                    save_embedding_csv(
                        model_out_dir, model=model_name, mode=mode,
                        per_param_metrics=res["per_param_metrics"],
                    )
                    for param, metrics in res["per_param_metrics"].items():
                        for metric, value in metrics.items():
                            append_summary_row(
                                summary_path, model=model_name, task="embedding",
                                mode=mode, metric=f"{param}/{metric}", value=value,
                            )

        wrapper.unload()

    print(f"\nDone. Summary: {summary_path}")


if __name__ == "__main__":
    main()
