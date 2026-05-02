"""TESS regression hyperparameter sweep script.

Supports all Torch + JAX model types end-to-end predicting stellar rotation frequency (frot):
  mlp          — MLPRegressor (flattened seq → MLP)
  s4d          — S4Model (S4D stack + mean pool)
  cnn          — ConvAE (encoder + global avg pool + Linear(C, 1))
  cnn_attn     — ConvAttnAE (encoder + bottleneck MHA + pool + Linear(C, 1))
  transformer  — TransformerClassifier (encoder + masked mean pool + head)
  linoss_imex  — LinOSS with IMEX discretization (JAX/Equinox)
  linoss_damped — LinOSS with damped_IMEX discretization (JAX/Equinox)

All models output a single scalar; MSE loss is used for training.
Sweep optimization target: val/r2 (coefficient of determination, maximize).

Sweep configs live in configs/TESS/sweep/sweep_reg_*.yaml.

Usage
-----
# Create a sweep (once, on login node or local machine):
    wandb sweep configs/TESS/sweep/sweep_reg_mlp.yaml   # prints sweep_id

# Launch agents on Engaging:
    bash benchmarks/TESS/run_sweep.sh --model_type mlp --sweep_id <id>

# Single run for debugging (wandb disabled):
    WANDB_MODE=disabled uv run python benchmarks/TESS/sweep_regression.py \\
        --model_type s4d --size xs --lr 1e-3 --batch_size 32

# LinOSS requires the jax extra:
    uv run --extra jax python benchmarks/TESS/sweep_regression.py \\
        --model_type linoss_imex ...

Environment variables
---------------------
TESS_DATA_DIR   Override for --data_dir (default: data/TESS/.cache/TESS)
TESS_CKPT_DIR   Override for --ckpt_dir
"""

from __future__ import annotations

import argparse
import os
import sys
import types
from pathlib import Path

_BENCH_DIR = Path(__file__).resolve().parent
if str(_BENCH_DIR) not in sys.path:
    sys.path.insert(0, str(_BENCH_DIR))

from sweep_utils import (
    JAX_MODELS,
    ThrottledTQDMProgressBar,
    _REPO_ROOT,
    add_infra_args,
    add_sweep_args,
    build_cnn,
    build_cnn_attn,
    build_linoss,
    build_mlp,
    build_s4d,
    build_transformer,
    collect_benchmark_param_counters,
    dump_sweep_run_config,
    tess_sweep_artifact_dir,
)

if str(_REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT / "src"))

import lightning as L
import wandb
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from lightning.pytorch.loggers import CSVLogger, WandbLogger

from dataloader.tess_dataloader import TESSRegressionDataModule
from tasks.TESS.tess_regression import TESSRegressionMSE


# ── Task builders ─────────────────────────────────────────────────────────────

def _build_torch_task(model, lr: float, weight_decay: float) -> L.LightningModule:
    task = TESSRegressionMSE(model=model, lr=lr, lr_decay=0.99)
    _wd = weight_decay

    def _configure_optimizers(self):
        import torch.optim as optim
        opt = optim.AdamW(self.parameters(), lr=self.lr, weight_decay=_wd)
        sched = optim.lr_scheduler.ExponentialLR(opt, gamma=self.lr_decay)
        return {"optimizer": opt, "lr_scheduler": {"scheduler": sched, "interval": "epoch"}}

    task.configure_optimizers = types.MethodType(_configure_optimizers, task)
    return task


def _build_linoss_task(
    model, lr: float, weight_decay: float, seed: int
) -> L.LightningModule:
    from tasks.TESS.tess_linoss import TESSLinOSSRegressionMSE
    return TESSLinOSSRegressionMSE(
        model=model,
        lr=lr,
        weight_decay=weight_decay,
        clip_grad_norm=1.0,
        seed=seed,
    )


# ── Argument parsing ──────────────────────────────────────────────────────────

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="TESS regression sweep")
    add_infra_args(p)
    p.add_argument("--data_dir",
                   default=os.environ.get("TESS_DATA_DIR",
                                          "data/TESS/.cache/TESS"))
    add_sweep_args(p)
    return p.parse_args()


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    args = _parse_args()

    run = wandb.init(
        project="TimeSeriesPhysics",
        config={
            "model_type":   args.model_type,
            "size":         args.size,
            "lr":           args.lr,
            "dropout":      args.dropout,
            "weight_decay": args.weight_decay,
            "batch_size":   args.batch_size,
            "seed":         args.seed,
        },
    )
    cfg = wandb.config

    L.seed_everything(cfg.seed, workers=True)

    is_jax = cfg.model_type in JAX_MODELS

    # ── Build model & task ────────────────────────────────────────────────────
    if cfg.model_type == "mlp":
        model = build_mlp(cfg.size, d_output=1, seq_len=args.seq_len, dropout=cfg.dropout)
        task  = _build_torch_task(model, cfg.lr, cfg.weight_decay)

    elif cfg.model_type == "s4d":
        model = build_s4d(cfg.size, d_output=1, dropout=cfg.dropout)
        task  = _build_torch_task(model, cfg.lr, cfg.weight_decay)

    elif cfg.model_type == "cnn":
        model = build_cnn(cfg.size, d_output=1, dropout=cfg.dropout)
        task  = _build_torch_task(model, cfg.lr, cfg.weight_decay)

    elif cfg.model_type == "cnn_attn":
        model = build_cnn_attn(cfg.size, d_output=1, dropout=cfg.dropout)
        task  = _build_torch_task(model, cfg.lr, cfg.weight_decay)

    elif cfg.model_type == "transformer":
        model = build_transformer(cfg.size, d_output=1, seq_len=args.seq_len, dropout=cfg.dropout)
        task  = _build_torch_task(model, cfg.lr, cfg.weight_decay)

    elif cfg.model_type == "linoss_imex":
        model = build_linoss(
            cfg.size,
            d_output=1,
            discretization="IMEX",
            seed=cfg.seed,
            dropout=cfg.dropout,
        )
        task  = _build_linoss_task(model, cfg.lr, cfg.weight_decay, cfg.seed)

    elif cfg.model_type == "linoss_damped":
        model = build_linoss(
            cfg.size,
            d_output=1,
            discretization="damped_IMEX",
            seed=cfg.seed,
            dropout=cfg.dropout,
        )
        task  = _build_linoss_task(model, cfg.lr, cfg.weight_decay, cfg.seed)

    else:
        raise ValueError(f"Unknown model_type: {cfg.model_type!r}")

    wandb.summary.update(collect_benchmark_param_counters(cfg.model_type, task))

    run_id = run.id if run is not None else "local"
    art_dir = tess_sweep_artifact_dir("regression", cfg.model_type, run_id)
    dump_sweep_run_config(art_dir / "run_config.yaml", args, run)

    # ── Data ──────────────────────────────────────────────────────────────────
    dm = TESSRegressionDataModule(
        data_dir=args.data_dir,
        batch_size=cfg.batch_size,
        num_workers=args.num_workers,
        seq_len=args.seq_len,
    )

    # ── Callbacks ─────────────────────────────────────────────────────────────
    ckpt_dir = Path(os.environ.get("TESS_CKPT_DIR", args.ckpt_dir)) / "regression" / cfg.model_type / run_id

    early_stop = EarlyStopping(monitor="val/loss", patience=args.patience, mode="min")

    if is_jax:
        from tasks.TESS.tess_linoss import JAXModelCheckpoint
        ckpt_cb = JAXModelCheckpoint(dirpath=str(ckpt_dir), monitor="val/loss", mode="min")
    else:
        ckpt_cb = ModelCheckpoint(
            dirpath=str(ckpt_dir), filename="best",
            monitor="val/loss", mode="min", save_top_k=1,
        )

    # ── Trainer ───────────────────────────────────────────────────────────────
    csv_logger = CSVLogger(save_dir=str(art_dir), name="metrics_csv")
    logger = [csv_logger, WandbLogger(experiment=run)]
    trainer = L.Trainer(
        max_epochs=args.max_epochs,
        accelerator="gpu",
        devices=1,
        log_every_n_steps=10,
        callbacks=[early_stop, ckpt_cb, ThrottledTQDMProgressBar()],
        logger=logger,
        enable_progress_bar=True,
    )

    trainer.fit(task, datamodule=dm)

    if is_jax:
        # Restore the best JAX checkpoint before testing so test metrics match
        # the best-val-loss model, not the final training epoch.
        from models.utils.jax.load_model import load_model as jax_load_model
        best_path = ckpt_dir / "best.eqx"
        if best_path.exists():
            task.jax_model, task.jax_model_state = jax_load_model(
                path=str(best_path),
                model=task.jax_model,
                model_state=task.jax_model_state,
            )
        else:
            print(f"WARNING: best JAX checkpoint not found at {best_path}; testing on final epoch weights.")

    trainer.test(task, datamodule=dm, ckpt_path="best" if not is_jax else None)

    wandb.finish()


if __name__ == "__main__":
    main()
