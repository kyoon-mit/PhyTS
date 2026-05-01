"""TESS classification hyperparameter sweep script.

Supports all six model types end-to-end (no pretraining):
  mlp          — MLPRegressor (flattened seq → MLP)
  s4d          — S4Model (S4D stack + mean pool)
  cnn          — ConvAE with num_classes (encoder + global avg pool + Linear head)
  cnn_attn     — ConvAttnAE with num_classes (encoder + bottleneck MHA + pool + Linear head)
  linoss_imex  — LinOSS with IMEX discretization (JAX/Equinox)
  linoss_damped — LinOSS with damped_IMEX discretization (JAX/Equinox)

Called by ``wandb agent`` during a sweep, or directly for single runs / debugging.
All hyperparameters flow through CLI args (which seed wandb.config); the sweep
controller overwrites them per trial.

Sweep configs live in configs/TESS/sweep/sweep_cls_*.yaml.

Usage
-----
# Create a sweep (once, on login node or local machine):
    wandb sweep configs/TESS/sweep/sweep_cls_mlp.yaml   # prints sweep_id

# Launch agents on Engaging:
    bash benchmarks/TESS/run_sweep.sh --model_type mlp --sweep_id <id>

# Single run for debugging (wandb disabled):
    WANDB_MODE=disabled uv run python benchmarks/TESS/sweep_classification.py \\
        --model_type cnn --size xs --lr 1e-3 --batch_size 32 --dropout 0.1

# LinOSS requires the jax extra:
    uv run --extra jax python benchmarks/TESS/sweep_classification.py \\
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

# Allow importing sweep_utils from the same directory.
_BENCH_DIR = Path(__file__).resolve().parent
if str(_BENCH_DIR) not in sys.path:
    sys.path.insert(0, str(_BENCH_DIR))

from sweep_utils import (
    JAX_MODELS,
    ThrottledTQDMProgressBar,
    add_infra_args,
    add_sweep_args,
    build_cnn,
    build_cnn_attn,
    build_linoss,
    build_mlp,
    build_s4d,
    collect_benchmark_param_counters,
    dump_sweep_run_config,
    tess_sweep_artifact_dir,
    _REPO_ROOT,
)

if str(_REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT / "src"))

import lightning as L
import wandb
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from lightning.pytorch.loggers import CSVLogger, WandbLogger

from dataloader.tess_dataloader import TESSClassificationDataModule
from tasks.TESS.tess_classification import TESSClassificationCE


# ── Task builders ─────────────────────────────────────────────────────────────

def _build_torch_task(model, num_classes: int, lr: float,
                      weight_decay: float) -> L.LightningModule:
    task = TESSClassificationCE(model=model, num_classes=num_classes, lr=lr, lr_decay=0.99)
    # Patch AdamW weight_decay into configure_optimizers without subclassing.
    _wd = weight_decay

    def _configure_optimizers(self):
        import torch.optim as optim
        opt = optim.AdamW(self.parameters(), lr=self.lr, weight_decay=_wd)
        sched = optim.lr_scheduler.ExponentialLR(opt, gamma=self.lr_decay)
        return {"optimizer": opt, "lr_scheduler": {"scheduler": sched, "interval": "epoch"}}

    task.configure_optimizers = types.MethodType(_configure_optimizers, task)
    return task


def _build_linoss_task(model, num_classes: int, lr: float,
                       seed: int) -> L.LightningModule:
    from tasks.TESS.tess_linoss import TESSLinOSSClassificationCE
    return TESSLinOSSClassificationCE(
        model=model, num_classes=num_classes, lr=lr, clip_grad_norm=1.0, seed=seed,
    )


# ── Argument parsing ──────────────────────────────────────────────────────────

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="TESS classification sweep")
    add_infra_args(p)
    p.add_argument("--data_dir",
                   default=os.environ.get("TESS_DATA_DIR",
                                          "data/TESS/.cache/TESS"))
    p.add_argument("--num_classes", type=int, default=8)
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
            "num_classes":  args.num_classes,
        },
    )
    cfg = wandb.config

    L.seed_everything(cfg.seed, workers=True)

    is_jax = cfg.model_type in JAX_MODELS

    # ── Build model & task ────────────────────────────────────────────────────
    if cfg.model_type == "mlp":
        model = build_mlp(cfg.size, args.num_classes, args.seq_len, cfg.dropout)
        task  = _build_torch_task(model, args.num_classes, cfg.lr, cfg.weight_decay)

    elif cfg.model_type == "s4d":
        model = build_s4d(cfg.size, args.num_classes, cfg.dropout)
        task  = _build_torch_task(model, args.num_classes, cfg.lr, cfg.weight_decay)

    elif cfg.model_type == "cnn":
        model = build_cnn(cfg.size, args.num_classes)
        task  = _build_torch_task(model, args.num_classes, cfg.lr, cfg.weight_decay)

    elif cfg.model_type == "cnn_attn":
        model = build_cnn_attn(cfg.size, args.num_classes)
        task  = _build_torch_task(model, args.num_classes, cfg.lr, cfg.weight_decay)

    elif cfg.model_type == "linoss_imex":
        model = build_linoss(cfg.size, args.num_classes, "IMEX", cfg.seed)
        task  = _build_linoss_task(model, args.num_classes, cfg.lr, cfg.seed)

    elif cfg.model_type == "linoss_damped":
        model = build_linoss(cfg.size, args.num_classes, "damped_IMEX", cfg.seed)
        task  = _build_linoss_task(model, args.num_classes, cfg.lr, cfg.seed)

    else:
        raise ValueError(f"Unknown model_type: {cfg.model_type!r}")

    wandb.summary.update(collect_benchmark_param_counters(cfg.model_type, task))

    run_id = run.id if run is not None else "local"
    art_dir = tess_sweep_artifact_dir("classification", cfg.model_type, run_id)
    dump_sweep_run_config(art_dir / "run_config.yaml", args, run)

    # ── Data ──────────────────────────────────────────────────────────────────
    dm = TESSClassificationDataModule(
        data_dir=args.data_dir,
        batch_size=cfg.batch_size,
        num_workers=args.num_workers,
        seq_len=args.seq_len,
    )

    # ── Callbacks ─────────────────────────────────────────────────────────────
    ckpt_dir = Path(os.environ.get("TESS_CKPT_DIR", args.ckpt_dir)) / "classification" / cfg.model_type / run_id

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
    trainer.test(task, datamodule=dm, ckpt_path="best" if not is_jax else None)

    wandb.finish()


if __name__ == "__main__":
    main()
