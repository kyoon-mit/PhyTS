"""Shared Lightning callbacks for non-TTY environments (HTCondor, SLURM log files)."""

import sys
import time

from lightning.pytorch import Callback, LightningModule, Trainer


class LogProgressCallback(Callback):
    """Prints epoch/step progress to stderr so it appears in HTCondor .err log files.

    Replaces tqdm, which silently disables itself when stdout is not a TTY.
    """

    def __init__(self, log_every_n_epochs: int = 1):
        self.log_every_n_epochs = log_every_n_epochs
        self._epoch_start_time: float = 0.0

    def on_train_epoch_start(self, trainer: Trainer, pl_module: LightningModule) -> None:
        self._epoch_start_time = time.monotonic()

    def on_validation_epoch_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        if trainer.current_epoch % self.log_every_n_epochs != 0:
            return

        elapsed = time.monotonic() - self._epoch_start_time
        metrics = trainer.logged_metrics
        train_loss = metrics.get("train/loss_epoch", metrics.get("train/loss", float("nan")))
        val_loss = metrics.get("val/loss_epoch", metrics.get("val/loss", float("nan")))

        print(
            f"[epoch {trainer.current_epoch:>4d}/{trainer.max_epochs}]"
            f"  train_loss={float(train_loss):.4f}"
            f"  val_loss={float(val_loss):.4f}"
            f"  ({elapsed:.1f}s)",
            file=sys.stderr,
            flush=True,
        )
