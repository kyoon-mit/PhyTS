"""Per-job training time limit for preemptible / time-sliced cluster jobs."""
from pathlib import Path
from typing import Any, Optional

import lightning.pytorch as pl
from lightning.pytorch.callbacks import Timer


class JobTimer(Timer):
    """Stop `fit` at an epoch boundary once *this job* has trained for `duration`.

    Unlike `Timer`, the elapsed time is not stored in / restored from checkpoints, so every
    resumed job gets a fresh budget. When the limit (and not early stopping or max_epochs)
    ends the fit, an empty `marker` file is written so the launcher knows to requeue.

    Args:
        duration: "DD:HH:MM:SS" string, timedelta or dict (see `Timer`).
        marker: path of the file written when the time limit stops training.
    """

    def __init__(self, duration: Any, marker: Optional[str] = None, verbose: bool = True) -> None:
        super().__init__(duration=duration, interval="epoch", verbose=verbose)
        self.marker = Path(marker) if marker else None

    def state_dict(self) -> dict[str, Any]:
        return {}

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        pass

    def on_train_epoch_end(self, trainer: "pl.Trainer", *args: Any, **kwargs: Any) -> None:
        if trainer.should_stop:  # early stopping already ended the run
            return
        if trainer.current_epoch + 1 >= (trainer.max_epochs or float("inf")):
            return
        super().on_train_epoch_end(trainer, *args, **kwargs)
        if trainer.should_stop and self.marker is not None and trainer.is_global_zero:
            self.marker.parent.mkdir(parents=True, exist_ok=True)
            self.marker.touch()
