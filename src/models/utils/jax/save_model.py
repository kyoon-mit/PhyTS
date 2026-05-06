"""JAXCheckpointManager: A PyTorch Lightning callback for saving JAX model checkpoints."""

import logging
import os
import sys
from pathlib import Path
from typing import Literal

import equinox as eqx
from lightning.pytorch import Callback, LightningModule
from lightning.pytorch.trainer import Trainer

logger = logging.getLogger(__name__)


def get_save_dir(trainer: Trainer) -> str:
    """Return a sensible root directory for checkpoint output."""
    if trainer.log_dir is not None:
        return trainer.log_dir
    return trainer.default_root_dir


def _stderr(msg: str) -> None:
    print(f"[JAXCheckpointManager] {msg}", file=sys.stderr, flush=True)


class JAXCheckpointManager(Callback):
    """PyTorch Lightning callback for saving JAX/equinox model checkpoints.

    Saves .eqx files via eqx.tree_serialise_leaves.  Metric-based saving keeps
    the top-k best checkpoints; until save_top_k checkpoints exist every
    validation epoch is saved unconditionally.
    """

    last_step_saved: int = -1
    output_dir: Path
    metric_checkpoints: dict[float, Path]
    step_checkpoints: dict[int, Path]
    best_metric: float | None = None

    def __init__(
        self,
        monitor: str | None = None,
        mode: Literal["min", "max"] = "max",
        save_top_k: int = 20,
        step_top_k: int = 10,
        every_n_steps: int | None = None,
        warmup_steps: int | None = None,
        dirpath: str | Path | None = None,
    ):
        self.monitor = monitor
        self.mode = mode
        self.save_top_k = save_top_k
        self.step_top_k = step_top_k
        self.cfg_every_n_steps = every_n_steps
        self.cfg_warmup_steps = warmup_steps
        self.dirpath = Path(dirpath) if dirpath is not None else None

        self.metric_checkpoints: dict[float, Path] = {}
        self.step_checkpoints: dict[int, Path] = {}

        if self.monitor is None and self.cfg_every_n_steps is None:
            self.cfg_every_n_steps = 1000

    def setup(self, trainer: Trainer, pl_module: LightningModule, stage: str) -> None:
        if self.dirpath is not None:
            self.output_dir = self.dirpath
        else:
            self.output_dir = Path(get_save_dir(trainer)) / "checkpoints"
        _stderr(f"checkpoint dir: {self.output_dir}")
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def on_validation_epoch_end(self, trainer: Trainer, pl_module: LightningModule):
        if trainer.sanity_checking:
            _stderr("skipping save (sanity check)")
            return

        if self._should_skip_warmup(trainer):
            return

        if self.monitor is not None:
            self._save_by_metric(trainer)

        if self.cfg_every_n_steps is not None:
            self._save_by_steps(trainer)

    def _should_skip_warmup(self, trainer: Trainer) -> bool:
        current_step = trainer.global_step
        if self.cfg_warmup_steps is not None and current_step < self.cfg_warmup_steps:
            _stderr(f"skipping save (warmup): step {current_step} < {self.cfg_warmup_steps}")
            return True
        return False

    def _get_metric(self, trainer: Trainer) -> float | None:
        """Read the monitored metric, trying both bare name and _epoch suffix."""
        metrics = trainer.logged_metrics
        for key in (self.monitor, f"{self.monitor}_epoch"):
            if key in metrics:
                return float(metrics[key])
        _stderr(
            f"metric '{self.monitor}' not found in logged_metrics "
            f"(available: {list(metrics.keys())})"
        )
        return None

    def _save_by_metric(self, trainer: Trainer):
        score = self._get_metric(trainer)
        if score is None:
            return

        n_saved = len(self.metric_checkpoints)
        epoch = trainer.current_epoch

        if n_saved < self.save_top_k:
            # Always save until we have save_top_k checkpoints.
            _stderr(
                f"epoch={epoch} score={score:.4f} — saving unconditionally "
                f"({n_saved}/{self.save_top_k} checkpoints so far)"
            )
            if self._is_better_metric(score):
                self.best_metric = score
            self._save_checkpoint(trainer, score=score, checkpoint_type="metric")
        elif self._is_better_metric(score):
            _stderr(
                f"epoch={epoch} score={score:.4f} improved from best={self.best_metric:.4f} — saving"
            )
            self.best_metric = score
            self._save_checkpoint(trainer, score=score, checkpoint_type="metric")
        else:
            _stderr(
                f"epoch={epoch} score={score:.4f} did not improve "
                f"(best={self.best_metric:.4f} mode={self.mode}) — skipping"
            )

    def _save_by_steps(self, trainer: Trainer):
        current_step = trainer.global_step
        if current_step - self.last_step_saved >= self.cfg_every_n_steps:
            _stderr(f"step-based save at step {current_step}")
            self._save_checkpoint(trainer, checkpoint_type="step")
            self.last_step_saved = current_step

    def _is_better_metric(self, score: float) -> bool:
        if self.best_metric is None:
            return True
        return score > self.best_metric if self.mode == "max" else score < self.best_metric

    def _save_checkpoint(
        self,
        trainer: Trainer,
        score: float | None = None,
        checkpoint_type: str = "step",
    ):
        if checkpoint_type == "metric":
            self._cleanup_old_checkpoints(checkpoint_type)

        model = trainer.lightning_module.jax_model
        state = trainer.lightning_module.jax_model_state
        opt_state = trainer.lightning_module.opt_state

        model_path = self._get_checkpoint_path(trainer, score)
        eqx.tree_serialise_leaves(model_path, (model, state, opt_state))

        current_step = trainer.global_step
        if checkpoint_type == "metric" and score is not None:
            self.metric_checkpoints[score] = model_path
            _stderr(f"saved metric checkpoint → {model_path.name} (step {current_step})")
        else:
            self.step_checkpoints[current_step] = model_path
            _stderr(f"saved step checkpoint → {model_path.name}")

    def _get_checkpoint_path(self, trainer: Trainer, score: float | None = None) -> Path:
        epoch = trainer.current_epoch
        step = trainer.global_step
        metric_part = f"_metric_{score:.6f}" if score is not None else ""
        return self.output_dir / f"checkpoint_epoch_{epoch:04d}_step_{step}{metric_part}.eqx"

    def _cleanup_old_checkpoints(self, checkpoint_type: str):
        if checkpoint_type == "metric":
            self._remove_excess_checkpoints(self.metric_checkpoints, self.save_top_k, use_mode=True)

    def _remove_excess_checkpoints(self, checkpoint_dict: dict, top_k: int, use_mode: bool):
        if len(checkpoint_dict) >= top_k:
            reverse_sort = self.mode == "max" if use_mode else False
            sorted_items = sorted(
                checkpoint_dict.items(),
                key=lambda x: x[0],
                reverse=reverse_sort,
            )
            key_to_remove, model_path = sorted_items[-1]
            _stderr(f"removing old checkpoint: {model_path.name}")
            self._delete_checkpoint_file(model_path)
            del checkpoint_dict[key_to_remove]

    def _delete_checkpoint_file(self, model_path: Path):
        if model_path.exists():
            os.remove(model_path)

    def get_best_checkpoint(self) -> tuple[Path, float]:
        """Returns (model_path, best_metric)"""
        if self.best_metric is None or self.best_metric not in self.metric_checkpoints:
            raise ValueError("No best metric checkpoint available")
        return self.metric_checkpoints[self.best_metric], self.best_metric
