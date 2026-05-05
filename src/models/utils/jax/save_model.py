"""JAXCheckpointManager: A PyTorch Lightning callback for saving JAX model checkpoints."""

import logging
import os
from pathlib import Path
from typing import Literal

import equinox as eqx
from lightning.pytorch import Callback, LightningModule
from lightning.pytorch.trainer import Trainer

logger = logging.getLogger(__name__)


def get_save_dir(trainer: Trainer) -> str:
    if trainer.log_dir is not None:
        return trainer.log_dir
    return trainer.default_root_dir


class JAXCheckpointManager(Callback):
    """PyTorch Lightning callback for saving JAX model checkpoints."""

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
    ):
        self.monitor = monitor
        self.mode = mode
        self.save_top_k = save_top_k
        self.step_top_k = step_top_k
        self.cfg_every_n_steps = every_n_steps
        self.cfg_warmup_steps = warmup_steps

        # Separate tracking for metric-based and step-based checkpoints
        self.metric_checkpoints: dict[float, Path] = {}
        self.step_checkpoints: dict[int, Path] = {}

        # Save every 1000 steps if no metric is provided.
        if self.monitor is None and self.cfg_every_n_steps is None:
            self.cfg_every_n_steps = 1000

    def setup(self, trainer: Trainer, pl_module: LightningModule, stage: str) -> None:

        save_dir = get_save_dir(trainer)
        self.output_dir = Path(save_dir) / "checkpoints"
        logger.info(f"JAXCheckpointManager will save checkpoints to: {self.output_dir}")
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def on_validation_epoch_end(self, trainer: Trainer, pl_module: LightningModule):
        # current_step = self.step_tracker.get_step()
        if self._should_skip_warmup(trainer):  # or current_step < 1:
            return

        # Handle metric-based saving
        if self.monitor is not None:
            self._save_by_metric(trainer)

        # Handle step-based saving
        if self.cfg_every_n_steps is not None:
            self._save_by_steps(trainer)

    def _should_skip_warmup(self, trainer: Trainer) -> bool:
        current_step = trainer.global_step
        if self.cfg_warmup_steps is not None and current_step < self.cfg_warmup_steps:
            logger.info(f"Skipping ckpt saving during warmup: step {current_step}")
            return True
        return False

    def _save_by_metric(self, trainer: Trainer):
        try:
            score = trainer.logged_metrics[self.monitor].item()
            if self._is_better_metric(score):
                self.best_metric = score
                self._save_checkpoint(trainer, score=score, checkpoint_type="metric")
        except KeyError:
            logger.info(
                f"Metric '{self.monitor}' not found in logged metrics,"
                f" saving every {self.cfg_every_n_steps or '1000'} steps"
            )
            if self.cfg_every_n_steps is None:
                self.cfg_every_n_steps = 1000

    def _save_by_steps(self, trainer: Trainer):
        current_step = trainer.global_step
        if current_step - self.last_step_saved >= self.cfg_every_n_steps:
            self._save_checkpoint(trainer, checkpoint_type="step")
            self.last_step_saved = current_step

    def _is_better_metric(self, score: float) -> bool:
        if self.best_metric is None:
            return True
        return score > self.best_metric if self.mode == "max" else score < self.best_metric

    def _get_input_shape(self, trainer: Trainer) -> tuple[int, int]:
        """Return ``(num_ifos, time_len)`` from the attached datamodule."""
        if hasattr(trainer, "datamodule") and trainer.datamodule is not None:
            dm = trainer.datamodule
        else:
            dm = trainer.lightning_module.trainer.datamodule
        num_ifos = len(dm.hparams.ifos)
        time_len = int(dm.hparams.sample_rate * dm.hparams.kernel_length)
        return num_ifos, time_len

    def _save_checkpoint(
        self,
        trainer: Trainer,
        score: float | None = None,
        checkpoint_type: str = "step",
    ):
        self._cleanup_old_checkpoints(checkpoint_type)

        model = trainer.lightning_module.jax_model
        state = trainer.lightning_module.jax_model_state
        opt_state = trainer.lightning_module.opt_state

        # Compute integrity checksum before writing to disk.
        checksum = None
        try:
            num_ifos, time_len = self._get_input_shape(trainer)
        except Exception as e:
            logger.warning(f"Could not compute checkpoint checksum: {e}. Saving without checksum.")
            num_ifos, time_len = None, None

        model_path = self._get_checkpoint_path(trainer, score, checksum, num_ifos, time_len)

        # Save model and state as tuple
        eqx.tree_serialise_leaves(model_path, (model, state, opt_state))

        current_step = trainer.global_step
        # Track checkpoint
        if checkpoint_type == "metric" and score is not None:
            self.metric_checkpoints[score] = model_path
            logger.info(f"Saved metric checkpoint (score={score}) at step {current_step}")
        else:
            self.step_checkpoints[current_step] = model_path
            logger.info(f"Saved step checkpoint at step {current_step}")

    def _get_checkpoint_path(
        self,
        trainer: Trainer,
        score: float | None = None,
        checksum: str | None = None,
        num_ifos: int | None = None,
        time_len: int | None = None,
    ) -> Path:
        metric_part = f"_metric_{score}" if score is not None else ""
        current_step = trainer.global_step
        checksum_part = f"_{num_ifos}_{time_len}_{checksum}" if checksum is not None else ""
        base_name = f"checkpoint_step_{current_step}{metric_part}{checksum_part}"
        return self.output_dir / f"{base_name}.eqx"

    def _cleanup_old_checkpoints(self, checkpoint_type: str):
        if checkpoint_type == "metric":
            self._remove_excess_checkpoints(self.metric_checkpoints, self.save_top_k, use_mode=True)

    def _remove_excess_checkpoints(self, checkpoint_dict: dict, top_k: int, use_mode: bool):
        if len(checkpoint_dict) >= top_k:
            # Sort checkpoints - for metrics use mode, for steps use asc. order
            reverse_sort = self.mode == "max" if use_mode else False
            sorted_items = sorted(
                checkpoint_dict.items(),
                key=lambda x: x[0],
                reverse=reverse_sort,
            )

            # Remove the worst/oldest checkpoint
            key_to_remove, model_path = sorted_items[-1]
            self._delete_checkpoint_file(model_path)
            del checkpoint_dict[key_to_remove]

    def _delete_checkpoint_file(self, model_path: Path):
        if model_path.exists():
            os.remove(model_path)

    def get_best_checkpoint(self) -> tuple[Path, float]:
        """Returns (model_path, best_metric)"""
        if self.best_metric is None or self.best_metric not in self.metric_checkpoints:
            raise ValueError("No best metric checkpoint available")
        model_path = self.metric_checkpoints[self.best_metric]
        return model_path, self.best_metric
