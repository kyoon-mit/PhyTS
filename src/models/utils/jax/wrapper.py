"""Wrapper to make JAX models compatible with PyTorch Lightning.

Convert tensors to JAX arrays using DLpack and handle JAX optimizer.
"""

import io
import logging
from pathlib import Path

import equinox as eqx
import jax
import lightning as L
import optax
import torch
from jaxtyping import PRNGKeyArray, PyTree

from .load_model import load_model
from .print_params import (
    count_array_elements,
    count_inexact_array_elements,
    print_param_tree,
)
from .training import LossFunction, jax_apply_training_step, jax_inference
from .utils import tensor_to_jax

# Define a type for batches of PyTorch tensors
type Batch = tuple[PyTree[torch.Tensor], ...]
type ModelInput = PyTree[torch.Tensor]

logger = logging.getLogger(__name__)


class JAXLightningModule(L.LightningModule):
    """Wrapper to make JAX models compatible with PyTorch Lightning.

    Convert tensors to JAX arrays using DLpack and handle JAX optimizer.
    """

    key: PRNGKeyArray

    jax_model: eqx.Module
    jax_model_state: eqx.nn.State
    jax_model_filter_spec: PyTree[bool]

    jax_optimizer: optax.GradientTransformation
    clip_grad_norm: float | None = None

    def __init__(
        self,
        model: eqx.Module,
        loss_fn: LossFunction,
        optimizer: optax.GradientTransformation,
        clip_grad_norm: float | None = None,
        model_filter_spec: PyTree[bool] | None = None,
        load_from_checkpoint: str | Path | None = None,
        seed: int = 0,
    ):
        super().__init__()

        # We disable automatic optimization to handle JAX optimizer manually.
        self.automatic_optimization = False

        # setup random key and model/optimizer state
        self.key = jax.random.key(seed)
        self.jax_model = model
        self.jax_model_state = eqx.nn.State(self.jax_model)
        self.jax_optimizer = optimizer
        self.loss_fn = loss_fn
        self.clip_grad_norm = clip_grad_norm

        # Set trainable vs. non trainable parameters for optimizer.
        # If no filter spec is provided, treat all inexact (float etc.) arrays are trainable.
        # To freeze some parameters, provide a filter spec PyTree[bool]
        # of the same structure as the model, where True indicates trainable parameters and False
        # indicates frozen parameters.
        self.jax_model_filter_spec = model_filter_spec or jax.tree_util.tree_map(
            eqx.is_inexact_array, self.jax_model
        )

        # load model and optimizer state from checkpoint if provided
        if load_from_checkpoint is not None:
            self.jax_model, self.jax_model_state = load_model(
                path=load_from_checkpoint,
                model=self.jax_model,
                model_state=self.jax_model_state,
            )

        # Parameter accounting after optional checkpoint hydration.
        mod_elems = count_inexact_array_elements(self.jax_model)
        state_elems = count_array_elements(self.jax_model_state)
        summary = (
            f"[{self.__class__.__name__}] jax_model floating leaves "
            f"(``eqx.is_inexact_array``; includes BN slots in the module tree): {mod_elems:,} | "
            f"jax_model_state array elements (``eqx.is_array``): {state_elems:,}"
        )
        print(summary, flush=True)
        logger.info("%s", summary)
        print_param_tree(self.jax_model, 3)

    def _prepare_batch(self, batch: Batch) -> tuple[PyTree[jax.Array], PyTree[jax.Array]]:
        """Return (x, y) from a JAX-converted batch. Override for dataset-specific layouts."""
        x, y, *_ = batch
        return x, y

    def _batched_keys(self, base_key: PRNGKeyArray, batch_size: int) -> PRNGKeyArray:
        """Split a scalar PRNG key into one key per sample for vmapped dropout."""
        return jax.random.split(base_key, batch_size)

    def _training_step_extra_logs(self, model_output: PyTree[jax.Array], y: PyTree[jax.Array]) -> None:
        """Log additional ``train/*`` metrics from forward outputs; override in task modules."""
        return

    def forward(self, x: ModelInput):
        """Forward pass. convert PyTorch tensor to JAX array and apply JAX model."""
        x_jax = tensor_to_jax(x)
        batch_size = jax.tree.leaves(x_jax)[0].shape[0]
        keys = self._batched_keys(jax.random.key(0), batch_size)
        return jax_inference(
            model=self.jax_model,
            x=x_jax,
            state=self.jax_model_state,
            key=keys,
        )

    def training_step(self, batch: Batch, batch_idx: int):
        """Training step.

        Convert batch to JAX arrays, apply training step, and update model state and optimizer.
        """
        batch = tensor_to_jax(batch)
        x, y = self._prepare_batch(batch)
        # Create a new key for this batch, then split per sample for vmap.
        step_key = jax.random.fold_in(self.key, batch_idx)
        batch_size = jax.tree.leaves(x)[0].shape[0]
        keys = self._batched_keys(step_key, batch_size)

        # TODO: (Benedict) Add async dispatch to speed up training.
        new_model, new_state, new_opt_state, loss, model_output = jax_apply_training_step(
            model=self.jax_model,  # type: ignore
            model_filter_spec=self.jax_model_filter_spec,
            state=self.jax_model_state,
            x=x,
            y=y,
            opt_state=self.opt_state,
            opt_update=self.jax_optimizer.update,
            loss_fn=self.loss_fn,
            key=keys,
        )

        self.jax_model = new_model
        self.jax_model_state = new_state
        self.opt_state = new_opt_state

        self.log(
            "train/loss",
            loss.item(),
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            sync_dist=True,
        )
        self._training_step_extra_logs(model_output, y)

        # Needed for lightning if self.automatic_optimization = False
        optimizers = self.optimizers()
        if isinstance(optimizers, (list, tuple)):
            for opt in optimizers:
                opt.step()
        else:
            optimizers.step()

        return torch.tensor(0.0)

    def validation_step(self, batch: Batch, batch_idx: int):
        """Validation step.

        Convert batch to JAX arrays and evaluate the JAX model.
        """
        batch = tensor_to_jax(batch)
        x, y = self._prepare_batch(batch)
        batch_size = jax.tree.leaves(x)[0].shape[0]
        keys = self._batched_keys(self.key, batch_size)
        outputs = jax_inference(
            model=self.jax_model,
            x=x,
            state=self.jax_model_state,
            key=keys,
        )
        loss = jax.jit(self.loss_fn)(outputs, y)
        self.log(
            "val/loss",
            loss.item(),
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            sync_dist=True,
        )

    def test_step(self, batch: Batch, batch_idx: int):
        """Test step.

        Convert batch to JAX arrays and evaluate the JAX model.
        """
        batch = tensor_to_jax(batch)
        x, y = self._prepare_batch(batch)
        batch_size = jax.tree.leaves(x)[0].shape[0]
        keys = self._batched_keys(self.key, batch_size)
        outputs = jax_inference(
            model=self.jax_model,
            x=x,
            state=self.jax_model_state,
            key=keys,
        )
        loss = jax.jit(self.loss_fn)(outputs, y)
        self.log(
            "test/loss",
            loss.item(),
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            sync_dist=True,
        )

    def configure_optimizers(self):
        """Configure optimizer and learning rate scheduler."""
        # Return the JAX optimizer and learning rate scheduler if provided
        self.jax_optimizer = optax.chain(
            optax.clip_by_global_norm(self.clip_grad_norm)
            if self.clip_grad_norm is not None
            else optax.identity(),
            self.jax_optimizer,
        )

        # separate trainable and non-trainable parameters for optimizer state initialization
        diff_model, _ = eqx.partition(self.jax_model, self.jax_model_filter_spec)
        self.opt_state = self.jax_optimizer.init(diff_model)

    # ─── checkpoint round-trip via Lightning's ModelCheckpoint ──────────────
    # JAX arrays are not torch parameters, so state_dict() is empty for this
    # module — the default Lightning checkpoint would not contain any model
    # weights. Serialise (jax_model, jax_model_state) into the checkpoint
    # dict so a stock ``ModelCheckpoint`` callback round-trips correctly.
    JAX_CKPT_KEY = "jax_state"

    def on_save_checkpoint(self, checkpoint: dict) -> None:
        buf = io.BytesIO()
        eqx.tree_serialise_leaves(buf, (self.jax_model, self.jax_model_state))
        checkpoint[self.JAX_CKPT_KEY] = buf.getvalue()

    def on_load_checkpoint(self, checkpoint: dict) -> None:
        blob = checkpoint.get(self.JAX_CKPT_KEY)
        if blob is None:
            logger.warning(
                "JAXLightningModule.on_load_checkpoint: no '%s' entry found "
                "in checkpoint; jax_model is left at its initial random "
                "weights.",
                self.JAX_CKPT_KEY,
            )
            return
        buf = io.BytesIO(blob)
        self.jax_model, self.jax_model_state = eqx.tree_deserialise_leaves(
            buf, (self.jax_model, self.jax_model_state)
        )
