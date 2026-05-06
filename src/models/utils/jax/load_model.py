"""Utilities for loading JAX model checkpoint."""

import logging
from pathlib import Path

import equinox as eqx

logger = logging.getLogger(__name__)


def load_model(
    path: Path | str,
    model: eqx.Module,
    model_state: eqx.nn.State,
) -> tuple[eqx.Module, eqx.nn.State]:
    """Load a JAX model and its state from a checkpoint file.

    Optimizer state saved in the checkpoint is ignored; the caller is responsible
    for initialising a fresh optimizer state after loading.
    """
    path = Path(path)
    logger.info(f"Loading model from {path}")
    loaded_model, loaded_state = eqx.tree_deserialise_leaves(path, (model, model_state))
    print(f"Loaded model and state from {path}", flush=True)
    return loaded_model, loaded_state
