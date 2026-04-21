"""Utilities for loading JAX model checkpoint."""

import logging
import os
from pathlib import Path

import equinox as eqx

logger = logging.getLogger(__name__)


def load_model(
    path: Path | str,
    model: eqx.Module,
    model_state: eqx.nn.State,
) -> tuple[eqx.Module, eqx.nn.State]:
    """Load a JAX model and its state from a checkpoint file."""
    path = Path(path)
    logger.info(f"Loading model from {path}")

    try:
        loaded_model, loaded_state = eqx.tree_deserialise_leaves(path, (model, model_state))
        print(f"Initialized model and state from {path}")
        return loaded_model, loaded_state
    except Exception:
        logger.warning(
            f"Failed to load model and state from {path} as tuple, trying to load separately"
        )

    loaded_model = eqx.tree_deserialise_leaves(path, model)
    logger.info(f"Initialized model from {path}")

    # Also load the corresponding state file if it exists
    if path.suffix == ".eqx":
        state_path = path.with_suffix(".eqx.state")
        if os.path.exists(state_path):
            model_state = eqx.tree_deserialise_leaves(state_path, model_state)
            logger.info(f"Loaded model state from {state_path}")
        else:
            logger.warning(f"Warning: State file {state_path} not found. Using fresh model state.")
    else:
        logger.warning(
            "Warning: Checkpoint path doesn't end with .eqx, cannot determine state file path."
        )

    return loaded_model, model_state
