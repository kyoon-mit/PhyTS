"""JAX training and inference utils."""

from abc import ABC
from typing import Callable

import equinox as eqx
import optax
from jaxtyping import Array, PRNGKeyArray, PyTree, Scalar

type LossFunction = Callable[[PyTree[Array], PyTree[Array]], Scalar]


class JAXModule(eqx.Module, ABC):
    """Base class for JAX models with state and optimisers."""

    def __call__(self, *args, **kwargs):
        """Override to handle JAX optimisers and state."""
        raise NotImplementedError("JAXModule is a base class; implement __call__ in subclasses.")


@eqx.filter_vmap(in_axes=(None, 0, None, 0), out_axes=(0, None), axis_name="batch")
def jax_fwd_batch(
    model: JAXModule,
    x: PyTree[Array],
    state: eqx.nn.State,
    key: PRNGKeyArray,
) -> tuple[PyTree[Array], eqx.nn.State]:
    """Batched forward pass with vmap. Model must return (logits, new_state)."""
    return model(x, state, key=key)


def jax_apply_loss_fn(
    diff_model: JAXModule,
    static_model: JAXModule,
    state: eqx.nn.State,
    x: PyTree[Array],
    y: PyTree[Array],
    loss_fn: LossFunction,
    key: PRNGKeyArray,
) -> tuple[Scalar, tuple[eqx.nn.State, PyTree[Array]]]:
    """Apply loss function to batch.

    Combines model parts, applies forward pass, and computes loss.
    """
    model = eqx.combine(diff_model, static_model)
    model_output, new_state = jax_fwd_batch(model, x, state, key)
    loss = loss_fn(model_output, y)
    return loss, (new_state, model_output)


@eqx.filter_jit
def jax_apply_training_step(
    model: JAXModule,
    model_filter_spec,
    state: eqx.nn.State,
    x: PyTree[Array],
    y: PyTree[Array],
    opt_state: optax.OptState,
    opt_update: optax.TransformUpdateFn,
    loss_fn: LossFunction,
    key: PRNGKeyArray,
) -> tuple[JAXModule, eqx.nn.State, optax.OptState, Scalar, PyTree[Array]]:
    """Jax Model training step."""
    diff_model, static_model = eqx.partition(model, model_filter_spec)

    (loss, (new_state, model_output)), grads = eqx.filter_value_and_grad(
        jax_apply_loss_fn, has_aux=True
    )(diff_model, static_model, state, x, y, loss_fn, key)

    updates, new_opt_state = opt_update(grads, opt_state, diff_model)
    new_model = eqx.combine(eqx.apply_updates(diff_model, updates), static_model)
    return new_model, new_state, new_opt_state, loss, model_output


@eqx.filter_jit
def jax_inference(
    model: eqx.Module,
    x: PyTree[Array],
    state: eqx.nn.State,
    key: PRNGKeyArray,
) -> tuple[PyTree[Array]]:
    """Jax Model inference step."""
    inference_model = eqx.tree_inference(model, value=True)
    outputs, _ = jax_fwd_batch(inference_model, x, state, key)
    return outputs
