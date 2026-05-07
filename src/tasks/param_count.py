"""Torch / architecture-specific parameter totals for Lightning ``hyper_parameters``."""

from __future__ import annotations

from typing import Any

import torch.nn as nn


def attach_scalar_hyperparams(lightning_module, extras: dict[str, Any]) -> None:
    """Merge ``extras`` into ``lightning_module.hparams`` after ``save_hyperparameters``.

    Lightning 2.6+ no longer forwards arbitrary ``**kwargs`` through
    :meth:`~lightning.pytorch.core.LightningModule.save_hyperparameters`.
    """

    hp = lightning_module.hparams
    for k, v in extras.items():
        hp[k] = v


def torch_nn_parameter_count(module: nn.Module) -> int:
    """``sum(p.numel() for p in module.parameters())`` (ignores buffers)."""
    return int(sum(p.numel() for p in module.parameters()))


def torch_nn_trainable_parameter_count(module: nn.Module) -> int:
    """Subset of ``torch_nn_parameter_count`` with ``requires_grad``."""
    return int(sum(p.numel() for p in module.parameters() if p.requires_grad))


def torch_model_parameter_hyper_dict(module: nn.Module) -> dict[str, Any]:
    """Totals to attach with :func:`attach_scalar_hyperparams` onto ``LightningModule.hparams``.

    Includes optional ``model_s4_analytic_num_parameters_nn`` when ``module``
    is an :class:`~models.s4d.S4Model`
    (matches :func:`~models.s4d.count_s4model_nn_parameters`).
    """
    out: dict[str, Any] = {
        "model_num_parameters_nn": torch_nn_parameter_count(module),
        "model_trainable_parameters_nn": torch_nn_trainable_parameter_count(module),
    }
    name = type(module).__name__
    if name == "S4Model":
        layer0 = module.s4_layers[0]
        from models.s4d import count_s4model_nn_parameters

        out["model_s4_analytic_num_parameters_nn"] = int(
            count_s4model_nn_parameters(
                d_input=int(module.encoder.weight.shape[1]),
                d_output=int(module.decoder.bias.shape[0]),
                d_model=int(module.encoder.weight.shape[0]),
                d_state=int(layer0.n),
                n_layers=len(module.s4_layers),
            )
        )
    return out


def torch_module_bundle_prefixed(prefix: str, module: nn.Module) -> dict[str, Any]:
    """Prefixes keys from :func:`torch_model_parameter_hyper_dict`.

    Examples
    --------
    With ``prefix='backbone'`` you get keys such as
    ``backbone_model_num_parameters_nn``.
    """
    inner = torch_model_parameter_hyper_dict(module)
    return {f"{prefix}_{key}": value for key, value in inner.items()}


def jax_equinox_model_hyper_dict(jax_model, jax_model_state) -> dict[str, int]:
    """Equinox subtree vs :class:`~equinox.nn.State` array element counts."""
    from models.utils.jax.print_params import count_array_elements, count_inexact_array_elements

    return {
        "jax_model_float_parameter_elements": int(count_inexact_array_elements(jax_model)),
        "jax_state_array_elements": int(count_array_elements(jax_model_state)),
    }
