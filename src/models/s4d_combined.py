"""S4D combined denoiser + regressor.

Composes a sequence-to-sequence denoiser with a pooled regressor so that the
combined module returns ``(x_denoised, y_preds)`` from a single noisy input.
This mirrors the ``S4DCombinedModel`` encoder used by the reference
``LitS4CombinedModel``
(https://github.com/chreissel/neutrino_project/blob/denoising/src/models/model.py).

Default sub-modules are :class:`models.s4d_seq2seq.S4ModelSeq2Seq`
(denoiser, ``(B, L, C) -> (B, L, C)``) and :class:`models.s4d.S4Model`
(regressor, ``(B, L, C) -> (B, regress_dim)``), but any modules with the
matching shapes can be passed in.
"""

from typing import Tuple

import torch
import torch.nn as nn


class S4DCombinedModel(nn.Module):
    """Denoiser + regressor pair returning ``(x_denoised, y_preds)``.

    The regressor consumes the *denoised* signal so its head can be reused on
    clean inputs at inference time.
    """

    def __init__(self, denoiser: nn.Module, regressor: nn.Module):
        super().__init__()
        self.denoiser = denoiser
        self.regressor = regressor

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x_denoised = self.denoiser(x)         # (B, L, C)
        y_preds = self.regressor(x_denoised)  # (B, regress_dim)
        return x_denoised, y_preds
