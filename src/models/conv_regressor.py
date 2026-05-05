"""1D CNN regressor for sequence -> vector regression.

Interface mirrors :class:`models.s4d.S4Model` and
:class:`models.mlp.MLPRegressor`:
``forward(x: (B, L, d_input)) -> (B, d_output)``.

Architecture: stack of ``Conv1d -> GroupNorm -> activation -> MaxPool1d``
blocks (channels widen along the stack), followed by adaptive average
pooling over the time axis and a linear head.
"""

from collections import OrderedDict

import torch.nn as nn


_ACTIVATIONS = {
    'gelu':       nn.GELU,
    'relu':       nn.ReLU,
    'leaky_relu': nn.LeakyReLU,
}


class Conv1DRegressor(nn.Module):
    """Stacked 1D CNN regressor pooling to a fixed-size vector.

    Args:
        d_input: number of input channels (e.g. 2 for I/Q).
        d_output: regression output dim.  For GaussianNLL set ``d_output=2``
            ``[mean, raw_var]``, matching the existing Project 8 pattern.
        channels: per-block output channels; one block per entry.
        kernel_size: must be odd (for SAME-style padding).
        pool_stride: temporal downsampling factor per block (set to 1 to
            disable pooling).
        dropout: dropout probability applied after each block (0 disables).
        activation: ``'gelu' | 'relu' | 'leaky_relu'``.
    """

    def __init__(
        self,
        d_input: int = 2,
        d_output: int = 2,
        channels: list[int] = [32, 64, 128, 256],
        kernel_size: int = 7,
        pool_stride: int = 2,
        dropout: float = 0.0,
        activation: str = 'gelu',
    ):
        super().__init__()
        if kernel_size % 2 == 0:
            raise ValueError('kernel_size must be odd')
        if not channels:
            raise ValueError('channels must contain at least one entry')
        if activation not in _ACTIVATIONS:
            raise ValueError(
                f"activation must be one of {sorted(_ACTIVATIONS)}, got {activation!r}"
            )
        act_cls = _ACTIVATIONS[activation]
        pad = kernel_size // 2

        blocks: OrderedDict[str, nn.Module] = OrderedDict()
        in_ch = d_input
        for i, out_ch in enumerate(channels):
            blocks[f'conv{i}'] = nn.Conv1d(in_ch, out_ch, kernel_size, padding=pad)
            blocks[f'norm{i}'] = nn.GroupNorm(1, out_ch)
            blocks[f'act{i}']  = act_cls()
            if pool_stride > 1:
                blocks[f'pool{i}'] = nn.MaxPool1d(pool_stride, stride=pool_stride)
            if dropout > 0.0:
                blocks[f'drop{i}'] = nn.Dropout1d(dropout)
            in_ch = out_ch
        self.backbone = nn.Sequential(blocks)

        self.pool = nn.AdaptiveAvgPool1d(1)
        self.head = nn.Linear(channels[-1], d_output)

    def forward(self, x):
        # x: (B, L, d_input) -> (B, d_input, L) for Conv1d
        x = x.transpose(1, 2)
        x = self.backbone(x)         # (B, C, L')
        x = self.pool(x).squeeze(-1) # (B, C)
        return self.head(x)          # (B, d_output)
