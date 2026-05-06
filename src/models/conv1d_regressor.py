"""1D ResNet regression model, mirroring ml4gw ResNet1D (aframe/LIGO community standard).

Architecture (default ResNet18-like with layers=[2,2,2,2]):
  - Stem: Conv1d(d_input, 64, 7, stride=2) + GroupNorm + ReLU + MaxPool(3, stride=2)
  - Residual stages: channels [64, 128, 256, 512], N blocks each, stride-2 downsampling
  - Head: AdaptiveAvgPool1d(1) + Linear(512, d_output)

Interface: (B, L, d_input) -> (B, d_output)

Comparable to S4D(d_model=512, n_layers=4):
  ResNet1DRegressor(d_input=2, layers=[2,2,2,2]) ≈ 11M params  (similar capacity)
  ResNet1DRegressor(d_input=2, layers=[1,1,1,1]) ≈ 2.7M params (lighter)
"""

import torch
import torch.nn as nn


def _conv(in_ch: int, out_ch: int, kernel_size: int, stride: int = 1) -> nn.Conv1d:
    pad = kernel_size // 2
    return nn.Conv1d(in_ch, out_ch, kernel_size, stride=stride, padding=pad, bias=False)


class _BasicBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, kernel_size: int, stride: int = 1):
        super().__init__()
        self.conv1 = _conv(in_ch, out_ch, kernel_size, stride)
        self.bn1 = nn.GroupNorm(min(32, out_ch), out_ch)
        self.act = nn.ReLU(inplace=True)
        self.conv2 = _conv(out_ch, out_ch, kernel_size)
        self.bn2 = nn.GroupNorm(min(32, out_ch), out_ch)
        self.downsample = None
        if stride != 1 or in_ch != out_ch:
            self.downsample = nn.Sequential(
                nn.Conv1d(in_ch, out_ch, 1, stride=stride, bias=False),
                nn.GroupNorm(min(32, out_ch), out_ch),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        out = self.act(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        if self.downsample is not None:
            identity = self.downsample(x)
        return self.act(out + identity)


def _make_stage(in_ch: int, out_ch: int, n_blocks: int, kernel_size: int) -> nn.Sequential:
    blocks = [_BasicBlock(in_ch, out_ch, kernel_size, stride=2)]
    for _ in range(1, n_blocks):
        blocks.append(_BasicBlock(out_ch, out_ch, kernel_size))
    return nn.Sequential(*blocks)


class ResNet1DRegressor(nn.Module):
    def __init__(
        self,
        d_input: int = 2,
        d_output: int = 2,
        layers: list[int] = (2, 2, 2, 2),
        kernel_size: int = 7,
    ):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv1d(d_input, 64, kernel_size=7, stride=2, padding=3, bias=False),
            nn.GroupNorm(min(32, 64), 64),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=3, stride=2, padding=1),
        )
        self.layer0 = nn.Sequential(*[_BasicBlock(64, 64, kernel_size) for _ in range(layers[0])])
        self.layer1 = _make_stage(64, 128, layers[1], kernel_size)
        self.layer2 = _make_stage(128, 256, layers[2], kernel_size)
        self.layer3 = _make_stage(256, 512, layers[3], kernel_size)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.head = nn.Linear(512, d_output)

        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, L, d_input) -> (B, d_input, L)
        x = x.transpose(1, 2)
        x = self.stem(x)
        x = self.layer0(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.pool(x).squeeze(-1)
        return self.head(x)


# Keep the original simple stacked conv as a lighter alternative
class Conv1DRegressor(ResNet1DRegressor):
    """Alias for backward compatibility — use ResNet1DRegressor directly."""
    pass
