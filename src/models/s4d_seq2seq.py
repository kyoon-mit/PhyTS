# This file is a derivative work of the S4 repository:
#   https://github.com/state-spaces/s4
#
# Copyright (c) 2023 The S4 Authors
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at:
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Modifications (c) 2026 Kyungseop Yoon (kyoon@mit.edu), 2026-03-30
#   - Derived S4ModelSeq2Seq from S4Model in s4d.py by removing temporal
#     mean-pooling, enabling sequence-to-sequence (denoising) tasks.
#   - The S4D kernel and S4D block are used unmodified from s4d.py.

"""S4D sequence-to-sequence model for denoising tasks."""

import torch.nn as nn

from models.s4d import S4D
from functions.dropout import DropoutNd


class S4ModelSeq2Seq(nn.Module):
    """Sequence-to-sequence variant of S4Model (no temporal pooling).

    Identical to S4Model from s4d.py except the mean-pooling step is removed,
    so the full sequence is preserved for denoising / seq2seq tasks.

    Input:  (B, L, d_input)
    Output: (B, L, d_output)

    Usage (LightningCLI YAML):
        model:
          class_path: models.s4d_seq2seq.S4ModelSeq2Seq
          init_args:
            d_input: 1
            d_output: 1
            d_model: 256
            d_state: 64
            n_layers: 4
            dropout: 0.2
    """

    def __init__(
        self,
        d_input: int,
        d_output: int = 1,
        d_model: int = 256,
        d_state: int = 64,
        n_layers: int = 4,
        dropout: float = 0.2,
        prenorm: bool = False,
        lr: float | None = None,
        dt_min: float = 0.001,
        dt_max: float = 0.1,
    ):
        super().__init__()
        self.prenorm   = prenorm
        self.encoder   = nn.Linear(d_input, d_model)
        self.s4_layers = nn.ModuleList()
        self.norms     = nn.ModuleList()
        self.dropouts  = nn.ModuleList()
        for _ in range(n_layers):
            self.s4_layers.append(
                S4D(d_model, d_state=d_state, dropout=dropout, transposed=True,
                    dt_min=dt_min, dt_max=dt_max, lr=lr)
            )
            self.norms.append(nn.LayerNorm(d_model))
            self.dropouts.append(DropoutNd(dropout))
        self.decoder = nn.Linear(d_model, d_output)

    def forward(self, x):
        """
        Input x:  (B, L, d_input)
        Returns:  (B, L, d_output)
        """
        x = self.encoder(x)      # (B, L, d_model)
        x = x.transpose(-1, -2)  # (B, d_model, L)

        for layer, norm, dropout in zip(self.s4_layers, self.norms, self.dropouts):
            z = x
            if self.prenorm:
                z = norm(z.transpose(-1, -2)).transpose(-1, -2)
            z, _ = layer(z)
            z = dropout(z)
            x = z + x
            if not self.prenorm:
                x = norm(x.transpose(-1, -2)).transpose(-1, -2)

        x = x.transpose(-1, -2)  # (B, L, d_model)
        x = self.decoder(x)      # (B, L, d_output)
        return x
