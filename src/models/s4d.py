# Derived from state-spaces/s4 (Apache 2.0)
#   https://github.com/state-spaces/s4/blob/main/models/s4/s4d.py
#   https://github.com/state-spaces/s4/blob/main/examples.py
# Copyright (c) 2023 The S4 Authors
#
# Modifications (c) 2026 Anonymous Authors
#   - Adjusted DropoutNd import path for this repository layout.
#   - Replaced `1j` complex literals with torch.complex() for torch.compile compatibility.
#   - Added d_output, d_model, n_layers args to S4Model; added param_count utility.
#   - Removed unused imports.

"""Minimal version of S4D with extra options and features stripped out, for pedagogical purposes."""

import math
import torch
import torch.nn as nn
from einops import repeat

from functions.dropout import DropoutNd


def count_s4model_nn_parameters(
    d_input: int,
    d_output: int,
    d_model: int,
    *,
    d_state: int = 64,
    n_layers: int = 4,
) -> int:
    """Return ``sum(p.numel() for p in model.parameters())`` for :class:`S4Model`.

    The closed form sums, for each :class:`S4D` block, the :class:`S4DKernel`
    parameters, the diagonal skip gain ``D``, and the 1×1 mixing conv; multiplies
    by ``n_layers``; and adds the encoder/decoder linears plus per-block
    :class:`~torch.nn.LayerNorm`. It matches :class:`~models.s4d_seq2seq.S4ModelSeq2Seq`
    (same stacking and linear layers; only pooling differs).

    Notes
    -----
    ``dropout``, ``lr``, ``prenorm``, ``dt_min``, and ``dt_max`` do not add
    learnable tensors (``DropoutNd`` / ``Identity``).

    Parameters
    ----------
    d_input
        Input feature dimension ``d_input``.
    d_output
        Scalar or vector output size ``d_output``.
    d_model
        Hidden width ``d_model`` (called ``h`` below).
    d_state
        S4D diagonal state dimension ``N`` (:math:`\\texttt{d\\_state}`).
    n_layers
        Number of stacked :class:`S4D` blocks ``L``.

    Returns
    -------
    int
        Total ``nn.Parameter`` elements::

            stack = L * (2*h**2 + 4*h*floor(N/2) + 4*h)
            total = stack + h*d_input + h + L*2*h + h*d_output + d_output

        Equivalent to folding ``floor(N/2)`` when ``d_state`` is odd: the kernel
        stores ``C`` and diagonal ``A`` with ``N // 2`` frequency bins, matching
        :class:`S4DKernel` (see ``torch.randn(H, N // 2, dtype=torch.cfloat)``).

        When ``d_state`` is even, ``4*h*floor(N/2)`` equals ``2*h*N``.  If ``lr=0``
        is passed into :class:`S4DKernel` so tensors become buffers, counts from
        this function won't match ``.parameters()`` (buffers are omitted).
    """
    h = d_model
    n_layers_i = int(n_layers)
    d_state_i = int(d_state)
    # S4DKernel uses N // 2 complex bins; parameter count follows q = floor(N/2).
    q = d_state_i // 2
    stack = n_layers_i * (2 * h * h + 4 * h * q + 4 * h)
    encoder = int(d_input) * h + h
    decoder = int(d_output) * h + int(d_output)
    norms = n_layers_i * 2 * h
    return stack + encoder + decoder + norms


class S4DKernel(nn.Module):
    """Generate convolution kernel from diagonal SSM parameters."""

    def __init__(self, d_model, N=64, dt_min=0.001, dt_max=0.1, lr=None):
        super().__init__()
        
        # Generate dt
        H = d_model
        log_dt = torch.rand(H) * (
            math.log(dt_max) - math.log(dt_min)
        ) + math.log(dt_min)

        C = torch.randn(H, N // 2, dtype=torch.cfloat)
        self.C = nn.Parameter(torch.view_as_real(C))
        self.register("log_dt", log_dt, lr)

        log_A_real = torch.log(0.5 * torch.ones(H, N//2))
        A_imag = math.pi * repeat(torch.arange(N//2), 'n -> h n', h=H)
        self.register("log_A_real", log_A_real, lr)
        self.register("A_imag", A_imag, lr)

    def forward(self, L):
        """
        returns: (..., c, L) where c is number of channels (default 1)
        """

        # Materialize parameters
        dt = torch.exp(self.log_dt) # (H)
        C = torch.view_as_complex(self.C) # (H N)
        # ORIGINAL CODE
        # A = -torch.exp(self.log_A_real) + 1j * self.A_imag # (H N)
        # MODIFIED CODE FOR torch.compile SAFETY
        A = torch.complex(-torch.exp(self.log_A_real), self.A_imag)

        # Vandermonde multiplication
        dtA = A * dt.unsqueeze(-1)  # (H N)
        K = dtA.unsqueeze(-1) * torch.arange(L, device=A.device) # (H N L)
        C = C * (torch.exp(dtA)-1.) / A
        K = 2 * torch.einsum('hn, hnl -> hl', C, torch.exp(K)).real

        return K

    def register(self, name, tensor, lr=None):
        """Register a tensor with a configurable learning rate and 0 weight decay"""

        if lr == 0.0:
            self.register_buffer(name, tensor)
        else:
            self.register_parameter(name, nn.Parameter(tensor))

            optim = {"weight_decay": 0.0}
            if lr is not None: optim["lr"] = lr
            setattr(getattr(self, name), "_optim", optim)

class S4D(nn.Module):
    def __init__(self, d_model, d_state=64, dropout=0.0, transposed=True, **kernel_args):
        super().__init__()

        self.h = d_model
        self.n = d_state
        self.d_output = self.h
        self.transposed = transposed

        self.D = nn.Parameter(torch.randn(self.h))

        # SSM Kernel
        self.kernel = S4DKernel(self.h, N=self.n, **kernel_args)

        # Pointwise
        self.activation = nn.GELU()
        # dropout_fn = nn.Dropout2d # NOTE: bugged in PyTorch 1.11
        dropout_fn = DropoutNd
        self.dropout = dropout_fn(dropout) if dropout > 0.0 else nn.Identity()

        # position-wise output transform to mix features
        self.output_linear = nn.Sequential(
            nn.Conv1d(self.h, 2*self.h, kernel_size=1),
            nn.GLU(dim=-2),
        )

    def forward(self, u, **kwargs): # absorbs return_output and transformer src mask
        """ Input and output shape (B, H, L) """
        if not self.transposed: u = u.transpose(-1, -2)
        L = u.size(-1)

        # Compute SSM Kernel
        k = self.kernel(L=L) # (H L)

        # Convolution
        k_f = torch.fft.rfft(k, n=2*L) # (H L)
        u_f = torch.fft.rfft(u, n=2*L) # (B H L)
        y = torch.fft.irfft(u_f*k_f, n=2*L)[..., :L] # (B H L)

        # Compute D term in state space equation - essentially a skip connection
        y = y + u * self.D.unsqueeze(-1)

        y = self.dropout(self.activation(y))
        y = self.output_linear(y)
        if not self.transposed: y = y.transpose(-1, -2)
        return y, None # Return a dummy state to satisfy this repo's interface, but this can be modified

class S4Model(nn.Module):
    def __init__(
        self,
        d_input,
        d_output=10,
        d_model=256,
        d_state=64,
        n_layers=4,
        dropout=0.2,
        prenorm=False,
        lr=None,
        dt_min=0.001,
        dt_max=0.1
    ):
        super().__init__()

        self.prenorm = prenorm

        # Linear encoder (d_input = 1 for grayscale and 3 for RGB)
        self.encoder = nn.Linear(d_input, d_model)

        # Stack S4 layers as residual blocks
        self.s4_layers = nn.ModuleList()
        self.norms = nn.ModuleList()
        self.dropouts = nn.ModuleList()
        for _ in range(n_layers):
            self.s4_layers.append(
                S4D(d_model, d_state=d_state, dropout=dropout, transposed=True,
                    dt_min=dt_min, dt_max=dt_max, lr=lr)
            )
            self.norms.append(nn.LayerNorm(d_model))
            self.dropouts.append(DropoutNd(dropout))

        # Linear decoder
        self.decoder = nn.Linear(d_model, d_output)

    def forward(self, x, mask=None):
        """
        Input x is shape (B, L, d_input)
        mask: optional (B, L) bool — True for real positions, False for padding.
              When provided, pooling averages only over real positions.
        """
        x = self.encoder(x)  # (B, L, d_input) -> (B, L, d_model)
        x = x.transpose(-1, -2)  # (B, L, d_model) -> (B, d_model, L)

        for layer, norm, dropout in zip(self.s4_layers, self.norms, self.dropouts):
            # Each iteration of this loop will map (B, d_model, L) -> (B, d_model, L)

            z = x
            if self.prenorm:
                # Prenorm
                z = norm(z.transpose(-1, -2)).transpose(-1, -2)

            # Apply S4 block: we ignore the state input and output
            z, _ = layer(z)

            # Dropout on the output of the S4 block
            z = dropout(z)

            # Residual connection
            x = z + x

            if not self.prenorm:
                # Postnorm
                x = norm(x.transpose(-1, -2)).transpose(-1, -2)

        x = x.transpose(-1, -2)  # (B, d_model, L) -> (B, L, d_model)

        # Pooling: masked or full mean over the sequence length
        if mask is not None:
            mask_f = mask.unsqueeze(-1).float()  # (B, L, 1)
            x = (x * mask_f).sum(dim=1) / mask_f.sum(dim=1).clamp(min=1)
        else:
            x = x.mean(dim=1)

        # Decode the outputs
        x = self.decoder(x)  # (B, d_model) -> (B, d_output)

        return x
