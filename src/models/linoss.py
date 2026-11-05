"""LinOSS: Linear Operator State Space Models for Sequence Modeling.

Original LinOSS: https://openreview.net/pdf?id=GRMfXcAAFh
Damped Variant: https://arxiv.org/abs/2505.12171
https://github.com/tk-rusch/linoss/tree/main
"""

import math
from typing import List, Optional

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
from jax import nn


def _simple_uniform_init(rng, shape, std=1.0):
    weights = jr.uniform(rng, shape) * 2.0 * std - std
    return weights


def _dense(x, w, matmul_dtype=None):
    """``x @ w``; if ``matmul_dtype`` is set (e.g. ``"bfloat16"``), cast the operands and accumulate in float32."""
    if matmul_dtype is None:
        return x @ w
    return jnp.matmul(x.astype(matmul_dtype), w.astype(matmul_dtype), preferred_element_type=jnp.float32)


class GLU(eqx.Module):
    """Gated Linear Unit (GLU) for nonlinearity in LinOSS blocks."""

    w1: eqx.nn.Linear
    w2: eqx.nn.Linear
    matmul_dtype: Optional[str] = eqx.field(static=True, default=None)

    def __init__(self, input_dim, output_dim, key, matmul_dtype=None):
        w1_key, w2_key = jr.split(key, 2)
        self.w1 = eqx.nn.Linear(input_dim, output_dim, use_bias=True, key=w1_key)
        self.w2 = eqx.nn.Linear(input_dim, output_dim, use_bias=True, key=w2_key)
        self.matmul_dtype = matmul_dtype

    def __call__(self, x):
        """GLU forward pass on ``(..., input_dim)``."""
        a = _dense(x, self.w1.weight.T, self.matmul_dtype) + self.w1.bias
        b = _dense(x, self.w2.weight.T, self.matmul_dtype) + self.w2.bias
        return a * jax.nn.sigmoid(b)


# Parallel scan operations
@jax.vmap
def binary_operator(q_i, q_j):
    """Binary operator for parallel scan of linear recurrence. Assumes a diagonal matrix A.

    Args:
        q_i: tuple containing A_i and Bu_i at position i       (P,), (P,)
        q_j: tuple containing A_j and Bu_j at position j       (P,), (P,)

    Returns:
        new element ( A_out, Bu_out )
    """
    A_i, b_i = q_i
    A_j, b_j = q_j

    N = A_i.size // 4
    iA_ = A_i[0 * N : 1 * N]
    iB_ = A_i[1 * N : 2 * N]
    iC_ = A_i[2 * N : 3 * N]
    iD_ = A_i[3 * N : 4 * N]
    jA_ = A_j[0 * N : 1 * N]
    jB_ = A_j[1 * N : 2 * N]
    jC_ = A_j[2 * N : 3 * N]
    jD_ = A_j[3 * N : 4 * N]
    A_new = jA_ * iA_ + jB_ * iC_
    B_new = jA_ * iB_ + jB_ * iD_
    C_new = jC_ * iA_ + jD_ * iC_
    D_new = jC_ * iB_ + jD_ * iD_
    Anew = jnp.concatenate([A_new, B_new, C_new, D_new])

    b_i1 = b_i[0:N]
    b_i2 = b_i[N:]

    new_b1 = jA_ * b_i1 + jB_ * b_i2
    new_b2 = jC_ * b_i1 + jD_ * b_i2
    new_b = jnp.concatenate([new_b1, new_b2])

    return Anew, new_b + b_j


def _apply_damped_linoss_imex(A_diag, G_diag, Bu_elements, step):  # noqa: N802
    """Compute the Damped LinOSS-IMEX recurrence.

    Args:
        A_diag: Diagonal state matrix.
        G_diag: Diagonal damping matrix.
        Bu_elements: Projected input sequence B @ u, shape (timesteps, state_dim).
        step: Discretization time-step.

    Returns:
        Hidden state sequence, shape (timesteps, state_dim).
    """

    Identity = jnp.ones_like(A_diag)
    S = Identity + step * G_diag
    M_11 = 1.0 / S
    M_12 = -step / S * A_diag
    M_21 = step / S
    M_22 = Identity - step**2 / S * A_diag

    M = jnp.concatenate([M_11, M_12, M_21, M_22])
    M_elements = M * jnp.ones((Bu_elements.shape[0], 4 * A_diag.shape[0]))

    F1 = step * (1.0 / S) * Bu_elements
    F2 = step**2 * (1.0 / S) * Bu_elements
    F = jnp.hstack((F1, F2))

    _, xs = jax.lax.associative_scan(binary_operator, (M_elements, F))
    ys = xs[:, A_diag.shape[0] :]

    return ys


def _map_theta_to_A(thetas, G_diag, steps):  # noqa: N802
    """Map theta parameter to diagonal state matrix A for damped LinOSS-IMEX.

    Args:
        thetas: Theta parameter values.
        G_diag: Diagonal damping matrix.
        steps: Discretization time-steps.

    Returns:
        Diagonal state matrix A.
    """
    A_plus = (
        4
        * jnp.sqrt(steps**4 * jnp.cos(thetas) ** (-2) + steps**5 * G_diag * jnp.cos(thetas) ** (-2))
        - steps**2
        * (
            -4
            - 2 * steps * G_diag
            - 4 * jnp.tan(thetas) ** 2
            - 2 * steps * G_diag * jnp.tan(thetas) ** 2
        )
    ) / (2 * steps**4 * (1 + jnp.tan(thetas) ** 2))
    A_minus = (
        -4
        * jnp.sqrt(steps**4 * jnp.cos(thetas) ** (-2) + steps**5 * G_diag * jnp.cos(thetas) ** (-2))
        - steps**2
        * (
            -4
            - 2 * steps * G_diag
            - 4 * jnp.tan(thetas) ** 2
            - 2 * steps * G_diag * jnp.tan(thetas) ** 2
        )
    ) / (2 * steps**4 * (1 + jnp.tan(thetas) ** 2))
    return jnp.where(thetas > jnp.pi / 2, A_plus, A_minus)


def apply_linoss_im(A_diag, Bu_elements, step):
    r"""Compute the LxP hidden states of LinOSS-IM given the LxP projected input.

    Args:
        A_diag  (float32):   diagonal state matrix   (P,)
        Bu_elements (complex64): projected input sequence B @ u    (L, P)
        step    (float):     discretization time-step $\Delta_t$  (P,)

    Returns:
        ys (complex64): hidden states      (L, P)
    """

    schur_comp = 1.0 / (1.0 + step**2.0 * A_diag)
    M_IM_11 = 1.0 - step**2.0 * A_diag * schur_comp
    M_IM_12 = -1.0 * step * A_diag * schur_comp
    M_IM_21 = step * schur_comp
    M_IM_22 = schur_comp

    M_IM = jnp.concatenate([M_IM_11, M_IM_12, M_IM_21, M_IM_22])

    M_IM_elements = M_IM * jnp.ones((Bu_elements.shape[0], 4 * A_diag.shape[0]))

    F1 = M_IM_11 * Bu_elements * step
    F2 = M_IM_21 * Bu_elements * step
    F = jnp.hstack((F1, F2))

    _, xs = jax.lax.associative_scan(binary_operator, (M_IM_elements, F))
    return xs[:, A_diag.shape[0] :]


def apply_linoss_imex(A_diag, Bu_elements, step):
    r"""Compute the LxP hidden states of LinOSS-IMEX given the LxP projected input.

    Args:
        A_diag  (float32):   diagonal state matrix   (P,)
        Bu_elements (complex64): projected input sequence B @ u    (L, P)
        step    (float):     discretization time-step $\Delta_t$  (P,)

    Returns:
        ys (complex64): hidden states      (L, P)
    """

    A_ = jnp.ones_like(A_diag)
    B_ = -1.0 * step * A_diag
    C_ = step
    D_ = 1.0 - (step**2.0) * A_diag

    M_IMEX = jnp.concatenate([A_, B_, C_, D_])

    M_IMEX_elements = M_IMEX * jnp.ones((Bu_elements.shape[0], 4 * A_diag.shape[0]))

    F1 = Bu_elements * step
    F2 = Bu_elements * (step**2.0)
    F = jnp.hstack((F1, F2))

    _, xs = jax.lax.associative_scan(binary_operator, (M_IMEX_elements, F))
    return xs[:, A_diag.shape[0] :]


class LinOSSLayer(eqx.Module):
    """Single LinOSS layer with IMEX, IM, or damped_IMEX discretization."""

    A_diag: jax.Array
    G_diag: Optional[jax.Array]
    B: jax.Array
    C: jax.Array
    D: jax.Array
    steps: jax.Array
    discretization: str
    matmul_dtype: Optional[str] = eqx.field(static=True, default=None)

    def __init__(
        self,
        ssm_size,
        H,
        discretization,
        r_min=0.0,
        theta_max=math.pi,
        matmul_dtype=None,
        *,
        key,
    ):
        B_key, C_key, D_key, A_key, step_key, G_key = jr.split(key, 6)

        if discretization == "damped_IMEX":
            self.steps = jr.normal(step_key, shape=(ssm_size,)) * 0.5
            steps = nn.sigmoid(self.steps)
            r_max = 1.0
            mags = jnp.sqrt(jr.uniform(G_key, shape=(ssm_size,)) * (r_max**2 - r_min**2) + r_min**2)
            self.G_diag = (1 - mags**2) / (steps * mags**2)
            G_diag = nn.relu(self.G_diag)
            theta = jr.uniform(A_key, shape=(ssm_size,)) * theta_max
            self.A_diag = _map_theta_to_A(theta, G_diag, steps)
        else:
            self.steps = jr.uniform(step_key, shape=(ssm_size,))
            self.G_diag = None
            self.A_diag = jr.uniform(A_key, shape=(ssm_size,))

        self.B = _simple_uniform_init(B_key, shape=(ssm_size, H, 2), std=1.0 / math.sqrt(H))
        self.C = _simple_uniform_init(C_key, shape=(H, ssm_size, 2), std=1.0 / math.sqrt(ssm_size))
        self.D = nn.initializers.normal(stddev=1.0)(D_key, (H,))
        self.discretization = discretization
        self.matmul_dtype = matmul_dtype

    def __call__(self, input_sequence):
        """LinOSS layer forward pass."""
        # B/C projections in real arithmetic: a complex (L, H) copy of the input costs 2x the
        # memory and 4x the FLOPs, and dominated the layer's runtime. Only the (L, P) state is complex.
        P = self.B.shape[0]
        B_cat = jnp.concatenate([self.B[..., 0], self.B[..., 1]], axis=0)  # (2P, H)
        Bu = _dense(input_sequence, B_cat.T, self.matmul_dtype)
        Bu_elements = Bu[:, :P] + 1j * Bu[:, P:]
        steps = nn.sigmoid(self.steps)

        if self.discretization == "damped_IMEX":
            G_diag = nn.relu(self.G_diag)
            A_boundary_low = (2 + steps * G_diag - 2 * jnp.sqrt(1 + steps * G_diag)) / steps**2
            A_boundary_high = (2 + steps * G_diag + 2 * jnp.sqrt(1 + steps * G_diag)) / steps**2
            A_diag = (
                A_boundary_low
                + nn.relu(self.A_diag - A_boundary_low)
                - nn.relu(self.A_diag - A_boundary_high)
            )
            ys = _apply_damped_linoss_imex(A_diag, G_diag, Bu_elements, steps)
        elif self.discretization == "IMEX":
            A_diag = nn.relu(self.A_diag)
            ys = apply_linoss_imex(A_diag, Bu_elements, steps)
        elif self.discretization == "IM":
            A_diag = nn.relu(self.A_diag)
            ys = apply_linoss_im(A_diag, Bu_elements, steps)
        else:
            raise NotImplementedError(
                f"Discretization {self.discretization!r} not implemented. "
                "Choose from 'IMEX', 'IM', 'damped_IMEX'."
            )

        # Re(C y) = C_re y_re - C_im y_im
        C_cat = jnp.concatenate([self.C[..., 0], -self.C[..., 1]], axis=1)  # (H, 2P)
        out = _dense(jnp.concatenate([ys.real, ys.imag], axis=-1), C_cat.T, self.matmul_dtype)
        return out + self.D * input_sequence


class LinOSSBlock(eqx.Module):
    """Single LinOSS block with a LinOSS layer, GLU nonlinearity, and residual connection."""

    norm: eqx.nn.BatchNorm
    ssm: LinOSSLayer
    glu: GLU
    drop: eqx.nn.Dropout

    def __init__(
        self,
        ssm_size,
        H,
        discretization,
        drop_rate=0.05,
        r_min=0.0,
        theta_max=math.pi,
        matmul_dtype=None,
        *,
        key,
    ):
        ssmkey, glukey = jr.split(key, 2)
        self.norm = eqx.nn.BatchNorm(input_size=H, axis_name="batch", channelwise_affine=False)
        self.ssm = LinOSSLayer(
            ssm_size,
            H,
            discretization,
            r_min=r_min,
            theta_max=theta_max,
            matmul_dtype=matmul_dtype,
            key=ssmkey,
        )
        self.glu = GLU(H, H, key=glukey, matmul_dtype=matmul_dtype)
        self.drop = eqx.nn.Dropout(p=drop_rate)

    def __call__(self, x, state, *, key):
        """Compute LinOSS block."""
        dropkey1, dropkey2 = jr.split(key, 2)
        skip = x
        x, state = self.norm(x.T, state)
        x = x.T
        x = self.ssm(x)
        x = self.drop(jax.nn.gelu(x), key=dropkey1)
        x = self.glu(x)
        x = self.drop(x, key=dropkey2)
        x = skip + x
        return x, state


class ConvStem(eqx.Module):
    """Strided conv downsampler: log2(downsample) conv layers (kernel 7, stride 2), GroupNorm + GELU between."""

    convs: List[eqx.nn.Conv1d]
    norms: List[eqx.nn.GroupNorm]

    def __init__(self, in_channels, out_channels, downsample, channels=64, kernel_size=7, *, key):
        n_layers = int(math.log2(downsample))
        if n_layers < 1 or 2**n_layers != downsample:
            raise ValueError(f"conv stem needs a power-of-two downsample >= 2; got {downsample}")
        dims = [in_channels] + [channels] * (n_layers - 1) + [out_channels]
        self.convs = [
            eqx.nn.Conv1d(c_in, c_out, kernel_size, stride=2, padding=kernel_size // 2, key=k)
            for c_in, c_out, k in zip(dims[:-1], dims[1:], jr.split(key, n_layers))
        ]
        self.norms = [eqx.nn.GroupNorm(min(32, c), c) for c in dims[1:-1]]

    def __call__(self, x):
        """(L, in_channels) -> (L // downsample, out_channels)."""
        x = x.T
        for conv, norm in zip(self.convs[:-1], self.norms):
            x = jax.nn.gelu(norm(conv(x)))
        return self.convs[-1](x).T


class LinOSS(eqx.Module):
    """Full LinOSS model with multiple LinOSS blocks and a final linear layer for output.

    ``task`` selects the output head:
      * ``"classification"``: mean-pool over time, linear, softmax.
      * ``"regression"``:    mean-pool over time, linear (no activation).
      * ``"forecasting"``:   per-step subsample + linear + tanh.

    ``patch_size > 1`` downsamples the input before the LinOSS blocks: non-overlapping
    patches of ``patch_size`` steps are flattened and linearly embedded (a strided conv
    with kernel = stride = ``patch_size``), so every block runs on ``L // patch_size``
    steps. Trailing steps that do not fill a patch are dropped. Only supported for the
    pooled heads (classification / regression).

    ``stem="conv"`` replaces that linear patch embedding with a :class:`ConvStem`
    (``patch_size`` must be a power of two; ``stem_channels`` sets its hidden width).

    ``matmul_dtype`` (e.g. ``"bfloat16"``) runs the B/C projections and GLU matmuls in
    that dtype with float32 accumulation; parameters and the recurrence stay float32.

    Masked pooling: if the input passed to ``__call__`` has one more channel than
    ``input_dim`` (i.e. shape ``(L, N+1)``), the last channel is treated as a
    float validity mask (1 = valid, 0 = padding) and used for masked mean pooling
    instead of plain mean pooling.  The signal channels ``x[..., :N]`` are fed
    to the encoder as usual.  This convention lets callers pass the mask without
    changing the JAX training infrastructure.
    """

    linear_encoder: Optional[eqx.nn.Linear]
    blocks: List[LinOSSBlock]
    linear_layer: eqx.nn.Linear
    task: str
    output_step: int
    input_dim: int
    patch_size: int = eqx.field(static=True, default=1)
    conv_stem: Optional[ConvStem] = None
    stateful: bool = True
    nondeterministic: bool = True
    lip2: bool = False

    def __init__(
        self,
        num_blocks,
        N,
        ssm_size,
        H,
        output_dim,
        task,
        output_step,
        discretization,
        r_min=0.0,
        theta_max=math.pi,
        drop_rate=0.05,
        patch_size=1,
        stem="linear",
        stem_channels=64,
        matmul_dtype=None,
        *,
        key=None,
        seed=0,
    ):
        if key is None:
            key = jr.key(seed)

        linear_encoder_key, *block_keys, linear_layer_key, weightkey = jr.split(key, num_blocks + 3)
        if stem == "conv":
            self.linear_encoder = None
            self.conv_stem = ConvStem(N, H, patch_size, channels=stem_channels, key=linear_encoder_key)
        elif stem == "linear":
            self.linear_encoder = eqx.nn.Linear(N * patch_size, H, key=linear_encoder_key)
        else:
            raise ValueError(f"stem must be 'linear' or 'conv'; got {stem!r}")
        self.blocks = [
            LinOSSBlock(
                ssm_size,
                H,
                discretization,
                drop_rate=drop_rate,
                r_min=r_min,
                theta_max=theta_max,
                matmul_dtype=matmul_dtype,
                key=key,
            )
            for key in block_keys
        ]
        self.linear_layer = eqx.nn.Linear(H, output_dim, key=linear_layer_key)
        if task not in ("classification", "regression", "forecasting", "denoising"):
            raise ValueError(
                f"task must be one of 'classification', 'regression', 'forecasting', 'denoising'; got {task!r}"
            )
        if patch_size > 1 and task not in ("classification", "regression"):
            raise ValueError(f"patch_size > 1 is only supported for pooled tasks; got task={task!r}")
        self.task = task
        self.output_step = output_step
        self.input_dim = N
        self.patch_size = patch_size

    def __call__(self, x, state, key):
        """Compute LinOSS.

        x: (L, N) signal, or (L, N+1) where last channel is a float validity mask
           (1 = valid cadence, 0 = zero-padded).  The mask is extracted before the
           encoder and used for masked mean pooling; it does not affect encoder weights.
        """
        # Extract mask channel if appended by the task's _prepare_batch.
        if x.shape[-1] > self.input_dim:
            mask = x[..., -1]               # (L,) float: 1=valid, 0=padding
            x = x[..., : self.input_dim]    # (L, N) signal
        else:
            mask = None

        if self.conv_stem is not None:
            x = self.conv_stem(x)
        else:
            if self.patch_size > 1:
                n_patches = x.shape[0] // self.patch_size
                x = x[: n_patches * self.patch_size].reshape(n_patches, self.patch_size * self.input_dim)
            x = jax.vmap(self.linear_encoder)(x)

        dropkeys = jr.split(key, len(self.blocks))
        for block, key in zip(self.blocks, dropkeys):
            x, state = block(x, state, key=key)
        if self.task == "classification":
            x = jnp.mean(x, axis=0)
            x = jax.nn.softmax(self.linear_layer(x), axis=0)
        elif self.task == "regression":
            x = jnp.mean(x, axis=0)
            x = self.linear_layer(x)
        elif self.task == "denoising":
            x = jax.vmap(self.linear_layer)(x)  # (L, output_dim), no activation
        else:  # forecasting
            x = x[self.output_step - 1 :: self.output_step]
            x = jax.nn.tanh(jax.vmap(self.linear_layer)(x))
        return x, state
