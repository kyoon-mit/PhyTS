"""
LinOSS seq2seq denoiser for TIDMAD, wrapped in PyTorch Lightning.

The model receives the noisy SQUID signal (channel0001) and predicts the
clean reference (channel0002).

Built on JAXLightningModule: JAX optimization runs via optax; PyTorch Lightning
handles data loading, logging, and checkpointing (use JAXCheckpointManager).

Usage (LightningCLI YAML):
    model:
      class_path: tasks.TIDMAD.linoss_denoising.TIDMADLinOSSDenoising
      init_args:
        num_blocks: 2
        ssm_size: 32
        H: 64
        discretization: IMEX
        lr: 1.0e-3
        clip_grad_norm: 1.0

Batch convention (from TIDMADDataset.__getitem__):
    noisy   (B, L)  — channel0001
    clean   (B, L)  — channel0002
    params  (B, 3)  — unused during denoising training
"""

import equinox as eqx
import jax.numpy as jnp
import optax
import torch

from models.linoss import LinOSS
from models.utils.jax.wrapper import JAXLightningModule


def _jax_psd_loss(y_pred, y_true):
    """PSD loss: MSE between power spectral densities (along the time axis).

    y_pred, y_true: (B, L, 1)  — batch × time × channel
    Matches DenoisingPSD in toy_denoising.py and is directly aligned with the
    PSD-based SNR benchmark metric used in TIDMAD Benchmark 1.
    """
    psd_pred = jnp.abs(jnp.fft.rfft(y_pred, axis=1)) ** 2
    psd_true = jnp.abs(jnp.fft.rfft(y_true, axis=1)) ** 2
    return jnp.mean((psd_pred - psd_true) ** 2)


class TIDMADLinOSSDenoising(JAXLightningModule):
    """LinOSS seq2seq denoiser for TIDMAD.

    All init args are primitives so the class is fully configurable from YAML.
    """

    def __init__(
        self,
        num_blocks: int = 2,
        ssm_size: int = 32,
        H: int = 64,
        discretization: str = "damped_IMEX",
        r_min: float = 0.0,
        theta_max: float = 3.141592653589793,
        lr: float = 1e-3,
        clip_grad_norm: float | None = 1.0,
        seed: int = 0,
    ):
        model = LinOSS(
            num_blocks=num_blocks,
            N=1,           # single input channel
            ssm_size=ssm_size,
            H=H,
            output_dim=1,  # single output channel
            task="denoising",
            output_step=1,
            discretization=discretization,
            r_min=r_min,
            theta_max=theta_max,
            seed=seed,
        )
        super().__init__(
            model=model,
            loss_fn=_jax_psd_loss,
            optimizer=optax.adamw(lr),
            clip_grad_norm=clip_grad_norm,
        )

    def _prepare_batch(self, batch):
        """Unpack TIDMAD batch and add channel dim for LinOSS.

        batch entries are already JAX arrays (converted in training_step).
        noisy, clean: (B, L) -> (B, L, 1)
        """
        noisy, clean, _ = batch
        return noisy[..., None], clean[..., None]

    def configure_optimizers(self):
        """Initialize JAX/optax optimizer state; return dummy PyTorch optimizer."""
        self.jax_optimizer = optax.chain(
            optax.clip_by_global_norm(self.clip_grad_norm)
            if self.clip_grad_norm is not None
            else optax.identity(),
            self.jax_optimizer,
        )
        diff_model, _ = eqx.partition(self.jax_model, self.jax_model_filter_spec)
        self.opt_state = self.jax_optimizer.init(diff_model)
        # Dummy PyTorch optimizer — Lightning requires one; actual updates are in JAX.
        dummy = torch.nn.Parameter(torch.zeros(1), requires_grad=False)
        return torch.optim.SGD([dummy], lr=0.0)
