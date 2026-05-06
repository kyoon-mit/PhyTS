"""Combined denoising + GaussianNLL-regression task for Project 8.

Mirrors the structure of ``LitS4CombinedModel`` from
[anonymous repository, available upon acceptance]
the encoder is expected to return ``(x_denoised, y_preds)`` and the total loss
is a weighted sum of a denoising loss and a regression loss::

    loss = lambda_denoise * denoising_loss(x_denoised, x_clean)
         + lambda_regress * GaussianNLL(y_preds, y)

Each lambda can either stay constant (default) or follow a
:class:`functions.curriculum_scheduler.LambdaScheduler` schedule passed as a
dict via ``lambda_denoise_schedule`` / ``lambda_regress_schedule``, e.g.::

    lambda_regress_schedule:
      schedule_type: step
      step_schedule:
        - [0,  0.0]   # epochs  0-29: regression off
        - [30, 0.5]   # epochs 30-59: regression at half weight
        - [60, 1.0]   # epochs 60+:   regression at full weight

The current lambda values are refreshed at the start of every training
epoch and logged as ``lambda/denoise`` and ``lambda/regress``.

``denoising_loss``
selects one of:

  * ``'MSELoss'``                — plain time-domain MSE.
  * ``'MixtureMSESpectralLoss'`` — convex mixture (default), parameterised
    by ``denoising_spectral_alpha``:
    ``alpha * MSE + (1 - alpha) * MSE(|rfft(x)|, |rfft(y)|)``.  Verbatim
    port of the loss in the original neutrino_project denoising branch.

The regression head is heteroscedastic: the regressor must produce ``(B, 2)``
interpreted as ``[mean, raw_var]``; ``F.softplus`` enforces positive variance,
matching ``Project8Regression``.

Expected DataModule batches: ``(X_noisy, X_clean, var)`` from
``Project8DataModule(combined=True)``.
"""

from typing import Optional

import lightning as L
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim

from functions.curriculum_scheduler import LambdaScheduler
from functions.losses import MixtureMSESpectralLoss


class Project8CombinedRegression(L.LightningModule):
    """Joint denoising + heteroscedastic regression for Project 8.

    Args:
        encoder: nn.Module returning ``(x_denoised, y_preds)``; the regression
            head must output ``(B, 2)`` ``[mean, raw_var]`` for one z-scored
            target.
        lambda_denoise / lambda_regress: constant loss weights.
        freeze_denoiser: if True, freeze ``encoder.denoiser`` (kept in eval
            mode throughout training so dropout stays off).
    """

    def __init__(
        self,
        encoder: nn.Module,
        learning_rate: float = 1e-3,
        gamma: float = 0.25,
        lr_patience: int = 5,
        threshold: float = 1e-3,
        lr_min: float = 1e-6,
        eps: float = 1e-6,
        lambda_denoise: float = 1.0,
        lambda_regress: float = 1.0,
        lambda_denoise_schedule: Optional[dict] = None,
        lambda_regress_schedule: Optional[dict] = None,
        trainer_max_epochs: int = 100,
        freeze_denoiser: bool = False,
        denoising_loss: str = 'MixtureMSESpectralLoss',
        denoising_spectral_alpha: float = 0.5,
    ):
        super().__init__()
        self.save_hyperparameters(ignore=['encoder'])
        self.encoder = encoder
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.lr_patience = lr_patience
        self.threshold = threshold
        self.lr_min = lr_min
        self.eps = eps
        self.lambda_denoise = lambda_denoise
        self.lambda_regress = lambda_regress
        self.trainer_max_epochs = trainer_max_epochs
        self.freeze_denoiser = freeze_denoiser

        def _make_scheduler(schedule_cfg, default_value):
            if schedule_cfg is None:
                return LambdaScheduler(
                    schedule_type='constant',
                    start_value=default_value,
                    total_epochs=trainer_max_epochs,
                )
            cfg = dict(schedule_cfg)
            cfg.setdefault('start_value', default_value)
            cfg.setdefault('total_epochs', trainer_max_epochs)
            return LambdaScheduler(**cfg)

        self.denoise_lambda_scheduler = _make_scheduler(
            lambda_denoise_schedule, lambda_denoise,
        )
        self.regress_lambda_scheduler = _make_scheduler(
            lambda_regress_schedule, lambda_regress,
        )

        if self.freeze_denoiser:
            if not hasattr(self.encoder, 'denoiser'):
                raise AttributeError(
                    "freeze_denoiser=True requires encoder to expose a "
                    "`denoiser` submodule (e.g. S4DCombinedModel)."
                )
            self.encoder.denoiser.eval()
            for p in self.encoder.denoiser.parameters():
                p.requires_grad = False

        # MixtureMSESpectralLoss takes its FFT along dim=1 (the time axis of
        # (B, L, C) batches), matching the reference implementation.
        if denoising_loss == 'MSELoss':
            self.denoising_criterion = nn.MSELoss()
        elif denoising_loss == 'MixtureMSESpectralLoss':
            self.denoising_criterion = MixtureMSESpectralLoss(
                alpha=denoising_spectral_alpha,
            )
        else:
            raise ValueError(
                f"Unknown denoising_loss {denoising_loss!r}; expected one of "
                "'MSELoss', 'MixtureMSESpectralLoss'."
            )

        self._target_name: Optional[str] = None
        self._target_mu: Optional[float] = None
        self._target_std: Optional[float] = None

    def train(self, mode: bool = True):
        super().train(mode)
        if self.freeze_denoiser:
            self.encoder.denoiser.eval()
        return self

    # ─── DataModule wiring ──────────────────────────────────────────────────
    def setup(self, stage: Optional[str] = None):
        dm = self.trainer.datamodule
        if not getattr(dm.hparams, 'combined', False):
            raise ValueError(
                "Project8CombinedRegression requires Project8DataModule to "
                "be instantiated with combined=True so batches are "
                "(X_noisy, X_clean, var)."
            )
        variables = list(dm.hparams.variables)
        if len(variables) != 1:
            raise ValueError(
                "Project8CombinedRegression expects exactly one variable in "
                f"the DataModule, got {variables}"
            )
        self._target_name = variables[0]
        if dm.mu is not None and dm.stds is not None:
            self._target_mu = float(dm.mu[0])
            self._target_std = float(dm.stds[0])
        else:
            self._target_mu, self._target_std = 0.0, 1.0

    # ─── curriculum schedulers ──────────────────────────────────────────────
    def on_train_epoch_start(self):
        epoch = self.trainer.current_epoch
        self.lambda_denoise = self.denoise_lambda_scheduler.get_lambda(epoch)
        self.lambda_regress = self.regress_lambda_scheduler.get_lambda(epoch)
        self.log('lambda/denoise', self.lambda_denoise,
                 on_step=False, on_epoch=True, logger=True)
        self.log('lambda/regress', self.lambda_regress,
                 on_step=False, on_epoch=True, logger=True)

    # ─── forward / step ─────────────────────────────────────────────────────
    def forward(self, x):
        return self.encoder(x)

    def _shared_step(self, batch):
        x_noisy, x_clean, var = batch
        x_denoised, y_preds = self.forward(x_noisy)

        loss_denoise = self.denoising_criterion(x_denoised, x_clean)

        y = var
        if y.ndim > 1 and y.shape[-1] == 1:
            y = y.squeeze(-1)
        mean, raw_var = torch.chunk(y_preds, 2, dim=-1)
        mu = mean.squeeze(-1)
        v  = F.softplus(raw_var).squeeze(-1)
        loss_regress = F.gaussian_nll_loss(mu, y, v, eps=self.eps)

        loss = self.lambda_denoise * loss_denoise + self.lambda_regress * loss_regress
        return loss, loss_denoise, loss_regress, mu, y

    def _log_rmse(self, mu, y, prefix):
        rmse_z = (mu - y).pow(2).mean().sqrt()
        self.log(f'{prefix}/rmse_z/{self._target_name}', rmse_z,
                 on_step=False, on_epoch=True)
        rmse_real = rmse_z * (self._target_std + 1e-8)
        self.log(f'{prefix}/rmse/{self._target_name}', rmse_real,
                 on_step=False, on_epoch=True)

    def training_step(self, batch, batch_idx):
        loss, loss_denoise, loss_regress, mu, y = self._shared_step(batch)
        self.log('train/loss',         loss,         on_step=False, on_epoch=True, prog_bar=True)
        self.log('train/loss_denoise', loss_denoise, on_step=False, on_epoch=True)
        self.log('train/loss_regress', loss_regress, on_step=False, on_epoch=True)
        self._log_rmse(mu, y, 'train')
        return loss

    def validation_step(self, batch, batch_idx):
        loss, loss_denoise, loss_regress, mu, y = self._shared_step(batch)
        self.log('val/loss',         loss,         on_step=False, on_epoch=True, prog_bar=True)
        self.log('val/loss_denoise', loss_denoise, on_step=False, on_epoch=True)
        self.log('val/loss_regress', loss_regress, on_step=False, on_epoch=True)
        self._log_rmse(mu, y, 'val')

    def test_step(self, batch, batch_idx):
        loss, loss_denoise, loss_regress, mu, y = self._shared_step(batch)
        self.log('test/loss',         loss,         on_step=False, on_epoch=True, prog_bar=True)
        self.log('test/loss_denoise', loss_denoise, on_step=False, on_epoch=True)
        self.log('test/loss_regress', loss_regress, on_step=False, on_epoch=True)
        self._log_rmse(mu, y, 'test')

    def configure_optimizers(self):
        params = (p for p in self.parameters() if p.requires_grad)
        optimizer = optim.AdamW(params, lr=self.learning_rate)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=self.gamma,
            patience=self.lr_patience,
            threshold=self.threshold,
            min_lr=self.lr_min,
        )
        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'monitor': 'val/loss',
                'interval': 'epoch',
            },
        }
