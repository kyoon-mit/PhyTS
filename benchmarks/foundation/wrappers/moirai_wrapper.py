"""
MOIRAI wrapper -- Salesforce's universal probabilistic forecaster.

Package:        uni2ts
HuggingFace:    Salesforce/moirai-1.1-R-{small,base,large}, Salesforce/moirai-2.0-R-small

MOIRAI expects GluonTS ListDataset input.  We convert numpy batches via
`_to_gluonts` (defined in base.py).  The predictor iterates through the
dataset and returns `SampleForecast` objects; we stack their samples into
(B, num_samples, horizon) and use the median as the point forecast.

Embedding: MOIRAI does not have a native embedding head.  Optional fallback
via forward hook on the encoder is possible but not required for the
benchmark (set supports_embed = False).
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import torch

from .base import (
    BaseFoundationModel,
    EmbeddingResult,
    ForecastResult,
)


_SIZE_TO_HF = {
    "small": "Salesforce/moirai-1.1-R-small",
    "base":  "Salesforce/moirai-1.1-R-base",
    "large": "Salesforce/moirai-1.1-R-large",
}


class MoiraiWrapper(BaseFoundationModel):
    name = "moirai"
    supports_forecast = True
    supports_denoise = False
    supports_embed = True      # via forward hook on encoder.norm

    def __init__(self) -> None:
        super().__init__()
        self._predictor = None
        self._module = None
        self._num_samples = 20
        self._context_len = 512
        self._horizon = 128
        self._patch_size: int | str = "auto"

    def load(
        self,
        device: str = "cuda",
        model_size: str = "base",
        *,
        hf_id: Optional[str] = None,
        context_len: int = 512,
        horizon: int = 128,
        patch_size: str | int = "auto",
        num_samples: int = 20,
        **_unused,
    ) -> None:
        try:
            from uni2ts.model.moirai import MoiraiForecast, MoiraiModule
        except ImportError as e:
            raise ImportError(
                "uni2ts is required for MoiraiWrapper. "
                "Install: pip install uni2ts"
            ) from e
        self.device = device
        self._num_samples = num_samples
        self._context_len = context_len
        self._horizon = horizon
        self._patch_size = patch_size
        hf = hf_id or _SIZE_TO_HF[model_size]

        self._module = MoiraiModule.from_pretrained(hf)
        forecast_model = MoiraiForecast(
            module=self._module,
            prediction_length=horizon,
            context_length=context_len,
            patch_size=patch_size,
            num_samples=num_samples,
            target_dim=1,
            feat_dynamic_real_dim=0,
            past_feat_dynamic_real_dim=0,
        )
        # Move the nn.Module to the requested device and put it in eval mode.
        forecast_model.eval()
        for p in forecast_model.parameters():
            p.requires_grad_(False)
        self.model = forecast_model
        self._predictor = forecast_model.create_predictor(
            batch_size=32, device=device,
        )

    # ── Forecasting ───────────────────────────────────────────────────────

    @torch.no_grad()
    def forecast(self, context: np.ndarray, horizon: int) -> ForecastResult:
        if self._predictor is None:
            raise RuntimeError("Call .load() before forecast().")
        ds = self._to_gluonts(context, freq="s")
        it = self._predictor.predict(ds)
        samples_list = []
        for fc in it:
            # Each `fc` is a SampleForecast with .samples of shape (num_samples, H)
            samples_list.append(np.asarray(fc.samples, dtype=np.float32))
        samples = np.stack(samples_list, axis=0)           # (B, num_samples, H)
        # Trim or verify horizon
        samples = samples[:, :, :horizon]
        point = np.median(samples, axis=1).astype(np.float32)  # (B, H)
        return ForecastResult(point_forecast=point, samples=samples)

    # ── Embedding via encoder hidden-state hook ───────────────────────────

    @torch.no_grad()
    def embed(self, signal: np.ndarray, *, pool: str = "mean") -> EmbeddingResult:
        """Run a dummy forecast and hook `encoder.norm` to capture hidden states.

        Requires the wrapper to have been loaded with a fixed `patch_size`
        (not "auto"); the auto path runs the encoder once per candidate patch
        size and makes the captures ambiguous. We pool over the context
        tokens only (excluding the horizon tokens the encoder sees for
        masked-prediction) to get a clean per-sample representation.
        """
        if self._predictor is None:
            raise RuntimeError("Call .load() before embed().")
        if self._patch_size == "auto":
            raise RuntimeError(
                "MOIRAI embed() requires a fixed `patch_size` at load() time "
                "(e.g. 32). The auto-search path produces multiple encoder "
                "passes with different token counts."
            )

        captured: list[torch.Tensor] = []

        def _hook(_m, _inp, out):
            h = out[0] if isinstance(out, tuple) else out
            if torch.is_tensor(h):
                captured.append(h.detach())

        handle = self._module.encoder.norm.register_forward_hook(_hook)
        try:
            self.forecast(signal, horizon=self._horizon)
        finally:
            handle.remove()

        if not captured:
            raise RuntimeError("MOIRAI encoder hook captured no output.")
        h = captured[0]  # (B, T, d); T = (context_len + horizon) / patch_size
        # Keep only context tokens
        p = int(self._patch_size)
        ctx_tokens = self._context_len // p
        h = h[:, :ctx_tokens]
        emb = self._pool_time(h, pool=pool)
        return EmbeddingResult(embeddings=emb.cpu().numpy().astype(np.float32))

    # ── Grad-flowing embedding for fine-tuning ───────────────────────────

    def embed_torch(self, signal: np.ndarray, *, pool: str = "mean") -> torch.Tensor:
        """Same as embed() but returns a torch tensor with grads enabled.

        MoiraiModule.forward expects a fully-pre-processed input (target,
        observed_mask, sample_id, time_id, variate_id, prediction_mask,
        patch_size) which the GluonTS predictor builds internally. To avoid
        replicating that pipeline, we let the predictor run a no-grad
        forecast pass while a forward pre-hook captures the encoder's
        already-built inputs (x, attn_mask, var_id, time_id), then we
        re-call ``self._module.encoder`` on those inputs with grads enabled.
        Cost: one extra encoder forward per batch, but the patch-embedding
        / token-id construction is reused.

        LoRA modules injected into ``encoder.layers.*.self_attn.{q,k,v,o}_proj``
        produce gradient flow through the LoRA params; everything earlier in
        the pipeline is frozen, which matches the LoRA fine-tuning intent.
        """
        if self._predictor is None:
            raise RuntimeError("Call .load() before embed_torch().")
        if self._patch_size == "auto":
            raise RuntimeError(
                "MOIRAI embed_torch() requires a fixed `patch_size` at "
                "load() time (e.g. 32)."
            )

        captured_inputs: list[tuple] = []

        def _pre_hook(_m, args, kwargs):
            captured_inputs.append((args, dict(kwargs)))

        handle = self._module.encoder.register_forward_pre_hook(
            _pre_hook, with_kwargs=True,
        )
        try:
            # forecast() is @torch.no_grad and uses the GluonTS predictor.
            self.forecast(signal, horizon=self._horizon)
        finally:
            handle.remove()

        if not captured_inputs:
            raise RuntimeError("MOIRAI encoder pre-hook captured nothing.")

        args, kwargs = captured_inputs[0]
        # Captured tensors are no-grad leaves; encoder weights (incl. LoRA
        # params) carry the grad signal.
        h = self._module.encoder(*args, **kwargs)  # (B, T, d)

        p = int(self._patch_size)
        ctx_tokens = self._context_len // p
        h = h[:, :ctx_tokens]
        return self._pool_time(h, pool=pool)

    def get_finetune_backbone(self) -> torch.nn.Module:
        if self._module is None:
            raise RuntimeError("Call .load() before get_finetune_backbone().")
        return self._module
