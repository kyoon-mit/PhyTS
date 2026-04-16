"""
Chronos wrapper -- Amazon's probabilistic time-series foundation model.

Package:        chronos-forecasting
HuggingFace:    amazon/chronos-t5-{tiny,small,base,large} or amazon/chronos-bolt-{tiny,small,base}

Chronos tokenises real-valued time series via quantile binning then runs a
T5-style seq2seq transformer.  Key facts:

* `ChronosPipeline.predict(context_tensor, prediction_length)` returns a
  tensor of shape `(B, num_samples, prediction_length)` -- native
  probabilistic output we pass through as `samples`, and the median is used
  as the point forecast.
* The encoder hidden states are exposed via `pipeline.embed(context)` which
  returns `(B, T, d_model)` where T depends on the input length.  We
  mean-pool over T to get `(B, d_model)`.

Both `chronos-t5-*` and `chronos-bolt-*` variants work; bolt is faster but
the API is identical.
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


# Default HF IDs per size. Users can override via `hf_id` kwarg on load().
_SIZE_TO_HF = {
    "tiny":  "amazon/chronos-t5-tiny",
    "small": "amazon/chronos-t5-small",
    "base":  "amazon/chronos-t5-base",
    "large": "amazon/chronos-t5-large",
}


class ChronosWrapper(BaseFoundationModel):
    name = "chronos"
    supports_forecast = True
    supports_denoise = False       # uses denoise_via_forecast fallback
    supports_embed = True          # native

    def __init__(self) -> None:
        super().__init__()
        self.pipeline = None
        self._num_samples = 20

    # ── Loading ───────────────────────────────────────────────────────────

    def load(
        self,
        device: str = "cuda",
        model_size: str = "base",
        *,
        hf_id: Optional[str] = None,
        num_samples: int = 20,
        **_unused,
    ) -> None:
        try:
            from chronos import ChronosPipeline
        except ImportError as e:
            raise ImportError(
                "chronos-forecasting is required for ChronosWrapper. "
                "Install: pip install chronos-forecasting"
            ) from e
        self.device = device
        self._num_samples = num_samples
        hf = hf_id or _SIZE_TO_HF.get(model_size)
        if hf is None:
            raise ValueError(
                f"Unknown Chronos size '{model_size}'. "
                f"Choose from: {list(_SIZE_TO_HF)} or pass hf_id=..."
            )
        # Bring weights to the target device.
        torch_dtype = torch.float32  # float32 for consistent metrics
        self.pipeline = ChronosPipeline.from_pretrained(
            hf, device_map=device, torch_dtype=torch_dtype,
        )
        # Expose underlying model for unload / finetune hooks.
        self.model = getattr(self.pipeline, "model", None)

    def unload(self) -> None:
        self.pipeline = None
        super().unload()

    # ── Forecasting (native, probabilistic) ───────────────────────────────

    @torch.no_grad()
    def forecast(self, context: np.ndarray, horizon: int) -> ForecastResult:
        if self.pipeline is None:
            raise RuntimeError("Call .load() before forecast().")
        ctx = torch.from_numpy(context.astype(np.float32))
        # ChronosPipeline.predict accepts (B, L) or list of 1-D tensors
        samples = self.pipeline.predict(
            inputs=ctx,
            prediction_length=horizon,
            num_samples=self._num_samples,
            limit_prediction_length=False,
        )  # (B, num_samples, H)
        samples_np = samples.cpu().numpy().astype(np.float32)
        point = np.median(samples_np, axis=1).astype(np.float32)    # (B, H)
        return ForecastResult(point_forecast=point, samples=samples_np)

    # ── Embedding (native) ────────────────────────────────────────────────

    @torch.no_grad()
    def embed(self, signal: np.ndarray) -> EmbeddingResult:
        if self.pipeline is None:
            raise RuntimeError("Call .load() before embed().")
        ctx = torch.from_numpy(signal.astype(np.float32))
        # pipeline.embed returns (embeddings, tokenizer_state) where
        # embeddings is (B, T, d_model).
        out = self.pipeline.embed(ctx)
        if isinstance(out, tuple):
            emb = out[0]
        else:
            emb = out
        # Mean-pool over time axis to get (B, d_model)
        if emb.dim() == 3:
            emb = emb.mean(dim=1)
        return EmbeddingResult(embeddings=emb.cpu().numpy().astype(np.float32))
