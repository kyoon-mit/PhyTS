"""
Time-MoE wrapper -- autoregressive sparse-MoE time series model.

Package:        transformers (trust_remote_code=True)
HuggingFace:    Maple728/TimeMoE-50M (default), Maple728/TimeMoE-200M

Usage notes
-----------
* Inputs are pre-normalised (per-sample z-score) float tensors of shape
  (B, context_len).  The model auto-regresses `max_new_tokens=horizon`
  steps; outputs have shape (B, context_len + horizon) and the forecast
  is `output[:, -horizon:]`.
* After forecasting we denormalise with the per-sample mean/std captured
  before the forward pass.
* Embedding: Time-MoE does not expose a dedicated embedding head.  We
  extract the final decoder hidden state (`output_hidden_states=True`) and
  mean-pool over time to get (B, d_model).
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
    "small": "Maple728/TimeMoE-50M",
    "base":  "Maple728/TimeMoE-50M",
    "large": "Maple728/TimeMoE-200M",
}


class TimeMoEWrapper(BaseFoundationModel):
    name = "timemoe"
    supports_forecast = True
    supports_denoise = False
    supports_embed = True            # via hidden-state pooling

    def __init__(self) -> None:
        super().__init__()
        self._hf_id = ""

    def load(
        self,
        device: str = "cuda",
        model_size: str = "base",
        *,
        hf_id: Optional[str] = None,
        **_unused,
    ) -> None:
        try:
            from transformers import AutoModelForCausalLM
        except ImportError as e:
            raise ImportError(
                "transformers>=4.30 is required for TimeMoEWrapper."
            ) from e
        self.device = device
        self._hf_id = hf_id or _SIZE_TO_HF[model_size]
        self.model = AutoModelForCausalLM.from_pretrained(
            self._hf_id,
            trust_remote_code=True,
            torch_dtype=torch.float32,
        ).to(device).eval()

        # Compatibility shim: Time-MoE's bundled modeling_time_moe.py was
        # written for transformers<=4.44 and calls `DynamicCache.seen_tokens`
        # and `DynamicCache.get_max_length()`, both removed upstream.  We
        # restore them as thin wrappers around the new API.
        try:
            from transformers.cache_utils import DynamicCache
            if not hasattr(DynamicCache, "seen_tokens"):
                def _seen_tokens(self):
                    try:
                        return self.get_seq_length()
                    except Exception:
                        return 0
                DynamicCache.seen_tokens = property(_seen_tokens)
            if not hasattr(DynamicCache, "get_max_length"):
                def _get_max_length(self):
                    return None   # unlimited / unspecified
                DynamicCache.get_max_length = _get_max_length
            if not hasattr(DynamicCache, "get_usable_length"):
                def _get_usable_length(self, new_seq_length, layer_idx=0):
                    # Old behavior: min(max_len - new_seq_length, current_len)
                    # New Cache classes track per-layer; we approximate with
                    # total seen tokens.
                    try:
                        return self.get_seq_length(layer_idx)
                    except Exception:
                        return 0
                DynamicCache.get_usable_length = _get_usable_length
        except Exception:
            pass

        # Compatibility shim: Time-MoE's ts_generation_mixin.py expects
        # `_extract_past_from_model_output` to return just the cache object.
        # In transformers >=4.44, the method returns (cache_name, cache) and
        # only accepts `(self, outputs)` (no standardize_cache_format kwarg).
        # We patch it to accept the extra kwarg AND return just the cache,
        # restoring the pre-4.44 contract that Time-MoE's mixin relies on.
        try:
            from transformers.generation.utils import GenerationMixin
            import inspect
            _orig_extract = GenerationMixin._extract_past_from_model_output
            sig = inspect.signature(_orig_extract)
            def _extract_patched(self, outputs, standardize_cache_format=False):
                result = _orig_extract(self, outputs)
                # transformers>=4.44 returns (cache_name, cache) — unwrap to just cache
                if isinstance(result, tuple) and len(result) == 2 and isinstance(result[0], str):
                    return result[1]
                return result
            GenerationMixin._extract_past_from_model_output = _extract_patched
        except Exception:
            pass

    # ── Forecasting ───────────────────────────────────────────────────────

    @torch.no_grad()
    def forecast(self, context: np.ndarray, horizon: int) -> ForecastResult:
        if self.model is None:
            raise RuntimeError("Call .load() before forecast().")
        # Per-sample z-score normalisation
        normed, mean, std = self._instance_normalize(context.astype(np.float32))
        x = torch.from_numpy(normed).to(self.device)
        out = self.model.generate(x, max_new_tokens=horizon)
        # out shape: (B, context_len + horizon)
        forecast_normed = out[:, -horizon:].cpu().numpy().astype(np.float32)
        forecast = self._instance_denormalize(forecast_normed, mean, std)
        return ForecastResult(point_forecast=forecast.astype(np.float32), samples=None)

    # ── Embedding via final hidden state ──────────────────────────────────

    @torch.no_grad()
    def embed(self, signal: np.ndarray, *, pool: str = "mean") -> EmbeddingResult:
        if self.model is None:
            raise RuntimeError("Call .load() before embed().")
        emb = self._embed_forward(signal, pool=pool)
        return EmbeddingResult(embeddings=emb.detach().cpu().numpy().astype(np.float32))

    def embed_torch(self, signal: np.ndarray, *, pool: str = "mean") -> torch.Tensor:
        if self.model is None:
            raise RuntimeError("Call .load() before embed_torch().")
        return self._embed_forward(signal, pool=pool)

    def _embed_forward(self, signal: np.ndarray, *, pool: str) -> torch.Tensor:
        normed, _, _ = self._instance_normalize(signal.astype(np.float32))
        x = torch.from_numpy(normed).to(self.device)
        out = self.model(x, output_hidden_states=True)
        hs = out.hidden_states[-1]          # (B, T, d_model)
        return self._pool_time(hs, pool=pool)
