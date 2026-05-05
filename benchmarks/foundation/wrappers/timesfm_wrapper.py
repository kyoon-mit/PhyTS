"""
TimesFM wrapper -- Google's time-series foundation model (PyTorch variant).

Package:        timesfm (pip install timesfm)
HuggingFace:    google/timesfm-2.0-500m-pytorch, google/timesfm-1.0-200m-pytorch, ...

API variants
------------
The library has gone through two versions of its public surface:

* Legacy:  `timesfm.TimesFm(hparams=..., checkpoint=...)` then
           `model.forecast(inputs=[np.array, ...], freq=[0, ...])`
* 2.5+:    `timesfm.TimesFM_2p5_200M_torch.from_pretrained(...)` then
           `model.compile(ForecastConfig(...))` and `model.forecast(...)`

We try the modern API first and fall back to legacy.

Inputs are **lists of 1-D numpy arrays** (context per sample), output is the
point forecast tensor `(B, horizon)` and, if requested, quantile forecasts
`(B, horizon, n_quantiles)`.

Embedding: TimesFM does not ship a native embedding call.  We fall back to a
forward hook on the final transformer block via `embed_via_hidden_states`.
"""

from __future__ import annotations

from typing import Any, Optional

import numpy as np
import torch

from .base import (
    BaseFoundationModel,
    EmbeddingResult,
    ForecastResult,
)


_SIZE_TO_HF = {
    "small": "google/timesfm-1.0-200m-pytorch",
    "base":  "google/timesfm-1.0-200m-pytorch",   # 200M is default; use large for 500M
    "large": "google/timesfm-2.0-500m-pytorch",
}
_HF_TO_LAYERS = {
    "google/timesfm-1.0-200m-pytorch": 20,
    "google/timesfm-2.0-500m-pytorch": 50,
}


class TimesFMWrapper(BaseFoundationModel):
    name = "timesfm"
    supports_forecast = True
    supports_denoise = False
    supports_embed = True    # via forward hook on the final transformer layer

    def __init__(self) -> None:
        super().__init__()
        self._forecast_fn = None                       # closure of whichever API loaded
        self._context_len = 512
        self._horizon = 128

    def load(
        self,
        device: str = "cuda",
        model_size: str = "base",
        *,
        hf_id: Optional[str] = None,
        context_len: int = 512,
        horizon: int = 128,
        **_unused,
    ) -> None:
        try:
            import timesfm
        except ImportError as e:
            raise ImportError(
                "timesfm is required for TimesFMWrapper. "
                "Install: pip install timesfm"
            ) from e

        self.device = device
        self._context_len = context_len
        self._horizon = horizon
        hf = hf_id or _SIZE_TO_HF.get(model_size, _SIZE_TO_HF["base"])

        # Try the modern (2.5+) API first.
        try:
            from_pretrained = getattr(timesfm, "TimesFM_2p5_200M_torch", None)
            if from_pretrained is not None and hasattr(from_pretrained, "from_pretrained"):
                m = from_pretrained.from_pretrained(hf)
                cfg = timesfm.ForecastConfig(
                    max_context=context_len,
                    max_horizon=horizon,
                    normalize_inputs=True,
                )
                m.compile(cfg)
                self.model = m
                self._forecast_fn = lambda inp: m.forecast(
                    horizon=horizon, inputs=inp,
                )
                return
        except Exception:
            pass

        # Legacy API: TimesFm(hparams=..., checkpoint=...).
        n_layers = _HF_TO_LAYERS.get(hf, 20)
        hparams = timesfm.TimesFmHparams(
            backend=("gpu" if "cuda" in device else "cpu"),
            per_core_batch_size=32,
            horizon_len=horizon,
            context_len=context_len,
            num_layers=n_layers,
        )
        ckpt = timesfm.TimesFmCheckpoint(huggingface_repo_id=hf)
        m = timesfm.TimesFm(hparams=hparams, checkpoint=ckpt)
        self.model = m
        self._forecast_fn = lambda inp: m.forecast(inputs=inp, freq=[0] * len(inp))

    # ── Forecasting ───────────────────────────────────────────────────────

    @torch.no_grad()
    def forecast(self, context: np.ndarray, horizon: int) -> ForecastResult:
        if self._forecast_fn is None:
            raise RuntimeError("Call .load() before forecast().")
        # TimesFM takes a list of 1-D numpy arrays.
        B = context.shape[0]
        inputs = [context[i].astype(np.float32) for i in range(B)]
        out = self._forecast_fn(inputs)
        # Output contract differs across versions:
        #  - legacy: (point_forecast, quantile_forecast) tuple, both np arrays
        #  - modern: (point_forecast, quantile_forecast) tuple as well
        if isinstance(out, tuple):
            point, quants = out[0], out[1]
        else:
            point, quants = out, None
        point = np.asarray(point, dtype=np.float32)[:, :horizon]
        samples = None
        if quants is not None:
            quants = np.asarray(quants, dtype=np.float32)
            # Quantile tensor layout: (B, H, Q); treat quantiles as samples.
            if quants.ndim == 3:
                samples = np.transpose(quants, (0, 2, 1))[:, :, :horizon]
        return ForecastResult(point_forecast=point, samples=samples)

    # ── Embedding via last transformer layer hook ────────────────────────

    @torch.no_grad()
    def embed(self, signal: np.ndarray, *, pool: str = "last") -> EmbeddingResult:
        """Hook the last `stacked_transformer.layers[-1]` output during a
        horizon-1 forecast and pool to (B, d).

        TimesFM is a causal decoder on non-overlapping patches; the last patch
        is the only position that has seen the full context, so `pool="last"`
        is the principled default.
        """
        if self._forecast_fn is None:
            raise RuntimeError("Call .load() before embed().")
        inner = getattr(self.model, "_model", None)
        if inner is None or not hasattr(inner, "stacked_transformer"):
            raise RuntimeError(
                "TimesFM embed() only supports the legacy PatchedTimeSeriesDecoder "
                "path (self.model._model.stacked_transformer). "
                "The 2.5+ API is not supported yet."
            )
        # Hook the whole stack (not an individual layer): the layer-level output
        # is a multi-head-split 4D tensor; the stack-level output is (B, T, d).
        target = inner.stacked_transformer

        captured: list[torch.Tensor] = []

        def _hook(_m, _inp, out):
            h = out[0] if isinstance(out, tuple) else out
            captured.append(h.detach())

        handle = target.register_forward_hook(_hook)
        try:
            # Trigger a forward via the public forecast API (horizon=1 is cheap).
            inputs = [signal[i].astype(np.float32) for i in range(signal.shape[0])]
            self._forecast_fn(inputs)
        finally:
            handle.remove()

        if not captured:
            raise RuntimeError("TimesFM forward hook captured no output.")
        # Legacy API invokes the decoder once per call; take the last capture
        # and drop any trailing positions beyond the original batch size (the
        # library may pad for efficiency).
        h = captured[-1]
        if h.dim() != 3:
            h = h.flatten(start_dim=1)[:signal.shape[0]]
            return EmbeddingResult(embeddings=h.cpu().numpy().astype(np.float32))
        h = h[: signal.shape[0]]
        emb = self._pool_time(h, pool=pool)
        return EmbeddingResult(embeddings=emb.cpu().numpy().astype(np.float32))

    # ── Grad-flowing embedding for fine-tuning ───────────────────────────

    def embed_torch(self, signal: np.ndarray, *, pool: str = "last") -> torch.Tensor:
        """Same as embed() but returns a torch tensor with grads enabled.

        Bypasses the public forecast API (which is wrapped in inference mode
        internally) by calling _preprocess_input + stacked_transformer
        directly on the legacy PatchedTimeSeriesDecoder.
        """
        if self._forecast_fn is None:
            raise RuntimeError("Call .load() before embed_torch().")
        inner = self._get_finetune_inner()
        backbone_device = next(inner.parameters()).device

        # Pad/truncate context to a multiple of patch_len (left-pad, so the
        # most recent samples sit at the right -- causal decoder).
        patch_len = int(inner.config.patch_len)
        ctx_len = self._context_len
        if ctx_len % patch_len != 0:
            ctx_len = (ctx_len // patch_len) * patch_len

        x = self._prep_context(signal.astype(np.float32), ctx_len)  # (B, ctx_len)
        x_t = torch.from_numpy(x).to(backbone_device)
        pad_t = torch.zeros_like(x_t)
        freq_t = torch.zeros((x_t.shape[0], 1), dtype=torch.long, device=backbone_device)

        model_input, patched_padding, _, _ = inner._preprocess_input(x_t, pad_t)
        model_input = model_input + inner.freq_emb(freq_t)
        h = inner.stacked_transformer(model_input, patched_padding)  # (B, N, d)
        return self._pool_time(h, pool=pool)

    def get_finetune_backbone(self) -> torch.nn.Module:
        return self._get_finetune_inner()

    def _get_finetune_inner(self) -> torch.nn.Module:
        inner = getattr(self.model, "_model", None)
        if inner is None or not hasattr(inner, "stacked_transformer"):
            raise RuntimeError(
                "TimesFM fine-tuning only supports the legacy "
                "PatchedTimeSeriesDecoder (self.model._model). "
                "The 2.5+ API is not supported yet."
            )
        return inner

    @staticmethod
    def _prep_context(x: np.ndarray, target_len: int) -> np.ndarray:
        """(B, L) numpy -> (B, target_len) numpy, left-pad / left-truncate."""
        B, L = x.shape
        if L == target_len:
            return x
        if L < target_len:
            return np.concatenate(
                [np.zeros((B, target_len - L), dtype=x.dtype), x], axis=-1
            )
        return x[:, -target_len:]
