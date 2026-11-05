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
    supports_embed = True

    def __init__(self) -> None:
        super().__init__()
        self._predictor = None
        self._module = None
        self._num_samples = 20
        self._context_len = 512

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

    # ── Embedding (encoder representations) ───────────────────────────────

    @torch.no_grad()
    def embed(self, signal: np.ndarray, patch_size: int = 32) -> EmbeddingResult:
        """(B, L) -> (B, d_model): mean over patch tokens of the encoder output.

        Mirrors MoiraiModule.forward up to the encoder (scaler -> in_proj ->
        encoder) for a single fully observed variate, no prediction window.
        L must be a multiple of `patch_size` (one of module.patch_sizes).
        """
        from uni2ts.common.torch_util import packed_attention_mask

        mod = self._module
        if mod is None:
            raise RuntimeError("Call .load() before embed().")
        if patch_size not in tuple(mod.patch_sizes):
            raise ValueError(f"patch_size {patch_size} not in {tuple(mod.patch_sizes)}")
        dev = next(mod.parameters()).device
        B, L = signal.shape
        if L % patch_size:
            raise ValueError(f"length {L} is not a multiple of patch_size {patch_size}")
        n, max_p = L // patch_size, max(mod.patch_sizes)
        x = torch.from_numpy(np.ascontiguousarray(signal, dtype=np.float32)).to(dev)
        target = torch.zeros(B, n, max_p, device=dev)
        target[..., :patch_size] = x.view(B, n, patch_size)
        observed = torch.zeros(B, n, max_p, dtype=torch.bool, device=dev)
        observed[..., :patch_size] = True
        sample_id = torch.ones(B, n, dtype=torch.long, device=dev)
        time_id = torch.arange(n, device=dev).expand(B, n)
        variate_id = torch.zeros(B, n, dtype=torch.long, device=dev)
        pred_mask = torch.zeros(B, n, dtype=torch.bool, device=dev)
        psize = torch.full((B, n), patch_size, dtype=torch.long, device=dev)
        loc, scale = mod.scaler(target, observed * ~pred_mask.unsqueeze(-1), sample_id, variate_id)
        reprs = mod.in_proj((target - loc) / scale, psize)
        reprs = mod.encoder(reprs, packed_attention_mask(sample_id), time_id=time_id, var_id=variate_id)
        return EmbeddingResult(embeddings=reprs.mean(dim=1).float().cpu().numpy())

