"""
Granite TTM R1 wrapper -- IBM's Tiny Time Mixer forecasting model.

Package:        granite-tsfm (provides `tsfm_public`)
HuggingFace:    ibm-granite/granite-timeseries-ttm-r1

TTM-R1 uses fixed-shape heads: the pre-trained variants expose context
lengths of 512 or 1024 with a prediction length of 96 (selected via the HF
`revision` branch -- `main` or `1024_96_v1`). Inputs are shaped
(B, context_length, num_channels); we run
the univariate case with num_channels=1 and pad/truncate callers' context to
the variant's context length. For horizons longer than the variant's
prediction length we roll the forecast autoregressively, re-feeding the
emitted window into the context.

Embedding is not a natively supported task for TTM; we fall back to
`embed_via_hidden_states` on the backbone's final mixer block.
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


# Per-size defaults. Users can override via `hf_id` / `revision` on load().
# All variants share the same HF repo; the `revision` branch selects the
# (context_length, prediction_length) variant. TTM-R1 publishes two branches:
#   main           -> context 512,  prediction 96
#   1024_96_v1     -> context 1024, prediction 96
_SIZE_TO_REVISION = {
    "small":  "main",
    "base":   "main",
    "large":  "1024_96_v1",
}
_REVISION_TO_CTX = {
    "main":        512,
    "1024_96_v1":  1024,
}
_DEFAULT_HF_ID = "ibm-granite/granite-timeseries-ttm-r1"


class GraniteTTMWrapper(BaseFoundationModel):
    name = "granite_ttm"
    supports_forecast = True
    supports_denoise = False       # uses denoise_via_forecast fallback
    supports_embed = True          # via backbone hidden-state pooling

    def __init__(self) -> None:
        super().__init__()
        self._hf_id: str = ""
        self._revision: str = "main"
        self._ctx_len: int = 512
        self._pred_len: int = 96

    # ── Loading ───────────────────────────────────────────────────────────

    def load(
        self,
        device: str = "cuda",
        model_size: str = "base",
        *,
        hf_id: Optional[str] = None,
        revision: Optional[str] = None,
        **_unused,
    ) -> None:
        try:
            from tsfm_public import TinyTimeMixerForPrediction
        except ImportError as e:
            raise ImportError(
                "granite-tsfm is required for GraniteTTMWrapper. "
                "Install: pip install granite-tsfm"
            ) from e

        self.device = device
        self._hf_id = hf_id or _DEFAULT_HF_ID
        self._revision = revision or _SIZE_TO_REVISION.get(model_size, "main")
        self._ctx_len = _REVISION_TO_CTX.get(self._revision, 512)

        self.model = TinyTimeMixerForPrediction.from_pretrained(
            self._hf_id, revision=self._revision,
        ).to(device).eval()
        # prediction_length comes from the checkpoint config
        self._pred_len = int(getattr(self.model.config, "prediction_length", 96))

    # ── Forecasting ───────────────────────────────────────────────────────

    @torch.no_grad()
    def forecast(self, context: np.ndarray, horizon: int) -> ForecastResult:
        if self.model is None:
            raise RuntimeError("Call .load() before forecast().")

        # Per-sample z-score normalisation; the model itself has internal
        # scaling but we match the pattern used by other wrappers so the
        # caller always sees inputs in the dataset's original units.
        normed, mean, std = self._instance_normalize(context.astype(np.float32))
        B = normed.shape[0]

        # Roll the forecast window by window until we cover `horizon` steps.
        buf = self._prep_input(normed, target_len=self._ctx_len)  # (B, ctx, 1)
        remaining = horizon
        out_chunks: list[np.ndarray] = []
        while remaining > 0:
            step = min(remaining, self._pred_len)
            out = self.model(past_values=buf)
            pred = out.prediction_outputs                          # (B, pred_len, 1)
            chunk = pred[:, :step, :]                              # (B, step, 1)
            out_chunks.append(chunk.cpu().numpy().astype(np.float32))
            remaining -= step
            if remaining > 0:
                # Slide the context: drop the oldest `step` samples, append
                # the emitted ones.
                buf = torch.cat([buf[:, step:, :], chunk], dim=1)

        fcst = np.concatenate(out_chunks, axis=1).squeeze(-1)       # (B, horizon)
        fcst = self._instance_denormalize(fcst, mean, std)
        return ForecastResult(point_forecast=fcst.astype(np.float32), samples=None)

    # ── Embedding via backbone hidden state ───────────────────────────────

    @torch.no_grad()
    def embed(self, signal: np.ndarray) -> EmbeddingResult:
        if self.model is None:
            raise RuntimeError("Call .load() before embed().")
        normed, _, _ = self._instance_normalize(signal.astype(np.float32))
        buf = self._prep_input(normed, target_len=self._ctx_len)    # (B, ctx, 1)
        out = self.model.backbone(buf)
        hs = out.last_hidden_state                                  # (B, C, N, d) or (B, N, d)
        # Mean-pool over every non-batch, non-feature axis to land at (B, d).
        if hs.dim() == 4:
            emb = hs.mean(dim=(1, 2))
        elif hs.dim() == 3:
            emb = hs.mean(dim=1)
        else:
            emb = hs.flatten(start_dim=1)
        return EmbeddingResult(embeddings=emb.cpu().numpy().astype(np.float32))

    # ── Helpers ───────────────────────────────────────────────────────────

    def _prep_input(self, x: np.ndarray, target_len: int) -> torch.Tensor:
        """(B, L) numpy -> (B, target_len, 1) float tensor on device.

        Pads with zeros on the left or truncates from the left so the most
        recent sample aligns with position target_len - 1 (the position TTM
        treats as "present" when emitting its forecast).
        """
        B, L = x.shape
        if L == target_len:
            arr = x
        elif L < target_len:
            arr = np.concatenate(
                [np.zeros((B, target_len - L), dtype=x.dtype), x], axis=-1
            )
        else:
            arr = x[:, -target_len:]
        t = torch.from_numpy(arr.astype(np.float32)).unsqueeze(-1).to(self.device)
        return t
