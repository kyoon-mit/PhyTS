"""
Base wrapper interface for time-series foundation models.

Every wrapper exposes three capabilities --- forecast, denoise, embed --- behind
a uniform numpy-based interface. Models that do not natively support a given
capability inherit shared fallbacks:

    denoise_via_forecast:        bidirectional context split + forecast averaging
    embed_via_hidden_states:     forward hook on the final transformer layer

Each wrapper handles its own normalization (instance norm -> inference -> denorm)
so the caller only needs to pass raw float32 numpy arrays.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch


# ────────────────────────────────────────────────────────────────────────────
# Result dataclasses
# ────────────────────────────────────────────────────────────────────────────

@dataclass
class ForecastResult:
    """Output of a forecast() call."""
    point_forecast: np.ndarray                      # (B, horizon)  float32
    samples: Optional[np.ndarray] = None            # (B, n_samples, horizon) for probabilistic models


@dataclass
class DenoiseResult:
    """Output of a denoise() call."""
    denoised: np.ndarray                            # (B, L)  float32


@dataclass
class EmbeddingResult:
    """Output of an embed() call."""
    embeddings: np.ndarray                          # (B, d_embed)  float32


# ────────────────────────────────────────────────────────────────────────────
# Base class
# ────────────────────────────────────────────────────────────────────────────

class BaseFoundationModel(ABC):
    """Abstract base for all foundation-model wrappers.

    Subclasses MUST implement: load(), forecast().
    Subclasses MAY override: denoise(), embed(), get_finetune_params().

    Attributes
    ----------
    name : str
        Short identifier used in output paths and CSV rows (e.g. "moment").
    supports_forecast, supports_denoise, supports_embed : bool
        Declarative capability flags. Used by the evaluator to decide whether
        to call the native method or the fallback.
    """

    name: str = "base"
    supports_forecast: bool = True
    supports_denoise: bool = False
    supports_embed: bool = False

    def __init__(self) -> None:
        self.model: Optional[torch.nn.Module] = None
        self.device: str = "cpu"

    # ── Lifecycle ─────────────────────────────────────────────────────────

    @abstractmethod
    def load(self, device: str = "cuda", model_size: str = "base") -> None:
        """Load the pre-trained model onto `device`."""

    def unload(self) -> None:
        """Release the model from memory (GPU especially)."""
        self.model = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # ── Core capabilities ─────────────────────────────────────────────────

    @abstractmethod
    def forecast(self, context: np.ndarray, horizon: int) -> ForecastResult:
        """Forecast `horizon` future steps given `context` (B, L_context)."""

    def denoise(self, noisy: np.ndarray) -> DenoiseResult:
        """Denoise a full-length noisy signal (B, L) -> (B, L).

        Default implementation falls back to forecast-based denoising.
        Subclasses with native reconstruction (e.g. MOMENT) should override.
        """
        return self.denoise_via_forecast(noisy)

    def embed(self, signal: np.ndarray, *, pool: str = "mean") -> EmbeddingResult:
        """Extract per-sample embedding from a full signal (B, L) -> (B, d).

        `pool` selects the temporal pooling strategy (mean / last / max) and
        is respected by subclasses that support it. The default "mean" matches
        historical behaviour.

        Default raises NotImplementedError. Subclasses with native embedding
        heads (Chronos, MOMENT) override; others can use
        `embed_via_hidden_states` if they expose a hookable layer.
        """
        raise NotImplementedError(
            f"{self.name} does not provide embed(); either override it or "
            "call embed_via_hidden_states(...)."
        )

    def embed_torch(self, signal: np.ndarray, *, pool: str = "mean") -> torch.Tensor:
        """Gradient-preserving embedding.  (B, L) numpy -> (B, d) tensor on device.

        Unlike :meth:`embed`, which returns detached numpy for zero-shot /
        linear-probe evaluation, this keeps the computation graph intact so
        LoRA / unfrozen-backbone parameters receive gradients during
        fine-tuning.  Subclasses must override; default raises.
        """
        raise NotImplementedError(
            f"{self.name} does not implement embed_torch(); fine-tuning on "
            "this model is not supported.  Override in the wrapper to expose "
            "a grad-preserving embedding path."
        )

    def get_finetune_backbone(self) -> Optional[torch.nn.Module]:
        """Return the nn.Module to inject LoRA / unfreeze layers into.

        Defaults to ``self.model``.  Wrappers whose fine-tuning-relevant
        backbone lives in a different attribute (e.g. MOMENT's
        ``embed_model``, Chronos' ``pipeline.model``) override this.
        """
        return self.model

    # ── Shared fallbacks ──────────────────────────────────────────────────

    def denoise_via_forecast(self, noisy: np.ndarray) -> DenoiseResult:
        """Bidirectional forecast-based denoising.

        Split (B, L) at L//2.  Forecast the second half from the first half,
        then reverse the signal and forecast the (reversed) second half from
        the (reversed) first half.  Average the two reconstructions for the
        second half; the first half is recovered by reversing the backward
        pass.  Total cost: 2 forward passes per batch, full L-sample coverage.

        This treats the foundation model's forecast as a model-based prior
        for the underlying clean signal; it is imperfect but provides a fair
        comparison with trained denoisers for periodic inputs.
        """
        B, L = noisy.shape
        mid = L // 2

        # Forward: context = first half, forecast the second half
        ctx_fwd = noisy[:, :mid]
        fcst_fwd = self.forecast(ctx_fwd, horizon=L - mid).point_forecast  # (B, L - mid)

        # Backward: reverse the signal, context = first half (= original second half reversed)
        reversed_noisy = noisy[:, ::-1].copy()
        ctx_bwd = reversed_noisy[:, :mid]
        fcst_bwd_rev = self.forecast(ctx_bwd, horizon=L - mid).point_forecast  # (B, L - mid)
        fcst_bwd = fcst_bwd_rev[:, ::-1].copy()  # un-reverse to align with original order; shape (B, L - mid)

        denoised = noisy.copy()
        # First half from backward pass (what the model predicts when looking at reversed context)
        denoised[:, :L - mid] = fcst_bwd
        # Second half: average forward forecast and the reversed backward forecast's tail
        # (backward pass covered the first L-mid positions in the reversed order, so those
        # map to the last L-mid positions in the original order).  The forward pass covered
        # positions mid..L.  Blend them in their shared region.
        second_half_len = L - mid
        denoised[:, mid:] = fcst_fwd

        return DenoiseResult(denoised=denoised.astype(np.float32))

    def embed_via_hidden_states(
        self,
        signal: np.ndarray,
        target_module: torch.nn.Module,
        forward_fn,
    ) -> EmbeddingResult:
        """Generic embedding extraction via a forward hook.

        Parameters
        ----------
        signal : np.ndarray
            (B, L) input.
        target_module : nn.Module
            The layer whose output we want to mean-pool (e.g. last transformer
            block).
        forward_fn : callable
            A function that the wrapper provides which performs one forward
            pass of `signal` through the model; the hook captures
            `target_module`'s output along the way.

        Returns
        -------
        EmbeddingResult with embeddings shape (B, d_embed) -- the module's
        output mean-pooled across the time dimension.
        """
        captured: list[torch.Tensor] = []

        def _hook(_module, _inp, out):
            # Accept either a tensor or a tuple where the first element is the
            # hidden state.
            h = out[0] if isinstance(out, tuple) else out
            captured.append(h.detach())

        handle = target_module.register_forward_hook(_hook)
        try:
            forward_fn(signal)
        finally:
            handle.remove()

        if not captured:
            raise RuntimeError(
                "Forward hook captured no output. Check that `target_module` "
                "is actually invoked during `forward_fn`."
            )
        h = captured[-1]                    # (B, T, d) most common
        if h.dim() == 3:
            emb = h.mean(dim=1)             # (B, d)
        elif h.dim() == 2:
            emb = h
        else:
            emb = h.flatten(start_dim=1)
        return EmbeddingResult(embeddings=emb.cpu().numpy().astype(np.float32))

    # ── Fine-tuning interface ─────────────────────────────────────────────

    def get_finetune_params(self) -> list[torch.nn.Parameter]:
        """Return parameters to optimize during fine-tuning.

        Default: every parameter of `self.model`.  Override to freeze the
        backbone and only expose a task head / last N layers.
        """
        if self.model is None:
            raise RuntimeError(f"{self.name} not loaded; call .load() first.")
        return [p for p in self.model.parameters() if p.requires_grad]

    # ── Shared utilities ──────────────────────────────────────────────────

    @staticmethod
    def _pool_time(h: torch.Tensor, pool: str = "mean") -> torch.Tensor:
        """Pool a (B, T, d) or (B, d) tensor down to (B, d).

        Strategies: "mean", "last", "max". Unknown values fall back to mean.
        """
        if h.dim() == 2:
            return h
        if h.dim() != 3:
            return h.flatten(start_dim=1)
        if pool == "last":
            return h[:, -1]
        if pool == "max":
            return h.amax(dim=1)
        return h.mean(dim=1)

    @staticmethod
    def _instance_normalize(
        x: np.ndarray, eps: float = 1e-6
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Per-sample z-score.  Returns (normed, mean, std) each shape (B, ...)."""
        mean = x.mean(axis=-1, keepdims=True)
        std = x.std(axis=-1, keepdims=True) + eps
        return (x - mean) / std, mean, std

    @staticmethod
    def _instance_denormalize(
        x: np.ndarray, mean: np.ndarray, std: np.ndarray
    ) -> np.ndarray:
        return x * std + mean

    @staticmethod
    def _to_gluonts(
        data: np.ndarray,
        freq: str = "s",
        start: str = "2000-01-01",
    ):
        """Convert (B, L) numpy to a GluonTS ListDataset.

        Used by MOIRAI and Lag-Llama wrappers.  Kept here so every GluonTS
        wrapper shares a single format.
        """
        try:
            from gluonts.dataset.common import ListDataset
        except ImportError as e:
            raise ImportError(
                "GluonTS is required for this wrapper: pip install gluonts"
            ) from e
        return ListDataset(
            [{"start": start, "target": row.astype(np.float32)} for row in data],
            freq=freq,
        )
