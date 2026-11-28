"""Zero-shot foundation-model protocol (paper Appendix D): frozen window embeddings + linear probe.

Every backbone embeds windows of 512 samples, the shortest common context, with the token pooling
that matches its pretraining objective (Table 9): the token mean for the encoders (MOMENT,
Chronos, MOIRAI, Granite-TTM) and the last token for the causal decoders (Time-MoE, TimesFM).
MOIRAI is embedded as in a patch-32, horizon-128 forecast pass. A benchmark script averages the
window embeddings of each series and fits a closed-form probe, with features (and the regression
target) z-scored using the statistics of the split the probe is fitted on.
"""

from __future__ import annotations

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression, Ridge

MODELS = ("moment", "chronos", "timesfm", "timemoe", "moirai", "granite_ttm")
CHUNK_LEN = 512
POOL = {"moment": "mean", "chronos": "mean", "moirai": "mean", "granite_ttm": "mean",
        "timemoe": "last", "timesfm": "last"}
MOIRAI_HORIZON = 128


def load_wrapper(name: str, device: str):
    """Base-size checkpoint of one backbone, set up for 512-sample windows."""
    if name == "moment":
        from foundation.wrappers.moment_wrapper import MomentWrapper
        w = MomentWrapper()
        w.load(device=device, model_size="base", seq_len=CHUNK_LEN)
    elif name == "chronos":
        from foundation.wrappers.chronos_wrapper import ChronosWrapper
        w = ChronosWrapper()
        w.load(device=device, model_size="base")
    elif name == "timesfm":
        from foundation.wrappers.timesfm_wrapper import TimesFMWrapper
        w = TimesFMWrapper()
        w.load(device=device, model_size="base", context_len=CHUNK_LEN, horizon=128)
    elif name == "timemoe":
        from foundation.wrappers.timemoe_wrapper import TimeMoEWrapper
        w = TimeMoEWrapper()
        w.load(device=device, model_size="base")
    elif name == "moirai":
        from foundation.wrappers.moirai_wrapper import MoiraiWrapper
        w = MoiraiWrapper()
        w.load(device=device, model_size="base", context_len=CHUNK_LEN)
        w._module.to(device).eval()   # embed() runs the module directly, not the predictor
    elif name == "granite_ttm":
        from foundation.wrappers.granite_ttm_wrapper import GraniteTTMWrapper
        w = GraniteTTMWrapper()
        w.load(device=device, model_size="base")
        if w._ctx_len != CHUNK_LEN:
            raise ValueError(f"TTM context {w._ctx_len} != window length {CHUNK_LEN}")
    else:
        raise ValueError(f"unknown model {name!r}; choose from {MODELS}")
    return w


def embed(wrapper, name: str, windows: np.ndarray, batch_size: int = 64) -> np.ndarray:
    """(N, 512) windows -> (N, d) embeddings."""
    kw = {} if POOL[name] == "mean" else {"pool": POOL[name]}
    if name == "moirai":
        kw["horizon"] = MOIRAI_HORIZON
    with torch.inference_mode():
        return np.concatenate([
            np.asarray(wrapper.embed(windows[i:i + batch_size], **kw).embeddings, dtype=np.float32)
            for i in range(0, len(windows), batch_size)
        ])


def _zscore(fit: np.ndarray, *others: np.ndarray) -> list[np.ndarray]:
    """Standardise with the fit split's statistics; non-finite entries become 0."""
    mu, sd = fit.mean(axis=0), fit.std(axis=0) + 1e-6
    return [np.nan_to_num((x - mu) / sd, nan=0.0, posinf=0.0, neginf=0.0) for x in (fit, *others)]


def logistic_probe(X_fit, y_fit, X_eval, C: float = 1.0) -> np.ndarray:
    """Multinomial logistic regression on z-scored features; returns the predicted labels."""
    X_fit, X_eval = _zscore(X_fit, X_eval)
    return LogisticRegression(C=C, max_iter=2000).fit(X_fit, y_fit).predict(X_eval)


def ridge_probe(X_fit, y_fit, X_eval, alpha: float = 1.0) -> np.ndarray:
    """Ridge regression on z-scored features and target; returns predictions in target units."""
    X_fit, X_eval = _zscore(X_fit, X_eval)
    mu, sd = y_fit.mean(), y_fit.std() + 1e-8
    return Ridge(alpha=alpha).fit(X_fit, (y_fit - mu) / sd).predict(X_eval) * sd + mu
