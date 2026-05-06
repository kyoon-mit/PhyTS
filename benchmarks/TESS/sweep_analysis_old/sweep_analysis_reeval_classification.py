"""Backward-compatible import path for classification re-eval.

Implementation lives in :mod:`sweep_digest.retest`.
"""

from __future__ import annotations

import sys
from pathlib import Path

_TESS = Path(__file__).resolve().parent.parent
if str(_TESS) not in sys.path:
    sys.path.insert(0, str(_TESS))

from sweep_digest.retest import (  # noqa: E402
    classification_sweep_ckpt_dir,
    gather_classification_preds_for_plots,
    reeval_classification_test_metrics,
    test_metrics_numpy,
)

__all__ = [
    "classification_sweep_ckpt_dir",
    "gather_classification_preds_for_plots",
    "reeval_classification_test_metrics",
    "test_metrics_numpy",
]
