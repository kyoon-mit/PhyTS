"""TESS eval figures shared with :mod:`benchmarks.TESS.eval_pipeline` and training (wandb).

Regression: scatter true vs predicted frot + residual histogram.
Classification: confusion matrix + per-class accuracy bars.

Validation figures log every :const:`VAL_PLOT_TO_WANDB_EVERY_N_EPOCHS`; test figures log once per
``trainer.test`` when :func:`log_test_plots_to_wandb` is called (W&B active).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

if TYPE_CHECKING:
    import lightning as L

# Log validation figures to wandb only when (epoch index) % N == 0 — i.e. after epochs 0, 10, 20, …
VAL_PLOT_TO_WANDB_EVERY_N_EPOCHS = 10


def regression_results_dict(y_true: np.ndarray, y_hat: np.ndarray) -> dict:
    """Build the results dict used by :func:`make_regression_figure` (matches eval pipeline)."""
    y_true = np.asarray(y_true, dtype=np.float64).reshape(-1)
    y_hat = np.asarray(y_hat, dtype=np.float64).reshape(-1)
    residuals = y_hat - y_true
    rmse = float(np.sqrt(np.mean(residuals**2)))
    mae = float(np.mean(np.abs(residuals)))
    ss_res = float(np.sum(residuals**2))
    ss_tot = float(np.sum((y_true - y_true.mean()) ** 2))
    r2 = float(1.0 - ss_res / max(ss_tot, 1e-10))
    return {
        "y_true": y_true,
        "y_hat": y_hat,
        "residuals": residuals,
        "rmse": rmse,
        "mae": mae,
        "r2": r2,
    }


def classification_results_dict(
    y_true: np.ndarray,
    y_hat: np.ndarray,
    label_names: list[str],
) -> dict:
    """Build results dict for :func:`make_classification_figure` (matches eval pipeline)."""
    y_true = np.asarray(y_true, dtype=np.int64).reshape(-1)
    y_hat = np.asarray(y_hat, dtype=np.int64).reshape(-1)
    acc = float(np.mean(y_true == y_hat))
    per_class_acc: dict[str, float] = {}
    for c, name in enumerate(label_names):
        mask = y_true == c
        per_class_acc[name] = (
            float(np.mean(y_hat[mask] == c)) if np.any(mask) else float("nan")
        )
    return {
        "y_true": y_true,
        "y_hat": y_hat,
        "acc": acc,
        "per_class_acc": per_class_acc,
    }


def make_regression_figure(results: dict, model_name: str) -> matplotlib.figure.Figure:
    """1×2 panel: scatter + residual histogram (same layout as eval pipeline)."""
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    fig.suptitle(
        f"{model_name}  |  RMSE={results['rmse']:.4f}  R²={results['r2']:.3f}"
    )

    ax = axes[0]
    lim = np.percentile(
        np.concatenate([results["y_true"], results["y_hat"]]), [1, 99]
    )
    ax.scatter(results["y_true"], results["y_hat"], s=4, alpha=0.3)
    ax.plot(lim, lim, "r--", linewidth=1)
    ax.set_xlabel("frot true")
    ax.set_ylabel("frot predicted")
    ax.set_title("Scatter")

    ax = axes[1]
    ax.hist(results["residuals"], bins=60, histtype="step", linewidth=1.5)
    ax.set_xlabel(r"$\hat{f}_{rot} - f_{rot}$")
    ax.set_ylabel("Count")
    ax.set_title("Residuals")

    plt.tight_layout()
    return fig


def make_classification_figure(
    results: dict, model_name: str, label_names: list[str]
) -> matplotlib.figure.Figure:
    """1×2 panel: confusion matrix + per-class accuracy (same layout as eval pipeline)."""
    n = len(label_names)
    y_true = results["y_true"].astype(int)
    y_hat = results["y_hat"].astype(int)
    conf = np.zeros((n, n), dtype=int)
    for t, p in zip(y_true, y_hat):
        conf[int(t), int(p)] += 1

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle(f"{model_name}  |  Accuracy={results['acc']:.3f}")

    ax = axes[0]
    im = ax.imshow(conf, cmap="Blues")
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(label_names, rotation=45, ha="right", fontsize=8)
    ax.set_yticklabels(label_names, fontsize=8)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("Confusion matrix")
    plt.colorbar(im, ax=ax)

    ax = axes[1]
    class_accs = [results["per_class_acc"].get(name, float("nan")) for name in label_names]
    ax.barh(range(n), class_accs)
    ax.set_yticks(range(n))
    ax.set_yticklabels(label_names, fontsize=8)
    ax.set_xlabel("Accuracy")
    ax.set_title("Per-class accuracy")
    ax.set_xlim(0, 1)

    plt.tight_layout()
    return fig


def _dm_label_names(pl_module, split: Literal["val", "test"]) -> list[str] | None:
    """Resolve class names from the Lightning datamodule split (TESS classification)."""
    tr = getattr(pl_module, "trainer", None)
    dm = getattr(tr, "datamodule", None) if tr else None
    if dm is None:
        return None
    ds = getattr(dm, split, None)
    if ds is not None and hasattr(ds, "label_names"):
        # May be a class attribute (e.g. TESSClassificationDataset) — materialize to list.
        return list(ds.label_names)
    return None


def _val_label_names(pl_module) -> list[str] | None:
    return _dm_label_names(pl_module, "val")


def log_validation_plots_to_wandb(
    pl_module: L.LightningModule,
    *,
    kind: Literal["regression", "classification"],
    y_true: np.ndarray,
    y_hat: np.ndarray,
    label_names: list[str] | None = None,
    model_name: str | None = None,
) -> None:
    """If a wandb run is active, log validation figures aligned with Lightning's W&B charts.

    Figures are uploaded only every :attr:`VAL_PLOT_TO_WANDB_EVERY_N_EPOCHS` validation
    (after epochs 0, 10, 20, … when that constant is 10).
    """
    try:
        import wandb
    except ImportError:
        return
    if wandb.run is None:
        return
    tr = getattr(pl_module, "trainer", None)
    if tr is None or getattr(tr, "sanity_checking", False):
        return

    ep = int(getattr(pl_module, "current_epoch", getattr(tr, "current_epoch", 0)))
    if ep % VAL_PLOT_TO_WANDB_EVERY_N_EPOCHS != 0:
        return

    name = model_name or pl_module.__class__.__name__
    # Match Lightning's WandbLogger: it logs metrics with keys like trainer/global_step
    # and does not pass wandb.log(step=...). Raw wandb.log(..., step=epoch) fights wandb's
    # internal step counter (train + val epoch logs advance it twice per epoch).
    gs = int(tr.global_step)

    if kind == "regression":
        results = regression_results_dict(y_true, y_hat)
        fig = make_regression_figure(results, name)
        wandb.log({"val/regression_plot": wandb.Image(fig), "trainer/global_step": gs})
        plt.close(fig)
        return

    # classification
    names = label_names or _val_label_names(pl_module)
    if names is None:
        n_cls = int(getattr(pl_module, "num_classes", 0))
        if n_cls <= 0:
            n_cls = int(max(np.max(y_true), np.max(y_hat))) + 1
        names = [str(i) for i in range(n_cls)]
    results = classification_results_dict(y_true, y_hat, names)
    fig = make_classification_figure(results, name, names)
    wandb.log({"val/classification_plot": wandb.Image(fig), "trainer/global_step": gs})
    plt.close(fig)


def log_test_plots_to_wandb(
    pl_module: L.LightningModule,
    *,
    kind: Literal["regression", "classification"],
    y_true: np.ndarray,
    y_hat: np.ndarray,
    label_names: list[str] | None = None,
    model_name: str | None = None,
) -> None:
    """If a wandb run is active, log one-off test figures (no epoch throttle).

    Logs ``test/regression_plot`` or ``test/classification_plot`` as :class:`wandb.Image`.
    """
    try:
        import wandb
    except ImportError:
        return
    if wandb.run is None:
        return
    tr = getattr(pl_module, "trainer", None)
    if tr is None:
        return

    name = model_name or pl_module.__class__.__name__
    gs = int(tr.global_step)

    if kind == "regression":
        results = regression_results_dict(y_true, y_hat)
        fig = make_regression_figure(results, name)
        wandb.log({"test/regression_plot": wandb.Image(fig), "trainer/global_step": gs})
        plt.close(fig)
        return

    names = label_names or _dm_label_names(pl_module, "test") or _dm_label_names(
        pl_module, "val"
    )
    if names is None:
        n_cls = int(getattr(pl_module, "num_classes", 0))
        if n_cls <= 0:
            n_cls = int(max(np.max(y_true), np.max(y_hat))) + 1
        names = [str(i) for i in range(n_cls)]
    results = classification_results_dict(y_true, y_hat, names)
    fig = make_classification_figure(results, name, names)
    wandb.log({"test/classification_plot": wandb.Image(fig), "trainer/global_step": gs})
    plt.close(fig)
