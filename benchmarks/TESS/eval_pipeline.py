"""TESS Benchmark Evaluation: Regression (frot) and Classification (label).

Evaluates all trained model variants on held-out test sets and writes
per-sample CSVs plus summary plots.

Usage:
    python benchmarks/TESS/eval_pipeline.py \\
        --data_dir data/TESS/.cache/TESS \\
        --out_dir  benchmarks/TESS

Model checkpoints and configs are discovered automatically from checkpoints/
and configs/TESS/ using hardcoded names. Pass --skip_regression or
--skip_classification to evaluate only one task.
"""

import argparse
import importlib
from pathlib import Path

import numpy as np
import torch
import yaml
import matplotlib.pyplot as plt


# ── Model loading ─────────────────────────────────────────────────────────────

def load_model(ckpt_path: str, cfg_path: str, device: torch.device) -> torch.nn.Module:
    """Reconstruct a model from LightningCLI YAML and load its state dict."""
    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)
    model_cfg = cfg["model"]["init_args"]["model"]
    module_name, class_name = model_cfg["class_path"].rsplit(".", 1)
    cls = getattr(importlib.import_module(module_name), class_name)
    model = cls(**model_cfg.get("init_args", {}))
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=True)
    sd = {
        k[len("model."):]: v
        for k, v in ckpt["state_dict"].items()
        if k.startswith("model.")
    }
    model.load_state_dict(sd)
    model.eval()
    return model.to(device)


def load_task(ckpt_path: str, cfg_path: str, device: torch.device):
    """Load a full LightningModule (task + model) from checkpoint.

    Used for frozen-backbone tasks where the state dict includes both
    backbone and head under their respective submodule prefixes.
    """
    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)

    task_cfg = cfg["model"]
    task_module, task_class = task_cfg["class_path"].rsplit(".", 1)
    task_cls = getattr(importlib.import_module(task_module), task_class)

    # Resolve nested model/head from init_args
    init_args = dict(task_cfg.get("init_args", {}))

    def _instantiate(obj_cfg: dict):
        mod, cls_name = obj_cfg["class_path"].rsplit(".", 1)
        cls = getattr(importlib.import_module(mod), cls_name)
        return cls(**obj_cfg.get("init_args", {}))

    if "model" in init_args and isinstance(init_args["model"], dict):
        init_args["model"] = _instantiate(init_args["model"])
    if "head" in init_args and isinstance(init_args["head"], dict):
        init_args["head"] = _instantiate(init_args["head"])

    task = task_cls(**init_args)
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=True)
    task.load_state_dict(ckpt["state_dict"], strict=False)
    task.eval()
    return task.to(device)


# ── Regression evaluation ─────────────────────────────────────────────────────

@torch.no_grad()
def evaluate_regression(task, dataloader, device: torch.device) -> dict:
    y_true_list, y_hat_list = [], []
    for flux, mask, frot in dataloader:
        flux = flux.to(device)
        y_hat = task(flux).squeeze(-1).cpu()
        y_true_list.append(frot)
        y_hat_list.append(y_hat)
    y_true = torch.cat(y_true_list).numpy()
    y_hat = torch.cat(y_hat_list).numpy()
    residuals = y_hat - y_true
    rmse = float(np.sqrt(np.mean(residuals**2)))
    mae = float(np.mean(np.abs(residuals)))
    ss_res = np.sum(residuals**2)
    ss_tot = np.sum((y_true - y_true.mean())**2)
    r2 = float(1.0 - ss_res / max(ss_tot, 1e-10))
    return {"y_true": y_true, "y_hat": y_hat, "residuals": residuals,
            "rmse": rmse, "mae": mae, "r2": r2}


def plot_regression(results: dict, model_name: str, out_dir: Path):
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    fig.suptitle(f"{model_name}  |  RMSE={results['rmse']:.4f}  R²={results['r2']:.3f}")

    ax = axes[0]
    lim = np.percentile(np.concatenate([results["y_true"], results["y_hat"]]), [1, 99])
    ax.scatter(results["y_true"], results["y_hat"], s=4, alpha=0.3)
    ax.plot(lim, lim, "r--", linewidth=1)
    ax.set_xlabel("frot true"); ax.set_ylabel("frot predicted"); ax.set_title("Scatter")

    ax = axes[1]
    ax.hist(results["residuals"], bins=60, histtype="step", linewidth=1.5)
    ax.set_xlabel(r"$\hat{f}_{rot} - f_{rot}$"); ax.set_ylabel("Count"); ax.set_title("Residuals")

    plt.tight_layout()
    out_path = out_dir / f"{model_name}_regression.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved {out_path}")


# ── Classification evaluation ─────────────────────────────────────────────────

@torch.no_grad()
def evaluate_classification(task, dataloader, device: torch.device,
                             label_names: list[str]) -> dict:
    y_true_list, y_hat_list, probs_list = [], [], []
    for flux, mask, label in dataloader:
        flux = flux.to(device)
        logits = task(flux).cpu()
        probs = torch.softmax(logits, dim=-1)
        y_hat = logits.argmax(dim=-1)
        y_true_list.append(label)
        y_hat_list.append(y_hat)
        probs_list.append(probs)
    y_true = torch.cat(y_true_list).numpy()
    y_hat = torch.cat(y_hat_list).numpy()
    probs = torch.cat(probs_list).numpy()
    acc = float(np.mean(y_true == y_hat))
    per_class_acc = {}
    for c, name in enumerate(label_names):
        mask = y_true == c
        per_class_acc[name] = float(np.mean(y_hat[mask] == c)) if mask.sum() > 0 else float("nan")
    return {"y_true": y_true, "y_hat": y_hat, "probs": probs,
            "acc": acc, "per_class_acc": per_class_acc}


def plot_classification(results: dict, model_name: str, label_names: list[str], out_dir: Path):
    n = len(label_names)
    conf = np.zeros((n, n), dtype=int)
    for t, p in zip(results["y_true"], results["y_hat"]):
        conf[int(t), int(p)] += 1

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle(f"{model_name}  |  Accuracy={results['acc']:.3f}")

    ax = axes[0]
    im = ax.imshow(conf, cmap="Blues")
    ax.set_xticks(range(n)); ax.set_yticks(range(n))
    ax.set_xticklabels(label_names, rotation=45, ha="right", fontsize=8)
    ax.set_yticklabels(label_names, fontsize=8)
    ax.set_xlabel("Predicted"); ax.set_ylabel("True"); ax.set_title("Confusion matrix")
    plt.colorbar(im, ax=ax)

    ax = axes[1]
    class_accs = [results["per_class_acc"].get(n, float("nan")) for n in label_names]
    ax.barh(range(n), class_accs)
    ax.set_yticks(range(n)); ax.set_yticklabels(label_names, fontsize=8)
    ax.set_xlabel("Accuracy"); ax.set_title("Per-class accuracy")
    ax.set_xlim(0, 1)

    plt.tight_layout()
    out_path = out_dir / f"{model_name}_classification.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved {out_path}")


# ── CSV output ────────────────────────────────────────────────────────────────

def save_regression_csv(results: dict, model_name: str, out_dir: Path):
    import csv
    path = out_dir / f"{model_name}_regression_results.csv"
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["model", "frot_true", "frot_hat", "residual"])
        for t, h, r in zip(results["y_true"], results["y_hat"], results["residuals"]):
            w.writerow([model_name, float(t), float(h), float(r)])
    print(f"Saved {path}")


def save_classification_csv(results: dict, model_name: str,
                             label_names: list[str], out_dir: Path):
    import csv
    path = out_dir / f"{model_name}_classification_results.csv"
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["model", "label_true_idx", "label_hat_idx",
                    "label_true", "label_hat", "correct"])
        for t, h in zip(results["y_true"], results["y_hat"]):
            ti, hi = int(t), int(h)
            w.writerow([model_name, ti, hi,
                        label_names[ti], label_names[hi], int(ti == hi)])
    print(f"Saved {path}")


# ── Main ──────────────────────────────────────────────────────────────────────

REGRESSION_MODELS = [
    ("mlp",          "checkpoints/tess_mlp_regression/best.ckpt",
                     "configs/TESS/train_tess_mlp_regression.yaml"),
    ("s4d",          "checkpoints/tess_s4d_regression/best.ckpt",
                     "configs/TESS/train_tess_s4d_regression.yaml"),
    ("s4d_head",     "checkpoints/tess_s4d_head_regression/best.ckpt",
                     "configs/TESS/train_tess_s4d_head_regression.yaml"),
]

CLASSIFICATION_MODELS = [
    ("mlp",          "checkpoints/tess_mlp_classification/best.ckpt",
                     "configs/TESS/train_tess_mlp_classification.yaml"),
    ("s4d",          "checkpoints/tess_s4d_classification/best.ckpt",
                     "configs/TESS/train_tess_s4d_classification.yaml"),
    ("s4d_head",     "checkpoints/tess_s4d_head_classification/best.ckpt",
                     "configs/TESS/train_tess_s4d_head_classification.yaml"),
]

FROZEN_TASKS = {"s4d_head"}   # use load_task instead of load_model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir",   default="data/TESS/.cache/TESS")
    parser.add_argument("--out_dir",    default="benchmarks/TESS")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--device",     default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--skip_regression",     action="store_true")
    parser.add_argument("--skip_classification", action="store_true")
    args = parser.parse_args()

    device = torch.device(args.device)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── Regression ──────────────────────────────────────────────────────────
    if not args.skip_regression:
        from dataloader.tess_dataloader import TESSRegressionDataModule
        dm = TESSRegressionDataModule(data_dir=args.data_dir, batch_size=args.batch_size)
        dm.setup("test")
        loader = dm.test_dataloader()

        print("\n=== Regression ===")
        for model_name, ckpt_path, cfg_path in REGRESSION_MODELS:
            if not Path(ckpt_path).exists():
                print(f"  [{model_name}] checkpoint not found, skipping: {ckpt_path}")
                continue
            loader_fn = load_task if model_name in FROZEN_TASKS else load_model
            model = loader_fn(ckpt_path, cfg_path, device)

            results = evaluate_regression(model, loader, device)
            print(f"  [{model_name}] RMSE={results['rmse']:.4f}  MAE={results['mae']:.4f}  R²={results['r2']:.3f}")
            plot_regression(results, model_name, out_dir)
            save_regression_csv(results, model_name, out_dir)

    # ── Classification ──────────────────────────────────────────────────────
    if not args.skip_classification:
        from dataloader.tess_dataloader import TESSClassificationDataModule, TESSClassificationDataset
        dm = TESSClassificationDataModule(data_dir=args.data_dir, batch_size=args.batch_size)
        dm.setup("test")
        loader = dm.test_dataloader()
        label_names = dm.test.label_names

        print("\n=== Classification ===")
        for model_name, ckpt_path, cfg_path in CLASSIFICATION_MODELS:
            if not Path(ckpt_path).exists():
                print(f"  [{model_name}] checkpoint not found, skipping: {ckpt_path}")
                continue
            loader_fn = load_task if model_name in FROZEN_TASKS else load_model
            model = loader_fn(ckpt_path, cfg_path, device)

            results = evaluate_classification(model, loader, device, label_names)
            print(f"  [{model_name}] Accuracy={results['acc']:.3f}")
            for cls_name, cls_acc in results["per_class_acc"].items():
                print(f"    {cls_name}: {cls_acc:.3f}")
            plot_classification(results, model_name, label_names, out_dir)
            save_classification_csv(results, model_name, label_names, out_dir)


if __name__ == "__main__":
    main()
