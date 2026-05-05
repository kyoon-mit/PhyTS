"""One-shot overview table for the TESS benchmark.

Combines classification accuracy and regression R² across three regimes —
zero-shot probe, LoRA fine-tune, full fine-tune — into a single Markdown
table.  Lag-Llama is excluded by default (too slow to fit the rest of the
schedule).

Skips smoke-test artefacts (``args.smoke_test == True`` or fewer than
``--min_epochs`` epochs of fine-tuning history) so a stale 2-batch run
doesn't poison the table — those cells render as ``pending``.

Usage
-----
    python benchmarks/foundation/make_tess_overview.py
    python benchmarks/foundation/make_tess_overview.py --tex_out plots/tess/overview.tex
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

MODEL_LABELS = {
    "moment":      "MOMENT",
    "chronos":     "Chronos",
    "timesfm":     "TimesFM",
    "timemoe":     "Time-MoE",
    "moirai":      "MOIRAI",
    "granite_ttm": "Granite-TTM",
}
MODEL_ORDER = ["moment", "chronos", "timesfm", "timemoe", "moirai", "granite_ttm"]
PENDING = "pending"


def _read(p: Path) -> dict | None:
    if not p.exists():
        return None
    with open(p) as f:
        return json.load(f)


def _is_smoke(d: dict, *, min_epochs: int) -> bool:
    args = d.get("args", {}) or {}
    if args.get("smoke_test"):
        return True
    history = d.get("history")
    if history is not None and len(history) < min_epochs:
        return True
    return False


def _collect(dirs: list[Path], *, val_key: str, lower_better: bool, min_epochs: int) -> dict[str, dict]:
    """For each model, return the metrics dict from the run with the best
    ``val_key`` across the given dirs.  Skips smoke-test artefacts."""
    best: dict[str, tuple[float, dict]] = {}
    for d in dirs:
        if not d.exists():
            continue
        for sub in sorted(d.iterdir()):
            if not sub.is_dir() or sub.name.startswith("_"):
                continue
            data = _read(sub / "metrics.json")
            if data is None:
                continue
            if _is_smoke(data, min_epochs=min_epochs):
                continue
            v = data["metrics"].get(val_key)
            if v is None:
                continue
            score = -v if lower_better else v
            cur = best.get(sub.name)
            if cur is None or score > cur[0]:
                best[sub.name] = (score, data)
    return {k: v[1] for k, v in best.items()}


def _fmt(d: dict | None, key: str) -> str:
    if d is None:
        return PENDING
    v = d["metrics"].get(key)
    if v is None:
        return PENDING
    return f"{v:.4f}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--zs_cls_dir", type=Path, default=Path("plots/tess/linear_probe"))
    ap.add_argument("--lora_cls_dirs", type=Path, nargs="*", default=[
        Path("plots/tess/lora"),
        Path("plots/tess/lora_brigand"),
        Path("plots/tess/lora_krakas"),
        Path("plots/tess/lora_ramee"),
    ])
    ap.add_argument("--full_cls_dirs", type=Path, nargs="*", default=[Path("plots/tess/full_ramee")])

    ap.add_argument("--zs_reg_dir", type=Path, default=Path("plots/tess/linear_probe_regression"))
    ap.add_argument("--lora_reg_dirs", type=Path, nargs="*", default=[
        Path("plots/tess/lora_regression"),
        Path("plots/tess/lora_regression_brigand"),
        Path("plots/tess/lora_regression_ramee"),
    ])
    ap.add_argument("--full_reg_dirs", type=Path, nargs="*", default=[Path("plots/tess/full_regression_ramee")])

    ap.add_argument("--min_epochs", type=int, default=5,
                    help="Treat fine-tune runs with fewer than this many epochs as smoke tests.")
    ap.add_argument("--tex_out", type=Path, default=None)
    args = ap.parse_args()

    # Classification: best by val_accuracy (higher is better).
    zs_cls   = _collect([args.zs_cls_dir],   val_key="val_accuracy", lower_better=False, min_epochs=0)
    lora_cls = _collect(args.lora_cls_dirs,  val_key="val_accuracy", lower_better=False, min_epochs=args.min_epochs)
    full_cls = _collect(args.full_cls_dirs,  val_key="val_accuracy", lower_better=False, min_epochs=args.min_epochs)

    # Regression: best by val_mse (lower is better) for fine-tunes; zero-shot
    # regression evaluator stores val_r2 / val_mse — pick val_mse for parity.
    zs_reg   = _collect([args.zs_reg_dir],   val_key="val_mse", lower_better=True, min_epochs=0)
    lora_reg = _collect(args.lora_reg_dirs,  val_key="val_mse", lower_better=True, min_epochs=args.min_epochs)
    full_reg = _collect(args.full_reg_dirs,  val_key="val_mse", lower_better=True, min_epochs=args.min_epochs)

    headers = [
        "Model",
        "Cls Acc (ZS)", "Cls Acc (LoRA)", "Cls Acc (Full)",
        "Reg R² (ZS)",  "Reg R² (LoRA)", "Reg R² (Full)",
    ]
    rows = []
    for key in MODEL_ORDER:
        rows.append([
            MODEL_LABELS[key],
            _fmt(zs_cls.get(key),   "accuracy"),
            _fmt(lora_cls.get(key), "accuracy"),
            _fmt(full_cls.get(key), "accuracy"),
            _fmt(zs_reg.get(key),   "r2"),
            _fmt(lora_reg.get(key), "r2"),
            _fmt(full_reg.get(key), "r2"),
        ])

    widths = [max(len(h), max(len(r[i]) for r in rows)) for i, h in enumerate(headers)]

    def _line(cells):
        out = []
        for c, w, i in zip(cells, widths, range(len(cells))):
            out.append(c.ljust(w) if i == 0 else c.rjust(w))
        return "| " + " | ".join(out) + " |"

    print(_line(headers))
    print("|" + "|".join("-" * (w + 2) for w in widths) + "|")
    for r in rows:
        print(_line(r))

    print()
    print("Cls metric: accuracy on test set (best by val_accuracy across runs).")
    print("Reg metric: R² on test set (best by val_mse across runs).")
    print(f"'{PENDING}' = no real run on disk yet (smoke-test artefacts ignored).")
    print("Lag-Llama excluded.")

    if args.tex_out is not None:
        args.tex_out.parent.mkdir(parents=True, exist_ok=True)
        # Pending → em-dash for paper-ready tables.
        def _tex_cell(s: str) -> str:
            return r"\textemdash" if s == PENDING else s

        # 7-column booktabs table with grouped multicolumn header
        # (Classification | Regression).
        lines = [
            r"\begin{table}[t]",
            r"\centering",
            r"\small",
            r"\caption{%",
            r"  Foundation-model benchmark on the PhyTS-bench TESS subset.",
            r"  \emph{Classification} predicts one of 8 stellar-variability classes",
            r"  (\textsc{aperiodic}, \textsc{contact\_rot}, \textsc{dsct\_bcep},",
            r"  \textsc{eclipse}, \textsc{gdor\_spb}, \textsc{instrument/junk},",
            r"  \textsc{rrlyr\_ceph}, \textsc{solarlike}) and reports test-set accuracy.",
            r"  \emph{Regression} predicts the rotation frequency $f_{\mathrm{rot}}$",
            r"  (cycles per day) and reports test-set $R^2$.",
            r"  Both tasks use the same TIC-grouped train/val/test split (seed 42)",
            r"  so no star appears in more than one split.",
            r"  Light curves are padded or truncated to 1024 samples.",
            r"  Three regimes are compared: zero-shot linear probe on frozen",
            r"  embeddings, LoRA fine-tuning ($r{=}8$, $\alpha{=}16$, head LR $10^{-3}$,",
            r"  backbone LR $10^{-4}$), and full fine-tuning of all backbone parameters",
            r"  (backbone LR $10^{-5}$). Best run per model is reported, selected by",
            r"  validation accuracy (classification) or validation MSE (regression).",
            r"  \textemdash{} indicates a run not yet completed. Lag-Llama is omitted",
            r"  (single inference pass exceeds the available compute budget).",
            r"  }",
            r"  \label{tab:tess_benchmark}",
            r"  \begin{tabular}{lccc@{\hskip 1.5em}ccc}",
            r"    \toprule",
            r"    & \multicolumn{3}{c}{Classification: Accuracy $\uparrow$}"
            r" & \multicolumn{3}{c}{Regression: $R^2$ $\uparrow$} \\",
            r"    \cmidrule(lr){2-4} \cmidrule(lr){5-7}",
            r"    Model & Zero-shot & LoRA & Full FT & Zero-shot & LoRA & Full FT \\",
            r"    \midrule",
        ]
        for r in rows:
            cells = [r[0]] + [_tex_cell(c) for c in r[1:]]
            lines.append("    " + " & ".join(cells) + r" \\")
        lines += [
            r"    \bottomrule",
            r"  \end{tabular}",
            r"\end{table}",
        ]
        args.tex_out.write_text("\n".join(lines) + "\n")
        print(f"\nWrote LaTeX → {args.tex_out}")


if __name__ == "__main__":
    main()
