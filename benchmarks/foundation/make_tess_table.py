"""Render the TESS classification benchmark as a single comparison table.

Pulls accuracy / balanced accuracy / macro-F1 from
``plots/tess/linear_probe/<model>/metrics.json`` (zero-shot probe) and from
the LoRA / fine-tune output directories (defaults: ``plots/tess/lora*``)
and prints a side-by-side table — Markdown to stdout, LaTeX optionally to
a file.

Usage
-----
    python benchmarks/foundation/make_tess_table.py \\
        --zero_shot_dir plots/tess/linear_probe \\
        --lora_dirs     plots/tess/lora plots/tess/lora_brigand plots/tess/lora_krakas \\
        --full_dirs     plots/tess/full \\
        --metric accuracy \\
        --tex_out plots/tess/table.tex

If multiple LoRA dirs hold the same model, the run with the highest
``val_accuracy`` is kept (so you don't have to manually pick the best).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

MODEL_LABELS = {
    "moment":      "MOMENT",
    "chronos":     "Chronos",
    "timemoe":     "Time-MoE",
    "granite_ttm": "Granite-TTM",
    "timesfm":     "TimesFM",
    "moirai":      "MOIRAI",
    "lagllama":    "Lag-Llama",
}
MODEL_ORDER = ["moment", "chronos", "timesfm", "timemoe", "moirai", "lagllama", "granite_ttm"]


def _read_metrics(model_dir: Path) -> dict | None:
    path = model_dir / "metrics.json"
    if not path.exists():
        return None
    with open(path) as f:
        return json.load(f)


def _collect(dir_paths: list[Path]) -> dict[str, dict]:
    """Return {model_name: best_metrics_dict} across all given dirs."""
    best: dict[str, dict] = {}
    for d in dir_paths:
        if not d.exists():
            continue
        for sub in sorted(d.iterdir()):
            if not sub.is_dir() or sub.name.startswith("_"):
                continue
            data = _read_metrics(sub)
            if data is None:
                continue
            cur = best.get(sub.name)
            if cur is None:
                best[sub.name] = data
            else:
                # Pick higher val_accuracy.
                if data["metrics"].get("val_accuracy", -1) > cur["metrics"].get("val_accuracy", -1):
                    best[sub.name] = data
    return best


def _fmt(x) -> str:
    if x is None:
        return "—"
    try:
        return f"{x:.4f}"
    except (TypeError, ValueError):
        return str(x)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--zero_shot_dir", type=Path, default=Path("plots/tess/linear_probe"))
    ap.add_argument("--lora_dirs", type=Path, nargs="*", default=[
        Path("plots/tess/lora"),
        Path("plots/tess/lora_brigand"),
        Path("plots/tess/lora_krakas"),
    ])
    ap.add_argument("--full_dirs", type=Path, nargs="*", default=[Path("plots/tess/full")])
    ap.add_argument("--metric", default="accuracy",
                    choices=["accuracy", "balanced_accuracy", "macro_f1"])
    ap.add_argument("--tex_out", type=Path, default=None)
    args = ap.parse_args()

    zs = _collect([args.zero_shot_dir])
    lora = _collect(args.lora_dirs)
    full = _collect(args.full_dirs)

    metric = args.metric
    metric_label = {
        "accuracy":          "Accuracy",
        "balanced_accuracy": "Balanced Acc.",
        "macro_f1":          "Macro-F1",
    }[metric]

    rows: list[tuple[str, float | None, float | None, float | None]] = []
    for key in MODEL_ORDER:
        if key not in zs and key not in lora and key not in full:
            continue
        rows.append((
            MODEL_LABELS.get(key, key),
            zs.get(key,   {}).get("metrics", {}).get(metric),
            lora.get(key, {}).get("metrics", {}).get(metric),
            full.get(key, {}).get("metrics", {}).get(metric),
        ))

    # ── Markdown ───────────────────────────────────────────────────────
    headers = ["Model", f"Zero-shot {metric_label}", f"LoRA {metric_label}", f"Full FT {metric_label}"]
    widths = [max(len(h), 12) for h in headers]
    for r in rows:
        widths[0] = max(widths[0], len(r[0]))
    print("| " + " | ".join(h.ljust(w) for h, w in zip(headers, widths)) + " |")
    print("|" + "|".join("-" * (w + 2) for w in widths) + "|")
    for name, z, l, f in rows:
        cells = [name.ljust(widths[0]), _fmt(z).rjust(widths[1]),
                 _fmt(l).rjust(widths[2]), _fmt(f).rjust(widths[3])]
        print("| " + " | ".join(cells) + " |")

    # Annotate provenance.
    print()
    print(f"Metric: {metric}.  Zero-shot from {args.zero_shot_dir}.  "
          f"LoRA pooled across {[str(p) for p in args.lora_dirs]} "
          f"(best val_accuracy per model).  "
          f"Full FT from {[str(p) for p in args.full_dirs]}.")

    # ── LaTeX (optional) ───────────────────────────────────────────────
    if args.tex_out is not None:
        args.tex_out.parent.mkdir(parents=True, exist_ok=True)
        lines = [
            r"\begin{tabular}{lccc}",
            r"\toprule",
            r"Model & Zero-shot " + metric_label
              + r" & LoRA " + metric_label
              + r" & Full FT " + metric_label + r" \\",
            r"\midrule",
        ]
        for name, z, l, f in rows:
            cells = [name, _fmt(z), _fmt(l), _fmt(f)]
            lines.append(" & ".join(cells) + r" \\")
        lines += [r"\bottomrule", r"\end{tabular}"]
        args.tex_out.write_text("\n".join(lines) + "\n")
        print(f"\nWrote LaTeX → {args.tex_out}")


if __name__ == "__main__":
    main()
