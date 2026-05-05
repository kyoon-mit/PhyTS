"""Render Kepler Q9 v3 linear-probe results as a LaTeX table.

Usage
-----
    python benchmarks/foundation/make_kepler_table.py \
        --results_dirs plots/kepler_q9v3/linear_probe \
                       plots/kepler_q9v3/linear_probe_large \
        --out plots/kepler_q9v3/linear_probe/table.tex
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
MODEL_ORDER = ["moment", "chronos", "timemoe", "granite_ttm", "timesfm", "moirai", "lagllama"]
SIZE_ORDER = ["small", "base", "large"]

# HuggingFace checkpoint actually used per (model, size), for the caption / readers.
SIZE_HF = {
    ("moment",      "base"):  "MOMENT-1-base",
    ("moment",      "large"): "MOMENT-1-large",
    ("chronos",     "base"):  "chronos-t5-base",
    ("chronos",     "large"): "chronos-t5-large",
    ("timemoe",     "base"):  "TimeMoE-50M",
    ("timemoe",     "large"): "TimeMoE-200M",
    ("timesfm",     "base"):  "timesfm-1.0-200m",
    ("timesfm",     "large"): "timesfm-2.0-500m",
    ("granite_ttm", "base"):  "TTM-r1 (512/96)",
    ("granite_ttm", "large"): "TTM-r1 (1024/96)",
    ("moirai",      "base"):  "moirai-1.1-R-base",
    ("moirai",      "large"): "moirai-1.1-R-large",
    ("lagllama",    "base"):  "lag-llama",
}


def _row_label(model: str, size: str) -> str:
    base = MODEL_LABELS.get(model, model)
    return f"{base}-{size}"


def _bold_max(values: list[float]) -> list[str]:
    if not values or all(v != v for v in values):
        return [f"{v:.3f}" for v in values]
    best = max(v for v in values if v == v)
    return [
        (r"\textbf{" + f"{v:.3f}" + "}") if v == best else f"{v:.3f}"
        for v in values
    ]


def _collect_rows(results_dirs: list[Path]) -> list[tuple[str, str, dict]]:
    """Walk every metrics.json under each results_dir, return (model, size, data)."""
    found: dict[tuple[str, str], dict] = {}
    for rd in results_dirs:
        if not rd.exists():
            continue
        for sub in rd.iterdir():
            mfile = sub / "metrics.json"
            if not mfile.exists():
                continue
            with open(mfile) as f:
                d = json.load(f)
            model = sub.name.lower().replace("-", "_")
            size = (d.get("args") or {}).get("model_size", "base")
            # Keep first occurrence per (model, size); later dirs don't override.
            found.setdefault((model, size), d)

    def _key(item):
        (m, s), _ = item
        mi = MODEL_ORDER.index(m) if m in MODEL_ORDER else len(MODEL_ORDER)
        si = SIZE_ORDER.index(s) if s in SIZE_ORDER else len(SIZE_ORDER)
        return (mi, si)

    return [(m, s, d) for (m, s), d in sorted(found.items(), key=_key)]


def build_table(results_dirs: list[Path], class_names: list[str]) -> str:
    rows = _collect_rows(results_dirs)
    if not rows:
        raise SystemExit(f"No metrics.json files found under {results_dirs}")

    # ── Summary table (overall metrics) ──────────────────────────────────
    accs = [r[2]["metrics"]["accuracy"] for r in rows]
    bals = [r[2]["metrics"]["balanced_accuracy"] for r in rows]
    f1s  = [r[2]["metrics"]["macro_f1"] for r in rows]
    d_embeds = [r[2]["d_embed"] for r in rows]
    pools = [r[2].get("pool", "mean") for r in rows]

    acc_s = _bold_max(accs)
    bal_s = _bold_max(bals)
    f1_s  = _bold_max(f1s)

    arch_label = {
        "moment": "enc.", "chronos": "enc-dec.", "moirai": "enc.",
        "granite_ttm": "mixer",
        "timemoe": "dec.", "timesfm": "dec.", "lagllama": "dec.",
    }

    lines = []
    lines.append(r"% Zero-shot linear probe on Kepler Q9 v3 (9-class stellar variability)")
    lines.append(r"\begin{table}[h]")
    lines.append(r"\centering")
    lines.append(r"\caption{Zero-shot linear probe on Kepler Q9 v3 "
                 r"(stratified 70/15/15 split, seed=42). "
                 r"Each foundation model's backbone is frozen; "
                 r"embeddings are extracted from three overlapping 512-sample "
                 r"windows and pooled (mean for encoders, last-token for causal "
                 r"decoders) before fitting an L2-regularised logistic regression. "
                 r"Model rows are labelled by checkpoint size; bold = best per column; "
                 r"$d$ = embedding dimension.}")
    lines.append(r"\label{tab:kepler_q9v3_linear_probe}")
    lines.append(r"\begin{tabular}{llccccc}")
    lines.append(r"\toprule")
    lines.append(r"Model & Arch. & Pool & $d$ & Accuracy & Bal.\ Acc. & Macro F1 \\")
    lines.append(r"\midrule")
    prev_model = None
    for i, (m, s, _) in enumerate(rows):
        if prev_model is not None and m != prev_model:
            lines.append(r"\midrule")
        prev_model = m
        label = _row_label(m, s).replace("_", r"\_")
        lines.append(f"{label} & {arch_label.get(m, '-')} & "
                     f"{pools[i]} & {d_embeds[i]} & "
                     f"{acc_s[i]} & {bal_s[i]} & {f1_s[i]} \\\\")
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")
    lines.append("")

    # ── Per-class F1 table ───────────────────────────────────────────────
    lines.append(r"\begin{table}[h]")
    lines.append(r"\centering")
    lines.append(r"\caption{Per-class F1 on Kepler Q9 v3 (zero-shot linear probe).}")
    lines.append(r"\label{tab:kepler_q9v3_per_class_f1}")
    col_spec = "l" + "c" * len(rows)
    lines.append(r"\resizebox{\textwidth}{!}{")
    lines.append(r"\begin{tabular}{" + col_spec + "}")
    lines.append(r"\toprule")
    header_cells = [_row_label(m, s).replace("_", r"\_") for m, s, _ in rows]
    lines.append("Class & " + " & ".join(header_cells) + r" \\")
    lines.append(r"\midrule")

    per_class_f1: list[list[float]] = []
    per_class_support: list[int] = []
    for cname in class_names:
        row_f1 = []
        support = 0
        for _, _, d in rows:
            rep = d["report"].get(cname, {})
            row_f1.append(float(rep.get("f1-score", float("nan"))))
            support = int(rep.get("support", support))
        per_class_f1.append(row_f1)
        per_class_support.append(support)

    for ci, cname in enumerate(class_names):
        bolded = _bold_max(per_class_f1[ci])
        pretty_cname = cname.replace("_", r"\_")
        lines.append(f"{pretty_cname} ($n={per_class_support[ci]}$) & "
                     + " & ".join(bolded) + r" \\")

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"}")
    lines.append(r"\end{table}")

    return "\n".join(lines)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--results_dirs", nargs="+",
                   default=["plots/kepler_q9v3/linear_probe",
                            "plots/kepler_q9v3/linear_probe_large"],
                   help="One or more directories containing per-model metrics.json. "
                        "model_size is read from each metrics.json's saved args.")
    p.add_argument("--out", default=None,
                   help="Where to write the .tex file "
                        "(default: <first results_dir>/table.tex)")
    args = p.parse_args()

    results_dirs = [Path(d) for d in args.results_dirs]

    # Discover class_names from any available metrics.json
    class_names = None
    for rd in results_dirs:
        if not rd.exists():
            continue
        for sub in rd.iterdir():
            m = sub / "metrics.json"
            if m.exists():
                with open(m) as f:
                    class_names = json.load(f)["class_names"]
                break
        if class_names is not None:
            break
    if class_names is None:
        raise SystemExit(f"No metrics.json found under {results_dirs}")

    tex = build_table(results_dirs, class_names)
    out = Path(args.out) if args.out else results_dirs[0] / "table.tex"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(tex)
    print(f"Wrote {out}")
    print("\n--- preview ---\n")
    print(tex)


if __name__ == "__main__":
    main()
