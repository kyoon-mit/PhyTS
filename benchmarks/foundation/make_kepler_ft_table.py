"""Render Kepler Q9 v3 fine-tuning grid as a LaTeX table.

Rows are model configs (e.g. MOMENT-base, MOMENT-large), columns are
{zero-shot, LoRA r=8, LoRA r=16, full SFT}. Cells contain test accuracy
(or '--' if the corresponding run does not exist on disk).

Usage
-----
    python benchmarks/foundation/make_kepler_ft_table.py \
        --out plots/kepler_q9v3/ft_table.tex
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
MODEL_ORDER = ["moment", "chronos", "timemoe", "granite_ttm",
               "timesfm", "moirai", "lagllama"]
SIZE_ORDER = ["base", "large"]

REGIMES = ["zs", "lora8", "lora16", "full"]
REGIME_LABELS = {
    "zs":     r"Zero-shot",
    "lora8":  r"LoRA $r{=}8$",
    "lora16": r"LoRA $r{=}16$",
    "full":   r"Full SFT",
}

# Canonical results dir per (model, size, regime). Picks a single definitive
# run when multiple variants exist (e.g. lora/ vs lora_moment_r8/).
RESULTS = {
    # ── MOMENT ─────────────────────────────────────────────────────────
    ("moment",      "base",  "zs"):     "linear_probe",
    ("moment",      "base",  "lora8"):  "lora_moment_r8",
    ("moment",      "base",  "lora16"): "lora_moment_r16",
    ("moment",      "base",  "full"):   "full_ft",
    ("moment",      "large", "zs"):     "linear_probe_large",
    # ── Chronos ────────────────────────────────────────────────────────
    ("chronos",     "base",  "zs"):     "linear_probe",
    ("chronos",     "base",  "lora8"):  "lora_chronos_r8_bs4",
    ("chronos",     "base",  "lora16"): "lora_chronos_r16_bs4",
    ("chronos",     "base",  "full"):   "full_ft_chronos",
    ("chronos",     "large", "zs"):     "linear_probe_large",
    # ── Time-MoE (wrapper aliases base==small for the 50M ckpt) ────────
    ("timemoe",     "base",  "zs"):     "linear_probe",
    ("timemoe",     "base",  "lora8"):  "lora_timemoe_r8_augustijn",
    ("timemoe",     "base",  "lora16"): "lora_timemoe_r16_condor",
    ("timemoe",     "base",  "full"):   "full_ft_timemoe_condor",
    ("timemoe",     "large", "zs"):     "linear_probe_large",
    # ── Granite-TTM ────────────────────────────────────────────────────
    ("granite_ttm", "base",  "zs"):     "linear_probe",
    ("granite_ttm", "base",  "full"):   "full_ft_granite_ttm",
    ("granite_ttm", "large", "zs"):     "linear_probe_large",
    # ── TimesFM ────────────────────────────────────────────────────────
    ("timesfm",     "base",  "zs"):     "linear_probe",
    ("timesfm",     "base",  "lora8"):  "lora_timesfm_r8_bs4",
    ("timesfm",     "base",  "lora16"): "lora_timesfm_r16_bs4",
    ("timesfm",     "base",  "full"):   "full_ft_timesfm",
    ("timesfm",     "large", "zs"):     "linear_probe_large",
    ("timesfm",     "large", "lora8"):  "lora_timesfm_large_r8_krakas",
    ("timesfm",     "large", "lora16"): "lora_timesfm_large_r16_krakas",
    ("timesfm",     "large", "full"):   "full_ft_timesfm_large_krakas",
    # ── MOIRAI ─────────────────────────────────────────────────────────
    ("moirai",      "base",  "zs"):     "linear_probe",
    ("moirai",      "base",  "lora8"):  "lora_moirai_r8_condor",
    ("moirai",      "base",  "lora16"): "lora_moirai_r16_condor",
    ("moirai",      "base",  "full"):   "full_ft_moirai_condor",
    ("moirai",      "large", "zs"):     "linear_probe_large",
    ("moirai",      "large", "lora8"):  "lora_moirai_large_r8_deanston",
    ("moirai",      "large", "lora16"): "lora_moirai_large_r16_deanston",
}

# (model, size) pairs to render — in display order.
ROW_KEYS = [
    ("moment",      "base"),  ("moment",      "large"),
    ("chronos",     "base"),  ("chronos",     "large"),
    ("timemoe",     "base"),  ("timemoe",     "large"),
    ("granite_ttm", "base"),  ("granite_ttm", "large"),
    ("timesfm",     "base"),  ("timesfm",     "large"),
    ("moirai",      "base"),  ("moirai",      "large"),
]


def _row_label(model: str, size: str) -> str:
    return f"{MODEL_LABELS[model]}-{size}"


def _load_acc(root: Path, model: str, size: str, regime: str) -> float | None:
    sub = RESULTS.get((model, size, regime))
    if sub is None:
        return None
    p = root / sub / model / "metrics.json"
    if not p.exists():
        return None
    with open(p) as f:
        d = json.load(f)
    return float(d["metrics"]["accuracy"])


def build_table(root: Path) -> str:
    # Build the value matrix.
    matrix: list[list[float | None]] = []
    for m, s in ROW_KEYS:
        matrix.append([_load_acc(root, m, s, r) for r in REGIMES])

    # Bold the per-column max.
    bolded: list[list[str]] = [["--"] * len(REGIMES) for _ in ROW_KEYS]
    for c in range(len(REGIMES)):
        col = [matrix[r][c] for r in range(len(ROW_KEYS))]
        valid = [v for v in col if v is not None]
        best = max(valid) if valid else None
        for r, v in enumerate(col):
            if v is None:
                bolded[r][c] = "--"
            elif best is not None and v == best:
                bolded[r][c] = r"\textbf{" + f"{v:.3f}" + "}"
            else:
                bolded[r][c] = f"{v:.3f}"

    lines = []
    lines.append(r"% Fine-tuning grid on Kepler Q9 v3 (test accuracy)")
    lines.append(r"\begin{table}[h]")
    lines.append(r"\centering")
    lines.append(r"\caption{Test accuracy on Kepler Q9 v3 across fine-tuning regimes "
                 r"(stratified 70/15/15 split, seed=42, batch best-val checkpoint, "
                 r"early stopping patience 20). "
                 r"Zero-shot = frozen backbone + L2-regularised logistic regression on "
                 r"mean/last-pooled embeddings. "
                 r"LoRA = adapters on attention projections only "
                 r"(head\,lr$=10^{-3}$, backbone\,lr$=10^{-4}$). "
                 r"Full SFT = all parameters trainable "
                 r"(head\,lr$=10^{-3}$, backbone\,lr$=10^{-5}$). "
                 r"`--' = run not available. Bold = best per column.}")
    lines.append(r"\label{tab:kepler_q9v3_ft_grid}")
    lines.append(r"\begin{tabular}{l" + "c" * len(REGIMES) + "}")
    lines.append(r"\toprule")
    lines.append("Model & " + " & ".join(REGIME_LABELS[r] for r in REGIMES) + r" \\")
    lines.append(r"\midrule")
    prev_model = None
    for i, (m, s) in enumerate(ROW_KEYS):
        if prev_model is not None and m != prev_model:
            lines.append(r"\midrule")
        prev_model = m
        label = _row_label(m, s).replace("_", r"\_")
        lines.append(f"{label} & " + " & ".join(bolded[i]) + r" \\")
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")
    return "\n".join(lines)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--root", default="plots/kepler_q9v3",
                   help="Root containing the linear_probe / lora_* / full_ft_* dirs.")
    p.add_argument("--out", default="plots/kepler_q9v3/ft_table.tex")
    args = p.parse_args()

    tex = build_table(Path(args.root))
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(tex)
    print(f"Wrote {out}")
    print("\n--- preview ---\n")
    print(tex)


if __name__ == "__main__":
    main()
