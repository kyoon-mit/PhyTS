"""
Compare PhyTS per-sector frot vs TARS adopted rotation period.
Produces diagnostic plots justifying the use of TARS over frot.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

plt.rcParams.update({
    "font.size": 12,
    "axes.titlesize": 14,
    "axes.labelsize": 13,
    "figure.dpi": 150,
})

OUTDIR = "/rg/perets_prj/ilay.kamai/PhyTS/TimeSeriesPhysics/plots/tess_investigation"

# ── Load data ──
df_reg = pd.read_parquet("/rg/perets_prj/ilay.kamai/PhyTS/data/TESS/tess_regression.parquet")
df_tars = pd.read_feather("/home/ilay.kamai/work/tess/catalogs/TARS.feather")

# Merge on TIC
merged = df_reg.merge(
    df_tars[["TICID", "adopted_period", "adopted_period_unc",
             "flag_doubled_period", "flag_multiple_periods", "n_secs"]],
    left_on="TIC", right_on="TICID", how="inner",
)
merged["frot_tars"] = 1.0 / merged["adopted_period"]
merged["frot_err_tars"] = merged["adopted_period_unc"] / (merged["adopted_period"] ** 2)
merged["ratio"] = merged["frot"] / merged["frot_tars"]

# Per-star aggregation of PhyTS frot
per_star = df_reg.groupby("TIC").agg(
    frot_mean=("frot", "mean"),
    frot_std=("frot", "std"),
    frot_min=("frot", "min"),
    frot_max=("frot", "max"),
    n_sectors=("sector", "count"),
).reset_index()
per_star["frot_range"] = per_star["frot_max"] - per_star["frot_min"]
per_star["frot_cv"] = per_star["frot_std"] / per_star["frot_mean"].abs()
per_star = per_star[per_star["n_sectors"] > 1]  # only multi-sector stars

# Also merge TARS for per-star comparison
per_star_tars = per_star.merge(
    df_tars[["TICID", "adopted_period", "adopted_period_unc"]],
    left_on="TIC", right_on="TICID", how="inner",
)
per_star_tars["frot_tars"] = 1.0 / per_star_tars["adopted_period"]

# ═══════════════════════════════════════════════════════════════════════
# FIGURE 1: 4-panel overview
# ═══════════════════════════════════════════════════════════════════════
fig, axes = plt.subplots(2, 2, figsize=(14, 12))

# ── Panel A: frot vs frot_tars scatter (per light curve) ──
ax = axes[0, 0]
ax.scatter(merged["frot_tars"], merged["frot"], s=4, alpha=0.3, c="steelblue")
lim = max(merged["frot"].quantile(0.99), merged["frot_tars"].quantile(0.99))
ax.plot([0, lim], [0, lim], "k--", lw=1, label="1:1")
ax.plot([0, lim], [0, 2 * lim], "r:", lw=1, alpha=0.6, label="2:1 harmonic")
ax.plot([0, lim], [0, 0.5 * lim], "orange", ls=":", lw=1, alpha=0.6, label="1:2 sub-harmonic")
ax.set_xlabel("TARS $f_{rot}$ = 1/P$_{rot}$ [d$^{-1}$]")
ax.set_ylabel("PhyTS per-sector $f_{rot}$ [d$^{-1}$]")
ax.set_title("A) Per-sector frot vs TARS (298 stars, 2733 LCs)")
ax.set_xlim(0, lim * 1.05)
ax.set_ylim(-0.5, lim * 1.05)
ax.legend(fontsize=10)

# ── Panel B: Ratio histogram ──
ax = axes[0, 1]
ratios = merged["ratio"]
ratios_clipped = ratios[(ratios > 0) & (ratios < 5)]
ax.hist(ratios_clipped, bins=100, color="steelblue", edgecolor="white", lw=0.3)
for h, c, label in [(0.5, "orange", "0.5×"), (1.0, "k", "1.0×"), (2.0, "red", "2.0×")]:
    ax.axvline(h, color=c, ls="--", lw=1.5, label=label)
ax.set_xlabel("Ratio: PhyTS $f_{rot}$ / TARS $f_{rot}$")
ax.set_ylabel("Count")
ax.set_title("B) Frequency ratio distribution")
ax.legend(fontsize=10)

# ── Panel C: Within-star frot spread ──
ax = axes[1, 0]
ax.hist(per_star["frot_range"], bins=60, color="coral", edgecolor="white", lw=0.3)
ax.axvline(per_star["frot_range"].median(), color="k", ls="--", lw=1.5,
           label=f"median = {per_star['frot_range'].median():.3f} d$^{{-1}}$")
ax.set_xlabel("Within-star $f_{rot}$ range (max − min) [d$^{-1}$]")
ax.set_ylabel("Count (stars)")
ax.set_title(f"C) PhyTS $f_{{rot}}$ scatter across sectors ({len(per_star)} multi-sector stars)")
ax.legend(fontsize=10)

# ── Panel D: Per-star mean frot vs TARS frot ──
ax = axes[1, 1]
ps = per_star_tars
ax.errorbar(ps["frot_tars"], ps["frot_mean"], yerr=ps["frot_std"],
            fmt="o", ms=3, alpha=0.5, elinewidth=0.5, color="steelblue", ecolor="gray")
lim2 = max(ps["frot_mean"].quantile(0.99), ps["frot_tars"].quantile(0.99))
ax.plot([0, lim2], [0, lim2], "k--", lw=1, label="1:1")
ax.set_xlabel("TARS $f_{rot}$ = 1/P$_{rot}$ [d$^{-1}$]")
ax.set_ylabel("PhyTS mean $f_{rot}$ ± std [d$^{-1}$]")
ax.set_title("D) Per-star mean frot (with error bars = sector spread)")
ax.set_xlim(0, lim2 * 1.05)
ax.set_ylim(-0.5, lim2 * 1.05)
ax.legend(fontsize=10)

fig.tight_layout()
fig.savefig(f"{OUTDIR}/frot_vs_tars_overview.png", bbox_inches="tight")
print(f"Saved frot_vs_tars_overview.png")

# ═══════════════════════════════════════════════════════════════════════
# FIGURE 2: Same-star sector-to-sector inconsistency examples
# ═══════════════════════════════════════════════════════════════════════
# Pick 9 stars with the largest within-star frot range
worst = per_star.nlargest(9, "frot_range")

fig, axes = plt.subplots(3, 3, figsize=(15, 12))
for idx, (ax, (_, row)) in enumerate(zip(axes.flat, worst.iterrows())):
    tic = row["TIC"]
    star_data = df_reg[df_reg["TIC"] == tic].sort_values("sector")

    ax.errorbar(star_data["sector"], star_data["frot"], yerr=star_data["frot_err"],
                fmt="o-", ms=5, capsize=3, color="steelblue", label="PhyTS $f_{rot}$")

    # Add TARS value if available
    tars_match = df_tars[df_tars["TICID"] == tic]
    if len(tars_match):
        tars_frot = 1.0 / tars_match["adopted_period"].values[0]
        tars_err = tars_match["adopted_period_unc"].values[0] / tars_match["adopted_period"].values[0] ** 2
        ax.axhline(tars_frot, color="red", ls="--", lw=1.5, label=f"TARS = {tars_frot:.3f}")
        ax.axhspan(tars_frot - tars_err, tars_frot + tars_err, color="red", alpha=0.1)

    ax.set_title(f"TIC {tic}", fontsize=11)
    ax.set_xlabel("Sector")
    ax.set_ylabel("$f_{rot}$ [d$^{-1}$]")
    ax.legend(fontsize=8)

fig.suptitle("Worst-case PhyTS $f_{rot}$ sector-to-sector inconsistency\n(9 stars with largest within-star range)",
             fontsize=14, y=1.02)
fig.tight_layout()
fig.savefig(f"{OUTDIR}/frot_sector_inconsistency.png", bbox_inches="tight")
print(f"Saved frot_sector_inconsistency.png")

# ═══════════════════════════════════════════════════════════════════════
# FIGURE 3: Uncertainty comparison
# ═══════════════════════════════════════════════════════════════════════
fig, axes = plt.subplots(1, 3, figsize=(16, 5))

# Panel A: PhyTS frot_err distribution
ax = axes[0]
ax.hist(df_reg["frot_err"], bins=60, color="steelblue", edgecolor="white", lw=0.3)
ax.set_xlabel("PhyTS $f_{rot}$ uncertainty [d$^{-1}$]")
ax.set_ylabel("Count")
ax.set_title("A) PhyTS reported uncertainties")
ax.axvline(df_reg["frot_err"].median(), color="k", ls="--",
           label=f"median = {df_reg['frot_err'].median():.4f}")
ax.legend()

# Panel B: TARS relative uncertainty
ax = axes[1]
tars_rel = df_tars["adopted_period_unc"] / df_tars["adopted_period"]
ax.hist(tars_rel, bins=100, color="coral", edgecolor="white", lw=0.3, range=(0, 0.2))
ax.set_xlabel("TARS relative uncertainty ($\\sigma_P / P$)")
ax.set_ylabel("Count")
ax.set_title("B) TARS relative uncertainties (944k stars)")
ax.axvline(tars_rel.median(), color="k", ls="--",
           label=f"median = {tars_rel.median():.4f}")
ax.legend()

# Panel C: PhyTS actual sector scatter vs reported error
ax = axes[2]
ps = per_star_tars.copy()
ps["reported_err_mean"] = df_reg.groupby("TIC")["frot_err"].mean().reindex(ps["TIC"]).values
ax.scatter(ps["reported_err_mean"], ps["frot_std"], s=10, alpha=0.5, c="steelblue")
mx = max(ps["reported_err_mean"].max(), ps["frot_std"].max())
ax.plot([0, mx], [0, mx], "k--", lw=1, label="1:1")
ax.set_xlabel("Mean reported $f_{rot}$ error [d$^{-1}$]")
ax.set_ylabel("Actual sector-to-sector std [d$^{-1}$]")
ax.set_title("C) Reported error vs actual scatter")
ax.legend()

fig.tight_layout()
fig.savefig(f"{OUTDIR}/uncertainty_comparison.png", bbox_inches="tight")
print(f"Saved uncertainty_comparison.png")

# ═══════════════════════════════════════════════════════════════════════
# Print summary stats
# ═══════════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("SUMMARY STATISTICS")
print("=" * 60)
agree_10 = (merged["frot"] - merged["frot_tars"]).abs() < 0.1
print(f"Per-sector frot agrees with TARS within 0.1 d^-1: {agree_10.sum()}/{len(merged)} ({100*agree_10.mean():.1f}%)")

scatter_gt_err = (per_star["frot_std"] > per_star.merge(
    df_reg.groupby("TIC")["frot_err"].mean().reset_index(),
    on="TIC")["frot_err"]).sum()
print(f"Stars where actual scatter > reported error: {scatter_gt_err}/{len(per_star)} ({100*scatter_gt_err/len(per_star):.1f}%)")

print(f"PhyTS stars with >0.5 d^-1 range across sectors: {(per_star['frot_range'] > 0.5).sum()}/{len(per_star)}")
print(f"PhyTS stars with >50% coefficient of variation: {(per_star['frot_cv'] > 0.5).sum()}/{len(per_star)}")
print(f"Negative frot values in PhyTS: {(df_reg['frot'] < 0).sum()}/{len(df_reg)}")
