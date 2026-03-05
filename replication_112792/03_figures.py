"""
03_figures.py — Generate summary figures for the replication.
The paper has no numbered figures, but we create useful visualizations:
1. Application trends: MA vs other NE states (parallel trends check)
2. DiD coefficient plot for Table 1 Panel A
3. County heterogeneity: low vs high insurance counties
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from utils import *
import matplotlib.pyplot as plt

# ─────────────────────────────────────────────────────────────────────
# Figure 1: Application trends — MA vs Other NE states
# ─────────────────────────────────────────────────────────────────────
sdf = load_state_data()
sdf["treat"] = np.where(sdf["state"] == "MA", "MA", "Other NE")

# Create fiscal year
sdf["fy"] = np.nan
for yr, lo, hi in [(2003, 171, 174), (2004, 175, 178), (2005, 179, 182),
                    (2006, 183, 186), (2007, 187, 190), (2008, 191, 194),
                    (2009, 195, 198)]:
    sdf.loc[(sdf["qnum"] >= lo) & (sdf["qnum"] <= hi), "fy"] = yr

fig, axes = plt.subplots(2, 2, figsize=(12, 10))
for ax, var, title in zip(axes.flat,
                           ["allapps", "DIonly", "SSItotal", "SSDItotal"],
                           ["All Applications", "SSDI Only", "SSI Total", "SSDI Total"]):
    for treat, color, ls in [("MA", "blue", "-"), ("Other NE", "red", "--")]:
        mask = sdf["treat"] == treat
        means = []
        for yr in range(2003, 2010):
            sub = sdf.loc[mask & (sdf["fy"] == yr)]
            if len(sub) > 0:
                means.append(np.average(sub[var], weights=sub["wapop"]))
            else:
                means.append(np.nan)
        ax.plot(range(2003, 2010), means, color=color, ls=ls, marker="o", label=treat)
    ax.axvline(2006.5, color="gray", ls=":", alpha=0.7, label="Reform")
    ax.set_title(title)
    ax.set_xlabel("Fiscal Year")
    ax.set_ylabel("Rate per 1,000")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

fig.suptitle("Figure 1: Disability Application Trends — MA vs Other NE States", fontsize=13)
fig.tight_layout()
fig.savefig(os.path.join(OUT, "fig1_trends.png"), dpi=150)
print("Saved fig1_trends.png")

# ─────────────────────────────────────────────────────────────────────
# Figure 2: DiD coefficient plot — Table 1 Panel A
# ─────────────────────────────────────────────────────────────────────
# Re-run state regressions to collect coefficients
sdf = load_state_data()
states = sorted(sdf["state"].unique())
state_dummies = ["stnum1"] + [f"stnum{i}" for i in range(3, len(states) + 1)]
q_vals = sorted(sdf["qnum"].unique())
for q in q_vals:
    sdf[f"q_{q}"] = (sdf["qnum"] == q).astype(int)
q_dummies = [f"q_{q}" for q in q_vals[1:]]
interaction_vars = ["MAXpost1", "MAXpost2", "MAXpost3"]
post_vars = ["post1", "post2", "post3"]
xvars = interaction_vars + post_vars + ["ue"] + state_dummies + q_dummies

fig, ax = plt.subplots(figsize=(10, 6))
x_positions = {"FY2007": 0, "FY2008": 1, "FY2009": 2}
width = 0.2
colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]

for i, depvar in enumerate(APP_VARS_STATE):
    res, _ = run_wls(sdf, depvar, xvars)
    coefs = [res.params[v] for v in interaction_vars]
    ses = [res.bse[v] for v in interaction_vars]
    x = [j + i * width - 1.5 * width for j in range(3)]
    ax.bar(x, coefs, width=width, color=colors[i], alpha=0.7, label=depvar)
    ax.errorbar(x, coefs, yerr=[1.96 * s for s in ses], fmt="none", color="black", capsize=3)

ax.axhline(0, color="black", lw=0.5)
ax.set_xticks([0, 1, 2])
ax.set_xticklabels(["MA*FY2007", "MA*FY2008", "MA*FY2009"])
ax.set_ylabel("Coefficient (rate per 1,000)")
ax.set_title("Figure 2: DiD Coefficients — State-Level (Table 1 Panel A)")
ax.legend()
ax.grid(True, alpha=0.3, axis="y")
fig.tight_layout()
fig.savefig(os.path.join(OUT, "fig2_coef_plot.png"), dpi=150)
print("Saved fig2_coef_plot.png")

# ─────────────────────────────────────────────────────────────────────
# Figure 3: County heterogeneity — low vs high insurance
# ─────────────────────────────────────────────────────────────────────
cdf, q_vals_county = load_county_data()
cdf["fy"] = np.nan
for yr, lo, hi in [(2003, 171, 174), (2004, 175, 178), (2005, 179, 182),
                    (2006, 183, 186), (2007, 187, 190), (2008, 191, 194)]:
    cdf.loc[(cdf["q_fld"] >= lo) & (cdf["q_fld"] <= hi), "fy"] = yr

fig, axes = plt.subplots(1, 2, figsize=(12, 5))
for ax, hi_val, title in zip(axes, [1, 0], ["Low-Insurance Counties", "High-Insurance Counties"]):
    sub = cdf[cdf["lowHI"] == hi_val]
    for treat, color, ls in [("MA", "blue", "-"), ("non-MA", "red", "--")]:
        mask = sub["MA"] == (1 if treat == "MA" else 0)
        means = []
        for yr in range(2003, 2009):
            s = sub.loc[mask & (sub["fy"] == yr)]
            if len(s) > 0:
                means.append(np.average(s["allapps"], weights=s["wapop"]))
            else:
                means.append(np.nan)
        ax.plot(range(2003, 2009), means, color=color, ls=ls, marker="o", label=treat)
    ax.axvline(2006.5, color="gray", ls=":", alpha=0.7)
    ax.set_title(title)
    ax.set_xlabel("Fiscal Year")
    ax.set_ylabel("All Apps per 1,000")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

fig.suptitle("Figure 3: Application Trends by Insurance Status", fontsize=13)
fig.tight_layout()
fig.savefig(os.path.join(OUT, "fig3_heterogeneity.png"), dpi=150)
print("Saved fig3_heterogeneity.png")

plt.close("all")
print("\nAll figures saved to output/")
