"""Figure 1 of the paper: treatment effects on xenophobia-related and
labor-related term use, full sample / Biden voters / Trump voters."""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats

from utils import OUTDIR

df = pd.read_parquet(OUTDIR / "study1_cleaned.parquet")


def cell(sub, var):
    n = len(sub)
    m = sub[var].mean()
    se = sub[var].std(ddof=1) / np.sqrt(n)
    return m, se, n


def ttest(sub, var):
    a = sub.loc[sub["Tr"] == 0, var].values
    b = sub.loc[sub["Tr"] == 1, var].values
    t, p = stats.ttest_ind(a, b, equal_var=False)
    return t, p


def het_test(df, var):
    """Test whether the treatment effect differs between Biden and Trump voters
    by interacting Tr with vote_trump in OLS (robust SEs)."""
    import statsmodels.api as sm
    sub = df.dropna(subset=["vote_trump"]).copy()
    sub["Tr_x_trump"] = sub["Tr"] * sub["vote_trump"]
    X = sm.add_constant(sub[["Tr", "vote_trump", "Tr_x_trump"]])
    y = sub[var]
    res = sm.OLS(y, X).fit(cov_type="HC1")
    return res.pvalues["Tr_x_trump"]


groups = {
    "Full sample": df,
    "Biden voters": df[df["vote_trump"] == 0],
    "Trump voters": df[df["vote_trump"] == 1],
}

panels = [
    ("Panel A: Xenophobia-related terms", "contain_racism_terms"),
    ("Panel B: Labor-related terms", "contain_labor_terms"),
]

print("=" * 70)
print("Figure 1 replication (means and standard errors)")
print("=" * 70)
rows = []
for title, var in panels:
    print(f"\n{title}")
    for gname, gdf in groups.items():
        for tr, tlbl in [(0, "Before"), (1, "After")]:
            m, se, n = cell(gdf[gdf["Tr"] == tr], var)
            print(f"  {gname:14s} {tlbl:7s} mean={m:.4f}  se={se:.4f}  n={n}")
            rows.append({"var": var, "group": gname, "tr": tlbl, "mean": m, "se": se, "n": n})
        t, p = ttest(gdf, var)
        print(f"    {gname:14s} welch t-test p = {p:.4g}")
    p_het = het_test(df, var)
    print(f"  Heterogeneity (interaction) p-value: {p_het:.4g}")

pd.DataFrame(rows).to_csv(OUTDIR / "figure1_values.csv", index=False)


# build the figure
fig, axes = plt.subplots(2, 1, figsize=(7, 7))
for ax, (title, var) in zip(axes, panels):
    xs, heights, errs, labels = [], [], [], []
    pos = 0
    group_centers = []
    for gname, gdf in groups.items():
        g_start = pos
        for tr in [0, 1]:
            m, se, n = cell(gdf[gdf["Tr"] == tr], var)
            xs.append(pos)
            heights.append(m)
            errs.append(se)
            labels.append("Before" if tr == 0 else "After")
            pos += 1
        group_centers.append((g_start + pos - 1) / 2)
        pos += 1  # spacer
    ax.bar(xs, heights, yerr=errs, capsize=4, color="lightgray", edgecolor="black")
    ax.set_xticks(xs)
    ax.set_xticklabels(labels)
    for c, name in zip(group_centers, groups.keys()):
        ax.text(c, ax.get_ylim()[1] * 0.92, name, ha="center", fontsize=10, fontweight="bold")
    ax.set_title(title)
    ax.set_ylabel("Mean ± s.e.m.")
    if var == "contain_racism_terms":
        ax.set_ylim(0, 0.22)
    else:
        ax.set_ylim(0, 0.9)

plt.tight_layout()
plt.savefig(OUTDIR / "figure1.png", dpi=150, bbox_inches="tight")
print(f"\nSaved figure to {OUTDIR/'figure1.png'}")
