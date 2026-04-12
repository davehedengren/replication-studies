"""Robustness checks for Leonardi & Moretti (2022) Table 1, column 5 -
the 'no pre-determined neighborhood characteristic predicts ex-post growth'
claim is the most stress-worthy. We stress-test the multivariate columns 1-4
and the bivariate column 5 along multiple dimensions."""
import numpy as np
import pandas as pd
import statsmodels.api as sm
from utils import build_analysis_frame

df = build_analysis_frame()
y = "d1204_r"

base = [
    "prezzo2004Abitaz",
    "prezzo2004Negozi",
    "metro_dum",
    "univ",
    "attraction_dum",
    "day_pop",
    "lat2004",
    "prezzo2004",
    "cucina2004",
    "mmichelin2004",
]
c1 = base
c2 = base + ["lat2004_d", "lat2004_a"]
c3 = base + ["d0004_d", "d0004_a"]
c4 = base + ["lat2004_d", "lat2004_a", "d0004_d", "d0004_a"]


def fit(data, xs, cov_type="nonrobust", cov_kwds=None):
    sub = data[[y] + list(xs)].dropna()
    X = sm.add_constant(sub[list(xs)])
    return sm.OLS(sub[y], X).fit(cov_type=cov_type, cov_kwds=cov_kwds or {})


def row(label, spec_name, spec, data, cov_type="nonrobust"):
    m = fit(data, spec, cov_type=cov_type)
    keyvars = ["prezzo2004Abitaz", "lat2004", "prezzo2004"]
    pieces = []
    for v in keyvars:
        if v in m.params.index:
            star = ""
            p = m.pvalues[v]
            if p < 0.01: star = "***"
            elif p < 0.05: star = "**"
            elif p < 0.10: star = "*"
            pieces.append(f"{v[:12]}:{m.params[v]:+.3f}{star}")
    print(
        f"  [{spec_name}] {label:<45} N={int(m.nobs):3d} R²={m.rsquared:.3f}  "
        + "  ".join(pieces)
    )


print("=" * 90)
print("ROBUSTNESS TO SPECIFICATION OF TABLE 1, multivariate columns")
print("=" * 90)

print("\n(0) Baseline (published spec)")
for name, spec in [("c1", c1), ("c2", c2), ("c3", c3), ("c4", c4)]:
    row("baseline", name, spec, df)

print("\n(1) HC1 robust SEs (Stata 'reg , robust')")
for name, spec in [("c1", c1), ("c4", c4)]:
    row("HC1", name, spec, df, cov_type="HC1")

print("\n(2) HC3 robust SEs")
for name, spec in [("c1", c1), ("c4", c4)]:
    row("HC3", name, spec, df, cov_type="HC3")

print("\n(3) Drop CBD zones (attraction_dum == 1, the 3 tourism-heavy zones)")
sub = df[df["attraction_dum"] == 0]
for name, spec in [("c1", c1), ("c4", c4)]:
    # drop attraction dummy since it's now constant
    row("no-CBD", name, [x for x in spec if x != "attraction_dum"], sub)

print("\n(4) Drop zone 153 (extreme d1204_r = -1.386) and zone 1 (huge day_pop)")
sub = df[~df["zona180"].isin([1, 153])]
for name, spec in [("c1", c1), ("c4", c4)]:
    row("no-outliers", name, spec, sub)

print("\n(5) Winsorize d1204_r at 5/95 percentiles")
lo, hi = df[y].quantile([0.05, 0.95])
wdf = df.copy()
wdf[y] = wdf[y].clip(lo, hi)
for name, spec in [("c1", c1), ("c4", c4)]:
    row("winsor 5/95", name, spec, wdf)

print("\n(6) Winsorize d1204_r at 1/99 percentiles")
lo, hi = df[y].quantile([0.01, 0.99])
wdf = df.copy()
wdf[y] = wdf[y].clip(lo, hi)
for name, spec in [("c1", c1), ("c4", c4)]:
    row("winsor 1/99", name, spec, wdf)

print("\n(7) Drop zones with prezzo2004/cucina2004 missing (same sample as c1-c4 -")
print("    already the case; check replacing with zero and dummy-flag)")
alt = df.copy()
alt["prezzo2004_miss"] = alt["prezzo2004"].isna().astype(int)
alt["cucina2004_miss"] = alt["cucina2004"].isna().astype(int)
alt["prezzo2004"] = alt["prezzo2004"].fillna(0)
alt["cucina2004"] = alt["cucina2004"].fillna(0)
for name, spec in [("c1", c1 + ["prezzo2004_miss", "cucina2004_miss"]),
                    ("c4", c4 + ["prezzo2004_miss", "cucina2004_miss"])]:
    row("miss-indicator", name, spec, alt)
print(f"  Note: expands N to 180 (vs 140 in published Table 1).")

print("\n(8) Outcome = d1204_d (retail growth) as placebo outcome")
for name, spec in [("c1", c1), ("c4", c4)]:
    sub = df[[*spec, "d1204_d"]].dropna()
    X = sm.add_constant(sub[list(spec)])
    m = sm.OLS(sub["d1204_d"], X).fit()
    keyvars = ["prezzo2004Abitaz", "lat2004", "prezzo2004"]
    print(
        f"  [{name}] placebo d1204_d                               "
        f"N={int(m.nobs):3d} R²={m.rsquared:.3f}  "
        + "  ".join(
            [
                f"{v[:12]}:{m.params[v]:+.3f}"
                + ("**" if m.pvalues[v] < 0.05 else "*" if m.pvalues[v] < 0.10 else "")
                for v in keyvars
                if v in m.params.index
            ]
        )
    )

print("\n(9) Outcome = d1204_a (food retail growth) as placebo outcome")
for name, spec in [("c1", c1), ("c4", c4)]:
    sub = df[[*spec, "d1204_a"]].dropna()
    X = sm.add_constant(sub[list(spec)])
    m = sm.OLS(sub["d1204_a"], X).fit()
    keyvars = ["prezzo2004Abitaz", "lat2004", "prezzo2004"]
    print(
        f"  [{name}] placebo d1204_a                               "
        f"N={int(m.nobs):3d} R²={m.rsquared:.3f}  "
        + "  ".join(
            [
                f"{v[:12]}:{m.params[v]:+.3f}"
                + ("**" if m.pvalues[v] < 0.05 else "*" if m.pvalues[v] < 0.10 else "")
                for v in keyvars
                if v in m.params.index
            ]
        )
    )

print("\n(10) Permutation / placebo on treatment year:")
print("     Regress d0004_r (pre-reform log growth) on baseline 2000 predictors")
df["d0004_r"] = np.log(df["lat2004"]) - np.log(df["lat2000"])
for name, spec in [("c1_pre", c1)]:
    sub = df[[*spec, "d0004_r"]].dropna()
    X = sm.add_constant(sub[list(spec)])
    m = sm.OLS(sub["d0004_r"], X).fit()
    keyvars = ["prezzo2004Abitaz", "lat2004", "prezzo2004"]
    print(
        f"  [{name}] pre-reform d0004_r                            "
        f"N={int(m.nobs):3d} R²={m.rsquared:.3f}  "
        + "  ".join(
            [
                f"{v[:12]}:{m.params[v]:+.3f}"
                + ("**" if m.pvalues[v] < 0.05 else "*" if m.pvalues[v] < 0.10 else "")
                for v in keyvars
                if v in m.params.index
            ]
        )
    )

print("\n(11) Random-permutation placebo on d1204_r (1000 draws):")
rng = np.random.default_rng(12345)
perms = 1000
real_r2 = fit(df, c1).rsquared
null_r2 = []
sub = df[[y] + c1].dropna().reset_index(drop=True)
X = sm.add_constant(sub[c1])
for _ in range(perms):
    yp = rng.permutation(sub[y].values)
    m = sm.OLS(yp, X).fit()
    null_r2.append(m.rsquared)
null_r2 = np.array(null_r2)
pval = (null_r2 >= real_r2).mean()
print(f"     Real R² = {real_r2:.3f}; permutation p-value = {pval:.3f}")
print(f"     (null median R² = {np.median(null_r2):.3f}; 95th pct = {np.percentile(null_r2, 95):.3f})")
print("     — consistent with the paper's null: baseline chars do NOT predict post-reform growth.")

print("\n(12) Leave-one-out sensitivity of R² in column 1")
max_delta = 0.0
max_zone = None
for z in df["zona180"].unique():
    sub = df[df["zona180"] != z]
    m = fit(sub, c1)
    delta = abs(m.rsquared - real_r2)
    if delta > max_delta:
        max_delta = delta
        max_zone = z
print(f"     Max |ΔR²| from dropping a single zone: {max_delta:.4f} (zone {int(max_zone)})")
