"""Data audit for the experiment dataset.

Checks:
1. Coverage — N, treatment arms, variable completeness.
2. Balance on key covariates across tr_acs vs tr_cps.
3. Distributions — summary stats, plausibility bounds, tails.
4. Forecast choices are mutually exclusive and exhaust the sample.
5. Missingness patterns for dependent variables.
6. Signal construction: Del_unempincr ≈ unempincr_shown − unempincr_alt.
7. Sample symmetry: shown ≈ alt marginal distributions.
8. No duplicates on respondent id (if any id field present).
"""
import numpy as np
import pandas as pd
from scipy import stats
from utils import load_experiment, OUT

df = load_experiment()
print(f"Rows: {len(df)}  Columns: {df.shape[1]}")

# 1. Coverage
print("\n[1] Coverage")
print(f"  ACS arm: {(df['tr_acs']==1).sum()}   CPS arm: {(df['tr_cps']==1).sum()}")
missing = df[["unemp_nextrecession", "forecast_recession", "Del_unempincr",
              "high_conf", "z_manip_ind"]].isna().sum()
print("  Missing on key vars:")
print(missing.to_string())

# 2. Balance
print("\n[2] Balance across tr_acs vs tr_cps")
bal_vars = ["female", "age", "college", "northeast", "midwest", "south", "west",
            "high_conf", "prior_unempincr"]
for v in bal_vars:
    a = df.loc[df["tr_acs"] == 1, v].dropna()
    c = df.loc[df["tr_cps"] == 1, v].dropna()
    t, p = stats.ttest_ind(a, c, equal_var=False)
    print(f"  {v:22s} ACS={a.mean():7.3f}  CPS={c.mean():7.3f}  diff p={p:.3f}")

# 3. Distributions
print("\n[3] Distributions of continuous signals and outcomes")
dist_vars = ["unemp_nextrecession", "Del_unempincr", "unempincr_shown",
             "unempincr_alt", "unempbase_shown", "unempbase_alt",
             "prior_unempincr", "z_manip_ind"]
print(df[dist_vars].describe().T[["count", "mean", "std", "min", "50%", "max"]].round(3))

# 4. Forecast choices mutually exclusive
print("\n[4] Forecast choice mutual exclusivity")
choice_sum = df[["forecast_recession", "forecast_govspending",
                 "forecast_iratebonds", "forecast_inflation",
                 "noforecast"]].sum(axis=1)
print(f"  Rows where sum==1: {(choice_sum == 1).sum()}  / {len(df)}")
print(f"  Rows where othermacro = 1 - recession - noforecast: "
      f"{(df['othermacro'] == (1 - df['forecast_recession'] - df['noforecast'])).sum()}")

# 5. Signal construction identity
print("\n[5] Signal construction check")
diff = df["Del_unempincr"] - (df["unempincr_shown"] - df["unempincr_alt"])
print(f"  max |Del_unempincr - (shown - alt)|: {diff.abs().max():.8f}")
print(f"  shown mean: {df['unempincr_shown'].mean():.3f}   "
      f"alt mean: {df['unempincr_alt'].mean():.3f}")

# 6. Sample symmetry
print("\n[6] Noise-difference distribution (should have mean ≈ 0)")
m = df["Del_unempincr"].mean(); s = df["Del_unempincr"].std()
print(f"  Del_unempincr  mean={m:.3f}  sd={s:.3f}  "
      f"t(H0: mean=0) p={stats.ttest_1samp(df['Del_unempincr'].dropna(), 0).pvalue:.3f}")

# 7. Outliers
print("\n[7] Top-5 perceived next-recession risks (%):")
print(df["unemp_nextrecession"].sort_values(ascending=False).head().to_list())
print("    bottom-5:",
      df["unemp_nextrecession"].sort_values().head().to_list())
print(f"  Fraction at 0: {(df['unemp_nextrecession'] == 0).mean():.3f}  "
      f"at 100: {(df['unemp_nextrecession'] == 100).mean():.3f}")

# 8. Duplicates
print("\n[8] Duplicate rows across all columns:",
      df.duplicated().sum())

# Save tidy audit report
with open(OUT / "04_audit.txt", "w") as f:
    f.write(f"N rows = {len(df)}\n")
    f.write(df[dist_vars].describe().T.to_string())
print("\nSaved:", OUT / "04_audit.txt")
