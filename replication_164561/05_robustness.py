"""Robustness checks for Bursztyn et al. (2022) 'Scapegoating During Crises'.

Tailored to the paper's experimental design and text outcomes. The key
claim is a large and significant 'Before > After' difference in xenophobia
references and a smaller 'After > Before' difference in labor references."""

import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.api as sm

from utils import OUTDIR, CONTROLS, RACISM_STEMS, LABOR_STEMS

df = pd.read_parquet(OUTDIR / "study1_cleaned.parquet")
df["nwords"] = df["open_ended"].fillna("").str.split().apply(len)
rng = np.random.default_rng(20250101)


def ols(sub, y, extra=None):
    X = sub[["Tr"] + (extra or [])].copy()
    X = sm.add_constant(X)
    return sm.OLS(sub[y], X).fit(cov_type="HC1")


def show(label, res):
    b = res.params["Tr"]
    se = res.bse["Tr"]
    p = res.pvalues["Tr"]
    n = int(res.nobs)
    print(f"  {label:<60s} β={b:+.4f}  se={se:.4f}  p={p:.4g}  N={n}")


print("Baseline treatment effects (OLS, HC1):")
for y in ["contain_racism_terms", "contain_labor_terms"]:
    show(f"{y}", ols(df, y))

print("\n--- 1. Regression-adjusted (full controls) ---")
for y in ["contain_racism_terms", "contain_labor_terms"]:
    sub = df.dropna(subset=CONTROLS + [y])
    show(f"{y} + controls", ols(sub, y, extra=CONTROLS))

print("\n--- 2. Drop empty responses (nwords == 0) ---")
sub = df[df["nwords"] > 0]
for y in ["contain_racism_terms", "contain_labor_terms"]:
    show(f"{y} (n>{0})", ols(sub, y))

print("\n--- 3. Drop short responses (nwords < 5) ---")
sub = df[df["nwords"] >= 5]
for y in ["contain_racism_terms", "contain_labor_terms"]:
    show(f"{y} (>=5 words)", ols(sub, y))

print("\n--- 4. Drop long-tail outliers (top 1% of word count) ---")
cap = df["nwords"].quantile(0.99)
sub = df[df["nwords"] <= cap]
for y in ["contain_racism_terms", "contain_labor_terms"]:
    show(f"{y} (drop top 1% length)", ols(sub, y))

print("\n--- 5. Normalize for response length (per-word rate) ---")
df["racism_per100"] = 100 * df["contain_racism_terms"] / np.maximum(df["nwords"], 1)
df["labor_per100"] = 100 * df["contain_labor_terms"] / np.maximum(df["nwords"], 1)
sub = df[df["nwords"] > 0]
for y in ["racism_per100", "labor_per100"]:
    show(f"{y}", ols(sub, y))

print("\n--- 6. Logistic regression (marginal-like specification) ---")
import statsmodels.formula.api as smf
for y in ["contain_racism_terms", "contain_labor_terms"]:
    res = smf.logit(f"{y} ~ Tr", data=df).fit(disp=False)
    me = res.get_margeff(at="overall").summary_frame()
    print(f"  {y:<30s} AME(Tr) = {me.loc['Tr', 'dy/dx']:+.4f}  p={me.loc['Tr', 'Pr(>|z|)']:.4g}")

print("\n--- 7. Permutation test (5000 reshuffles of Tr within party) ---")
for y in ["contain_racism_terms", "contain_labor_terms"]:
    obs = df.groupby("Tr")[y].mean().loc[1] - df.groupby("Tr")[y].mean().loc[0]
    null_diffs = []
    vals = df[y].values
    trs = df["Tr"].values.copy()
    for _ in range(5000):
        rng.shuffle(trs)
        d = vals[trs == 1].mean() - vals[trs == 0].mean()
        null_diffs.append(d)
    null_diffs = np.array(null_diffs)
    pp = np.mean(np.abs(null_diffs) >= np.abs(obs))
    print(f"  {y:<30s} observed Δ = {obs:+.4f}   perm p = {pp:.4g}")

print("\n--- 8. Leave-one-stem-out sensitivity for racism terms ---")
for drop in RACISM_STEMS:
    stems = [s for s in RACISM_STEMS if s != drop]
    feat = df["story_cleaned"].apply(lambda t: int(any(s in t for s in stems)))
    tmp = df.assign(outcome=feat)
    res = ols(tmp, "outcome")
    show(f"drop '{drop}'", res)

print("\n--- 9. Leave-one-stem-out sensitivity for labor terms ---")
for drop in LABOR_STEMS:
    stems = [s for s in LABOR_STEMS if s != drop]
    feat = df["story_cleaned"].apply(lambda t: int(any(s in t for s in stems)))
    tmp = df.assign(outcome=feat)
    res = ols(tmp, "outcome")
    show(f"drop '{drop}'", res)

print("\n--- 10. Subgroup heterogeneity (by demographic) ---")
for g_name, g_mask in [
    ("Male",   df["male"] == 1),
    ("Female", df["male"] == 0),
    ("White",  df["white"] == 1),
    ("Non-white", df["white"] == 0),
    ("College",   df["college"] == 1),
    ("No college", df["college"] == 0),
    ("Age < median", df["age"] < df["age"].median()),
    ("Age >= median", df["age"] >= df["age"].median()),
]:
    sub = df[g_mask]
    show(f"racism | {g_name}", ols(sub, "contain_racism_terms"))

print("\n--- 11. Placebo stems (words that should not be affected by treatment) ---")
placebo_stems = ["family", "wife", "children", "church", "government", "america"]
for s in placebo_stems:
    feat = df["story_cleaned"].str.contains(s, regex=False).astype(int)
    tmp = df.assign(outcome=feat)
    res = ols(tmp, "outcome")
    show(f"contains '{s}'", res)

print("\n--- 12. Independents-only sample ---")
sub = df[df["pol"] == 3]
for y in ["contain_racism_terms", "contain_labor_terms"]:
    show(f"{y} | Independents", ols(sub, y))

print("\n--- Robustness complete ---")
