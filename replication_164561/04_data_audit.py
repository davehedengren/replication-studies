"""Data audit for Study 1 of Bursztyn et al. 2022 'Scapegoating During Crises'.

Checks: coverage, distributions, balance across treatment, missingness,
duplicates, response-length artefacts, and a manual sanity check of the
text-feature stems against the raw open_ended responses."""

import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.api as sm

from utils import load_clean, OUTDIR, RACISM_STEMS, LABOR_STEMS

# Start from the raw SPSS file to see what gets dropped.
import pyreadstat
raw, _ = pyreadstat.read_sav("164561-V1/replication_PP/raw/study1.sav")
raw.columns = [c.lower() for c in raw.columns]
print(f"Raw N: {len(raw)}")

df = load_clean()
print(f"Clean N: {len(df)}")
print(f"Drop funnel:")
print(f"  raw = {len(raw)}")
n = len(raw)
n_after_pid = raw["prolific_pid"].notna().sum()
print(f"  after dropping missing prolific_pid: {n_after_pid}")
raw2 = raw[raw["consent"] == 1]
print(f"  after consent==1: {len(raw2)}")
raw2 = raw2[raw2["attention"].astype(str).str.strip() == "1"]
print(f"  after attention==1: {len(raw2)}")
raw2 = raw2[raw2["treatment"].astype(str).str.strip().isin(["excuse", "noexcuse"])]
print(f"  after non-missing treatment: {len(raw2)}")
raw2 = raw2[~raw2["vote"].isin([3, 4])]
print(f"  after dropping vote in {{3,4}}: {len(raw2)}")

print("\n--- 1. COVERAGE ---")
print(f"N = {len(df)}")
print(f"Unique prolific_pid = {df['prolific_pid'].nunique()} (duplicates: {len(df) - df['prolific_pid'].nunique()})")
print(f"Treatment cells: {df['Tr'].value_counts().to_dict()}")
print(f"Party: Biden={(df['vote_trump']==0).sum()}, Trump={(df['vote_trump']==1).sum()}")

print("\n--- 2. DEMOGRAPHIC DISTRIBUTIONS ---")
for c in ["age", "male", "white", "black", "hispanic", "college", "fulltime",
          "income_midpoint", "ln_income"]:
    s = df[c]
    print(f"  {c:<18s} mean={s.mean():.3f}  sd={s.std():.3f}  min={s.min():.2f}  max={s.max():.2f}  "
          f"missing={s.isna().sum()}")

print("\n--- 3. BALANCE OF COVARIATES BY TREATMENT (t-tests) ---")
rows = []
for c in ["male", "age", "white", "college", "ln_income", "hispanic",
          "fulltime", "northeast", "midwest", "south", "west", "vote_trump",
          "democrat", "republican"]:
    a = df.loc[df["Tr"] == 0, c].dropna()
    b = df.loc[df["Tr"] == 1, c].dropna()
    t, p = stats.ttest_ind(a, b, equal_var=False)
    rows.append({"var": c, "before_mean": a.mean(), "after_mean": b.mean(),
                 "diff": b.mean() - a.mean(), "p": p})
bal = pd.DataFrame(rows)
print(bal.round(4).to_string(index=False))
n_sig = (bal["p"] < 0.05).sum()
print(f"\n# of variables with p<0.05: {n_sig} of {len(bal)} "
      f"(expected ~{0.05*len(bal):.1f} by chance)")

print("\n--- 4. MISSINGNESS ---")
for c in ["age", "educ", "race", "inc", "vote", "pol", "region", "employment",
          "open_ended", "story_cleaned"]:
    if c in df.columns:
        print(f"  {c}: missing = {df[c].isna().sum()}")

print("\n--- 5. JOINT BALANCE (F-test of treatment on all controls) ---")
from utils import CONTROLS
X = sm.add_constant(df[CONTROLS].dropna())
y = df.loc[X.index, "Tr"]
res = sm.OLS(y, X).fit(cov_type="HC1")
fstat = res.fvalue
fp = res.f_pvalue
print(f"F({len(CONTROLS)}, {int(res.df_resid)}) = {fstat:.3f}  p = {fp:.4f}")

print("\n--- 6. TEXT RESPONSE QUALITY ---")
df["nwords"] = df["open_ended"].fillna("").str.split().apply(len)
print(f"Word count: mean={df['nwords'].mean():.1f}  median={df['nwords'].median()}  "
      f"min={df['nwords'].min()}  max={df['nwords'].max()}")
empty = (df["nwords"] == 0).sum()
short = (df["nwords"] < 5).sum()
print(f"  Empty responses: {empty}; < 5 words: {short}")

# Word count difference by treatment
tt = stats.ttest_ind(df.loc[df["Tr"] == 0, "nwords"], df.loc[df["Tr"] == 1, "nwords"],
                     equal_var=False)
print(f"  Before word count: {df.loc[df['Tr']==0, 'nwords'].mean():.2f}")
print(f"  After word count:  {df.loc[df['Tr']==1, 'nwords'].mean():.2f}")
print(f"  t-test p: {tt.pvalue:.4f}")

print("\n--- 7. STEM-MATCHING SANITY CHECK ---")
# For each racism stem, count responses
for s in RACISM_STEMS:
    n = df["story_cleaned"].str.contains(s, regex=False).sum()
    print(f"  '{s}' -> {n} responses")
for s in LABOR_STEMS:
    n = df["story_cleaned"].str.contains(s, regex=False).sum()
    print(f"  '{s}' -> {n} responses")

# Example hits
print("\n  First 3 'bigot' matches:")
hits = df[df["story_cleaned"].str.contains("bigot", regex=False)].head(3)
for _, r in hits.iterrows():
    print(f"    - {r['open_ended'][:120]}")

print("\n--- 8. REGRESSION-ADJUSTED TREATMENT EFFECTS (main results with controls) ---")
for out in ["contain_racism_terms", "contain_labor_terms"]:
    sub = df.dropna(subset=CONTROLS + [out])
    X = sm.add_constant(sub[["Tr"] + CONTROLS])
    y = sub[out]
    res = sm.OLS(y, X).fit(cov_type="HC1")
    print(f"\n  Outcome: {out}")
    print(f"    Tr coefficient: {res.params['Tr']:.4f}  "
          f"se={res.bse['Tr']:.4f}  p={res.pvalues['Tr']:.4g}  "
          f"N={int(res.nobs)}")

print("\n--- 9. CONSISTENCY: vote_trump <-> pol ---")
xt = pd.crosstab(df["pol"], df["vote_trump"], dropna=False)
print(xt)
print("  (pol: 1=Rep, 2=Dem, 3=Ind; vote_trump: 1=Trump, 0=Biden)")

print("\n--- Data audit complete. ---")
