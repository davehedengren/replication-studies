"""Section 6: Preferences for Information Order (Figure 4 / Table 2).

Published (page 24-25):
- ChoiceFree: 55% see-incentive-first, 45% quality-first (costless)
- Professionals: 45% see-incentive-first, 55% quality-first
- Incentive First Costly: 41% see-incentive-first (coef = -0.14, t=-7.84)
- Quality First Costly: 30% see-incentive-first (coef = +0.15, t=5.17)
"""
from utils import prep_choice, hc3, COVARS, OUT
import statsmodels.api as sm
import pandas as pd

main = prep_choice()

# Raw shares by treatment
treat_labels = {0: "ChoiceFree", 1: "Professionals",
                2: "IncentiveFirstCostly", 3: "QualityFirstCostly"}
print("Raw preference shares (prefer see incentive first) by treatment:")
for t, lbl in treat_labels.items():
    sub = main[main["treatment"] == t]
    p = sub["choicebefore"].mean()
    print(f"  {lbl:25s} N={len(sub):4d}  P(incentive first)={p:.3f}  (quality first={1-p:.3f})")

# Table 2 col 1: OLS with HC3, all covariates
m = main.dropna(subset=["choicebefore"] + COVARS).copy()
X = m[COVARS + ["incentiveshigh_incentiveleft"]]
res = hc3(m["choicebefore"], X)
print("\nTable 2 col 1:")
print(res.summary().tables[1])
print(f"N = {int(res.nobs)}")

# Column 2: adds stdalpha, excludes professionals
m2 = main[(main["professionals"] == 0)].dropna(subset=["choicebefore", "stdalpha"] + COVARS).copy()
X2 = m2[COVARS + ["incentiveshigh_incentiveleft", "stdalpha"]]
X2 = X2.drop(columns=["professionalsfree"])  # 0 in this subset
res2 = hc3(m2["choicebefore"], X2)
print("\nTable 2 col 2 (AMT only, adds stdalpha):")
print(res2.summary().tables[1])
print(f"N = {int(res2.nobs)}")

# Column 3: adds selfishness interactions
m3 = main.dropna(subset=["choicebefore", "stdalpha"] + COVARS).copy()
X3 = m3[COVARS + ["incentiveshigh_incentiveleft", "stdalpha",
                  "selfishseeincentivecostly", "selfishseequalitycostly"]]
res3 = hc3(m3["choicebefore"], X3)
print("\nTable 2 col 3 (full):")
print(res3.summary().tables[1])
print(f"N = {int(res3.nobs)}")

# Adjusted margins (Figure 4): predict at mean covariates, per treatment
mean_vals = m[COVARS + ["incentiveshigh_incentiveleft"]].mean()
rows = []
for t, lbl in treat_labels.items():
    row = mean_vals.copy()
    row["professionalsfree"] = 1.0 if t == 1 else 0.0
    row["seeincentivecostly"] = 1.0 if t == 2 else 0.0
    row["seequalitycostly"] = 1.0 if t == 3 else 0.0
    x = pd.Series({"const": 1.0, **row.to_dict()})
    x = x[res.params.index]
    pred = float((x * res.params).sum())
    rows.append((lbl, pred, 1 - pred))

print("\nAdjusted (margins) P(see incentive first) by treatment:")
for lbl, p, q in rows:
    print(f"  {lbl:25s}  incentive={p:.3f}  quality={q:.3f}")

with open(OUT / "table2_results.txt", "w") as f:
    f.write("Table 2: Preferences for Information Order (OLS, HC3)\n")
    for label, r in [("Col1", res), ("Col2", res2), ("Col3", res3)]:
        f.write(f"\n{label} (N={int(r.nobs)}):\n")
        for k in ["professionalsfree", "seeincentivecostly", "seequalitycostly"]:
            if k in r.params.index:
                b = r.params[k]; se = r.bse[k]; t = r.tvalues[k]; p = r.pvalues[k]
                f.write(f"  {k:22s} {b:+.4f} (SE {se:.4f}) t={t:+.2f} p={p:.4f}\n")
    f.write("\nAdjusted margins:\n")
    for lbl, p, q in rows:
        f.write(f"  {lbl:25s}  incentive={p:.3f}  quality={q:.3f}\n")
