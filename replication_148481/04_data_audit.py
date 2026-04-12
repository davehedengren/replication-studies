"""Data audit for DellaVigna et al. (2022) Experiment 1.

Checks coverage, distributions, balance, and panel completeness."""

import numpy as np
import pandas as pd
from utils import clean_long, clean_wide

wide = clean_wide()
long, _ = clean_long()


def hdr(t):
    print(f"\n--- {t} " + "-" * (70 - len(t)))


print("=" * 72)
print("PHASE 3 — DATA AUDIT, Experiment 1 (N=446)")
print("=" * 72)

hdr("Coverage")
print(f"Workers (unique id)       : {long['id'].nunique()}")
print(f"Rounds per worker (min/max): {long.groupby('id').size().min()} / {long.groupby('id').size().max()}")
print(f"Sessions (group ids)      : {long['group'].nunique()}")
print(f"Session size (mean/min/max): "
      f"{wide.groupby('group').size().mean():.1f} / "
      f"{wide.groupby('group').size().min()} / "
      f"{wide.groupby('group').size().max()}")
print(f"Date range                : {wide['date2'].min().date()} → {wide['date2'].max().date()}")
print(f"Worker-round observations : {len(long)}")

hdr("Panel balance")
cells = pd.crosstab(long["order"], long["time"])
print(cells)
# Every order x time cell should be non-empty
missing = (cells == 0).sum().sum()
print(f"Order x round cells with 0 obs: {missing}  (expect 0)")

hdr("Output distribution (envelopes / 20 min)")
print(long["output"].describe())
q = long["output"].quantile([0.01, 0.05, 0.5, 0.95, 0.99, 0.999])
print("Quantiles:\n", q)
print(f"Max output : {long['output'].max():.0f}  (plausible? 20 min cap)")
print(f"Min output : {long['output'].min():.0f}")
n_zero = (long["output"] == 0).sum()
print(f"Zero-output observations : {n_zero}")

hdr("Output by round (sanity — learning curve + treatments)")
by_time = long.groupby(["order", "time"])["output"].agg(["mean", "count"]).round(2)
print(by_time)

hdr("Gift treatment balance")
treat = wide["gift_type"].value_counts().sort_index()
print(treat)
print(f"Per-treatment share (%): {(100 * treat / treat.sum()).round(1).to_dict()}")
# Pre-treatment output (rounds 5-8) should be balanced across treatments:
pre58 = long[(long["time"] > 4) & (long["time"] <= 8)]
bal = pre58.groupby("gift_type")["output"].agg(["mean", "std", "count"]).round(3)
print("\nPre-gift (rounds 5-8) output by assigned gift treatment:")
print(bal)
# Anova-style F test: is mean output equal across gift types in rounds 5-8?
import statsmodels.api as sm
X = pd.get_dummies(pre58["gift_type"], drop_first=True).astype(float)
X = sm.add_constant(X)
res = sm.OLS(pre58["output"].values, X.values).fit(
    cov_type="cluster", cov_kwds={"groups": pre58["group"].values}
)
f_test = res.f_test(np.eye(X.shape[1])[1:])
print(f"F-test (mean equal across gifts, rounds 5-8): F={float(f_test.statistic):.2f}  p={float(f_test.pvalue):.3f}")

hdr("Demographic balance (wide)")
demo = wide[["female", "age_num", "dempl", "donate", "dvolunteer"]].describe().round(3)
print(demo)
for gift in ["neu", "pos", "neg", "knd"]:
    m = wide[wide["gift_type"] == gift][["female", "age_num", "dempl"]].mean().round(3)
    print(f"  {gift}: {m.to_dict()}")

hdr("Missing data check")
# The cleaning drops any worker with any missing round; verify none remain
for i in range(1, 11):
    miss = wide[f"output{i}"].isna().sum()
    if miss:
        print(f"  Round {i}: {miss} missing")
print(f"  Workers with any missing round: {wide[[f'output{i}' for i in range(1, 11)]].isna().any(axis=1).sum()}")

hdr("Duplicates")
print(f"Duplicate id in wide: {wide['id'].duplicated().sum()}")
print(f"Duplicate (id, time) in long: {long[['id', 'time']].duplicated().sum()}")

hdr("Outlier screen (IQR x 3)")
q1, q3 = long["output"].quantile([0.25, 0.75])
iqr = q3 - q1
lo, hi = q1 - 3 * iqr, q3 + 3 * iqr
out_hi = (long["output"] > hi).sum()
out_lo = (long["output"] < lo).sum()
print(f"IQR = {iqr:.1f}, screen=[{lo:.1f}, {hi:.1f}]")
print(f"High outliers: {out_hi}, Low outliers: {out_lo}")
print("Top 10 outputs:", sorted(long["output"].nlargest(10).tolist()))

print("\nAudit complete.")
