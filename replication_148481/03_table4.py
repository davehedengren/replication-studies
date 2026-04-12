"""Replicate Table 4 of DellaVigna, List, Malmendier, Rao (2022):
"Experiment 1: Gift Exchange Effects — Reduced-form Regressions".

In the Stata code this is 3_reducedform.do → TableGiftExch_abs.csv (Panel A)
and TableGiftExch_log.csv (Panel B).  Here we reproduce the key columns:

    Col 1 : OLS, no controls                          (rounds 9-10 pooled)
    Col 2 : + preoutputmean (rounds 1-8 average)       (rounds 9-10 pooled)
    Col 3 : + output58mean (rounds 5-8 average)        (rounds 9-10 pooled)  **
    Col 4 : round 9 only, with output58mean
    Col 5 : round 10 only, with output58mean
    Col 6 : match OFF only, with output58mean
    Col 7 : match ON only, with output58mean

Regressor set : d_pos_gift, d_neg_gift, d_knd_gift, (control), constant.
SE clustered by session group.
"""

import numpy as np
import pandas as pd
import statsmodels.api as sm
from utils import clean_long, OUT

long, _ = clean_long()

# Per-worker pre-period averages for controls
long["preoutput"] = np.where(long["time"] <= 8, long["output"], np.nan)
long["output58"] = np.where((long["time"] > 4) & (long["time"] <= 8), long["output"], np.nan)
long["geoutput"] = np.where(long["time"] > 8, long["output"], np.nan)
for col in ["preoutput", "output58", "geoutput"]:
    long[f"{col}mean"] = long.groupby("id")[col].transform("mean")


def reg(df, controls, panel_b=False):
    rhs = ["const", "d_pos_gift", "d_neg_gift", "d_knd_gift"] + list(controls)
    d = df.copy()
    d["const"] = 1.0
    y = np.log(d["output"].values) if panel_b else d["output"].values
    X = d[rhs].values
    res = sm.OLS(y, X).fit(cov_type="cluster", cov_kwds={"groups": d["group"].values})
    return res, rhs


def collect(res, rhs, label, panel_b):
    out = {"spec": label, "panel": "B (log)" if panel_b else "A (levels)",
           "N": int(res.nobs), "R2": float(res.rsquared)}
    for name in rhs:
        i = rhs.index(name)
        out[f"{name}_b"] = res.params[i]
        out[f"{name}_se"] = res.bse[i]
    return out


rounds910 = long[(long["time"] == 9) | (long["time"] == 10)]
rows = []
for panel_b in [False, True]:
    # Col 1: no controls, rounds 9-10
    res, rhs = reg(rounds910, [], panel_b)
    rows.append(collect(res, rhs, "col1_no_ctrls_9_10", panel_b))
    # Col 2: pre-output mean (rounds 1-8)
    res, rhs = reg(rounds910, ["preoutputmean"], panel_b)
    rows.append(collect(res, rhs, "col2_preoutputmean_9_10", panel_b))
    # Col 3: post-lunch pre-period mean (rounds 5-8)
    res, rhs = reg(rounds910, ["output58mean"], panel_b)
    rows.append(collect(res, rhs, "col3_output58mean_9_10", panel_b))
    # Col 4 & 5: split by round
    for t in [9, 10]:
        sub = long[long["time"] == t]
        res, rhs = reg(sub, ["output58mean"], panel_b)
        rows.append(collect(res, rhs, f"col{4 if t == 9 else 5}_time{t}", panel_b))
    # Col 6 & 7: split by match on/off
    for m in [0, 1]:
        sub = long[long["matchgift_on"] == m]
        res, rhs = reg(sub, ["output58mean"], panel_b)
        rows.append(collect(res, rhs, f"col{6 if m == 0 else 7}_match{m}", panel_b))

df = pd.DataFrame(rows)


def pretty(v, s):
    return f"{v:+7.3f} ({s:.3f})"


print("=" * 96)
print("PANEL A — Levels (output)")
print("=" * 96)
cols = ["spec", "d_pos_gift_b", "d_pos_gift_se", "d_neg_gift_b", "d_neg_gift_se",
        "d_knd_gift_b", "d_knd_gift_se", "N", "R2"]
pa = df[df["panel"] == "A (levels)"]
print(f"{'Spec':28s}  {'Positive':>18s}  {'Negative':>18s}  {'In-kind':>18s}     {'N':>4s}   {'R2':>5s}")
for _, r in pa.iterrows():
    p = pretty(r["d_pos_gift_b"], r["d_pos_gift_se"])
    n = pretty(r["d_neg_gift_b"], r["d_neg_gift_se"])
    k = pretty(r["d_knd_gift_b"], r["d_knd_gift_se"])
    print(f"{r['spec']:28s}  {p:>18s}  {n:>18s}  {k:>18s}     {int(r['N']):4d}   {r['R2']:.3f}")

print("\n" + "=" * 96)
print("PANEL B — Log output")
print("=" * 96)
pb = df[df["panel"] == "B (log)"]
print(f"{'Spec':28s}  {'Positive':>18s}  {'Negative':>18s}  {'In-kind':>18s}     {'N':>4s}   {'R2':>5s}")
for _, r in pb.iterrows():
    p = pretty(r["d_pos_gift_b"], r["d_pos_gift_se"])
    n = pretty(r["d_neg_gift_b"], r["d_neg_gift_se"])
    k = pretty(r["d_knd_gift_b"], r["d_knd_gift_se"])
    print(f"{r['spec']:28s}  {p:>18s}  {n:>18s}  {k:>18s}     {int(r['N']):4d}   {r['R2']:.3f}")

df.to_csv(OUT / "table4_gift_effects.csv", index=False)
print(f"\nWrote {OUT / 'table4_gift_effects.csv'}")
