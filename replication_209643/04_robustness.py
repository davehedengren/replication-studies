"""Robustness checks for Liu (2025) Table 1 / Fact 1: the within-occupation share
of log wage variance from 1980 to 2000.

Checks (tailored to the stylized-fact claim):
  1. Alternative wage filter: trim top 1% / bottom 1% rather than $5/$1500.
  2. Winsorize log wage at 1st/99th percentile.
  3. Drop allocated (imputed) wage observations (hr_alloc==1 / aernhr_flag).
  4. Include part-timers (drop ft==1 filter).
  5. Alternative age range: 25-55 (prime-age).
  6. Use unweighted data.
  7. Use person-weight (wgt) instead of hour-weight (wgt_hrs).
  8. Drop 1980-only or 2000-only occupations (balanced panel of occs).
  9. Drop singleton (year,occ) cells (should not matter since so few).
 10. Alternative education: raw 5-level edu (don't merge edgtc into edclg).
 11. Men only / Women only subsets.

Each check reports the within-share at the 40-occupation level (the grouping
that matches the paper exactly in the baseline).
"""
import numpy as np
import pandas as pd
from utils import (
    load_morg_pooled, apply_sample_filter, build_edu, merge_crosswalk,
    map_40occ, OUT,
)
from importlib import import_module
compute_table1 = import_module("01_table1").compute_table1


def prep(df, occ_map=map_40occ):
    df = df[df["occ1990dd"].notna()].copy()
    df["perwt"] = np.floor(df["wgt_hrs"] / 100).astype(int)
    df = df[df["perwt"] > 0]
    df["logw"] = np.log(df["wage"])
    df["occ"] = df["occ1990dd"].map(occ_map)
    return df[["year", "occ", "edu", "young", "male", "perwt", "logw"]]


def baseline():
    df = load_morg_pooled()
    df = apply_sample_filter(df)
    df = build_edu(df)
    df = merge_crosswalk(df)
    return df


def run(df, label):
    d = prep(df)
    r = compute_table1(d)
    return {"check": label, "s1980": round(r["s1980"], 3),
            "s2000": round(r["s2000"], 3), "ds": round(r["ds"], 3),
            "N": int(len(d))}


def main():
    rows = []

    # Baseline
    df = baseline()
    rows.append(run(df, "baseline (40 occ)"))

    # 1. Trim at 1/99 percentile instead of $5/$1500
    df1 = load_morg_pooled()
    df1 = df1[(df1["age"] >= 21) & (df1["age"] <= 60)]
    df1 = df1[df1["mlr"].isin([1, 2])]
    df1 = df1[df1["ft"] == 1]
    df1 = df1[df1["hr_wage"].notna()].rename(columns={"hr_wage": "wage"})
    df1["young"] = (df1["age"] <= 40).astype(int)
    df1["male"] = (df1["sex"] == 1).astype(int)
    for yr in [1980, 2000]:
        mask = df1["year"] == yr
        lo = df1.loc[mask, "wage"].quantile(0.01)
        hi = df1.loc[mask, "wage"].quantile(0.99)
        df1 = df1[~mask | ((df1["wage"] >= lo) & (df1["wage"] <= hi))]
    df1 = build_edu(df1)
    df1 = merge_crosswalk(df1)
    rows.append(run(df1, "1. trim 1/99 pct"))

    # 2. Winsorize log wage at 1/99
    df2 = baseline()
    df2["wage"] = df2["wage"].astype(float)
    for yr in [1980, 2000]:
        mask = df2["year"] == yr
        lo = np.log(df2.loc[mask, "wage"]).quantile(0.01)
        hi = np.log(df2.loc[mask, "wage"]).quantile(0.99)
        df2.loc[mask, "wage"] = np.exp(np.log(df2.loc[mask, "wage"]).clip(lo, hi))
    rows.append(run(df2, "2. winsorize logw 1/99"))

    # 3. Drop allocated wages
    df3 = baseline()
    if "hr_alloc" in df3.columns:
        df3 = df3[df3["hr_alloc"].fillna(0) == 0]
    rows.append(run(df3, "3. drop allocated wages"))

    # 4. Include part-timers
    df4 = load_morg_pooled()
    df4 = df4[(df4["age"] >= 21) & (df4["age"] <= 60)]
    df4 = df4[df4["mlr"].isin([1, 2])]
    df4 = df4[df4["hr_wage"].notna()]
    df4 = df4[(df4["hr_wage"] > 5) & (df4["hr_wage"] < 1500)]
    df4 = df4.rename(columns={"hr_wage": "wage"})
    df4["young"] = (df4["age"] <= 40).astype(int)
    df4["male"] = (df4["sex"] == 1).astype(int)
    df4 = build_edu(df4)
    df4 = merge_crosswalk(df4)
    rows.append(run(df4, "4. include part-time"))

    # 5. Prime-age 25-55
    df5 = baseline()
    df5 = df5[(df5["age"] >= 25) & (df5["age"] <= 55)]
    rows.append(run(df5, "5. age 25-55"))

    # 6. Unweighted (set perwt=1)
    df6 = baseline()
    d6 = prep(df6)
    d6["perwt"] = 1
    r6 = compute_table1(d6)
    rows.append({"check": "6. unweighted",
                 "s1980": round(r6["s1980"], 3),
                 "s2000": round(r6["s2000"], 3),
                 "ds": round(r6["ds"], 3), "N": int(len(d6))})

    # 7. Use wgt (person weight) instead of wgt_hrs
    df7 = baseline()
    d7 = df7[df7["occ1990dd"].notna()].copy()
    d7["perwt"] = np.floor(d7["wgt"] / 100).astype(int)
    d7 = d7[d7["perwt"] > 0]
    d7["logw"] = np.log(d7["wage"])
    d7["occ"] = d7["occ1990dd"].map(map_40occ)
    r7 = compute_table1(d7[["year", "occ", "edu", "young", "male", "perwt", "logw"]])
    rows.append({"check": "7. person weight (wgt)",
                 "s1980": round(r7["s1980"], 3),
                 "s2000": round(r7["s2000"], 3),
                 "ds": round(r7["ds"], 3), "N": int(len(d7))})

    # 8. Balanced occupation panel
    df8 = baseline()
    df8 = df8[df8["occ1990dd"].notna()]
    occs80 = set(df8[df8["year"] == 1980]["occ1990dd"].unique())
    occs00 = set(df8[df8["year"] == 2000]["occ1990dd"].unique())
    common = occs80 & occs00
    df8 = df8[df8["occ1990dd"].isin(common)]
    rows.append(run(df8, "8. balanced occ panel"))

    # 9. Drop tiny (year,occ) cells (N<5) within the 40-occ aggregation
    df9 = baseline()
    d9 = prep(df9)
    cell = d9.groupby(["year", "occ"]).size()
    small = cell[cell < 5].index
    mask = ~d9.set_index(["year", "occ"]).index.isin(small)
    d9 = d9[mask]
    r9 = compute_table1(d9)
    rows.append({"check": "9. drop N<5 cells",
                 "s1980": round(r9["s1980"], 3),
                 "s2000": round(r9["s2000"], 3),
                 "ds": round(r9["ds"], 3), "N": int(len(d9))})

    # 10. Keep 5-level edu (don't merge edgtc)
    # Requires re-running build_edu without final collapse.
    df10 = load_morg_pooled()
    df10 = apply_sample_filter(df10)
    df10 = build_edu(df10)  # already collapsed
    # Undo collapse: re-tag GTC separately
    #   Use grdatn>=44 for 2000, edu_pre>17 for 1980
    df10 = merge_crosswalk(df10)
    # Just report: identical to baseline since edu only enters through `gg`.
    # Test an alternative — split edu=4 into edclg/edgtc based on grdatn.
    df10_split = df10.copy()
    post = df10_split["year"].isin([2000, 2002])
    is_gtc = np.where(post, df10_split["grdatn"] >= 44,
                       df10_split["_grdhi"] > 17)
    df10_split.loc[(df10_split["edu"] == 4) & is_gtc, "edu"] = 5
    rows.append(run(df10_split, "10. 5-level edu (split GTC)"))

    # 11. Men only
    df11 = baseline()
    df11 = df11[df11["sex"] == 1]
    rows.append(run(df11, "11a. men only"))
    df11w = baseline()
    df11w = df11w[df11w["sex"] == 2]
    rows.append(run(df11w, "11b. women only"))

    out = pd.DataFrame(rows)
    print(out.to_string(index=False))
    out.to_csv(OUT / "robustness.csv", index=False)


if __name__ == "__main__":
    main()
