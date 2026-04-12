"""Replicate Table 1 columns 4-6: within-occ share of RESIDUAL log wage variance.

Residuals are from a year-specific Mincer regression of log wage on (edu × age ×
sex × race) FEs, weighted by wgt_hrs. Paper uses 240 groups: 5 edu × 8 age × 2 sex
× 3 race. We then run the same variance decomposition on the residuals.
"""
import numpy as np
import pandas as pd
import statsmodels.api as sm
from utils import (
    load_morg_pooled, apply_sample_filter, build_edu, merge_crosswalk,
    map_20occ, map_30occ, map_40occ, OUT,
)
from importlib import import_module
compute_table1 = import_module("01_table1").compute_table1


def residualize(df):
    # 8 age bins [21..25],[26..30],...,[56..60]
    df = df.copy()
    df["age_bin"] = np.minimum(((df["age"] - 21) // 5).astype(int), 7)
    df["race3"] = np.where(df["black"] == 1, 1, np.where(df["other"] == 1, 2, 0))
    df["edu_int"] = df["edu"].astype(int)
    out = []
    for yr, g in df.groupby("year"):
        g = g.copy()
        cats = g[["edu_int", "age_bin", "sex", "race3"]].astype(str).agg("_".join, axis=1)
        X = pd.get_dummies(cats, drop_first=False).astype(float).values
        y = np.log(g["wage"].values)
        w = g["wgt_hrs"].values
        model = sm.WLS(y, X, weights=w).fit()
        g["logw_resid"] = y - model.predict(X)
        out.append(g)
    return pd.concat(out, ignore_index=True)


def main():
    df = load_morg_pooled()
    df = apply_sample_filter(df)
    df = build_edu(df)
    df = merge_crosswalk(df)
    df = df[df["occ1990dd"].notna()]
    df["perwt"] = np.floor(df["wgt_hrs"] / 100).astype(int)
    df = df[df["perwt"] > 0]

    df = residualize(df)
    df["logw"] = df["logw_resid"]
    base = df[["year", "occ1990dd", "edu", "young", "male", "perwt", "logw"]].copy()

    published = {  # residual log wages columns of Table 1
        "20 Occupations": (0.924, 0.914, 0.866),
        "30 Occupations": (0.913, 0.899, 0.829),
        "40 Occupations": (0.910, 0.898, 0.839),
        "383 Occupations": (0.865, 0.873, 0.912),
    }
    mappings = {
        "20 Occupations": map_20occ,
        "30 Occupations": map_30occ,
        "40 Occupations": map_40occ,
    }
    rows = []
    for name, fn in mappings.items():
        d = base.copy(); d["occ"] = d["occ1990dd"].map(fn)
        r = compute_table1(d)
        p1980, p2000, pds = published[name]
        rows.append({
            "grouping": name,
            "s1980_pub": p1980, "s1980_rep": round(r["s1980"], 3),
            "s2000_pub": p2000, "s2000_rep": round(r["s2000"], 3),
            "ds_pub": pds,      "ds_rep":    round(r["ds"], 3),
        })
    d = base.copy(); d["occ"] = d["occ1990dd"]
    r = compute_table1(d)
    p1980, p2000, pds = published["383 Occupations"]
    rows.append({
        "grouping": "383 Occupations",
        "s1980_pub": p1980, "s1980_rep": round(r["s1980"], 3),
        "s2000_pub": p2000, "s2000_rep": round(r["s2000"], 3),
        "ds_pub": pds,      "ds_rep":    round(r["ds"], 3),
    })
    out = pd.DataFrame(rows)
    print(out.to_string(index=False))
    out.to_csv(OUT / "table1_residual_wages.csv", index=False)


if __name__ == "__main__":
    main()
