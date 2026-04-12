"""Replicate Table 1: Contribution of Within-Occupation Log Wage Variance.

Stages (matching Table 1a.do):
  1. Load MORG 1980+2000 pooled, apply sample filters.
  2. Build demographic groups gg = (edu, young, male). Weight = mean of perwt in (gg, occ).
  3. Compute per-(year, occ) weighted mean and variance of log wage.
  4. Aggregate across occs with time-invariant weights to get between-occ var and
     mean within-occ var. Share = within/total.

Runs three aggregations (20, 30, 40 occupations) and the raw 383 occ1990dd level.
"""
import numpy as np
import pandas as pd
from utils import (
    load_morg_pooled, apply_sample_filter, build_edu, merge_crosswalk,
    map_20occ, map_30occ, map_40occ, OUT,
)


def compute_table1(df_occ):
    """df_occ has columns: year, occ, edu, young, male, perwt, logw."""
    d = df_occ.dropna(subset=["occ"]).copy()
    d["gg"] = (d["edu"].astype(int).astype(str) + "_" +
               d["young"].astype(str) + "_" + d["male"].astype(str))
    # weight = mean of perwt per (gg, occ), pooled across years (replicates
    # `by gg occ: egen weight = mean(perwt)` in Stata).
    d["weight"] = d.groupby(["gg", "occ"])["perwt"].transform("mean")

    # First collapse: per (year, occ) weighted mean/var of logw (aw=perwt),
    # and sum of weight (rawsum).
    rows = []
    for (yr, occ), g in d.groupby(["year", "occ"]):
        w = g["perwt"].values
        x = g["logw"].values
        W = w.sum()
        if W <= 0:
            continue
        mu = np.sum(x * w) / W
        N = len(x)
        # Stata aw variance: SS/W * N/(N-1)
        if N > 1:
            ss = np.sum(w * (x - mu) ** 2)
            v = ss / W * N / (N - 1)
        else:
            v = np.nan  # Stata's sd() returns missing for N=1 cells
        rows.append({
            "year": yr,
            "occ": occ,
            "mean_logw": mu,
            "var_logw": v,
            "weight": g["weight"].sum(),  # rawsum
        })
    occ_df = pd.DataFrame(rows)

    # Second collapse: per year weighted var of mean_logw (aw=weight),
    # weighted mean of var_logw. Stata drops missing var_logw cells from the
    # within mean but keeps them for the between calculation.
    out = {}
    for yr, g in occ_df.groupby("year"):
        w = g["weight"].values
        m = g["mean_logw"].values
        v = g["var_logw"].values
        W = w.sum()
        mu_w = np.sum(m * w) / W
        N = len(m)
        between = np.sum(w * (m - mu_w) ** 2) / W * N / (N - 1)
        # within: weighted mean of per-occ variance, dropping missing cells
        mask = np.isfinite(v)
        ww = w[mask]
        within = np.sum(v[mask] * ww) / ww.sum() if ww.sum() > 0 else np.nan
        total = between + within
        out[int(yr)] = dict(within=within, between=between, total=total,
                             share=within / total)
    s80, s00 = out[1980], out[2000]
    ds = (s00["within"] - s80["within"]) / (s00["total"] - s80["total"])
    return dict(
        s1980=s80["share"], s2000=s00["share"], ds=ds,
        within_1980=s80["within"], within_2000=s00["within"],
        total_1980=s80["total"], total_2000=s00["total"],
    )


def main():
    df = load_morg_pooled()
    df = apply_sample_filter(df)
    df = build_edu(df)
    df = merge_crosswalk(df)
    df = df[df["occ1990dd"].notna()]
    df["perwt"] = np.floor(df["wgt_hrs"] / 100).astype(int)
    df = df[df["perwt"] > 0]
    df["logw"] = np.log(df["wage"])

    base = df[["year", "occ1990dd", "edu", "young", "male", "perwt", "logw"]].copy()

    results = {}
    # 20 occupations
    d = base.copy(); d["occ"] = d["occ1990dd"].map(map_20occ)
    results["20 Occupations"] = compute_table1(d)
    # 30 occupations
    d = base.copy(); d["occ"] = d["occ1990dd"].map(map_30occ)
    results["30 Occupations"] = compute_table1(d)
    # 40 occupations
    d = base.copy(); d["occ"] = d["occ1990dd"].map(map_40occ)
    results["40 Occupations"] = compute_table1(d)
    # 383 occ1990dd
    d = base.copy(); d["occ"] = d["occ1990dd"]
    results["383 Occupations"] = compute_table1(d)

    # Published values from Table 1 (log wages):
    published = {
        "20 Occupations": (0.820, 0.767, 0.561),
        "30 Occupations": (0.800, 0.732, 0.468),
        "40 Occupations": (0.790, 0.730, 0.493),
        "383 Occupations": (0.714, 0.681, 0.549),
    }

    rows = []
    for name, r in results.items():
        p1980, p2000, pds = published[name]
        rows.append({
            "grouping": name,
            "s1980_pub": p1980, "s1980_rep": round(r["s1980"], 3),
            "s2000_pub": p2000, "s2000_rep": round(r["s2000"], 3),
            "ds_pub": pds,       "ds_rep":    round(r["ds"], 3),
            "within_1980": round(r["within_1980"], 4),
            "between_1980": round(r["total_1980"] - r["within_1980"], 4),
            "within_2000": round(r["within_2000"], 4),
            "between_2000": round(r["total_2000"] - r["within_2000"], 4),
        })
    out = pd.DataFrame(rows)
    print(out.to_string(index=False))
    out.to_csv(OUT / "table1_log_wages.csv", index=False)


if __name__ == "__main__":
    main()
