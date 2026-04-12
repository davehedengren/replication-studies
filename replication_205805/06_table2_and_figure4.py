"""Replicate Table 2 (compositional effects) and Figure 4 event-study coefficients.

Table 2 uses QC micro data (50-115% FPL cells) to estimate the effect of
inclmt100 on enrollee characteristics. We build state-year weighted means of
each outcome and regress on inclmt100 + controls + state/year FE.

Figure 4 uses the same event-study sample and outputs the event-time
coefficients for the dependent variables ln_num_0to130 and ln_num_50to115.
"""
from __future__ import annotations
import numpy as np
import pandas as pd
import statsmodels.api as sm
from utils import DATA, qc_file_paths
import re


es = pd.read_parquet(DATA / "eventstudy_sample.parquet")

# -----------------------------------------------------------------------------
# Table 2: compositional means of enrollees in 50-115% FPL
# Build from raw QC files so we have access to fin_net_inc, sex, age, nkids.
# -----------------------------------------------------------------------------
print("Building QC person-level table for 50-115% FPL...")
frames = []
for p in qc_file_paths():
    df = pd.read_stata(str(p), convert_categoricals=False)
    df.columns = [c.lower() for c in df.columns]
    need_scalar = [c for c in ["state", "yrmonth", "fywgt", "tpov", "fsnetinc"] if c in df.columns]
    if "fsnetinc" not in df.columns:
        df["fsnetinc"] = np.nan
        need_scalar.append("fsnetinc")
    age_cols = [c for c in df.columns if re.fullmatch(r"age\d+", c)]
    sex_cols = [c for c in df.columns if re.fullmatch(r"sex\d+", c)]
    age = df[age_cols]
    sex = df[sex_cols] if sex_cols else None

    # Reshape long: each row per household becomes up to len(age_cols) person rows
    long_parts = []
    for i, a_col in enumerate(age_cols, start=1):
        s_col = f"sex{i}"
        sv = df[s_col] if s_col in df.columns else pd.Series(np.nan, index=df.index)
        sub = pd.DataFrame({
            "state": df["state"].values,
            "yrmonth": df["yrmonth"].values,
            "fywgt": df["fywgt"].values,
            "tpov": df["tpov"].values,
            "fsnetinc": df["fsnetinc"].values,
            "age": df[a_col].values,
            "sex": sv.values,
        })
        long_parts.append(sub)
    long = pd.concat(long_parts, ignore_index=True)
    long = long.dropna(subset=["age"])  # drop empty slots
    long = long[(long["tpov"] >= 50) & (long["tpov"] <= 115)]
    long["ym"] = long["yrmonth"].astype(float)
    long.loc[long["ym"] < 10000, "ym"] = long["ym"] + 190000
    long["year"] = (long["ym"] // 100).astype(int)
    frames.append(long[["state", "year", "fywgt", "age", "sex", "fsnetinc"]])
    print(f"  {p.name}: {len(long):,} rows")
qc_long = pd.concat(frames, ignore_index=True)
qc_long = qc_long[(qc_long["year"] >= 1996) & (qc_long["year"] <= 2016)]
qc_long["female"] = (qc_long["sex"] == 2).astype(float)
qc_long["fywgt"] = qc_long["fywgt"].astype(float)

def wmean(g, col):
    w = g["fywgt"]
    v = g[col]
    mask = v.notna() & w.notna()
    w = w[mask]; v = v[mask]
    if w.sum() == 0:
        return np.nan
    return (v * w).sum() / w.sum()

grouped = qc_long.groupby(["state", "year"])
comp = pd.DataFrame({
    "female": grouped.apply(lambda g: wmean(g, "female")),
    "age":    grouped.apply(lambda g: wmean(g, "age")),
    "fsnetinc": grouped.apply(lambda g: wmean(g, "fsnetinc")),
}).reset_index().rename(columns={"state": "statefip"})
comp["statefip"] = comp["statefip"].astype(int)
comp.to_parquet(DATA / "qc_comp_stateyear.parquet")

es2 = es.merge(comp, on=["statefip", "year"], how="left")
CONTROLS = ["unemployment", "gl_index", "outreachnon0", "lnoutreach"]

def reg(df, y):
    d = df.dropna(subset=[y] + CONTROLS + ["inclmt100", "statefip", "year"]).copy()
    state_d = pd.get_dummies(d["statefip"], prefix="s", drop_first=True).astype(float)
    year_d  = pd.get_dummies(d["year"], prefix="y", drop_first=True).astype(float)
    X = pd.concat([d[["inclmt100"] + CONTROLS].reset_index(drop=True),
                   state_d.reset_index(drop=True),
                   year_d.reset_index(drop=True)], axis=1)
    X = sm.add_constant(X, has_constant="add")
    yv = d[y].astype(float).reset_index(drop=True)
    groups = d["statefip"].reset_index(drop=True)
    res = sm.OLS(yv, X).fit(cov_type="cluster", cov_kwds={"groups": groups})
    return res.params["inclmt100"], res.bse["inclmt100"], int(res.nobs)

print("\n=== Table 2 (50-115% FPL enrollee composition) ===")
print(f"{'outcome':15s} {'coef':>10s} {'se':>10s}  {'N':>5s}   (published)")
for y, pub in [("female", "-0.001 (0.004)"),
               ("age",    "0.391 (0.420)"),
               ("fsnetinc","-28.557 (20.033)")]:
    b, se, n = reg(es2, y)
    print(f"{y:15s} {b:>10.3f} {se:>10.3f}  {n:>5d}   {pub}")


# -----------------------------------------------------------------------------
# Figure 4: event-time coefficients on ln enrollment
# -----------------------------------------------------------------------------
# Build event_time column
print("\n=== Figure 4 event-time coefficients ===")
es_fig = es.copy()
# Determine event year for treated states
es_fig["prev_inclmt"] = es_fig.groupby("statefip")["bbce_inclmt"].shift(1)
es_fig["is_change"] = ((es_fig["bbce_inclmt"] != es_fig["prev_inclmt"])
                       & es_fig["prev_inclmt"].notna()).astype(int)
event_years = (es_fig[es_fig["is_change"] == 1]
               .groupby("statefip")["year"].first().to_dict())
es_fig["event_year"] = es_fig["statefip"].map(event_years)
es_fig["event_time"] = es_fig["year"] - es_fig["event_year"]
# Untreated get event_time = -1
es_fig.loc[es_fig["event_year"].isna(), "event_time"] = -1
es_fig["event_time"] = es_fig["event_time"].astype(int)

# Post-event inclmt (the treated state's new eligibility level), constant per
# state; untreated states get 130 (no change).
post_inclmt_map = (es_fig[es_fig["event_time"] == 5]
                   .set_index("statefip")["bbce_inclmt"].to_dict())
# Fallback: use max inclmt observed in the state
for s in es_fig["statefip"].unique():
    if s not in post_inclmt_map:
        post_inclmt_map[s] = es_fig[es_fig["statefip"] == s]["bbce_inclmt"].max()
es_fig["post_inclmt"] = es_fig["statefip"].map(post_inclmt_map)
inclmt_dev = (es_fig["post_inclmt"] - 130) / 100.0  # treatment dose (post level)

taus = list(range(-5, 6))
for tau in taus:
    if tau == -1:
        continue
    es_fig[f"et_{tau}"] = inclmt_dev * (es_fig["event_time"] == tau).astype(float)

def run_event_study(df, y, x_ctrl):
    d = df.dropna(subset=[y, x_ctrl] + CONTROLS + ["statefip", "year"]).copy()
    et_cols = [f"et_{t}" for t in taus if t != -1]
    for c in et_cols:
        if c not in d.columns:
            d[c] = 0.0
    state_d = pd.get_dummies(d["statefip"], prefix="s", drop_first=True).astype(float)
    year_d  = pd.get_dummies(d["year"], prefix="y", drop_first=True).astype(float)
    X = pd.concat([d[et_cols + CONTROLS + [x_ctrl]].reset_index(drop=True),
                   state_d.reset_index(drop=True),
                   year_d.reset_index(drop=True)], axis=1)
    X = sm.add_constant(X, has_constant="add")
    yv = d[y].astype(float).reset_index(drop=True)
    groups = d["statefip"].reset_index(drop=True)
    res = sm.OLS(yv, X).fit(cov_type="cluster", cov_kwds={"groups": groups})
    return res, et_cols

print("\n0-130% FPL:")
res_a, cols = run_event_study(es_fig, "ln_num_0to130", "ln_denom_0to130")
for c in cols:
    t = int(c.split("_")[1])
    print(f"  t={t:+d}: {res_a.params[c]:>7.3f} ({res_a.bse[c]:>6.3f})")

print("\n50-115% FPL:")
res_b, cols = run_event_study(es_fig, "ln_num_50to115", "ln_denom_50to115")
for c in cols:
    t = int(c.split("_")[1])
    print(f"  t={t:+d}: {res_b.params[c]:>7.3f} ({res_b.bse[c]:>6.3f})")

# Save coefficient tables
pd.DataFrame({
    "tau": [int(c.split("_")[1]) for c in cols],
    "coef_0to130": [res_a.params[c] for c in cols],
    "se_0to130":   [res_a.bse[c] for c in cols],
    "coef_50to115": [res_b.params[c] for c in cols],
    "se_50to115":   [res_b.bse[c] for c in cols],
}).to_csv(DATA / "figure4_coefs.csv", index=False)
print("\nDone.")
