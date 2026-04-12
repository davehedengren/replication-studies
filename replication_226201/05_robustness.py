"""Robustness checks for Table 1 (ACM 2025) — spec (3) baseline.

Each check is a variant of the sector-IV + pre-trend-controls specification.
Results are written to `output/robustness_summary.csv`.
"""
import numpy as np
import pandas as pd

from utils import OUT, build_regression_panel, aw_cluster_iv, dummies_from

PANEL = build_regression_panel()
BASE_CTRLS = ["init_life_sci_uni", "init_uni_size", "init_fac_stud_ratio"]


def run_spec(df, xvar="xx", zvar="zz", controls=None, cluster_col="geo_state"):
    ctrls = controls or []
    req = ["yy", xvar, zvar, "fte_wts", "geo_state", "sector"] + ctrls
    d = df.dropna(subset=req).copy()
    if d.empty:
        return None
    D = dummies_from(d["geo_state"], "state", drop_first=True).values
    pnp = (d["sector"] == 2).astype(float).values.reshape(-1, 1)
    xmain = d[[xvar]].values
    xint = xmain * pnp
    zmain = d[[zvar]].values
    zint = zmain * pnp
    X_exog = np.column_stack([np.ones(len(d)).reshape(-1, 1), pnp, D,
                              d[ctrls].values if ctrls else np.empty((len(d), 0))])
    X_endog = np.column_stack([xmain, xint])
    Z_instr = np.column_stack([zmain, zint])
    cl = d[cluster_col].values
    r = aw_cluster_iv(d["yy"].values, X_exog, X_endog, Z_instr,
                      d["fte_wts"].values, cl)
    r["names"] = [xvar, f"{xvar}*pnp", "const", "pnp"]  # only the part we care about
    return {
        "beta_rd": r["beta"][0], "se_rd": r["se"][0],
        "beta_int": r["beta"][1], "se_int": r["se"][1],
        "n": r["n"], "R2": r["R2"],
    }


def tag(label, res):
    if res is None:
        return {"check": label, "beta_rd": np.nan, "se_rd": np.nan,
                "beta_int": np.nan, "se_int": np.nan, "n": 0, "R2": np.nan}
    res = dict(res); res["check"] = label
    return res


rows = []

# Baseline (= spec 3)
r = run_spec(PANEL, controls=BASE_CTRLS)
rows.append(tag("Baseline (spec 3)", r))

# 1. Drop unis where both xx=0 and zz=0 (no R&D signal at all)
m = PANEL[(PANEL["xx"] != 0) | (PANEL["zz"] != 0)]
rows.append(tag("1. Drop xx=zz=0 obs", run_spec(m, controls=BASE_CTRLS)))

# 2. Drop the single uni with the largest instrument share (Johns Hopkins-type)
top_z = PANEL.nlargest(1, "zz")["uni_ID"].values
m = PANEL[~PANEL["uni_ID"].isin(top_z)].copy()
# Rescale zz so shares re-normalize to 1 on the remaining sample
m["zz"] = m["zz"] / m["zz"].sum()
rows.append(tag("2. Drop top-zz uni", run_spec(m, controls=BASE_CTRLS)))

# 3. Trim top/bottom 1% of yy
lo, hi = PANEL["yy"].quantile([0.01, 0.99])
m = PANEL[(PANEL["yy"] >= lo) & (PANEL["yy"] <= hi)]
rows.append(tag("3. Trim 1%% yy tails", run_spec(m, controls=BASE_CTRLS)))

# 4. Winsorize xx at 1/99 within sample
m = PANEL.copy()
lo, hi = m["xx"].quantile([0.01, 0.99])
m["xx"] = m["xx"].clip(lo, hi)
rows.append(tag("4. Winsorize xx 1/99", run_spec(m, controls=BASE_CTRLS)))

# 5. Drop top 10 largest universities by fte_wts
big = PANEL.nlargest(10, "fte_wts")["uni_ID"]
m = PANEL[~PANEL["uni_ID"].isin(big)]
rows.append(tag("5. Drop 10 biggest unis", run_spec(m, controls=BASE_CTRLS)))

# 6. Unweighted
m = PANEL.copy()
m["fte_wts"] = 1.0
rows.append(tag("6. Unweighted (aw=1)", run_spec(m, controls=BASE_CTRLS)))

# 7. Cluster at uni_ID instead of state (upper-bound check)
r = run_spec(PANEL, controls=BASE_CTRLS, cluster_col="uni_ID")
rows.append(tag("7. Cluster at uni_ID", r))

# 8. Drop California + New York (big lifesci states)
m = PANEL[~PANEL["geo_state"].isin(["CA", "NY"])]
rows.append(tag("8. Drop CA & NY", run_spec(m, controls=BASE_CTRLS)))

# 9. Private non-profit only
m = PANEL[PANEL["sector"] == 2]
# In the PNP-only subsample, the xx*pnp interaction collapses to xx itself
# (pnp ≡ 1); re-run without sector interaction.
def run_single(df):
    req = ["yy", "xx", "zz", "fte_wts", "geo_state"] + BASE_CTRLS
    d = df.dropna(subset=req).copy()
    D = dummies_from(d["geo_state"], "state", drop_first=True).values
    X_exog = np.column_stack([np.ones(len(d)).reshape(-1, 1), D, d[BASE_CTRLS].values])
    X_endog = d[["xx"]].values
    Z_instr = d[["zz"]].values
    r = aw_cluster_iv(d["yy"].values, X_exog, X_endog, Z_instr,
                      d["fte_wts"].values, d["geo_state"].values)
    return {"beta_rd": r["beta"][0], "se_rd": r["se"][0],
            "beta_int": np.nan, "se_int": np.nan, "n": r["n"], "R2": r["R2"]}
rows.append(tag("9. PNP only, no interaction", run_single(m)))

# 10. Public only
m = PANEL[PANEL["sector"] == 1]
rows.append(tag("10. Public only, no interaction", run_single(m)))

# 11. No state FE
def run_nofe(df):
    req = ["yy", "xx", "zz", "fte_wts", "geo_state", "sector"] + BASE_CTRLS
    d = df.dropna(subset=req).copy()
    pnp = (d["sector"] == 2).astype(float).values.reshape(-1, 1)
    X_exog = np.column_stack([np.ones(len(d)).reshape(-1, 1), pnp,
                              d[BASE_CTRLS].values])
    X_endog = np.column_stack([d[["xx"]].values, d[["xx"]].values * pnp])
    Z_instr = np.column_stack([d[["zz"]].values, d[["zz"]].values * pnp])
    r = aw_cluster_iv(d["yy"].values, X_exog, X_endog, Z_instr,
                      d["fte_wts"].values, d["geo_state"].values)
    return {"beta_rd": r["beta"][0], "se_rd": r["se"][0],
            "beta_int": r["beta"][1], "se_int": r["se"][1],
            "n": r["n"], "R2": r["R2"]}
rows.append(tag("11. No state FE", run_nofe(PANEL)))

# 12. Placebo treatment: shuffle zz across unis (should kill coefficient)
rng = np.random.default_rng(20260411)
n_perm = 200
placebo_bs = []
for _ in range(n_perm):
    m = PANEL.copy()
    m["zz"] = m["zz"].sample(frac=1, random_state=rng.integers(1 << 31)).values
    r = run_spec(m, controls=BASE_CTRLS)
    if r is not None:
        placebo_bs.append(r["beta_rd"])
placebo_bs = np.array(placebo_bs)
baseline_b = rows[0]["beta_rd"]
pval = np.mean(np.abs(placebo_bs) >= abs(baseline_b))
rows.append({"check": f"12. Placebo zz shuffle ({n_perm}x)",
             "beta_rd": placebo_bs.mean(), "se_rd": placebo_bs.std(),
             "beta_int": np.nan, "se_int": np.nan,
             "n": int(PANEL.shape[0]), "R2": pval})
print(f"\nPlacebo (zz shuffled {n_perm}x): mean={placebo_bs.mean():.5f}, "
      f"sd={placebo_bs.std():.5f}, share |placebo| >= baseline = {pval:.3f}")

out = pd.DataFrame(rows)[["check", "beta_rd", "se_rd", "beta_int", "se_int", "n", "R2"]]
out["t_rd"] = out["beta_rd"] / out["se_rd"]
out.to_csv(OUT / "robustness_summary.csv", index=False)
print("\n" + out.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
print(f"\nSaved to {OUT / 'robustness_summary.csv'}")
