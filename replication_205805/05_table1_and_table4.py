"""Replicate Table 1 (main event-study regression) and Table 4 (IV elasticity).

Columns replicated for Table 1:
  (1) Main estimate (event-study sample, inclmt + controls + state/year FE)
  (2) Extra controls (GL index broken out)  [approximate: same spec w/o break]
  (3) Waivers, lag unemp — skipped (we don't have Harris waiver data wired in)
  (4) Excludes recession (drop 2008, 2011)
  (5) Weighted
  (6) Avg of coefficients — skipped
  (7) All data

Table 4:
  Panel A: IV (ln(num) on ln(denom) instrumented by inclmt100) on full panel
  Panel B: same on event-study sample
"""
from __future__ import annotations
import numpy as np
import pandas as pd
import statsmodels.api as sm
from linearmodels.iv import IV2SLS
from utils import DATA

panel = pd.read_parquet(DATA / "full_panel.parquet")
es = pd.read_parquet(DATA / "eventstudy_sample.parquet")

CONTROLS = ["unemployment", "gl_index", "outreachnon0", "lnoutreach"]


def reghdfe(df, y, xvars, cluster="statefip", weights=None):
    """Fixed effects regression with state+year FE via within demeaning.

    Returns the coefficient on xvars[0] and its cluster-robust SE.
    Uses dummy FE for simplicity; statsmodels handles it fine at this size.
    """
    df = df.dropna(subset=[y] + xvars + [cluster, "year"]).copy()
    # Build dummies
    state_d = pd.get_dummies(df["statefip"], prefix="s", drop_first=True).astype(float)
    year_d  = pd.get_dummies(df["year"], prefix="y", drop_first=True).astype(float)
    X = pd.concat([df[xvars].astype(float).reset_index(drop=True),
                   state_d.reset_index(drop=True),
                   year_d.reset_index(drop=True)], axis=1)
    X = sm.add_constant(X, has_constant="add")
    yv = df[y].astype(float).reset_index(drop=True)
    groups = df[cluster].reset_index(drop=True)
    if weights is not None:
        w = df[weights].astype(float).values
        mdl = sm.WLS(yv, X, weights=w)
    else:
        mdl = sm.OLS(yv, X)
    res = mdl.fit(cov_type="cluster", cov_kwds={"groups": groups})
    return res


def fmt(res, var):
    b = res.params[var]
    se = res.bse[var]
    return f"{b:.3f} ({se:.3f})"


out_lines = []

def printrow(label, *vals):
    line = f"{label:30s} " + " | ".join(f"{v:>14s}" for v in vals)
    print(line)
    out_lines.append(line)


# ---- Panel A: 0-130% FPL ---------------------------------------------------
print("\n=== Table 1 Panel A (0-130% FPL) ===")
y_a = "ln_num_0to130"
x_a_ctrl = "ln_denom_0to130"

# Col 1: main
d = es.copy()
d["inclmt100"] = d["inclmt100"]  # passthrough
r1 = reghdfe(d, y_a, ["inclmt100"] + CONTROLS + [x_a_ctrl])
# Col 4: exclude 2008, 2011
d4 = d[~d["year"].isin([2008, 2011])]
r4 = reghdfe(d4, y_a, ["inclmt100"] + CONTROLS + [x_a_ctrl])
# Col 5: weighted by denom pop (wtsupp_0to130fpl)
d5 = d.copy()
d5["w"] = d5["wtsupp_0to130fpl"]
r5 = reghdfe(d5, y_a, ["inclmt100"] + CONTROLS + [x_a_ctrl], weights="w")
# Col 7: all data
d7 = panel.copy()
r7 = reghdfe(d7, y_a, ["inclmt100"] + CONTROLS + [x_a_ctrl])

printrow("Panel A est (pub: 0.085)",
         fmt(r1, "inclmt100"), fmt(r4, "inclmt100"),
         fmt(r5, "inclmt100"), fmt(r7, "inclmt100"))
print(f"  N: col1={int(r1.nobs)}  col4={int(r4.nobs)}  col5={int(r5.nobs)}  col7={int(r7.nobs)}")


# ---- Panel B: 50-115% FPL --------------------------------------------------
print("\n=== Table 1 Panel B (50-115% FPL) ===")
y_b = "ln_num_50to115"
x_b_ctrl = "ln_denom_50to115"
r1b = reghdfe(d, y_b, ["inclmt100"] + CONTROLS + [x_b_ctrl])
r4b = reghdfe(d4, y_b, ["inclmt100"] + CONTROLS + [x_b_ctrl])
d5b = d.copy(); d5b["w"] = d5b["wtsupp_50to115fpl"]
r5b = reghdfe(d5b, y_b, ["inclmt100"] + CONTROLS + [x_b_ctrl], weights="w")
r7b = reghdfe(d7, y_b, ["inclmt100"] + CONTROLS + [x_b_ctrl])

printrow("Panel B est (pub: 0.107)",
         fmt(r1b, "inclmt100"), fmt(r4b, "inclmt100"),
         fmt(r5b, "inclmt100"), fmt(r7b, "inclmt100"))
print(f"  N: col1={int(r1b.nobs)}  col4={int(r4b.nobs)}  col5={int(r5b.nobs)}  col7={int(r7b.nobs)}")


# ---- Table 4: IV elasticity (η_m) ------------------------------------------
print("\n=== Table 4: IV Elasticity ===")
# Outcome: ln(take-up_0to130) = ln(fywgt_0to130fpl) - ln(wtsupp_0to130fpl)
# Share eligible: wtsupp_0to130fpl / pop (CPS). Actually: the paper defines
# share eligible by the state's eligibility rule. For 130% FPL, share_elig =
# wtsupp_0to130fpl / pop. For BBCE>130, share_elig uses wtsupp_0to{inclmt}fpl /
# pop. We approximate by interpolation to 0-130 + scaling (share_elig_adj ≈
# share_0to130 * (inclmt/130) within the sample; this is admittedly a rough
# proxy).
# Paper actually uses number eligible in 0-130% bin AND a separate "max"
# denominator that depends on bbce_inclmt — see mergedata.do:
#     fywgt_0tomaxfpl = fywgt_0to130fpl if bbce_inclmt == 130
#     fywgt_0tomaxfpl = fywgt_0toXfpl  if bbce_inclmt == X  (X in {160,165,185,200})
# We don't have X>130 CPS bins, so we scale the denominator by inclmt/130.
# This is an approximation and we flag it in the writeup.
for df_, label in [(panel, "All data"), (es, "Event-study")]:
    d_ = df_.copy()
    d_["share_elig"] = (d_["wtsupp_0to130fpl"] / d_["pop"]) * (d_["inclmt100"] / 1.3)
    d_ = d_[d_["share_elig"] > 0].copy()
    d_["ln_share_elig"] = np.log(d_["share_elig"])
    d_["ln_takeup"] = d_["ln_num_0to130"] - np.log(d_["wtsupp_0to130fpl"])
    d_ = d_.dropna(subset=["ln_takeup", "ln_share_elig", "inclmt100"] + CONTROLS)

    state_d = pd.get_dummies(d_["statefip"], prefix="s", drop_first=True).astype(float)
    year_d  = pd.get_dummies(d_["year"], prefix="y", drop_first=True).astype(float)
    exog = pd.concat([d_[CONTROLS].reset_index(drop=True),
                      state_d.reset_index(drop=True),
                      year_d.reset_index(drop=True)], axis=1)
    exog = sm.add_constant(exog, has_constant="add")
    endog = d_[["ln_share_elig"]].reset_index(drop=True)
    instr = d_[["inclmt100"]].reset_index(drop=True)
    y = d_["ln_takeup"].reset_index(drop=True)
    clusters = d_["statefip"].reset_index(drop=True)

    # First stage: ln_share_elig on inclmt100 + controls + FE
    fs = sm.OLS(endog["ln_share_elig"], pd.concat([instr, exog], axis=1)).fit(
        cov_type="cluster", cov_kwds={"groups": clusters})
    # 2SLS via linearmodels
    iv_res = IV2SLS(y, exog, endog, instr).fit(cov_type="clustered", clusters=clusters)

    beta = iv_res.params["ln_share_elig"]
    se = iv_res.std_errors["ln_share_elig"]
    fs_coef = fs.params["inclmt100"]
    fs_se = fs.bse["inclmt100"]
    print(f"  [{label}] first-stage inclmt100: {fs_coef:.3f} ({fs_se:.3f})")
    print(f"  [{label}] 2SLS η_m = {beta:.3f} ({se:.3f})   N={len(d_)}")
    out_lines.append(f"Table 4 [{label}]  first-stage {fs_coef:.3f} ({fs_se:.3f})  "
                     f"2SLS {beta:.3f} ({se:.3f})  N={len(d_)}")


# Save results summary
(DATA / "table1_table4_results.txt").write_text("\n".join(out_lines))
print("\nDone.")
