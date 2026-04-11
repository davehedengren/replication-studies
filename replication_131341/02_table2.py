"""Replicate Table 2 (Panels A, B, C) — the paper's main results.

Panel A: First-stage / manipulation-check regressions of perceived risk
         measures on Del_unempincr + controls (OLS, HC1).
Panel B: Reduced-form regressions of forecast-demand dummies on Del_unempincr.
Panel C: 2SLS of forecast-demand dummies on perceived unemp_nextrecession,
         instrumented by Del_unempincr.
"""
import numpy as np
import pandas as pd

from utils import (
    load_experiment, build_design, ols_hc1, iv_2sls,
    INFO_REGS, BASE_CONTROLS, PANEL_A_OUTCOMES, PANEL_BC_OUTCOMES, OUT, fmt,
)

df = load_experiment()

# ---- Panel A: First-stage / manipulation checks ---------------------------
print("=" * 78)
print("PANEL A: First-stage / manipulation checks (Del_unempincr coefficient)")
print("=" * 78)
panel_a_rows = []
# Paper requires sample where forecast_recession is non-missing
subset = df.dropna(subset=["forecast_recession"]).copy()
for y_var in PANEL_A_OUTCOMES:
    d, X = build_design(subset, extra=INFO_REGS, drop_na=INFO_REGS + [y_var, "forecast_recession"])
    y = d[y_var].astype(float)
    res = ols_hc1(y, X)
    b = res.params["Del_unempincr"]
    se = res.bse["Del_unempincr"]
    p = res.pvalues["Del_unempincr"]
    n = int(res.nobs)
    r2 = res.rsquared
    ym, ys = y.mean(), y.std()
    panel_a_rows.append(
        dict(outcome=y_var, beta=b, se=se, p=p, N=n, R2=r2, ymean=ym, ysd=ys)
    )
    print(f"  {y_var:22s} β={fmt(b)}  se={fmt(se)}  p={fmt(p)}  N={n}  R²={r2:.3f}  "
          f"ȳ={fmt(ym,2)}  sd={fmt(ys,2)}")

# ---- Panel B: Reduced form on forecast choices ----------------------------
print("\n" + "=" * 78)
print("PANEL B: Reduced-form Del_unempincr → forecast demand")
print("=" * 78)
panel_b_rows = []
for y_var in PANEL_BC_OUTCOMES:
    d, X = build_design(df, extra=INFO_REGS, drop_na=INFO_REGS + [y_var])
    y = d[y_var].astype(float)
    res = ols_hc1(y, X)
    b = res.params["Del_unempincr"]
    se = res.bse["Del_unempincr"]
    p = res.pvalues["Del_unempincr"]
    n = int(res.nobs)
    r2 = res.rsquared
    panel_b_rows.append(dict(outcome=y_var, beta=b, se=se, p=p, N=n, R2=r2))
    print(f"  {y_var:22s} β={fmt(b)}  se={fmt(se)}  p={fmt(p)}  N={n}  R²={r2:.3f}")

# ---- Panel C: IV (2SLS) ---------------------------------------------------
print("\n" + "=" * 78)
print("PANEL C: 2SLS — unemp_nextrecession instrumented by Del_unempincr")
print("=" * 78)
panel_c_rows = []
for y_var in PANEL_BC_OUTCOMES:
    res = iv_2sls(
        df, outcome=y_var, endog="unemp_nextrecession",
        instrument="Del_unempincr",
        exog_controls=["unempincr_alt", "Del_unempbase", "unempbase_alt"] + BASE_CONTROLS,
    )
    b = res.params["unemp_nextrecession"]
    se = res.std_errors["unemp_nextrecession"]
    p = res.pvalues["unemp_nextrecession"]
    n = int(res.nobs)
    fs_f = res.first_stage.diagnostics.loc["unemp_nextrecession", "f.stat"]
    ym = df[y_var].mean()
    ys = df[y_var].std()
    panel_c_rows.append(
        dict(outcome=y_var, beta=b, se=se, p=p, N=n, fs_F=fs_f, ymean=ym, ysd=ys)
    )
    print(f"  {y_var:22s} β={fmt(b)}  se={fmt(se)}  p={fmt(p)}  N={n}  "
          f"F={fs_f:.2f}  ȳ={fmt(ym,2)}")

pd.DataFrame(panel_a_rows).to_csv(OUT / "table2_panelA.csv", index=False)
pd.DataFrame(panel_b_rows).to_csv(OUT / "table2_panelB.csv", index=False)
pd.DataFrame(panel_c_rows).to_csv(OUT / "table2_panelC.csv", index=False)

print("\n--- Comparison to published Table 2 ---")
pub_a = {
    "unemp_nextrecession": (0.489, 0.134),
    "z_manip_1":           (0.012, 0.005),
    "z_manip_2":           (0.007, 0.005),
    "z_manip_3":           (0.013, 0.004),
    "z_manip_ind":         (0.016, 0.005),
}
pub_b = {
    "forecast_recession":   (0.006, 0.002),
    "forecast_govspending": (-0.002, 0.002),
    "forecast_iratebonds":  (-0.003, 0.001),
    "forecast_inflation":   (0.001, 0.002),
    "othermacro":           (-0.004, 0.002),
    "noforecast":           (-0.002, 0.002),
}
pub_c = {
    "forecast_recession":   (0.012, 0.006),
    "forecast_govspending": (-0.004, 0.004),
    "forecast_iratebonds":  (-0.006, 0.003),
    "forecast_inflation":   (0.002, 0.004),
    "othermacro":           (-0.008, 0.005),
    "noforecast":           (-0.004, 0.004),
}

def compare(rows, pub, label):
    print(f"\n{label}")
    print(f"  {'outcome':22s}  {'β(rep)':>8s} {'β(pub)':>8s}   {'se(rep)':>8s} {'se(pub)':>8s}")
    for r in rows:
        pb, ps = pub[r["outcome"]]
        print(f"  {r['outcome']:22s}  {r['beta']:8.3f} {pb:8.3f}   {r['se']:8.3f} {ps:8.3f}")

compare(panel_a_rows, pub_a, "Panel A:")
compare(panel_b_rows, pub_b, "Panel B:")
compare(panel_c_rows, pub_c, "Panel C:")
