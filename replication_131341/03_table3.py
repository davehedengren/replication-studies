"""Replicate Table 3 (heterogeneity by confidence in prior).

Panel A: manipulation-check outcomes, low-confidence subsample.
Panel B: same, high-confidence subsample.
Panels C/D: forecast-demand outcomes, split by confidence.
Interaction regression is used for the p-value on (a − b).
"""
import pandas as pd

from utils import (
    load_experiment, build_design, ols_hc1,
    INFO_REGS, BASE_CONTROLS, OCC_VAR,
    PANEL_A_OUTCOMES, PANEL_BC_OUTCOMES, OUT, fmt,
)
import numpy as np
import statsmodels.api as sm

df = load_experiment()


def fit_sub(df, y_var, sub_mask, drop_hiconf=False):
    """Fit reduced-form spec on subset. If drop_hiconf, omit the high_conf
    control (its within-group variance is zero)."""
    extra = INFO_REGS
    controls = [c for c in BASE_CONTROLS if not (drop_hiconf and c == "high_conf")]
    d = df.loc[sub_mask].dropna(subset=extra + controls + [y_var, OCC_VAR]).copy()
    occ = pd.get_dummies(d[OCC_VAR].astype(int), prefix="occ", drop_first=True).astype(float)
    occ = occ.loc[:, occ.sum(axis=0) > 0]
    X = pd.concat(
        [d[extra].astype(float).reset_index(drop=True),
         d[controls].astype(float).reset_index(drop=True),
         occ.reset_index(drop=True)],
        axis=1,
    )
    X = X.loc[:, X.std() > 0]
    X = sm.add_constant(X, has_constant="add")
    y = d[y_var].astype(float).reset_index(drop=True)
    return sm.OLS(y, X).fit(cov_type="HC1")


def diff_pvalue(df, y_var):
    """p-value on Del_unempincr*high_conf from the fully interacted regression."""
    d = df.dropna(subset=INFO_REGS + BASE_CONTROLS + [y_var, OCC_VAR]).copy()
    # interaction terms
    for v in INFO_REGS:
        d[f"{v}_hc"] = d[v] * d["high_conf"]
    interact_names = [f"{v}_hc" for v in INFO_REGS]
    occ = pd.get_dummies(d[OCC_VAR].astype(int), prefix="occ", drop_first=True).astype(float)
    occ = occ.loc[:, occ.sum(axis=0) > 0]
    occ_hc = occ.multiply(d["high_conf"].values, axis=0)
    occ_hc.columns = [c + "_hc" for c in occ.columns]
    # base controls interacted with high_conf — drop high_conf itself (linear with itself)
    base_controls_no_hc = [c for c in BASE_CONTROLS if c != "high_conf"]
    ctrl_hc = d[base_controls_no_hc].multiply(d["high_conf"].values, axis=0)
    ctrl_hc.columns = [c + "_hc" for c in base_controls_no_hc]
    X = pd.concat(
        [d[INFO_REGS + interact_names + BASE_CONTROLS].astype(float).reset_index(drop=True),
         occ.reset_index(drop=True),
         occ_hc.reset_index(drop=True),
         ctrl_hc.reset_index(drop=True)],
        axis=1,
    )
    X = X.loc[:, X.std() > 0]
    X = sm.add_constant(X, has_constant="add")
    y = d[y_var].astype(float).reset_index(drop=True)
    res = sm.OLS(y, X).fit(cov_type="HC1")
    return res.pvalues.get("Del_unempincr_hc", np.nan)


print("=" * 78)
print("Table 3 — heterogeneity by confidence in prior")
print("=" * 78)

results = []
for y_var in PANEL_A_OUTCOMES + PANEL_BC_OUTCOMES:
    res_low = fit_sub(df, y_var, df["high_conf"] == 0, drop_hiconf=True)
    res_high = fit_sub(df, y_var, df["high_conf"] == 1, drop_hiconf=True)
    p_diff = diff_pvalue(df, y_var)
    b_lo = res_low.params["Del_unempincr"]
    se_lo = res_low.bse["Del_unempincr"]
    n_lo = int(res_low.nobs)
    b_hi = res_high.params["Del_unempincr"]
    se_hi = res_high.bse["Del_unempincr"]
    n_hi = int(res_high.nobs)
    results.append(dict(
        outcome=y_var, b_low=b_lo, se_low=se_lo, n_low=n_lo,
        b_high=b_hi, se_high=se_hi, n_high=n_hi, p_diff=p_diff,
    ))
    print(f"  {y_var:22s}  low: {fmt(b_lo)} ({fmt(se_lo)}) N={n_lo}  "
          f"high: {fmt(b_hi)} ({fmt(se_hi)}) N={n_hi}  p(diff)={fmt(p_diff)}")

pd.DataFrame(results).to_csv(OUT / "table3.csv", index=False)
print("\nSaved table3.csv")
