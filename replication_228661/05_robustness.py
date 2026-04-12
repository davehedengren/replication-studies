"""Robustness checks for the empirical moments of paper 228661.

The paper uses these regressions as headline empirical facts that the
structural preferred-habitat model is then estimated to match. We cannot
re-run the Julia MLE, but we can stress-test the empirical targets
themselves. A target that swings wildly across plausible specification
choices is a weak anchor for structural estimation, even if the structural
code is correct.

Checks (tailored to the paper's predictability regressions):

  R1. Drop the zero lower bound / post-2008 period                 [subsample]
  R2. Drop the euro crisis window (2010-2012)                       [subsample]
  R3. Drop the COVID quarter and after (2020+)                      [subsample]
  R4. Start the sample at 1999 (euro adoption)                      [subsample]
  R5. Alternative HAC bandwidth (fixed L=4)                         [SE choice]
  R6. HC1 heteroskedasticity-robust SEs                             [SE choice]
  R7. Winsorize Δlog_E at 1% / 99%                                  [outliers]
  R8. Drop top 5 |Δlog_E| observations                              [outliers]
  R9. Placebo country swap: Fama-Bliss with DE dep on US ind        [sign flip]
  R10. Long-horizon UIP with sign-flipped level indicator           [placebo]
  R11. Generalized UIP tau=5 vs tau=20 ratio (risk-premium tilt)    [internal]
  R12. Level-slope UIP: does the slope coefficient change sign      [specification]
"""

from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm

from utils import (
    MATURITIES,
    OUT,
    bw_auto_newey_west,
    in_sample,
    load_quarterly,
)

TAUS = [2, 5, 10, 20]  # years


def _subset(df, label, mask):
    sub = df[mask].reset_index(drop=True)
    return label, sub


def _fb_coef(sub, m, dep, ind):
    d = pd.DataFrame({
        "dep": sub[f"R{dep}_{m}"] - sub[f"y{dep}_12"],
        "ind": sub[f"f{ind}_{m}"] - sub[f"y{ind}_12"],
    }).dropna()
    if len(d) < 20:
        return np.nan
    res, _ = bw_auto_newey_west(d["dep"].values, d["ind"].values)
    return res.params[1]


def _g_uip_coef(sub, m):
    d = pd.DataFrame({
        "dep": sub["D_log_E"] + sub[f"RF_{m}"] - sub[f"RH_{m}"],
        "ind": sub["yF_12"] - sub["yH_12"],
    }).dropna()
    if len(d) < 20:
        return np.nan
    res, _ = bw_auto_newey_west(d["dep"].values, d["ind"].values)
    return res.params[1]


def _long_uip_coef(sub, m):
    q = round(m / 3)
    d = pd.DataFrame({
        "dep": (sub["log_E"].shift(-q) - sub["log_E"]) / (m / 12.0),
        "ind": sub[f"yH_{m}"] - sub[f"yF_{m}"],
    }).dropna()
    if len(d) < 20:
        return np.nan
    res, _ = bw_auto_newey_west(d["dep"].values, d["ind"].values)
    return res.params[1]


def _ls_uip_coefs(sub, m):
    q = round(m / 3)
    d = pd.DataFrame({
        "dep": (sub["log_E"].shift(-q) - sub["log_E"]) / (m / 12.0),
        "lvl": sub[f"yH_{m}"] - sub[f"yF_{m}"],
        "slp": (sub["yH_120"] - sub["yH_12"]) - (sub["yF_120"] - sub["yF_12"]),
    }).dropna()
    if len(d) < 20:
        return (np.nan, np.nan)
    res, _ = bw_auto_newey_west(d["dep"].values, d[["lvl", "slp"]].values)
    return (res.params[1], res.params[2])


def sample_sensitivity(df):
    print("\n--- R1-R4: subsample sensitivity of core coefficients ---")
    cases = [
        ("baseline 1986Q2-2021Q1", df["yq"] >= pd.Timestamp("1986-04-01")),
        ("R1 drop post-2008 (pre-2008 only)",
         (df["yq"] >= pd.Timestamp("1986-04-01")) & (df["yq"] < pd.Timestamp("2008-01-01"))),
        ("R2 drop euro crisis 2010-2012",
         (df["yq"] >= pd.Timestamp("1986-04-01")) &
         ~((df["yq"] >= pd.Timestamp("2010-01-01")) & (df["yq"] <= pd.Timestamp("2012-12-31")))),
        ("R3 drop COVID 2020+",
         (df["yq"] >= pd.Timestamp("1986-04-01")) & (df["yq"] < pd.Timestamp("2020-01-01"))),
        ("R4 euro era only 1999+",
         df["yq"] >= pd.Timestamp("1999-01-01")),
    ]
    cols = ["sample", "N", "FB_HH_τ5", "FB_FF_τ5", "G_UIP_τ5",
            "long_UIP_τ5", "ls_UIP_lvl_τ5", "ls_UIP_slp_τ5"]
    rows = []
    for label, mask in cases:
        sub = df[mask].reset_index(drop=True)
        m = 60
        fb_HH = _fb_coef(sub, m, "H", "H")
        fb_FF = _fb_coef(sub, m, "F", "F")
        g = _g_uip_coef(sub, m)
        lu = _long_uip_coef(sub, m)
        lsl, lss = _ls_uip_coefs(sub, m)
        rows.append([label, len(sub), fb_HH, fb_FF, g, lu, lsl, lss])
    return pd.DataFrame(rows, columns=cols)


def se_sensitivity(df):
    print("\n--- R5-R6: SE choice sensitivity (FB at τ=5, 10) ---")
    sub = in_sample(df)
    rows = []
    for m in [60, 120]:
        for dep, ind in [("H", "H"), ("F", "F")]:
            dd = pd.DataFrame({
                "dep": sub[f"R{dep}_{m}"] - sub[f"y{dep}_12"],
                "ind": sub[f"f{ind}_{m}"] - sub[f"y{ind}_12"],
            }).dropna()
            X = sm.add_constant(dd["ind"].values)
            y = dd["dep"].values
            r_nw, L = bw_auto_newey_west(y, dd["ind"].values)
            r_l4 = sm.OLS(y, X).fit(cov_type="HAC",
                                    cov_kwds={"maxlags": 4, "kernel": "bartlett"})
            r_hc1 = sm.OLS(y, X).fit(cov_type="HC1")
            rows.append([m, dep + ind, r_nw.params[1],
                         r_nw.bse[1], r_l4.bse[1], r_hc1.bse[1], L])
    return pd.DataFrame(rows, columns=["maturity", "reg", "beta",
                                       "SE_NW_Andrews", "SE_NW_L4", "SE_HC1", "L"])


def outlier_sensitivity(df):
    print("\n--- R7-R8: Δlog_E outlier sensitivity for generalised UIP ---")
    sub = in_sample(df).copy()
    rows = []
    rows.append(["R7 baseline", len(sub), _g_uip_coef(sub, 60), _long_uip_coef(sub, 60)])
    # winsorize Δlog_E at 1/99
    s = sub.copy()
    q_lo, q_hi = s["D_log_E"].quantile([0.01, 0.99])
    s["D_log_E"] = s["D_log_E"].clip(q_lo, q_hi)
    rows.append(["R7 winsorize D_log_E 1/99", len(s), _g_uip_coef(s, 60), _long_uip_coef(s, 60)])
    # drop top 5 |D_log_E|
    s = sub.copy()
    order = s["D_log_E"].abs().rank(method="first", ascending=False)
    s = s[order > 5].reset_index(drop=True)
    rows.append(["R8 drop top 5 |ΔlogE|", len(s), _g_uip_coef(s, 60), _long_uip_coef(s, 60)])
    return pd.DataFrame(rows, columns=["case", "N", "G_UIP_τ5", "long_UIP_τ5"])


def placebo_checks(df):
    print("\n--- R9-R10: placebo/sign checks ---")
    sub = in_sample(df)
    rows = []
    # R9: Stata computes FB_HF by regressing US excess return on DE forward
    # premium -- we flip that to its natural "no relation" placebo. The
    # paper's claim is that cross-country FB coefficients are smaller than
    # within; we confirm they are near zero/opposite sign.
    for (dep, ind) in [("H", "F"), ("F", "H")]:
        rows.append([f"FB {dep}/{ind} τ=5", _fb_coef(sub, 60, dep, ind)])
    # R10: flip sign of the level regressor in the long-horizon UIP
    q = round(60 / 3)
    d = pd.DataFrame({
        "dep": (sub["log_E"].shift(-q) - sub["log_E"]) / 5.0,
        "ind": -(sub["yH_60"] - sub["yF_60"]),
    }).dropna()
    res, _ = bw_auto_newey_west(d["dep"].values, d["ind"].values)
    rows.append([f"long-UIP sign-flipped τ=5", res.params[1]])
    return pd.DataFrame(rows, columns=["check", "coef"])


def tilt_and_slope(df):
    print("\n--- R11-R12: internal consistency of UIP tilt and slope coefficient ---")
    sub = in_sample(df)
    rows = []
    for tau in TAUS:
        m = tau * 12
        rows.append([tau, _g_uip_coef(sub, m),
                     _long_uip_coef(sub, m), *_ls_uip_coefs(sub, m)])
    return pd.DataFrame(rows, columns=["tau", "G_UIP", "long_UIP",
                                       "ls_UIP_lvl", "ls_UIP_slp"])


def main():
    df = load_quarterly()
    rs = [
        sample_sensitivity(df),
        se_sensitivity(df),
        outlier_sensitivity(df),
        placebo_checks(df),
        tilt_and_slope(df),
    ]
    names = ["sample_sens", "se_sens", "outlier", "placebo", "tilt"]
    for name, r in zip(names, rs):
        print(r.to_string(index=False))
        r.to_csv(OUT / f"robust_{name}.csv", index=False)
        print()


if __name__ == "__main__":
    main()
