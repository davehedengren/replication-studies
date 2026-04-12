"""Python translation of compute_regression_coeffs_quarterly.do.

Reproduces the four blocks of predictability regressions the paper uses as
empirical moments for its structural estimation and plots in Figures 2-3:

  - Fama-Bliss (FB): excess-return-on-forward-premium across maturities
  - Campbell-Shiller (CS): yield-change-on-slope across maturities
  - Generalized UIP (G_UIP): exchange-rate change on short-rate differential
  - Long-horizon UIP (long_UIP): tau-year exchange rate change on yield diff
  - Level-Slope UIP (long_LS_UIP): same + yield-curve slope difference

All four regressions use the post-1986Q2 sample with Newey-West HAC SEs and
Stata's ivreg2 "bw(auto)" rule. Results are compared maturity-by-maturity
against the xlsx files produced by the upstream Stata run.
"""

import numpy as np
import pandas as pd

from utils import (
    MATURITIES,
    OUT,
    bw_auto_newey_west,
    in_sample,
    load_published_moments,
    load_quarterly,
)


def run_fb(df):
    rows = []
    for m in MATURITIES:
        if m < 24:  # min_maturity + m_short = 24
            rows.append({"tau": m // 12})
            continue
        sub = df.copy()
        for c in "HF":
            sub[f"FBdep_{c}"] = sub[f"R{c}_{m}"] - sub[f"y{c}_12"]
            sub[f"FBind_{c}"] = sub[f"f{c}_{m}"] - sub[f"y{c}_12"]
        out = {"tau": m // 12}
        for dep in "HF":
            for ind in "HF":
                d = sub[[f"FBdep_{dep}", f"FBind_{ind}"]].dropna()
                res, _ = bw_auto_newey_west(d[f"FBdep_{dep}"].values,
                                            d[f"FBind_{ind}"].values)
                out[f"FB_{dep}{ind}"] = res.params[1]
                out[f"hac_{dep}{ind}"] = res.bse[1]
        rows.append(out)
    return pd.DataFrame(rows)


def run_cs(df):
    rows = []
    for m in MATURITIES:
        if m < 24:
            rows.append({"tau": m // 12})
            continue
        m_dm = m - 12
        sub = df.copy()
        for c in "HF":
            sub[f"y{c}_{m_dm}_F4"] = sub[f"y{c}_{m_dm}"].shift(-4)
            sub[f"CSdep_{c}"] = sub[f"y{c}_{m_dm}_F4"] - sub[f"y{c}_{m}"]
            sub[f"CSind_{c}"] = (12.0 / (m - 12)) * (sub[f"y{c}_{m}"] - sub[f"y{c}_12"])
        out = {"tau": m // 12}
        for dep in "HF":
            for ind in "HF":
                d = sub[[f"CSdep_{dep}", f"CSind_{ind}"]].dropna()
                res, _ = bw_auto_newey_west(d[f"CSdep_{dep}"].values,
                                            d[f"CSind_{ind}"].values)
                out[f"CS_{dep}{ind}"] = res.params[1]
                out[f"hac_{dep}{ind}"] = res.bse[1]
        rows.append(out)
    return pd.DataFrame(rows)


def run_g_uip(df):
    rows = []
    for m in MATURITIES:
        if m < 24:
            rows.append({"tau": m // 12})
            continue
        sub = df.copy()
        sub["G_dep"] = sub["D_log_E"] + sub[f"RF_{m}"] - sub[f"RH_{m}"]
        sub["G_ind"] = sub["yF_12"] - sub["yH_12"]
        d = sub[["G_dep", "G_ind"]].dropna()
        res, _ = bw_auto_newey_west(d["G_dep"].values, d["G_ind"].values)
        rows.append({"tau": m // 12, "G_UIP": res.params[1], "hac": res.bse[1]})
    return pd.DataFrame(rows)


def run_long_uip(df):
    rows = []
    for m in MATURITIES:
        sub = df.copy()
        q = round(m / 3)
        sub["logE_F"] = sub["log_E"].shift(-q)
        sub["L_dep"] = (sub["logE_F"] - sub["log_E"]) / (m / 12.0)
        sub["L_ind"] = sub[f"yH_{m}"] - sub[f"yF_{m}"]
        d = sub[["L_dep", "L_ind"]].dropna()
        res, _ = bw_auto_newey_west(d["L_dep"].values, d["L_ind"].values)
        rows.append({"tau": m // 12, "UIP_level": res.params[1], "hac": res.bse[1]})
    return pd.DataFrame(rows)


def run_long_ls_uip(df):
    rows = []
    for m in MATURITIES:
        sub = df.copy()
        q = round(m / 3)
        sub["logE_F"] = sub["log_E"].shift(-q)
        sub["L_dep"] = (sub["logE_F"] - sub["log_E"]) / (m / 12.0)
        sub["L_lvl"] = sub[f"yH_{m}"] - sub[f"yF_{m}"]
        sub["L_slp"] = (sub["yH_120"] - sub["yH_12"]) - (sub["yF_120"] - sub["yF_12"])
        d = sub[["L_dep", "L_lvl", "L_slp"]].dropna()
        res, _ = bw_auto_newey_west(d["L_dep"].values, d[["L_lvl", "L_slp"]].values)
        rows.append({
            "tau": m // 12,
            "UIP_level": res.params[1], "UIP_slope": res.params[2],
            "hac_level": res.bse[1], "hac_slope": res.bse[2],
        })
    return pd.DataFrame(rows)


def compare(computed, published, cols, label, max_rel=5e-2):
    merged = computed.merge(published, on="tau", suffixes=("_py", "_st"))
    print(f"\n== {label} ==")
    worst = 0.0
    for c in cols:
        py, st = merged[f"{c}_py"], merged[f"{c}_st"]
        diff = (py - st).abs()
        denom = st.abs().replace(0, np.nan)
        rel = (diff / denom).fillna(diff)
        print(f"  {c}: max|py-st|={diff.max():.4g}  max rel={rel.max():.4g}")
        worst = max(worst, diff.max())
    print(f"  overall max abs diff = {worst:.4g}")
    return merged


def main():
    df = in_sample(load_quarterly())

    pub = load_published_moments()

    fb = run_fb(df)
    cs = run_cs(df)
    g = run_g_uip(df)
    lu = run_long_uip(df)
    ls = run_long_ls_uip(df)

    compare(fb, pub["FB"].rename(columns={"tau_arr": "tau"}),
            ["FB_HH", "FB_HF", "FB_FH", "FB_FF",
             "hac_HH", "hac_HF", "hac_FH", "hac_FF"], "Fama-Bliss")
    compare(cs, pub["CS"].rename(columns={"tau_arr": "tau"}),
            ["CS_HH", "CS_HF", "CS_FH", "CS_FF",
             "hac_HH", "hac_HF", "hac_FH", "hac_FF"], "Campbell-Shiller")
    compare(g, pub["G_UIP"].rename(columns={"tau_arr": "tau"}),
            ["G_UIP", "hac"], "Generalized UIP")
    compare(lu, pub["long_UIP"].rename(columns={"tau_arr": "tau"}),
            ["UIP_level", "hac"], "Long-horizon UIP")
    compare(ls, pub["long_LS_UIP"].rename(columns={"tau_arr": "tau"}),
            ["UIP_level", "UIP_slope", "hac_level", "hac_slope"],
            "Level-Slope UIP")

    fb.to_csv(OUT / "fb_py.csv", index=False)
    cs.to_csv(OUT / "cs_py.csv", index=False)
    g.to_csv(OUT / "g_uip_py.csv", index=False)
    lu.to_csv(OUT / "long_uip_py.csv", index=False)
    ls.to_csv(OUT / "long_ls_uip_py.csv", index=False)


if __name__ == "__main__":
    main()
