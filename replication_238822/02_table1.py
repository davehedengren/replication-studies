"""Replicate Table 1 (main effect of digitization on tax per acre) and Table 2
(2SLS with proportion of villages digitized as endogenous, digitization as IV).
"""

import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf

from utils import load, OUT


def drop_singletons(df, unit, time):
    """Mimic reghdfe singleton dropping. reghdfe iteratively drops rows whose
    unit or time value appears only once, because such rows cannot contribute
    to within-group variation."""
    while True:
        n0 = len(df)
        cu = df.groupby(unit).size()
        ct = df.groupby(time).size()
        df = df[df[unit].isin(cu[cu > 1].index) & df[time].isin(ct[ct > 1].index)]
        if len(df) == n0:
            return df


def twfe_ols(df, y, x, unit, time, cluster):
    cols = list(dict.fromkeys([y, x, unit, time, cluster]))
    d = df[cols].dropna().copy()
    d = drop_singletons(d, unit, time)
    ud = pd.get_dummies(d[unit], prefix="u", drop_first=True).astype(float)
    td = pd.get_dummies(d[time], prefix="t", drop_first=True).astype(float)
    X = pd.concat([pd.Series(1.0, index=d.index, name="const"),
                   d[[x]].astype(float), ud, td], axis=1)
    res = sm.OLS(d[y].astype(float).values, X.values).fit(
        cov_type="cluster", cov_kwds={"groups": d[cluster].values}
    )
    idx = list(X.columns).index(x)
    return res.params[idx], res.bse[idx], int(res.nobs), res.pvalues[idx]


def median_reg(df, y, x, unit, time, cluster):
    """LAD with district + year FE dummies. SE via a small cluster bootstrap."""
    cols = list(dict.fromkeys([y, x, unit, time, cluster]))
    d = df[cols].dropna().copy()
    d = drop_singletons(d, unit, time)
    ud = pd.get_dummies(d[unit], prefix="u", drop_first=True).astype(float)
    td = pd.get_dummies(d[time], prefix="t", drop_first=True).astype(float)
    X = pd.concat([pd.Series(1.0, index=d.index, name="const"),
                   d[[x]].astype(float), ud, td], axis=1)
    mod = sm.QuantReg(d[y].astype(float).values, X.values)
    res = mod.fit(q=0.5, max_iter=5000)
    idx = list(X.columns).index(x)
    # cluster bootstrap SE (200 reps to keep runtime short)
    rng = np.random.default_rng(44000)
    groups = d[cluster].values
    uniq = np.unique(groups)
    boots = []
    for b in range(200):
        samp = rng.choice(uniq, size=len(uniq), replace=True)
        rows = np.concatenate([np.where(groups == g)[0] for g in samp])
        try:
            r = sm.QuantReg(d[y].values[rows], X.values[rows]).fit(q=0.5, max_iter=2000)
            boots.append(r.params[idx])
        except Exception:
            continue
    se = float(np.std(boots, ddof=1))
    return res.params[idx], se, int(res.nobs)


def iv_2sls(df, y, endog, iv, unit, time, cluster):
    """Manual 2SLS with district + year FE: instrument endog with iv."""
    cols = list(dict.fromkeys([y, endog, iv, unit, time, cluster]))
    d = df[cols].dropna().copy()
    d = drop_singletons(d, unit, time)
    ud = pd.get_dummies(d[unit], prefix="u", drop_first=True).astype(float)
    td = pd.get_dummies(d[time], prefix="t", drop_first=True).astype(float)
    base = pd.concat([pd.Series(1.0, index=d.index, name="const"), ud, td], axis=1)
    # first stage: endog ~ iv + FEs
    X1 = pd.concat([d[[iv]].astype(float), base], axis=1)
    f1 = sm.OLS(d[endog].astype(float).values, X1.values).fit(
        cov_type="cluster", cov_kwds={"groups": d[cluster].values}
    )
    endog_hat = f1.fittedvalues
    # first-stage F on iv coefficient (Kleibergen-Paap reduces to cluster-robust
    # first-stage F with one endogenous and one instrument)
    k_iv = list(X1.columns).index(iv)
    F = (f1.params[k_iv] / f1.bse[k_iv]) ** 2
    # second stage: y ~ endog_hat + FEs
    X2 = pd.concat([pd.Series(endog_hat, index=d.index, name=endog), base], axis=1)
    s2 = sm.OLS(d[y].astype(float).values, X2.values).fit(
        cov_type="cluster", cov_kwds={"groups": d[cluster].values}
    )
    k_e = list(X2.columns).index(endog)
    return {
        "first_stage_beta": float(f1.params[k_iv]),
        "first_stage_se": float(f1.bse[k_iv]),
        "KP_F": float(F),
        "iv_beta": float(s2.params[k_e]),
        "iv_se": float(s2.bse[k_e]),
        "N": int(len(d)),
    }


def main():
    out = []

    def log(msg=""):
        print(msg)
        out.append(str(msg))

    log("==================== TABLE 1 ====================")
    log("Tax collection per acre on digitization (TWFE and stacked DID)")
    log("Published: OLS -6.57 (SE 3.69) unstacked | -6.74 (SE 3.83) stacked")
    log("Published: Median -5.21 (SE 2.43) unstacked | -5.60 (SE 1.97) stacked")
    log("")
    tax = load("Tax_District_Yr.dta")
    b, se, n, p = twfe_ols(tax, "tax_acres", "digitization", "district", "fin_yr", "district")
    log(f"[Col 1 OLS unstacked ]  beta={b:7.4f}  se={se:.4f}  p={p:.3f}  N={n}")
    bm, sem, nm = median_reg(tax, "tax_acres", "digitization", "district", "fin_yr", "district")
    log(f"[Col 2 Median unstk  ]  beta={bm:7.4f}  se={sem:.4f} (boot) N={nm}")

    stk = load("Tax_District_Yr_stacked.dta")
    b2, se2, n2, p2 = twfe_ols(stk, "tax_acres", "digitization", "district_h", "fin_yr_h", "district")
    log(f"[Col 3 OLS stacked   ]  beta={b2:7.4f}  se={se2:.4f}  p={p2:.3f}  N={n2}")
    bm2, sem2, nm2 = median_reg(stk, "tax_acres", "digitization", "district_h", "fin_yr_h", "district")
    log(f"[Col 4 Median stacked]  beta={bm2:7.4f}  se={sem2:.4f} (boot) N={nm2}")

    log("")
    log("==================== TABLE 2 (2SLS + OLS) ====================")
    log("Published: first stage 37.42*** (unstk), 39.97*** (stk); KP-F 55.7 / 112.1")
    log("Published: 2SLS -0.176 (unstk), -0.169 (stk); OLS -0.0756 (unstk), -0.0864 (stk)")
    log("")
    r_u = iv_2sls(tax, "tax_acres", "x", "digitization", "district", "fin_yr", "district")
    log(f"[Unstacked] 1st stage beta={r_u['first_stage_beta']:.3f} se={r_u['first_stage_se']:.3f}  "
        f"KP-F={r_u['KP_F']:.1f}  2SLS beta={r_u['iv_beta']:.4f} se={r_u['iv_se']:.4f}  N={r_u['N']}")
    r_s = iv_2sls(stk, "tax_acres", "x", "digitization", "district_h", "fin_yr_h", "district")
    log(f"[Stacked  ] 1st stage beta={r_s['first_stage_beta']:.3f} se={r_s['first_stage_se']:.3f}  "
        f"KP-F={r_s['KP_F']:.1f}  2SLS beta={r_s['iv_beta']:.4f} se={r_s['iv_se']:.4f}  N={r_s['N']}")

    # Plain OLS of tax on x (percentage of villages digitized)
    b, se, n, p = twfe_ols(tax, "tax_acres", "x", "district", "fin_yr", "district")
    log(f"[OLS x unstacked    ]  beta={b:.4f}  se={se:.4f}  N={n}")
    b, se, n, p = twfe_ols(stk, "tax_acres", "x", "district_h", "fin_yr_h", "district")
    log(f"[OLS x stacked      ]  beta={b:.4f}  se={se:.4f}  N={n}")

    (OUT / "02_table1.txt").write_text("\n".join(out))


if __name__ == "__main__":
    main()
