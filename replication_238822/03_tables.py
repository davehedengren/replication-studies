"""Replicate Tables 3-6: tax base (T3), reported base (T4), bureaucrat
performance (T5), and revenue-circle level (T6)."""

import numpy as np
import pandas as pd
import statsmodels.api as sm

from utils import load, OUT


def drop_singletons(df, unit, time):
    while True:
        n0 = len(df)
        cu = df.groupby(unit).size()
        ct = df.groupby(time).size()
        df = df[df[unit].isin(cu[cu > 1].index) & df[time].isin(ct[ct > 1].index)]
        if len(df) == n0:
            return df


def twfe_ols(df, y, x, unit, time, cluster, weights=None):
    cols = list(dict.fromkeys([y, x, unit, time, cluster] + ([weights] if weights else [])))
    d = df[cols].dropna().copy()
    d = drop_singletons(d, unit, time)
    ud = pd.get_dummies(d[unit], prefix="u", drop_first=True).astype(float)
    td = pd.get_dummies(d[time], prefix="t", drop_first=True).astype(float)
    X = pd.concat([pd.Series(1.0, index=d.index, name="const"),
                   d[[x]].astype(float), ud, td], axis=1)
    if weights is not None:
        w = d[weights].astype(float).values
        res = sm.WLS(d[y].astype(float).values, X.values, weights=w).fit(
            cov_type="cluster", cov_kwds={"groups": d[cluster].values}
        )
    else:
        res = sm.OLS(d[y].astype(float).values, X.values).fit(
            cov_type="cluster", cov_kwds={"groups": d[cluster].values}
        )
    idx = list(X.columns).index(x)
    return float(res.params[idx]), float(res.bse[idx]), int(res.nobs), float(res.pvalues[idx])


def main():
    out = []

    def log(msg=""):
        print(msg)
        out.append(str(msg))

    # ============================================================
    log("==================== TABLE 3: Tax base ====================")
    log("Published: profit 4.909 (3.212) | NDVI 0.00724 (0.00570) | "
        "irrigated -0.0000514 (0.0490) | log land 0.0635 (0.0444)")
    log("")

    # Column 1: farm profit per acre (HIES), cultivating sample, hh_weight
    h = load("hh_level_data_hies.dta")
    h = h[h["cultivatingsample"] == 1].copy()
    b, se, n, p = twfe_ols(h, "profit_per_area_wins", "digitization",
                           "district", "fin_yr", "district", weights="hh_weight")
    log(f"[Col 1 profit     ]  beta={b:.4f}  se={se:.4f}  N={n}")

    # Column 2: NDVI
    ndvi = load("NDVI_2006_2014.dta")
    b, se, n, p = twfe_ols(ndvi, "NDVI_veg", "digitization", "district", "fin_yr", "district")
    log(f"[Col 2 NDVI       ]  beta={b:.5f}  se={se:.5f}  N={n}")

    # Columns 3-4: PSLM individual data (secEFGB_3_25.dta)
    e = load("secEFGB_3_25.dta")
    log(f"  Loaded secEFGB: {len(e)} rows")
    # Column 3: irrigated (drop==3, recode so irrigated=1, else=0)
    e4 = e.dropna(subset=["secF1_Q1_c4"]).copy()
    e4 = e4[e4["secF1_Q1_c4"] != 3].copy()
    e4["secF1_Q1_c4"] = e4["secF1_Q1_c4"] - 1
    b, se, n, p = twfe_ols(e4, "secF1_Q1_c4", "digitization", "district", "fin_yr",
                           "district", weights="weights")
    log(f"[Col 3 irrigated  ]  beta={b:.6f}  se={se:.6f}  N={n}")

    # Column 4: ln_area (log land owned)
    b, se, n, p = twfe_ols(e, "ln_area", "digitization", "district", "fin_yr",
                           "district", weights="weights")
    log(f"[Col 4 log land   ]  beta={b:.4f}  se={se:.4f}  N={n}")

    # ============================================================
    log("")
    log("==================== TABLE 4: Reported base assessed by bureaucrats ====================")
    log("Published: log assessed area -0.100 (0.0338) | log tax demand -0.600 (0.211)")
    log("")

    rc = load("reportedcultivated.dta")
    b, se, n, p = twfe_ols(rc, "ln_area", "digitization", "district", "fin_yr", "district")
    log(f"[Col 1 ln assessed area ]  beta={b:.4f}  se={se:.4f}  N={n}")

    tax = load("Tax_District_Yr.dta")
    tax = tax[tax["max_cdemand2_PKR"] > 0].copy()
    tax["ln_demand"] = np.log(tax["max_cdemand2_PKR"])
    b, se, n, p = twfe_ols(tax, "ln_demand", "digitization", "district", "fin_yr", "district")
    log(f"[Col 2 ln tax demand    ]  beta={b:.4f}  se={se:.4f}  N={n}")

    # ============================================================
    log("")
    log("==================== TABLE 5: Bureaucrat performance ====================")
    log("Published: cum_tax -35.42*** (11.52) | >=50% -0.394*** (0.128) | "
        ">=75% -0.417*** (0.122) | zero months 0.263** (0.116) | N=304")
    log("")

    bu = load("Bureaucrat_district_Yr.dta")
    for dv in ["cum_tax", "perf_dummy50", "perf_dummy75", "frac_zeroperf"]:
        b, se, n, p = twfe_ols(bu, dv, "digitization", "district", "fin_yr", "district")
        log(f"[Col {dv:15s}]  beta={b:.4f}  se={se:.4f}  p={p:.4f}  N={n}")

    # ============================================================
    log("")
    log("==================== TABLE 6: Revenue circle level ====================")
    log("Published: unstacked -0.222** (0.101) | stacked -0.263* (0.144)")
    log("")
    rv = load("revcircle_yr.dta")
    b, se, n, p = twfe_ols(rv, "tax_acres", "computerization_id_dummy",
                           "id_SAR", "fin_yr", "id_SAR")
    log(f"[Col 1 OLS unstacked ]  beta={b:.4f}  se={se:.4f}  N={n}")

    rvs = load("revcircle_yr_stacked.dta")
    b, se, n, p = twfe_ols(rvs, "tax_acres", "computerization_id_dummy",
                           "id_SAR_h", "fin_yr_h", "id_SAR")
    log(f"[Col 3 OLS stacked   ]  beta={b:.4f}  se={se:.4f}  N={n}")

    (OUT / "03_tables.txt").write_text("\n".join(out))


if __name__ == "__main__":
    main()
