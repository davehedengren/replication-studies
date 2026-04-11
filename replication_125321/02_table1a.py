"""Replicate Table 1 Panel A: RD estimates for PM10 and AOD.
Equation (1) in the paper: P_ict = beta1 * 1{t >= Auto_ict} + f(t-Auto)
                              + beta4 W + alpha_i + month_t + u."""

import pandas as pd
from rdrobust import rdrobust
from utils import OUT, WEATHER_VARS, residualize

PUB_A_PM10 = {
    "(1) All, raw":        {"coef": 34.7, "se": 10.7, "n": 91470,  "bw": 109},
    "(2) All, residual":   {"coef": 34.9, "se":  5.8, "n": 232326, "bw": 263},
    "(3) Wave 1":          {"coef": 27.5, "se":  9.8, "n": 81950,  "bw": 140},
    "(4) Wave 2":          {"coef": 64.7, "se":  9.9, "n": 68456,  "bw": 234},
    "(5) Deadline":        {"coef": 57.1, "se":  8.6, "n": 86042,  "bw": 184},
}
PUB_A_AOD = {
    "(1) All, raw":        {"coef":  0.065, "se": 0.044, "n": 5057, "bw": 6},
    "(2) All, residual":   {"coef": -0.005, "se": 0.021, "n": 5851, "bw": 7},
    "(3) Wave 1":          {"coef":  0.026, "se": 0.031, "n": 3173, "bw": 7},
    "(4) Wave 2":          {"coef": -0.030, "se": 0.029, "n": 2316, "bw": 6},
    "(5) Deadline":        {"coef": -0.003, "se": 0.025, "n": 4894, "bw": 10},
}


def rd(df, y, running, cluster_col="code_city", p=1, q=2):
    sub = df[[y, running, cluster_col]].dropna()
    res = rdrobust(
        y=sub[y], x=sub[running],
        c=0, p=p, q=q, kernel="tri",
        cluster=sub[cluster_col],
        masspoints="off", bwselect="mserd",
        stdvars=True,  # matches Stata rdrobust default; Python default differs
    )
    coef = float(res.coef.iloc[0, 0])
    se = float(res.se.iloc[0, 0])
    n = int(res.N_h[0] + res.N_h[1])
    bw = float(res.bws.iloc[0, 0])
    return coef, se, n, bw


def main():
    print("=== Table 1 Panel A: Daily PM10 ===")
    sd = pd.read_parquet(OUT / "station_day_rd.parquet")
    sd["resid_pm10_smw"] = residualize(
        sd, "pm10", fe_cols=["pm10_n", "month"], controls=WEATHER_VARS
    )

    results = []
    # col 1: raw PM10 — use only rows with all covariates present (to match Stata)
    mask_cov = sd[WEATHER_VARS].notna().all(axis=1)
    c1 = sd.loc[mask_cov]
    results.append(("(1) All, raw",) + rd(c1, "pm10", "T"))
    # col 2: residualized
    results.append(("(2) All, residual",) + rd(sd, "resid_pm10_smw", "T"))
    # col 3: wave 1
    results.append(("(3) Wave 1",) + rd(sd[sd.phase == 1], "resid_pm10_smw", "T"))
    # col 4: wave 2
    results.append(("(4) Wave 2",) + rd(sd[sd.phase == 2], "resid_pm10_smw", "T"))
    # col 5: deadline sample
    ddl = sd[sd.auto_date.isin([19359, 19724])]
    results.append(("(5) Deadline",) + rd(ddl, "resid_pm10_smw", "T"))

    print(f"{'col':<20} {'coef':>10} {'se':>10} {'n':>10} {'bw':>10}   vs pub")
    for label, coef, se, n, bw in results:
        pub = PUB_A_PM10[label]
        tag = "MATCH" if abs(coef - pub["coef"]) / max(1e-9, pub["coef"]) < 0.10 else "DIFF"
        print(f"{label:<20} {coef:10.2f} {se:10.2f} {n:10,d} {bw:10.0f}   "
              f"pub={pub['coef']:6.1f} ({pub['se']:.1f}), n={pub['n']:,} [{tag}]")

    # Panel A row 2: Monthly AOD
    print("\n=== Table 1 Panel A: Monthly AOD ===")
    sm = pd.read_parquet(OUT / "station_month.parquet")
    sm["resid_aod_smw"] = residualize(
        sm, "aod", fe_cols=["pm10_n", "month"], controls=WEATHER_VARS
    )
    mask_cov = sm[WEATHER_VARS].notna().all(axis=1) & sm["resid_aod_smw"].notna()
    c1 = sm.loc[mask_cov]

    aod_res = []
    aod_res.append(("(1) All, raw",) + rd(c1, "aod", "n_month"))
    aod_res.append(("(2) All, residual",) + rd(sm, "resid_aod_smw", "n_month"))
    aod_res.append(("(3) Wave 1",) + rd(sm[sm.phase == 1], "resid_aod_smw", "n_month"))
    aod_res.append(("(4) Wave 2",) + rd(sm[sm.phase == 2], "resid_aod_smw", "n_month"))
    aod_ddl = sm[sm.auto_date.isin([19359, 19724])]
    aod_res.append(("(5) Deadline",) + rd(aod_ddl, "resid_aod_smw", "n_month"))

    print(f"{'col':<20} {'coef':>12} {'se':>10} {'n':>10} {'bw':>10}   vs pub")
    for label, coef, se, n, bw in aod_res:
        pub = PUB_A_AOD[label]
        ok = abs(coef - pub["coef"]) < 0.05
        print(f"{label:<20} {coef:12.4f} {se:10.4f} {n:10,d} {bw:10.0f}   "
              f"pub={pub['coef']:7.3f} ({pub['se']:.3f}), n={pub['n']:,} "
              f"[{'MATCH' if ok else 'DIFF'}]")

    out = pd.DataFrame(
        results + aod_res,
        columns=["col", "coef", "se", "n", "bw"],
    )
    out["panel"] = ["PM10"] * 5 + ["AOD"] * 5
    out.to_csv(OUT / "table1a_replicated.csv", index=False)
    print(f"\nSaved {OUT / 'table1a_replicated.csv'}")


if __name__ == "__main__":
    main()
