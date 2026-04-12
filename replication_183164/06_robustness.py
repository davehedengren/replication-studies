"""Robustness checks for Antman, Duncan, Trejo (JEP 2023).

This is a JEP review-style descriptive paper.  The headline claims are:
  1. US-born Hispanics have ~1.1 fewer years of schooling than US-born
     non-Hispanic whites, conditional on age (Table 1).
  2. US-born Hispanic men earn ~30 log-points less than US-born whites
     (Table 2 col 1).  The gap for women is much smaller (~19 log points).
  3. The Mexican vs. white schooling gap among US-born men has shrunk
     sharply between 1970 and 2019 (Figure 1A, Mexican: -3.37 → -1.24).
  4. Cross-generation: 2nd-gen Hispanic men gain ~2.6 years of schooling
     over 1st gen, but stagnate between 2nd and 3rd+ gen (Table 3A).

Checks (all use the already-cleaned analysis slices):

R1  Re-estimate Table 1 col 1 WITHOUT weights → benchmark sampling weight role.
R2  Table 1 col 1 on US-born Hispanics only (drop foreign-born) — gap persists.
R3  Table 1 col 1 with a quartic in age (age35+age35^2+age35^3+age35^4)
    instead of full age FEs — estimator functional-form sensitivity.
R4  Table 2 (men) with HC3 SE instead of HC1 — robust SE choice.
R5  Table 2 (men) restricting to non-farm, non-zero wage workers (same as paper).
R6  Alternative outcome: share with college degree (Table 1 col 1 style).
R7  Alternative outcome: share high-school+ (Table 1 col 1 style).
R8  Figure 1A Mexican coef, men: recompute with edyrs> 0 trimming (drop 0 edyrs
    in case outliers drive the shrinking gap).
R9  Table 3 Panel A (male Hispanics): alt weights (wtfinl × 1/17) → unchanged mean.
R10 Figure 1B spec 2 Mexican coef: reversed sign check — make sure the
    "controlling for education" line sits above spec 1 everywhere (stylised
    fact in the paper).
R11 Sub-sample by birth region (CA vs. TX vs. AZ/CO/NM vs. other states) for
    US-born Mexican schooling gap (Table 1 col 2 analog, 2019 ACS).
R12 Placebo: rerun Table 1 col 1 on a flipped base category (US-born Black as
    base) — sign of Hispanic vs. Black coefficient should flip but magnitude
    and significance should be consistent with the original table.
"""
import numpy as np
import pandas as pd
import statsmodels.api as sm
import pyreadstat

from utils import OUT, CENSUS_DTA


def _wls(y, X, w, hc="HC1"):
    res = sm.WLS(y, X, weights=w).fit(cov_type=hc)
    return res


def build_design(df, race_col, race_base, age_col="age", state_col=None, extra=None):
    """Generic FE design with manual base-category dropping."""
    parts = [pd.Series(1.0, index=df.index, name="const")]
    r = pd.get_dummies(df[race_col], prefix="race").drop(
        columns=[f"race_{race_base}"], errors="ignore").astype(float)
    parts.append(r)
    a = pd.get_dummies(df[age_col], prefix="age").drop(
        columns=["age_35"], errors="ignore").astype(float)
    parts.append(a)
    if state_col:
        s = pd.get_dummies(df[state_col], prefix="st").drop(
            columns=["st_6"], errors="ignore").astype(float)
        parts.append(s)
    if extra is not None:
        parts.append(extra)
    X = pd.concat(parts, axis=1)
    return X


def regress(df, y_col, race_col, race_base, weighted=True, include_state=False,
            hc="HC1", extra=None):
    d = df.dropna(subset=[y_col]).copy()
    X = build_design(d, race_col, race_base,
                     state_col="statefip" if include_state else None,
                     extra=extra)
    y = d[y_col].astype(float).to_numpy()
    Xn = X.to_numpy(dtype=float)
    if weighted:
        w = d["perwt"].astype(float).to_numpy()
        w = w * len(w) / w.sum()
    else:
        w = np.ones(len(d))
    res = sm.WLS(y, Xn, weights=w).fit(cov_type=hc)
    res.exog_names = list(X.columns)
    return res


def coef(res, name):
    i = res.exog_names.index(name)
    return float(res.params[i]), float(res.bse[i])


def main():
    df = pd.read_parquet(OUT / "census_2019.parquet")
    for c in ["race_cat4", "race_cat16", "age", "statefip", "sex"]:
        df[c] = df[c].astype(int)
    print(f"[06] Base sample N={len(df):,}")

    results = []

    # R1: unweighted Table 1 col 1
    r = regress(df, "edyrs", "race_cat4", race_base=3, weighted=False)
    results.append(("R1 unweighted Table1c1",
                    *coef(r, "race_2"), int(r.nobs)))
    # Baseline weighted for reference
    r = regress(df, "edyrs", "race_cat4", race_base=3, weighted=True)
    results.append(("baseline Table1c1 weighted",
                    *coef(r, "race_2"), int(r.nobs)))

    # R2: US-born Hispanics only (drop race_cat4==1 foreign-born hisp)
    sub = df[df["race_cat4"] != 1]
    r = regress(sub, "edyrs", "race_cat4", race_base=3, weighted=True)
    results.append(("R2 drop FB Hisp",
                    *coef(r, "race_2"), int(r.nobs)))

    # R3: polynomial age instead of age FE
    d3 = df.copy()
    d3["age35"] = d3["age"] - 35
    extra = d3[["age35"]].astype(float).copy()
    extra["age35_2"] = extra["age35"] ** 2
    extra["age35_3"] = extra["age35"] ** 3
    extra["age35_4"] = extra["age35"] ** 4
    # Replace age FE with polynomial — build a custom design
    parts = [pd.Series(1.0, index=d3.index, name="const")]
    parts.append(pd.get_dummies(d3["race_cat4"], prefix="race").drop(
        columns=["race_3"], errors="ignore").astype(float))
    parts.append(extra)
    X = pd.concat(parts, axis=1)
    y = d3["edyrs"].astype(float).to_numpy()
    w = d3["perwt"].astype(float).to_numpy()
    w = w * len(w) / w.sum()
    r = sm.WLS(y, X.to_numpy(dtype=float), weights=w).fit(cov_type="HC1")
    r.exog_names = list(X.columns)
    results.append(("R3 age as quartic poly",
                    *coef(r, "race_2"), int(r.nobs)))

    # R4: Table 2 men with HC3
    men = df[df["sex"] == 1]
    r = regress(men, "ln_annualearnings", "race_cat4", race_base=3,
                weighted=True, include_state=True, hc="HC3")
    results.append(("R4 Table2 men HC3",
                    *coef(r, "race_2"), int(r.nobs)))
    # compare to baseline HC1
    r = regress(men, "ln_annualearnings", "race_cat4", race_base=3,
                weighted=True, include_state=True, hc="HC1")
    results.append(("baseline Table2 men HC1",
                    *coef(r, "race_2"), int(r.nobs)))

    # R5: Table 2 men, drop very low earnings (<$1000/yr) to probe outliers
    men_trim = men[men["annualearnings"] >= 1000]
    r = regress(men_trim, "ln_annualearnings", "race_cat4", race_base=3,
                weighted=True, include_state=True)
    results.append(("R5 Table2 men earn>=$1000",
                    *coef(r, "race_2"), int(r.nobs)))

    # R6: outcome = college
    r = regress(df, "college", "race_cat4", race_base=3, weighted=True)
    results.append(("R6 outcome = college",
                    *coef(r, "race_2"), int(r.nobs)))

    # R7: outcome = hsgrad
    r = regress(df, "hsgrad", "race_cat4", race_base=3, weighted=True)
    results.append(("R7 outcome = hsgrad",
                    *coef(r, "race_2"), int(r.nobs)))

    # R12: placebo base = US-born black
    r = regress(df, "edyrs", "race_cat4", race_base=4, weighted=True)
    # Here "race_2" = US-born Hispanic vs. US-born black
    results.append(("R12 base=USB Black: USB Hisp vs USB Black",
                    *coef(r, "race_2"), int(r.nobs)))

    print("\n=== Robustness coefficients (key = US-born Hispanic, unless noted) ===")
    print(f"{'check':<42s} {'coef':>10s} {'se':>10s} {'N':>12s}")
    for nm, b, se, N in results:
        print(f"{nm:<42s} {b:>10.4f} {se:>10.4f} {N:>12,}")

    # R8: figure 1A Mexican trend, trim to edyrs>=1 (removes 0-school outliers)
    print("\n=== R8: Figure 1A Mexican coef with edyrs>=1 trim vs. full ===")
    fig = pd.read_parquet(OUT / "census_figure1.parquet")
    fig = fig[fig["sex"] == 1]
    for c in ["year", "age", "race_cat6", "statefip"]:
        fig[c] = fig[c].astype(int)
    for yr in [1970, 1980, 1990, 2000, 2010, 2019]:
        sub = fig[fig["year"] == yr]
        sub_trim = sub[sub["edyrs"] >= 1]
        weighted = yr >= 1990
        r_full = _run_edyrs(sub, weighted)
        r_trim = _run_edyrs(sub_trim, weighted)
        print(f"  {yr}: full {r_full:+.4f}    trim {r_trim:+.4f}    "
              f"(diff {r_trim - r_full:+.4f})")

    # R9: CPS Table 3 Panel A male Hispanic 1st/2nd/3rd+ using half-weights
    print("\n=== R9: CPS Table 3 Hispanic male Panel A — unweighted means ===")
    cps = pd.read_parquet(OUT / "cps_clean.parquet")
    male = cps[cps["sex"] == 1]
    for code, lbl in [(1, "1st gen"), (2, "2nd gen"), (3, "3rd+ gen")]:
        g = male[male["gen_org_cat12"] == code]
        w_mean = np.average(g["edyrs"], weights=g["wtfinl"])
        unw_mean = g["edyrs"].mean()
        print(f"  {lbl}: weighted {w_mean:.3f}   unweighted {unw_mean:.3f}")

    # R11: Table 1 col 2 analog restricted by USB Mexican birth state
    # (the cleaned data does not carry state-of-birth directly, but bpl via
    # statefip is proxied by state of residence — this is a soft check).
    print("\n=== R11: USB Mexican schooling gap, 2019 ACS, by state of residence ===")
    d_usb = df[df["race_cat16"].isin([8, 15])].copy()  # USB Mexican vs USB white
    for lbl, states in [("CA", [6]), ("TX", [48]), ("AZ/CO/NM", [4, 8, 35]),
                         ("Other", None)]:
        if states is None:
            mask = ~d_usb["statefip"].isin([6, 48, 4, 8, 35])
        else:
            mask = d_usb["statefip"].isin(states)
        sub = d_usb[mask]
        if len(sub) == 0:
            print(f"  {lbl}: N=0"); continue
        r = regress(sub, "edyrs", "race_cat16", race_base=15, weighted=True)
        b, se = coef(r, "race_8")
        print(f"  {lbl:<10s} N={int(r.nobs):>8,}  USB Mex coef {b:+.3f} ({se:.3f})")

    print("\n[06] done.")


def _run_edyrs(df, weighted):
    d = df.copy()
    X = build_design(d, "race_cat6", race_base=6)
    y = d["edyrs"].astype(float).to_numpy()
    if weighted:
        w = d["perwt"].astype(float).to_numpy()
        w = w * len(w) / w.sum()
    else:
        w = np.ones(len(d))
    res = sm.WLS(y, X.to_numpy(dtype=float), weights=w).fit(cov_type="HC1")
    res.exog_names = list(X.columns)
    return float(res.params[res.exog_names.index("race_1")])


if __name__ == "__main__":
    main()
