"""Robustness checks for the Bandiera et al. (2022) JEP descriptive facts.

The paper presents stylised aggregates rather than causal estimates, so these
checks stress-test: (a) aggregation choice (weighted vs unweighted, alternate
weights), (b) sample composition (drop large/outlier countries, source), (c)
group definition (with/without North Africa), and (d) cohort-size regression
sensitivity.
"""
import numpy as np
import pandas as pd

from utils import (
    load,
    mark_africa,
    prep_pooled_age18to24,
    weighted_mean,
    OUT,
)


def _cat_shares(df: pd.DataFrame, weight: pd.Series | None, agg: str = "sal_nonagri") -> dict:
    if weight is None:
        m = df.mean(numeric_only=True)
        return {
            "sal_agri": 100 * m["employee_paid_agri"],
            "sal_nonagri": 100 * m["employee_paid_noagri"],
            "se_agri": 100 * m["SE_agri"],
            "se_nonagri": 100 * m["SE_noagri"],
        }
    return {
        "sal_agri": 100 * weighted_mean(df["employee_paid_agri"], weight),
        "sal_nonagri": 100 * weighted_mean(df["employee_paid_noagri"], weight),
        "se_agri": 100 * weighted_mean(df["SE_agri"], weight),
        "se_nonagri": 100 * weighted_mean(df["SE_noagri"], weight),
    }


def check_1_unweighted(df: pd.DataFrame) -> pd.DataFrame:
    """Figure 2 bottom with country-level unweighted means (paper Appendix Fig A2)."""
    rows = []
    for name, g in df.groupby("africa"):
        r = _cat_shares(g, None)
        r["group"] = "Africa" if name == 1 else "Other"
        rows.append(r)
    return pd.DataFrame(rows).set_index("group")


def check_2_alt_weights(df: pd.DataFrame) -> pd.DataFrame:
    """Weight by pop-in-workforce vs raw pop vs unweighted."""
    out = {}
    d = df.copy()
    d["popw_all"] = np.round(d["pop"] / 100_000)
    d["popw_work"] = np.round((d["pop"] / 100_000) * d["inwork"])
    for label, w in [
        ("pop", d["popw_all"]),
        ("pop_inwork", d["popw_work"]),
        ("unweighted", None),
    ]:
        for name, g in d.groupby("africa"):
            ww = w.loc[g.index] if w is not None else None
            r = _cat_shares(g, ww)
            r["group"] = "Africa" if name == 1 else "Other"
            r["weight"] = label
            out.setdefault(label, []).append(r)
    rows = []
    for k, v in out.items():
        rows.extend(v)
    return pd.DataFrame(rows)[["weight", "group", "sal_agri", "sal_nonagri", "se_agri", "se_nonagri"]]


def check_3_drop_north_africa(df: pd.DataFrame) -> pd.DataFrame:
    """Exclude Egypt & Morocco from Africa — robustness to North Africa definition."""
    d = df[~df["countrycode"].isin(["EGY", "MAR"])].copy()
    d["popw_work"] = np.round((d["pop"] / 100_000) * d["inwork"])
    rows = []
    for name, g in d.groupby("africa"):
        r = _cat_shares(g, g["popw_work"])
        r["group"] = "SSA only" if name == 1 else "Other"
        r["N"] = len(g)
        rows.append(r)
    return pd.DataFrame(rows)


def check_4_drop_ipums_only(df: pd.DataFrame) -> pd.DataFrame:
    """Africa sample by source: DHS-only vs IPUMS-only vs pooled."""
    d = df[df["africa"] == 1].copy()
    d["popw_work"] = np.round((d["pop"] / 100_000) * d["inwork"])
    rows = []
    for label, sel in [
        ("pooled", d),
        ("IPUMS only", d[d["source"].astype(str) == "IPUMS"]),
        ("DHS only", d[d["source"].astype(str) == "DHS"]),
    ]:
        r = _cat_shares(sel, sel["popw_work"])
        r["subset"] = label
        r["N"] = len(sel)
        rows.append(r)
    return pd.DataFrame(rows)


def check_5_leave_one_out(df: pd.DataFrame) -> pd.DataFrame:
    """Leave-one-country-out for the African salaried non-agri share."""
    d = df[df["africa"] == 1].copy()
    d["popw_work"] = np.round((d["pop"] / 100_000) * d["inwork"])
    base = 100 * weighted_mean(d["employee_paid_noagri"], d["popw_work"])
    rows = []
    for c in d["country_string"].unique():
        gg = d[d["country_string"] != c]
        rows.append(
            {
                "dropped": c,
                "sal_nonagri_africa": 100 * weighted_mean(gg["employee_paid_noagri"], gg["popw_work"]),
            }
        )
    return pd.DataFrame(rows).assign(baseline=base).sort_values("sal_nonagri_africa")


def check_6_gender(df: pd.DataFrame) -> pd.DataFrame:
    """Male vs female salaried share among 18-24 workers (Appendix Fig A3)."""
    g = load("gender_age18to24")
    g = mark_africa(g)
    g = g[g["age18to24"] == 1].copy()
    g = g.dropna(
        subset=[
            "employee_paid_agri",
            "employee_paid_noagri",
            "inwork",
            "paidwk_all",
            "lgdppc",
        ]
    )
    g = g[g["country_string"] != "Venezuela"]
    dup = g.duplicated(subset=["countrycode", "year", "age18to24", "gender"], keep=False) \
        if "gender" in g.columns else None
    # Figure out the gender column
    gender_col = next((c for c in g.columns if c.lower() in ("gender", "sex", "female", "male_female")), None)
    if gender_col is None:
        return pd.DataFrame({"note": ["no gender column"]})
    g["SE_agri"] = g["self_paid_agri"] + g["unpaidwk_agri"]
    g["SE_noagri"] = g["self_paid_noagri"] + g["unpaidwk_noagri"]
    g["popw"] = np.round((g["pop"] / 100_000) * g["inwork"])
    rows = []
    for (a, sex), sub in g.groupby(["africa", gender_col]):
        r = _cat_shares(sub, sub["popw"])
        r["group"] = "Africa" if a == 1 else "Other"
        r["gender"] = str(sex)
        rows.append(r)
    return pd.DataFrame(rows)[["group", "gender", "sal_agri", "sal_nonagri", "se_agri", "se_nonagri"]]


def check_7_cohort_size_continents(df_young: pd.DataFrame) -> pd.DataFrame:
    """Run the Figure 5 cohort-size OLS separately by country — is the pattern
    driven by a few outliers? Reports country-level slopes (admin within country)."""
    import statsmodels.api as sm

    ay = load("pooled_admin_young")
    a1 = load("pooled_admin1")
    ay = ay[ay["self_paid_agri"].notna() & ay["lgdppc"].notna()].copy()
    ay["maxy"] = ay.groupby("countrycode")["year"].transform("max")
    ay = ay[ay["year"] == ay["maxy"]]
    ay = ay.drop(columns=["age18to24"])
    keys = ["countrycode", "admin1", "year", "source"]
    ay = ay.merge(a1[keys + ["age18to24"]], on=keys, how="left")
    ay = ay[ay["age18to24"] <= 0.4]

    rows = []
    for cc, sub in ay.groupby("countrycode"):
        if len(sub) < 4:
            continue
        y = sub["employee_paid_noagri"]
        X = sm.add_constant(sub["age18to24"])
        res = sm.OLS(y, X).fit()
        rows.append(
            {
                "country": cc,
                "n_provinces": len(sub),
                "beta_cohort": float(res.params["age18to24"]),
                "country_mean": 100 * sub["employee_paid_noagri"].mean(),
            }
        )
    return pd.DataFrame(rows).sort_values("beta_cohort")


def check_8_cohort_size_linear_vs_quadratic() -> pd.DataFrame:
    """Local polynomial in Figure 5 is smooth; here we compare linear vs
    quadratic fits as a parsimonious summary."""
    import statsmodels.api as sm

    ay = load("pooled_admin_young")
    a1 = load("pooled_admin1")
    ay = ay[ay["self_paid_agri"].notna() & ay["lgdppc"].notna()].copy()
    ay["maxy"] = ay.groupby("countrycode")["year"].transform("max")
    ay = ay[ay["year"] == ay["maxy"]]
    ay = ay.drop(columns=["age18to24"])
    keys = ["countrycode", "admin1", "year", "source"]
    ay = ay.merge(a1[keys + ["age18to24"]], on=keys, how="left")
    ay = ay[ay["age18to24"] <= 0.4]
    ay["SE_agri"] = ay["self_paid_agri"] + ay["unpaidwk_agri"]
    ay["SE_noagri"] = ay["self_paid_noagri"] + ay["unpaidwk_noagri"]
    ay["cohort_sq"] = ay["age18to24"] ** 2
    rows = []
    for v in ["employee_paid_agri", "employee_paid_noagri", "SE_agri", "SE_noagri"]:
        lin = sm.OLS(ay[v], sm.add_constant(ay[["age18to24"]])).fit(cov_type="HC1")
        quad = sm.OLS(ay[v], sm.add_constant(ay[["age18to24", "cohort_sq"]])).fit(cov_type="HC1")
        rows.append(
            {
                "outcome": v,
                "lin_beta": float(lin.params["age18to24"]),
                "lin_se": float(lin.bse["age18to24"]),
                "lin_R2": float(lin.rsquared),
                "quad_beta_lin": float(quad.params["age18to24"]),
                "quad_beta_sq": float(quad.params["cohort_sq"]),
                "quad_R2": float(quad.rsquared),
            }
        )
    return pd.DataFrame(rows)


def check_9_wealth_gradient() -> pd.DataFrame:
    """Province×wealth: cohort-size slope on salaried non-ag by quintile."""
    import statsmodels.api as sm

    df = load("pooled_adminXwealth")
    df["wealth_q"] = np.nan
    for i in range(1, 6):
        col = f"wealth_pca_q_{i}"
        df.loc[df[col] == 1, "wealth_q"] = i
    df = df[df["wealth_q"].notna() & df["self_paid_agri"].notna() & df["lgdppc"].notna()]
    df["maxy"] = df.groupby("countrycode")["year"].transform("max")
    df = df[df["year"] == df["maxy"]]
    # get province cohort size
    admin = load("pooled_admin1")[["countrycode", "admin1", "year", "source", "age18to24"]]
    df = df.drop(columns=["age18to24"]).merge(admin, on=["countrycode", "admin1", "year", "source"], how="left")
    df = df[df["age18to24"] <= 0.4]
    rows = []
    for q, g in df.groupby("wealth_q"):
        res = sm.OLS(g["employee_paid_noagri"], sm.add_constant(g["age18to24"])).fit(cov_type="HC1")
        rows.append(
            {
                "quintile": int(q),
                "n_cells": len(g),
                "beta_cohort": float(res.params["age18to24"]),
                "se": float(res.bse["age18to24"]),
                "level_mean": 100 * g["employee_paid_noagri"].mean(),
            }
        )
    return pd.DataFrame(rows)


def check_10_internet_robustness() -> pd.DataFrame:
    """Figure 7 alternative splits: tertiles rather than median, and with/without
    Egypt & Morocco. Cross-checks the 0.261/0.170/0.313 pattern."""
    df = load("pooled_age18to24")
    df = mark_africa(df)
    df = df[(df["region"] == 6) | (df["countrycode"].isin(["EGY", "MAR"]))]
    df = df[df["age18to24"] == 1].copy()
    cc = load("pooled_cross_country")[["countrycode", "year", "source", "age18to24"]].rename(
        columns={"age18to24": "cohort_size"}
    )
    df = df.merge(cc, on=["countrycode", "year", "source"], how="inner")
    dup = df.duplicated(subset=["countrycode", "year"], keep=False)
    df = df[~(dup & (df["source"].astype(str) == "DHS"))]
    itu = load("ITU").rename(columns={"CountryCode": "countrycode", "Year": "year", "v171": "internetuse"})
    df = df.merge(itu[["countrycode", "year", "internetuse"]], on=["countrycode", "year"], how="inner")
    df = df[df["internetuse"].notna() & df["cohort_size"].notna() & df["employee_paid_noagri"].notna()]
    df["maxy"] = df.groupby("countrycode")["year"].transform("max")
    df = df[df["year"] == df["maxy"]].copy()

    rows = []
    for label, sub in [
        ("all 22 countries", df),
        ("drop N. Africa", df[~df["countrycode"].isin(["EGY", "MAR"])]),
    ]:
        ch_med = sub["cohort_size"].median()
        it_med = sub["internetuse"].median()
        sub = sub.copy()
        sub["large"] = (sub["cohort_size"] > ch_med).astype(int)
        sub["highnet"] = (sub["internetuse"] > it_med).astype(int)
        rows.append(
            {
                "subset": label,
                "n": len(sub),
                "avg": sub["employee_paid_noagri"].mean(),
                "large_lownet": sub.loc[(sub.large == 1) & (sub.highnet == 0), "employee_paid_noagri"].mean(),
                "large_highnet": sub.loc[(sub.large == 1) & (sub.highnet == 1), "employee_paid_noagri"].mean(),
                "small_lownet": sub.loc[(sub.large == 0) & (sub.highnet == 0), "employee_paid_noagri"].mean(),
                "small_highnet": sub.loc[(sub.large == 0) & (sub.highnet == 1), "employee_paid_noagri"].mean(),
            }
        )
    return pd.DataFrame(rows)


if __name__ == "__main__":
    df = prep_pooled_age18to24()
    d_y = df[df["age18to24"] == 1].copy()

    results = {}
    print("\n[1] Unweighted country means (Figure 2 bottom):")
    r = check_1_unweighted(d_y)
    print(r.round(1).to_string())
    results["1_unweighted"] = r

    print("\n[2] Alternative weights for Figure 2 shares:")
    r = check_2_alt_weights(d_y)
    print(r.round(1).to_string(index=False))
    results["2_weights"] = r

    print("\n[3] Drop Egypt & Morocco from Africa:")
    r = check_3_drop_north_africa(d_y)
    print(r.round(1).to_string(index=False))
    results["3_no_north_africa"] = r

    print("\n[4] Africa shares by data source:")
    r = check_4_drop_ipums_only(d_y)
    print(r.round(1).to_string(index=False))
    results["4_by_source"] = r

    print("\n[5] Leave-one-country-out (African sal_nonagri):")
    r = check_5_leave_one_out(d_y)
    print(r.round(2).to_string(index=False))
    results["5_loo"] = r

    print("\n[6] Gender breakdown (appendix Fig A3):")
    try:
        r = check_6_gender(d_y)
        print(r.round(1).to_string(index=False))
        results["6_gender"] = r
    except Exception as e:
        print("gender check failed:", e)

    print("\n[7] Country-level Figure 5 slopes (salaried non-ag ~ cohort size):")
    r = check_7_cohort_size_continents(d_y)
    print(r.round(3).to_string(index=False))
    results["7_country_slopes"] = r

    print("\n[8] Linear vs quadratic cohort-size fit:")
    r = check_8_cohort_size_linear_vs_quadratic()
    print(r.round(3).to_string(index=False))
    results["8_linquad"] = r

    print("\n[9] Wealth × cohort size gradient:")
    r = check_9_wealth_gradient()
    print(r.round(3).to_string(index=False))
    results["9_wealth"] = r

    print("\n[10] Figure 7 robustness:")
    r = check_10_internet_robustness()
    print(r.round(3).to_string(index=False))
    results["10_internet"] = r

    for k, v in results.items():
        v.to_csv(OUT / f"robust_{k}.csv", index=False if not isinstance(v.index, pd.MultiIndex) and v.index.name is None else True)
