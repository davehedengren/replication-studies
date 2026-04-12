"""Reproduce the remaining figures / numbers:
  - Figure 5 province scatter + local polynomial cohort-size relationship
  - Figure 6 province×wealth cohort-size relationship (sample counts only)
  - Figure 7 internet × cohort size bars (0.261 / 0.170 / 0.313)
"""
import numpy as np
import pandas as pd

from utils import load, mark_africa, OUT


def _figure5_panel() -> pd.DataFrame:
    ay = load("pooled_admin_young")
    a1 = load("pooled_admin1")
    ay = ay[ay["self_paid_agri"].notna() & ay["lgdppc"].notna()].copy()
    ay["maxy"] = ay.groupby("countrycode")["year"].transform("max")
    ay = ay[ay["year"] == ay["maxy"]]
    ay = ay.drop(columns=["age18to24"])
    keys = ["countrycode", "admin1", "year", "source"]
    ay = ay.merge(a1[keys + ["age18to24"]], on=keys, how="left")
    ay["SE_agri"] = ay["self_paid_agri"] + ay["unpaidwk_agri"]
    ay["SE_noagri"] = ay["self_paid_noagri"] + ay["unpaidwk_noagri"]
    return ay


def figure5_counts() -> dict:
    df = _figure5_panel()
    return {
        "provinces_all": int(df.groupby(["countrycode", "admin1"]).ngroups),
        "countries": int(df["countrycode"].nunique()),
        "cohort_min_pct": float(df["age18to24"].min()) * 100,
        "cohort_max_pct": float(df["age18to24"].max()) * 100,
        "provinces_in_plot": int((df["age18to24"] <= 0.4).sum()),
    }


def figure5_regressions() -> pd.DataFrame:
    """OLS of each outcome on cohort size across African provinces (linear
    analogue of the local polynomial in Figure 5)."""
    import statsmodels.api as sm

    df = _figure5_panel()
    df = df[df["age18to24"] <= 0.4]
    rows = []
    for v in ["employee_paid_agri", "employee_paid_noagri", "SE_agri", "SE_noagri"]:
        y = df[v]
        X = sm.add_constant(df["age18to24"])
        res = sm.OLS(y, X, missing="drop").fit(cov_type="HC1")
        rows.append(
            {
                "outcome": v,
                "beta_cohort": float(res.params["age18to24"]),
                "se": float(res.bse["age18to24"]),
                "N": int(res.nobs),
            }
        )
    return pd.DataFrame(rows)


def figure7() -> pd.DataFrame:
    """Internet × cohort bar plot numbers (p.95).
    Targets: average 0.261, below-median internet large-cohort 0.170,
             above-median internet large-cohort 0.313. 22 African countries.
    """
    df = load("pooled_age18to24")
    df = mark_africa(df)
    # sample: Africa + Egypt/Morocco, ages 18-24
    df = df[(df["region"] == 6) | (df["countrycode"].isin(["EGY", "MAR"]))]
    df = df[df["age18to24"] == 1].copy()

    # merge in cohort size from cross-country file
    cc = load("pooled_cross_country")[["countrycode", "year", "source", "age18to24"]]
    cc = cc.rename(columns={"age18to24": "cohort_size"})
    df = df.merge(cc, on=["countrycode", "year", "source"], how="inner")

    dup = df.duplicated(subset=["countrycode", "year"], keep=False)
    df = df[~(dup & (df["source"].astype(str) == "DHS"))]

    itu = load("ITU").rename(columns={"CountryCode": "countrycode", "Year": "year", "v171": "internetuse"})
    df = df.merge(itu[["countrycode", "year", "internetuse"]], on=["countrycode", "year"], how="inner")

    df = df[df["internetuse"].notna() & df["cohort_size"].notna() & df["employee_paid_noagri"].notna()]
    df["maxy"] = df.groupby("countrycode")["year"].transform("max")
    df = df[df["year"] == df["maxy"]].copy()

    n_countries = int(df["countrycode"].nunique())

    # medians
    cohort_median = df["cohort_size"].median()
    internet_median = df["internetuse"].median()
    df["large_cohort"] = (df["cohort_size"] > cohort_median).astype(int)
    df["high_internet"] = (df["internetuse"] > internet_median).astype(int)

    avg_africa = df["employee_paid_noagri"].mean()
    large_low_int = df.loc[(df["large_cohort"] == 1) & (df["high_internet"] == 0), "employee_paid_noagri"].mean()
    large_high_int = df.loc[(df["large_cohort"] == 1) & (df["high_internet"] == 1), "employee_paid_noagri"].mean()

    return pd.DataFrame(
        [
            {"bar": "average (all African)", "share": avg_africa, "N": n_countries},
            {"bar": "large cohort, below-median internet", "share": large_low_int, "N": int(((df.large_cohort == 1) & (df.high_internet == 0)).sum())},
            {"bar": "large cohort, above-median internet", "share": large_high_int, "N": int(((df.large_cohort == 1) & (df.high_internet == 1)).sum())},
        ]
    )


if __name__ == "__main__":
    print("Figure 5 sample:", figure5_counts())
    print("\nFigure 5 OLS (African provinces, cohort size 18-24):")
    print(figure5_regressions().round(4).to_string(index=False))
    print("\nFigure 7:")
    f7 = figure7()
    print(f7.round(3).to_string(index=False))
    f7.to_csv(OUT / "figure7.csv", index=False)
    figure5_regressions().to_csv(OUT / "figure5_regressions.csv", index=False)
