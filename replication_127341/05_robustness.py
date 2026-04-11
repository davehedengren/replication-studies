"""Robustness checks for the gravity elasticity estimates (equation 20).

Baseline (sigma-1)*kappa_1 = 1.53, (sigma-1)*epsilon = 0.45.
Each check re-runs OLS with RCo + RCd fixed effects and reports the two
coefficients plus N.
"""
import numpy as np
import pandas as pd
import statsmodels.api as sm
from utils import OUT


def gravity_ols(df, ses="hc1"):
    X = pd.DataFrame({"lndist": df["lndist"].values, "yod": df["yod"].values})
    for prefix, col in [("o", "RCo"), ("d", "RCd")]:
        dummies = pd.get_dummies(df[col].astype(int), prefix=prefix,
                                 drop_first=True, dtype=float)
        X = pd.concat([X.reset_index(drop=True),
                       dummies.reset_index(drop=True)], axis=1)
    X = sm.add_constant(X)
    y = df["lns"].values
    if ses == "cluster_RCo":
        fit = sm.OLS(y, X).fit(cov_type="cluster",
                               cov_kwds={"groups": df["RCo"].astype(int).values})
    elif ses == "cluster_date":
        fit = sm.OLS(y, X).fit(cov_type="cluster",
                               cov_kwds={"groups":
                                         df["statadate"].astype("int64").values})
    else:
        fit = sm.OLS(y, X).fit(cov_type="HC1")
    return {
        "kappa_sigma": -fit.params["lndist"],
        "kappa_se":     fit.bse["lndist"],
        "epsilon_sigma": fit.params["yod"],
        "epsilon_se":    fit.bse["yod"],
        "N": int(fit.nobs),
    }


def main():
    df = pd.read_parquet(OUT / "sel_gravity.parquet")
    base = df[df["year"] == 2020].reset_index(drop=True)

    checks = []

    # (0) Baseline (HC1) for reference — slightly different SEs from two-way
    # cluster, but identical point estimates.
    checks.append(("baseline (2020, HC1 SEs)", gravity_ols(base)))

    # (1) Pre-pandemic only (2018-2019) — in principle chi ≈ 1 throughout,
    # so the epsilon coefficient should be noisier.
    pre = df[df["year"] < 2020].reset_index(drop=True)
    checks.append(("2018-2019 only", gravity_ols(pre)))

    # (2) Full panel (all years pooled).
    checks.append(("full panel 2018-2020", gravity_ols(df)))

    # (3) Post-first-case only (ipat == 1).
    post = base[base["ipat"] == 1].reset_index(drop=True)
    checks.append(("post-first-case (ipat=1)", gravity_ols(post)))

    # (4) Drop off-diagonal distance 0 rows (robustness on lndist transform).
    off = base[base["RCo"] != base["RCd"]].reset_index(drop=True)
    checks.append(("off-diagonal only", gravity_ols(off)))

    # (5) Drop central Gangnam district (10123) — leave-one-out.
    no_gn = base[(base["RCo"] != 10123) & (base["RCd"] != 10123)]\
        .reset_index(drop=True)
    checks.append(("drop Gangnam (10123)", gravity_ols(no_gn)))

    # (6) Winsorise expenditures at 1st/99th pct (recompute lns).
    w = base.copy()
    lo, hi = w["eod"].quantile([0.01, 0.99])
    w["eod"] = w["eod"].clip(lo, hi)
    w["lns"] = np.log(w["eod"])
    checks.append(("winsorise eod 1/99 pct", gravity_ols(w)))

    # (7) Alternative SE: cluster on RCo only.
    checks.append(("2020, cluster(RCo)",
                   gravity_ols(base, ses="cluster_RCo")))

    # (8) Alternative SE: cluster on date.
    checks.append(("2020, cluster(date)",
                   gravity_ols(base, ses="cluster_date")))

    # (9) Replace yod (level chi) with log(chi). The paper writes equation 20
    # with ln(chi), but the Stata code uses chi in levels. Check whether this
    # changes the qualitative picture.
    lg = base.copy()
    lg["yod"] = np.log(lg["yod"].clip(lower=1e-6))
    checks.append(("log(chi) instead of chi", gravity_ols(lg)))

    # (10) Weekdays only (dow 0-4), matching the Stata weekday convention.
    wkday = base[base["statadate"].dt.dayofweek < 5].reset_index(drop=True)
    checks.append(("weekdays only", gravity_ols(wkday)))

    # (11) Placebo: randomly permute yod across OD-date cells.
    rng = np.random.default_rng(127341)
    plc = base.copy()
    plc["yod"] = rng.permutation(plc["yod"].values)
    checks.append(("placebo (shuffle yod)", gravity_ols(plc)))

    hdr = f"{'check':<32}{'(s-1)*kappa':>14}{'SE':>10}{'(s-1)*eps':>14}{'SE':>10}{'N':>10}"
    print(hdr)
    print("-" * len(hdr))
    for label, r in checks:
        print(f"{label:<32}{r['kappa_sigma']:>14.3f}{r['kappa_se']:>10.4f}"
              f"{r['epsilon_sigma']:>14.3f}{r['epsilon_se']:>10.4f}"
              f"{r['N']:>10,}")

    pd.DataFrame([{"check": l, **r} for l, r in checks]).to_csv(
        OUT / "robustness.csv", index=False)
    print(f"\nWrote {OUT / 'robustness.csv'}")


if __name__ == "__main__":
    main()
