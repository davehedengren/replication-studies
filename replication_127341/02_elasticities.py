"""Replicate equation (20): gravity estimates of distance and commute elasticities.

Stata original:
    reghdfe lns lndist yod if year(statadate) == 2020, a(RCo RCd) cluster(RCo RCd)

We absorb RCo and RCd FEs by adding them as categorical dummies (dropping first
of each to avoid collinearity) and cluster SEs two-way on RCo and RCd.

Published (section 3.3):
    (sigma-1) * kappa_1 = 1.53  (SE 0.066)   [coefficient on -lndist]
    (sigma-1) * epsilon = 0.45  (SE 0.067)   [coefficient on yod = chi]
    Sigma = 5  =>  kappa_1 = 0.38, epsilon = 0.11
"""
import numpy as np
import pandas as pd
import statsmodels.api as sm
from utils import OUT, PUB


def twoway_cluster_cov(X, resid, g1, g2):
    """Cameron-Gelbach-Miller two-way cluster robust covariance.

    V = V(g1) + V(g2) - V(g1 & g2).
    """
    XtX_inv = np.linalg.inv(X.T @ X)

    def meat(groups):
        scores = X * resid[:, None]
        df = pd.DataFrame(scores)
        df["g"] = groups
        agg = df.groupby("g").sum().values
        return agg.T @ agg

    M = meat(g1) + meat(g2) - meat(pd.Categorical(
        list(zip(g1, g2))).codes)
    return XtX_inv @ M @ XtX_inv


def main():
    df = pd.read_parquet(OUT / "sel_gravity.parquet")
    df = df[df["year"] == 2020].reset_index(drop=True)
    print(f"Sample: {len(df):,} obs ({df['RCo'].nunique()} origins x "
          f"{df['RCd'].nunique()} destinations)")

    # Two-way fixed effects via dummies (drop first of each to avoid collinearity).
    X = pd.DataFrame({
        "lndist": df["lndist"].values,
        "yod":    df["yod"].values,
    })
    origin_d = pd.get_dummies(df["RCo"].astype(int), prefix="o",
                              drop_first=True, dtype=float)
    dest_d = pd.get_dummies(df["RCd"].astype(int), prefix="d",
                            drop_first=True, dtype=float)
    X = pd.concat([X, origin_d, dest_d], axis=1)
    X = sm.add_constant(X)
    y = df["lns"].values

    # Point estimates (OLS is consistent; use for coefficients).
    fit = sm.OLS(y, X.values).fit()
    beta = fit.params
    resid = fit.resid
    coef_lndist = beta[X.columns.get_loc("lndist")]
    coef_yod = beta[X.columns.get_loc("yod")]

    # Two-way cluster SEs on RCo and RCd.
    cov = twoway_cluster_cov(
        X.values, resid,
        df["RCo"].astype(int).values,
        df["RCd"].astype(int).values,
    )
    se = np.sqrt(np.diag(cov))
    se_lndist = se[X.columns.get_loc("lndist")]
    se_yod = se[X.columns.get_loc("yod")]

    kappa_sigma = -coef_lndist
    kappa_sigma_se = se_lndist
    epsilon_sigma = coef_yod
    epsilon_sigma_se = se_yod

    # With sigma = 5 so sigma - 1 = 4.
    kappa1 = kappa_sigma / 4
    epsilon = epsilon_sigma / 4

    # Same-district spending share in March 2020 (Seoul).
    mar = df[(df["statadate"].dt.month == 3)]
    totals = mar.groupby(["statadate", "RCo"])["eod"].sum()
    diag = mar[mar["RCo"] == mar["RCd"]].groupby(["statadate", "RCo"])["eod"].sum()
    own_share = (diag.sum() / totals.sum()) * 100

    print("\n--- Elasticity estimates (equation 20) ---")
    hdr = f"{'':<24}{'replicated':>14}{'published':>14}"
    print(hdr)
    print("-" * len(hdr))
    for label, rep, pub in [
        ("(sigma-1)*kappa_1",     kappa_sigma,     PUB["kappa_sigma"]),
        ("  SE",                  kappa_sigma_se,  PUB["kappa_sigma_se"]),
        ("(sigma-1)*epsilon",     epsilon_sigma,   PUB["epsilon_sigma"]),
        ("  SE",                  epsilon_sigma_se,PUB["epsilon_sigma_se"]),
        ("kappa_1 (sigma=5)",     kappa1,          PUB["kappa1"]),
        ("epsilon (sigma=5)",     epsilon,         PUB["epsilon"]),
        ("own-share (March, %)",  own_share,       PUB["own_share_pct"]),
    ]:
        print(f"{label:<24}{rep:>14.3f}{pub:>14.3f}")

    pd.DataFrame([{
        "kappa_sigma": kappa_sigma, "kappa_sigma_se": kappa_sigma_se,
        "epsilon_sigma": epsilon_sigma, "epsilon_sigma_se": epsilon_sigma_se,
        "kappa1": kappa1, "epsilon": epsilon,
        "own_share_pct": own_share, "N": len(df),
    }]).to_csv(OUT / "elasticities.csv", index=False)


if __name__ == "__main__":
    main()
