"""Robustness checks for the main Table B.II specification (OLS, country+year FE).

Baseline = specification II: deviation ~ Revenue + Debt + Debt*Rev/100 + Target
           + country FE + year FE. The headline claim is that the target
coefficient is large, negative, and highly significant ("−0.458 (0.062)").

We stress-test that finding with 10 checks. All checks output the Target
coefficient, its SE, and p-value so the reader can see which survive.
"""
import os
import warnings
import numpy as np
import pandas as pd
import statsmodels.api as sm

from utils import OUT, RHS_BASE, fit_ols_fe


def ols_with_yfe(d, outcome="deviation", extras=None, cluster=None):
    """Plain OLS with country + year dummies; return Target (b, se, p, N)."""
    d = d.dropna(subset=[outcome] + RHS_BASE + list(extras or [])).copy()
    parts = [d[RHS_BASE].reset_index(drop=True)]
    if extras:
        parts.append(d[list(extras)].reset_index(drop=True))
    parts.append(pd.get_dummies(d["Country"], drop_first=False, dtype=float).reset_index(drop=True))
    yrs = pd.get_dummies(d["year"], drop_first=True, dtype=float).reset_index(drop=True)
    yrs.columns = [f"year_{c}" for c in yrs.columns]
    parts.append(yrs)
    X = pd.concat(parts, axis=1)
    y = d[outcome].values
    kwargs = {}
    if cluster is not None:
        kwargs = dict(cov_type="cluster", cov_kwds={"groups": d[cluster].values})
    m = sm.OLS(y, X.values).fit(**kwargs)
    tgt = list(X.columns).index("Center.Target")
    return m.params[tgt], m.bse[tgt], m.pvalues[tgt], int(m.nobs)


def print_row(label, b, se, p, n):
    stars = "***" if p < 0.01 else ("**" if p < 0.05 else ("*" if p < 0.1 else " "))
    print(f"  {label:45s}  b={b:+7.3f}  se={se:.3f}  p={p:.3f}{stars}  N={n}")


def main():
    df = pd.read_csv(os.path.join(OUT, "panel.csv"))
    df = df.dropna(subset=["deviation"] + RHS_BASE).copy()

    print("=" * 78)
    print("ROBUSTNESS CHECKS — Table B.II baseline")
    print("=" * 78)

    # Baseline
    b, se, p, n = ols_with_yfe(df)
    print("\n[baseline] col II (country + year FE)")
    print_row("Center.Target", b, se, p, n)

    print("\n[1] Drop each country (leave-one-out) — worst case on Target")
    rows = []
    for c in df["Country"].unique():
        sub = df[df["Country"] != c]
        bb, ss, pp, nn = ols_with_yfe(sub)
        rows.append((c, bb, ss, pp, nn))
    rows.sort(key=lambda r: r[1])
    for c, bb, ss, pp, nn in rows[:3] + rows[-3:]:
        print_row(f"drop {c}", bb, ss, pp, nn)
    min_b = min(r[1] for r in rows)
    max_b = max(r[1] for r in rows)
    print(f"  Target coef range across LOO countries: [{min_b:.3f}, {max_b:.3f}]")

    print("\n[2] Drop high-inflation outliers (Turkey & Iceland)")
    sub = df[~df["Country"].isin(["Turkey", "Iceland"])]
    print_row("drop Turkey, Iceland", *ols_with_yfe(sub))

    print("\n[3] Drop global financial crisis period (2007-2009)")
    sub = df[~df["year"].between(2007, 2009)]
    print_row("drop 2007-2009", *ols_with_yfe(sub))

    print("\n[4] Restrict to post-GFC (2010-2019)")
    sub = df[df["year"] >= 2010]
    print_row("year >= 2010", *ols_with_yfe(sub))

    print("\n[5] Income split: above vs below median GDP/capita PPP")
    # Developed flag in database_paper.xlsx is unfortunately all zeros, so we
    # use country-mean GDP/capita PPP as a proxy for advanced vs emerging.
    gdp_mean = df.groupby("Country")["GDP.capita.PPP"].mean()
    med = gdp_mean.median()
    high = gdp_mean[gdp_mean >= med].index.tolist()
    low = gdp_mean[gdp_mean < med].index.tolist()
    print_row("high-income countries", *ols_with_yfe(df[df["Country"].isin(high)]))
    print_row("low-income countries",  *ols_with_yfe(df[df["Country"].isin(low)]))

    print("\n[6] Winsorise deviation at 5%/95% tails")
    w = df.copy()
    lo, hi = w["deviation"].quantile([0.05, 0.95])
    w["deviation"] = w["deviation"].clip(lo, hi)
    print_row("winsorised 5/95", *ols_with_yfe(w))

    print("\n[7] Cluster SEs by country")
    print_row("cluster(Country)", *ols_with_yfe(df, cluster="Country"))

    print("\n[8] Placebo outcome: Revenue (should not be driven by Target)")
    # Swap deviation with Revenue, drop Revenue from the RHS to avoid identity.
    d = df.dropna(subset=["Revenue"] + ["Gross.Debt", "DebtRev", "Center.Target"]).copy()
    parts = [d[["Gross.Debt", "DebtRev", "Center.Target"]].reset_index(drop=True)]
    parts.append(pd.get_dummies(d["Country"], drop_first=False, dtype=float).reset_index(drop=True))
    yrs = pd.get_dummies(d["year"], drop_first=True, dtype=float).reset_index(drop=True)
    yrs.columns = [f"year_{c}" for c in yrs.columns]
    parts.append(yrs)
    X = pd.concat(parts, axis=1)
    y = d["Revenue"].values
    m = sm.OLS(y, X.values).fit()
    ti = list(X.columns).index("Center.Target")
    print_row("placebo: Revenue ~ X", m.params[ti], m.bse[ti], m.pvalues[ti], int(m.nobs))

    print("\n[9] Permutation test: shuffle Target within country, 500 draws")
    rng = np.random.default_rng(2024)
    draws = 500
    shuffled_b = np.empty(draws)
    d = df.dropna(subset=["deviation"] + RHS_BASE).copy()
    for i in range(draws):
        d2 = d.copy()
        d2["Center.Target"] = d2.groupby("Country")["Center.Target"].transform(
            lambda s: rng.permutation(s.values)
        )
        d2["DebtRev"] = d2["Gross.Debt"] * d2["Revenue"] / 100
        bb, _, _, _ = ols_with_yfe(d2)
        shuffled_b[i] = bb
    p_perm = float(np.mean(np.abs(shuffled_b) >= abs(b)))
    print(f"  permutation p-value for |b| ≥ {abs(b):.3f}: {p_perm:.3f}")
    print(f"  shuffled-b mean={shuffled_b.mean():.3f}, sd={shuffled_b.std():.3f}")

    print("\n[10] Logit outcome: run the same placebo permutation on Overshoot_y")
    d = df.dropna(subset=["Overshoot_y"] + RHS_BASE).copy()
    rng = np.random.default_rng(2024)
    draws = 300
    shuffled_b = []
    for i in range(draws):
        d2 = d.copy()
        d2["Center.Target"] = d2.groupby("Country")["Center.Target"].transform(
            lambda s: rng.permutation(s.values)
        )
        d2["DebtRev"] = d2["Gross.Debt"] * d2["Revenue"] / 100
        # Fit GLM like fit_logit_fe
        parts = [d2[RHS_BASE].reset_index(drop=True)]
        parts.append(pd.get_dummies(d2["Country"], drop_first=False, dtype=float).reset_index(drop=True))
        yrs = pd.get_dummies(d2["year"], drop_first=True, dtype=float).reset_index(drop=True)
        yrs.columns = [f"year_{c}" for c in yrs.columns]
        parts.append(yrs)
        X = pd.concat(parts, axis=1)
        y = d2["Overshoot_y"].astype(float).values
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                m = sm.GLM(y, X.values, family=sm.families.Binomial()).fit(maxiter=100)
            ti = list(X.columns).index("Center.Target")
            shuffled_b.append(m.params[ti])
        except Exception:
            continue
    shuffled_b = np.array(shuffled_b)
    # True observed logit col II from Table B.III
    actual_b = -1.242
    p_perm = float(np.mean(np.abs(shuffled_b) >= abs(actual_b)))
    print(f"  permutation p-value for |b| ≥ {abs(actual_b):.3f}: {p_perm:.3f}")
    print(f"  draws used: {len(shuffled_b)}, mean={shuffled_b.mean():.3f}, sd={shuffled_b.std():.3f}")


if __name__ == "__main__":
    main()
