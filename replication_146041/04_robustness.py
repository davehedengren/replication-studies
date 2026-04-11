"""Robustness checks on the Table 1/2 elasticity of relative skill efficiency
(irAQ53_hrs) wrt log GDP p.w. All checks operate on the 12-country micro sample
at year==2000 unless noted. The published headline elasticity is 1.408 (SE 0.394).
"""
import numpy as np
import pandas as pd

from utils import (
    load_calib, compute_wage_ratios, compute_H5L3, compute_irAQ,
    ols_log_elasticity, SIGMA, SIGMA_HIGH, SIGMA_LOW, OUT,
)


def prep():
    df = load_calib()
    df = compute_wage_ratios(df, "dum_skti_secall")
    H5, L3, H5L3 = compute_H5L3(df, "dum_skti_secall", "hrs_secall")
    df["H5L3_hrs"] = H5L3
    us = (df["country"] == "United States") & (df["year"] == 2000)
    for s, name in [(SIGMA, "irAQ53_hrs"), (SIGMA_LOW, "irAQ53rl_hrs"), (SIGMA_HIGH, "irAQ53rh_hrs")]:
        df[name] = compute_irAQ(df, df["wrat53_dum_skti_secall"], H5L3, s, us)
    base = df[(df["sample_micro"] == 1) & (df["year"] == 2000) & df["irAQ53_hrs"].notna()].copy()
    return df, base


def leave_one_out(base, col):
    outs = []
    for c in base["country"].tolist():
        sub = base[base["country"] != c]
        b, se, _ = ols_log_elasticity(sub, col)
        outs.append((c, b, se))
    return pd.DataFrame(outs, columns=["dropped", "coef", "se"])


def placebo_permutation(base, col, n_perm=5000, seed=0):
    rng = np.random.default_rng(seed)
    y = np.log(base[col].values)
    x0 = base["l_y"].values
    X = np.column_stack([np.ones(len(x0)), x0])
    beta_obs = np.linalg.lstsq(X, y, rcond=None)[0][1]
    n = len(y)
    perm_betas = np.empty(n_perm)
    for i in range(n_perm):
        xi = rng.permutation(x0)
        Xi = np.column_stack([np.ones(n), xi])
        perm_betas[i] = np.linalg.lstsq(Xi, y, rcond=None)[0][1]
    # two-sided p-value
    p = (np.abs(perm_betas) >= abs(beta_obs)).mean()
    return beta_obs, p, perm_betas


def winsorize(series, lo=0.1, hi=0.9):
    q1, q2 = series.quantile(lo), series.quantile(hi)
    return series.clip(q1, q2)


def main():
    df, base = prep()
    target_col = "irAQ53_hrs"

    print("=" * 70)
    print("Baseline elasticity (for reference)")
    print("=" * 70)
    b, se, n = ols_log_elasticity(base, target_col)
    print(f"  irAQ53_hrs ~ l_y:  coef={b:.3f}  se={se:.3f}  n={n}  (published 1.408 [0.394])")

    print("\n" + "=" * 70)
    print("1. Leave-one-country-out sensitivity")
    print("=" * 70)
    loo = leave_one_out(base, target_col)
    print(loo.to_string(index=False))
    print(f"  range [{loo['coef'].min():.3f}, {loo['coef'].max():.3f}]  "
          f"(most-influential drop moves it by {max(abs(loo['coef']-b)):.3f})")
    loo.to_csv(OUT / "rob_loo.csv", index=False)

    print("\n" + "=" * 70)
    print("2. Drop US as reference country")
    print("=" * 70)
    sub = base[base["country"] != "United States"]
    b2, se2, n2 = ols_log_elasticity(sub, target_col)
    print(f"  coef={b2:.3f}  se={se2:.3f}  n={n2}")

    print("\n" + "=" * 70)
    print("3. Drop Canada (second-richest country in sample)")
    print("=" * 70)
    sub = base[base["country"] != "Canada"]
    b3, se3, n3 = ols_log_elasticity(sub, target_col)
    print(f"  coef={b3:.3f}  se={se3:.3f}  n={n3}")

    print("\n" + "=" * 70)
    print("4. Drop India (poorest country in sample)")
    print("=" * 70)
    sub = base[base["country"] != "India"]
    b4, se4, n4 = ols_log_elasticity(sub, target_col)
    print(f"  coef={b4:.3f}  se={se4:.3f}  n={n4}")

    print("\n" + "=" * 70)
    print("5. HC1 vs HC3 heteroskedasticity-robust SEs (statsmodels)")
    print("=" * 70)
    import statsmodels.api as sm
    y = np.log(base[target_col].values)
    X = sm.add_constant(base["l_y"].values)
    for cov in ["HC0", "HC1", "HC2", "HC3"]:
        res = sm.OLS(y, X).fit(cov_type=cov)
        print(f"  {cov}:  coef={res.params[1]:.3f}  se={res.bse[1]:.3f}")

    print("\n" + "=" * 70)
    print("6. Alternative σ values")
    print("=" * 70)
    for col, label in [("irAQ53rl_hrs", "σ=1.3"), ("irAQ53_hrs", "σ=1.5"), ("irAQ53rh_hrs", "σ=2.0")]:
        b6, se6, n6 = ols_log_elasticity(base, col)
        print(f"  {label:>8}:  coef={b6:.3f}  se={se6:.3f}")

    print("\n" + "=" * 70)
    print("7. Winsorize irAQ at 10/90 quantiles")
    print("=" * 70)
    w = winsorize(base[target_col])
    wb = base.assign(**{target_col + "_w": w})
    bw, sew, nw = ols_log_elasticity(wb, target_col + "_w")
    print(f"  coef={bw:.3f}  se={sew:.3f}")

    print("\n" + "=" * 70)
    print("8. Placebo permutation test (two-sided p-value, 5000 draws)")
    print("=" * 70)
    bp, p, _ = placebo_permutation(base, target_col, 5000, seed=42)
    print(f"  observed β={bp:.3f}, permutation p={p:.4f}")

    print("\n" + "=" * 70)
    print("9. Placebo outcome: w5/w3 (skill premium itself) — should be ~ -0.14, not large")
    print("=" * 70)
    bp2, se9, _ = ols_log_elasticity(base, "wrat53_dum_skti_secall")
    print(f"  coef={bp2:.3f}  se={se9:.3f}  (paper: -0.138)")

    print("\n" + "=" * 70)
    print("10. Using 'bodies' weights vs 'hours' weights (measurement choice)")
    print("=" * 70)
    H5b, L3b, H5L3b = compute_H5L3(df, "dum_skti_secall", "bod_secall")
    us_mask = (df["country"] == "United States") & (df["year"] == 2000)
    df["irAQ_bod"] = compute_irAQ(df, df["wrat53_dum_skti_secall"], H5L3b, SIGMA, us_mask)
    base2 = df[(df["sample_micro"] == 1) & (df["year"] == 2000) & df["irAQ_bod"].notna()]
    bb, seb, nb = ols_log_elasticity(base2, "irAQ_bod")
    print(f"  bodies: coef={bb:.3f}  se={seb:.3f}  n={nb}")

    print("\n" + "=" * 70)
    print("11. Restrict to high-income-dispersion trio only (India/Indonesia/US/Canada)")
    print("=" * 70)
    sub = base[base["country"].isin(["India", "Indonesia", "United States", "Canada"])]
    b11, se11, n11 = ols_log_elasticity(sub, target_col)
    print(f"  coef={b11:.3f}  se={se11:.3f}  n={n11}")

    print("\n" + "=" * 70)
    print("12. Alternative year: use 2010 calibration data where available")
    print("=" * 70)
    # find countries with year==2010 calibration rows
    df10 = df[(df["sample_micro"] == 1) & (df["year"] == 2010) & df["irAQ53_hrs"].notna()]
    print(f"  countries at 2010: {sorted(df10['country'].tolist())}")
    if len(df10) >= 4:
        b12, se12, n12 = ols_log_elasticity(df10, target_col)
        print(f"  coef={b12:.3f}  se={se12:.3f}  n={n12}")
    else:
        print("  not enough countries in 2010 subset; skipped.")


if __name__ == "__main__":
    main()
