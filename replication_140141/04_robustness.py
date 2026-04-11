"""Phase 4: robustness checks for Table 1 headline result
(effect of share_65plus on services VA / employment share)."""
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from utils import OUT


def areg(df, y, extras):
    formula = f"{y} ~ " + " + ".join(extras + ["C(code)"])
    return smf.ols(formula, data=df).fit(
        cov_type="cluster", cov_kwds={"groups": df["code"]})


def b_se(m, name):
    return m.params[name], m.bse[name]


def main():
    df = pd.read_parquet(OUT / "macro_panel.parquet").dropna(
        subset=["va_share_agr", "share_65plus", "log_gdp_pc", "log_gdp_pc2"])
    extras_full = ["log_gdp_pc", "log_gdp_pc2", "share_65plus"]

    print("=" * 78)
    print("ROBUSTNESS: β on share_65plus for va_share_ser (Table 1 col 6, Panel B)")
    print("Published baseline: β = +1.309 (SE 0.352), N=707")
    print("=" * 78)
    rows = []

    # 0. Replicated baseline
    m = areg(df, "va_share_ser", extras_full)
    b, se = b_se(m, "share_65plus")
    rows.append(("0. Baseline (full sample)", b, se, int(m.nobs)))

    # 1. Add year FEs
    df1 = df.copy()
    m = smf.ols("va_share_ser ~ share_65plus + log_gdp_pc + log_gdp_pc2 + C(code) + C(year)",
                data=df1).fit(cov_type="cluster",
                              cov_kwds={"groups": df1["code"]})
    rows.append(("1. + year fixed effects", *b_se(m, "share_65plus"), int(m.nobs)))

    # 2. Drop USA
    d = df[df["code"] != "USA"]
    m = areg(d, "va_share_ser", extras_full)
    rows.append(("2. Drop USA", *b_se(m, "share_65plus"), int(m.nobs)))

    # 3. Drop Luxembourg (VA_ser outlier)
    d = df[df["code"] != "LUX"]
    m = areg(d, "va_share_ser", extras_full)
    rows.append(("3. Drop Luxembourg", *b_se(m, "share_65plus"), int(m.nobs)))

    # 4. Drop Japan + Korea (East Asian catch-up)
    d = df[~df["code"].isin(["JPN", "KOR"])]
    m = areg(d, "va_share_ser", extras_full)
    rows.append(("4. Drop JPN and KOR", *b_se(m, "share_65plus"), int(m.nobs)))

    # 5. Post-1985 only
    d = df[df["year"] >= 1985]
    m = areg(d, "va_share_ser", extras_full)
    rows.append(("5. Post-1985 subsample", *b_se(m, "share_65plus"), int(m.nobs)))

    # 6. Pre-1990 only
    d = df[df["year"] < 1990]
    m = areg(d, "va_share_ser", extras_full)
    rows.append(("6. Pre-1990 subsample", *b_se(m, "share_65plus"), int(m.nobs)))

    # 7. Linear income control only
    m = areg(df, "va_share_ser", ["log_gdp_pc", "share_65plus"])
    rows.append(("7. Linear log-GDP-pc only", *b_se(m, "share_65plus"), int(m.nobs)))

    # 8. Winsorize va_share_ser 1%/99%
    d = df.copy()
    lo, hi = d["va_share_ser"].quantile([0.01, 0.99])
    d["va_share_ser"] = d["va_share_ser"].clip(lo, hi)
    m = areg(d, "va_share_ser", extras_full)
    rows.append(("8. Winsorize 1/99%", *b_se(m, "share_65plus"), int(m.nobs)))

    # 9. HC1 (no cluster)
    m = smf.ols("va_share_ser ~ share_65plus + log_gdp_pc + log_gdp_pc2 + C(code)",
                data=df).fit(cov_type="HC1")
    rows.append(("9. HC1 instead of cluster", *b_se(m, "share_65plus"), int(m.nobs)))

    # 10. Placebo: shuffle share_65plus within years, see if coef collapses
    rng = np.random.default_rng(20240101)
    d = df.copy()
    d["share_65plus"] = (d.groupby("year")["share_65plus"]
                           .transform(lambda x: rng.permutation(x.values)))
    m = areg(d, "va_share_ser", extras_full)
    rows.append(("10. Placebo (shuffle share_65plus)",
                 *b_se(m, "share_65plus"), int(m.nobs)))

    # 11. Permutation p-value (1000 draws, sign of coefficient)
    rng = np.random.default_rng(7)
    base_b = rows[0][1]
    ds = []
    for i in range(500):
        d = df.copy()
        d["share_65plus"] = (d.groupby("code")["share_65plus"]
                               .transform(lambda x: rng.permutation(x.values)))
        m = areg(d, "va_share_ser", extras_full)
        ds.append(m.params["share_65plus"])
    ds = np.array(ds)
    pval = (np.abs(ds) >= abs(base_b)).mean()
    print(f"\n[Placebo permutation] within-country shuffle: "
          f"empirical two-sided p-value = {pval:.4f}  "
          f"(mean null β={ds.mean():+.3f}, sd={ds.std():.3f})")

    # 12. Outcome placebo: regress trade (WDI control) on aging — should be near zero
    d = df.dropna(subset=["trade"])
    m = areg(d, "trade", extras_full)
    rows.append(("12. Placebo outcome: trade/GDP",
                 *b_se(m, "share_65plus"), int(m.nobs)))

    print(f"\n{'Spec':<40} {'β':>10} {'SE':>8} {'N':>6}")
    print("-" * 70)
    for name, b, se, n in rows:
        print(f"{name:<40} {b:+10.3f} {se:8.3f} {n:6d}")


if __name__ == "__main__":
    main()
