"""Replicate the headline tables from Henry, Zhuravskaya & Guriev (AER 2022).

Targets:
  - Text descriptive numbers: 14.7% / 10.2% / 10.8% sharing rates
  - Table 2 col 1: OLS want_share_fb ~ C(survey), robust SEs, reference=Alt-Facts
  - Table 3 col 1: OLS want_share_facts ~ C(survey), survey in {2,3}, robust SEs
"""
import numpy as np
import pandas as pd
import statsmodels.api as sm

from replication_140161.utils import build_all, OUT, TREATMENT_LABELS


def robust_ols(y, X):
    """OLS with HC1 robust SEs (Stata's `, r`)."""
    Xc = sm.add_constant(X, has_constant="add")
    return sm.OLS(y, Xc, missing="drop").fit(cov_type="HC1")


def main():
    df = build_all()

    print("=" * 72)
    print("Sample construction (mirrors 1.infile_data.do Wave 1 filters)")
    print("=" * 72)
    print(df.groupby("survey").size().rename("N"))
    print(f"Total Wave 1 observations: {len(df)}")

    print()
    print("=" * 72)
    print("Descriptive: share of participants who said 'yes' to sharing alt-facts")
    print("=" * 72)
    table1 = (df.groupby("survey")["want_share_fb"]
                .agg(["mean", "count"])
                .rename(index=TREATMENT_LABELS))
    table1["published"] = [0.147, 0.102, 0.108]
    table1["diff"] = table1["mean"] - table1["published"]
    print(table1.round(4))

    print()
    print("=" * 72)
    print("Table 2, column 1 — ATE on sharing alt-facts (robust SEs)")
    print("=" * 72)
    X1 = df[["imposed", "voluntary"]]
    m1 = robust_ols(df["want_share_fb"], X1)
    summary = pd.DataFrame({
        "coef": m1.params,
        "se": m1.bse,
        "t": m1.tvalues,
        "p": m1.pvalues,
    })
    # Published: imposed -0.045 (0.016), voluntary -0.038 (0.016), N=2537, R² 0.004,
    # mean DV control = 0.147
    summary["published_coef"] = ["0.147", "-0.045", "-0.038"]
    summary["published_se"]   = ["(0.007)", "(0.016)", "(0.016)"]
    print(summary.round(4))
    print(f"N = {int(m1.nobs)}  |  R² = {m1.rsquared:.4f}  "
          f"(published N=2537, R²=0.004)")
    print(f"Mean DV in Alt-Facts (reference) = {m1.params['const']:.4f}  "
          "(published 0.147)")

    print()
    print("=" * 72)
    print("Table 3, column 1 — ATE on sharing fact-check (robust SEs)")
    print("=" * 72)
    sub = df[df["survey"].isin([2, 3])].copy()
    X3 = sub[["voluntary"]]
    m3 = robust_ols(sub["want_share_facts"], X3)
    summary3 = pd.DataFrame({
        "coef": m3.params,
        "se": m3.bse,
        "t": m3.tvalues,
        "p": m3.pvalues,
    })
    summary3["published_coef"] = ["0.143", "-0.028"]
    summary3["published_se"]   = ["(0.012)", "(0.016)"]
    print(summary3.round(4))
    print(f"N = {int(m3.nobs)}  |  R² = {m3.rsquared:.4f}  "
          f"(published N=1692, R²=0.002)")
    print(f"Mean DV in Imposed FC (reference) = {m3.params['const']:.4f}  "
          "(published 0.143)")

    # save a tidy CSV of the Table 2 col 1 results for later re-use
    summary.to_csv(OUT / "table2_col1.csv")
    summary3.to_csv(OUT / "table3_col1.csv")
    print()
    print(f"Wrote: {OUT/'table2_col1.csv'}")
    print(f"Wrote: {OUT/'table3_col1.csv'}")


if __name__ == "__main__":
    main()
