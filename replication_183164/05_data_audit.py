"""Data audit for the replication samples used by Antman, Duncan, Trejo (2023).

Checks coverage, distributions, missing patterns, and basic plausibility for:
  * 2019 ACS Table 1-2 sample (1.21M obs),
  * Figure 1 US-born men sample (1970-2019, 8.8M obs),
  * CPS 2003-2019 analytic sample (1.47M obs).
"""
import numpy as np
import pandas as pd
from utils import OUT


def line(s=""):
    print(s)


def audit_census_2019():
    df = pd.read_parquet(OUT / "census_2019.parquet")
    line("=" * 70)
    line("AUDIT 1: 2019 ACS sample used for Tables 1 and 2")
    line("=" * 70)
    line(f"N = {len(df):,}")
    line(f"Male / female split: {(df['sex']==1).sum():,} / {(df['sex']==2).sum():,}")
    by = df.groupby("race_cat4").size()
    labels = {1: "Foreign-born Hispanic", 2: "US-born Hispanic",
              3: "US-born non-Hisp White", 4: "US-born non-Hisp Black"}
    line("\nrace_cat4 counts:")
    for k, v in by.items():
        line(f"  {labels.get(int(k), int(k)):<28s} {v:>10,}")
    line("\nAge summary:")
    line(f"  min/max = {df['age'].min()}/{df['age'].max()}")
    line(f"  mean    = {df['age'].mean():.2f}")
    line("\nedyrs summary (unweighted):")
    desc = df["edyrs"].describe()
    for k in ["count", "mean", "std", "min", "25%", "50%", "75%", "max"]:
        line(f"  {k:<6s} = {desc[k]:.3f}")
    line(f"\nmissing edyrs   = {df['edyrs'].isna().sum():,}")
    line(f"ln_earnings NaN = {df['ln_annualearnings'].isna().sum():,}  "
         f"(share of {len(df):,} = {df['ln_annualearnings'].isna().mean():.3%})")
    line("  — NaN = zero annual earnings; excluded from Table 2 by construction.")
    # Positivity check
    neg = (df["annualearnings"] < 0).sum()
    line(f"Negative annual earnings rows: {neg:,}")
    # State coverage
    line(f"statefip unique values: {df['statefip'].nunique()}")


def audit_figure1():
    df = pd.read_parquet(OUT / "census_figure1.parquet")
    df = df[df["sex"] == 1].copy()
    line("\n" + "=" * 70)
    line("AUDIT 2: 1970-2019 US-born men sample (Figure 1)")
    line("=" * 70)
    line(f"N = {len(df):,}")
    counts = df.groupby("year").size()
    line("\nObservations per census/ACS year:")
    for yr, n in counts.items():
        line(f"  {int(yr)}: {n:>10,}")
    line("\nrace_cat6 shares by year (unweighted):")
    g = df.groupby(["year", "race_cat6"]).size().unstack(fill_value=0)
    shares = g.div(g.sum(axis=1), axis=0)
    line(shares.to_string(float_format=lambda x: f"{x:.3f}"))
    # Earnings coverage
    ln_na = df.groupby("year")["ln_annualearnings"].apply(lambda s: s.isna().mean())
    line("\nShare with zero earnings (NaN ln-earnings), by year:")
    for yr, v in ln_na.items():
        line(f"  {int(yr)}: {v:.3%}")


def audit_cps():
    df = pd.read_parquet(OUT / "cps_clean.parquet")
    line("\n" + "=" * 70)
    line("AUDIT 3: CPS 2003-2019 analytic sample (Table 3)")
    line("=" * 70)
    line(f"N = {len(df):,}")
    line("\nObservations per year:")
    for y, n in df.groupby("year").size().items():
        line(f"  {int(y)}: {n:>8,}")
    line("\ngen_org_cat12 counts (should be 1..9 only):")
    line(df["gen_org_cat12"].value_counts().sort_index().to_string())
    line("\ngen_org_cat15 counts (should be 1..9,13..15 only):")
    line(df["gen_org_cat15"].value_counts().sort_index().to_string())
    line(f"\nedyrs range: {df['edyrs'].min()} - {df['edyrs'].max()}")
    line(f"edyrs NaN: {df['edyrs'].isna().sum()}")
    # check that all respondents are 4th rotation group + hhintype 1 (already pre-filtered)
    line("\nSanity: age bounds should be 25..59")
    line(f"  min/max age = {df['age'].min()}/{df['age'].max()}")


def main():
    audit_census_2019()
    audit_figure1()
    audit_cps()
    line("\n[05] done.")


if __name__ == "__main__":
    main()
