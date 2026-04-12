"""Data audit for the Araujo et al. inflation-targeting panel (Appendix B).

Checks:
 1. Coverage: countries, years, observations per cell
 2. Plausibility bounds on macro variables
 3. Missing data by country and variable
 4. Panel balance and gaps
 5. Duplicate country-year cells
 6. Overshoot consistency (vs pre-computed `Overshoot` column)
 7. Target changes within countries
 8. GDP gap sanity vs the precomputed sheet
 9. STL vs X-13 GDP-gap sensitivity (since we substitute for R's `seas`)
"""
import os
import numpy as np
import pandas as pd

from utils import OUT, load_database, build_overshoot, build_gdp_gap_stl, build_gdp_gap_precomputed


def main():
    df = pd.read_csv(os.path.join(OUT, "panel.csv"))
    print("=" * 70)
    print("DATA AUDIT — replication 216641")
    print("=" * 70)

    # 1. Coverage
    print("\n[1] Coverage")
    print(f"  rows: {len(df)}, countries: {df['Country'].nunique()}, "
          f"year span: {int(df['year'].min())}–{int(df['year'].max())}")
    balance = df.groupby("Country")["year"].agg(["min", "max", "count"])
    print(f"  min years/country: {balance['count'].min()}, "
          f"max: {balance['count'].max()}")
    if (balance["count"] != 20).any():
        print("  countries with <20 years of data:")
        print(balance[balance["count"] != 20].to_string())
    else:
        print("  all 23 countries have exactly 20 yearly rows (balanced)")

    # 2. Plausibility bounds
    print("\n[2] Plausibility bounds on key variables")
    bounds = {
        "Gross.Debt":     (0, 250),
        "Revenue":        (0, 80),
        "CPI":            (-5, 100),
        "Center.Target":  (0, 30),
        "deviation":      (-10, 100),
        "GDP.Gap":        (-30, 30),
        "Reer.YoY":       (-1, 1),
    }
    for v, (lo, hi) in bounds.items():
        s = df[v].dropna()
        bad = s[(s < lo) | (s > hi)]
        print(f"  {v:14s}: n={len(s):3d}  min={s.min():8.3f}  max={s.max():8.3f}  "
              f"out-of-bounds={len(bad)}")

    # 3. Missing-data pattern
    print("\n[3] Missing-data counts by variable (out of 460 rows)")
    for v in ["Gross.Debt", "Revenue", "Center.Target", "CPI", "Reer.YoY",
              "GDP.Gap", "Overshoot_y"]:
        print(f"  {v:14s}: {df[v].isna().sum()} missing")

    # 4. Countries with systematic missingness
    print("\n[4] Countries dropped from Tables B.II/III (no Revenue coverage)")
    dropped = df.groupby("Country")["Revenue"].apply(lambda s: s.isna().all())
    for c in dropped[dropped].index:
        print(f"  - {c}")

    # 5. Duplicates
    print("\n[5] Duplicate (Country, year) rows")
    dups = df.duplicated(["Country", "year"]).sum()
    print(f"  {dups} duplicates")

    # 6. Overshoot consistency: pre-computed vs rebuilt
    print("\n[6] Overshoot indicator: original column vs rebuilt from target_data")
    orig = load_database()[["Country", "year", "Overshoot"]]
    rebuilt, _ = build_overshoot()
    merged = orig.merge(rebuilt, on=["Country", "year"], how="outer")
    agree = (merged["Overshoot"] == merged["Overshoot_y"]).sum()
    disagree_mask = (
        merged["Overshoot"].notna() & merged["Overshoot_y"].notna()
        & (merged["Overshoot"] != merged["Overshoot_y"])
    )
    print(f"  identical where both present: {agree} / {merged.notna().all(axis=1).sum()}")
    print(f"  disagreements where both present: {disagree_mask.sum()}")
    if disagree_mask.sum():
        print(merged[disagree_mask].head(10).to_string())

    # 7. Target changes
    print("\n[7] Countries that changed their inflation target at least once")
    changed = df.groupby("Country")["Center.Target"].nunique()
    changed = changed[changed > 1].sort_values(ascending=False)
    print(f"  {len(changed)} / {df['Country'].nunique()} countries "
          f"({100*len(changed)/df['Country'].nunique():.0f}%)")
    print(f"  paper reports 55% for the 20-country sample; we see "
          f"{100*len(changed)/df['Country'].nunique():.0f}% across all 23")

    # 8. GDP gap vs precomputed
    print("\n[8] GDP gap sanity: overall mean & sd (precomputed sheet)")
    g = df["GDP.Gap"].dropna()
    print(f"  n={len(g)}, mean={g.mean():.3f}, sd={g.std():.3f}")
    print("  (mean of an HP-filter cycle component should be ≈0)")

    # 9. STL-based GDP-gap reconstruction for comparison with X-13-based precomputed
    print("\n[9] STL vs precomputed GDP gap (correlation by country)")
    stl = build_gdp_gap_stl()
    pre = build_gdp_gap_precomputed()
    merged_gap = stl.merge(pre, on=["year", "Country"], suffixes=("_stl", "_pre"))
    per = merged_gap.dropna().groupby("Country").apply(
        lambda d: d["GDP.Gap_stl"].corr(d["GDP.Gap_pre"])
    )
    print(f"  median corr = {per.median():.3f}, min = {per.min():.3f}, max = {per.max():.3f}")
    low = per[per < 0.9]
    if len(low):
        print("  countries where STL substitute disagrees with X-13 precompute:")
        for c, v in low.items():
            print(f"    {c}: {v:.3f}")


if __name__ == "__main__":
    main()
