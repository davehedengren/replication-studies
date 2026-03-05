"""
04_data_audit.py — Data quality checks for the Dads & Daughters replication.

Checks coverage, distributions, logical consistency, missing patterns,
duplicates, and coding anomalies.
"""

import sys, os
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
from utils import ANALYSIS_DATA_PATH, RAW_DATA_DIR


def main():
    print("=" * 60)
    print("04_data_audit.py: Data Quality Audit")
    print("=" * 60)

    df = pd.read_parquet(ANALYSIS_DATA_PATH)

    # Also load raw data for cross-checks
    bl_raw = pd.read_stata(os.path.join(RAW_DATA_DIR, 'product_pricing_hh_final.dta'),
                           convert_categoricals=False)
    fu_raw = pd.read_stata(os.path.join(RAW_DATA_DIR, 'product_pricing_fu.dta'),
                           convert_categoricals=False)
    rand_raw = pd.read_stata(os.path.join(RAW_DATA_DIR, 'randomized_data.dta'),
                             convert_categoricals=False)

    issues = []

    # ================================================================
    # CHECK 1: Coverage
    # ================================================================
    print("\n--- CHECK 1: Coverage ---")
    print(f"  Total obs: {len(df)}")
    print(f"  Unique households: {df['hhid'].nunique()}")
    print(f"  Unique goods: {df['good'].nunique()}")
    print(f"  Goods: {sorted(df['good'].unique())}")

    print(f"\n  Obs per good:")
    for g in sorted(df['good'].unique()):
        n = (df['good'] == g).sum()
        print(f"    {g:15s}  N={n:5d}")

    print(f"\n  Wave breakdown:")
    print(f"    BL (fu=0): {(df['fu']==0).sum()}")
    print(f"    FU (fu=1): {(df['fu']==1).sum()}")

    main_sample = df[(df['childgood'] == 1) & (df['toys_bin'] == 0)]
    print(f"\n  Main analysis sample: {len(main_sample)} obs")
    print(f"    With all key vars: {main_sample.dropna(subset=['wtp_std','girl','mom','momXgirl']).shape[0]}")

    # ================================================================
    # CHECK 2: Variable distributions
    # ================================================================
    print("\n--- CHECK 2: Variable Distributions ---")

    cont_vars = ['wtp_raw', 'wtp', 'wtp_capped', 'wtp_std', 'adult_wtp_std']
    for v in cont_vars:
        if v in df.columns:
            vals = df[v].dropna()
            print(f"\n  {v}:")
            print(f"    N={len(vals)}, mean={vals.mean():.4f}, sd={vals.std():.4f}")
            print(f"    min={vals.min():.4f}, p25={vals.quantile(0.25):.4f}, "
                  f"med={vals.median():.4f}, p75={vals.quantile(0.75):.4f}, max={vals.max():.4f}")

            # Check for extreme outliers (>4 IQR from median)
            iqr = vals.quantile(0.75) - vals.quantile(0.25)
            lower = vals.quantile(0.25) - 4 * iqr
            upper = vals.quantile(0.75) + 4 * iqr
            n_extreme = ((vals < lower) | (vals > upper)).sum()
            if n_extreme > 0:
                pct = 100 * n_extreme / len(vals)
                print(f"    EXTREME OUTLIERS (4*IQR): {n_extreme} ({pct:.2f}%)")
                if pct > 5:
                    issues.append(f"High outlier rate in {v}: {n_extreme} ({pct:.1f}%)")

    # ================================================================
    # CHECK 3: Logical consistency
    # ================================================================
    print("\n--- CHECK 3: Logical Consistency ---")

    # WTP should be between 0 and 1 (fraction of market price)
    if 'wtp' in df.columns:
        n_neg = (df['wtp'] < 0).sum()
        n_over1 = (df['wtp'] > 1).sum()
        print(f"  wtp < 0: {n_neg} obs")
        print(f"  wtp > 1: {n_over1} obs (before capping, should be 0 after)")
        if n_neg > 0:
            issues.append(f"Found {n_neg} obs with negative WTP")

    # Binary variables should be 0/1
    for v in ['girl', 'mom', 'childgood', 'humcap', 'health', 'hypo', 'young', 'toys_bin', 'fu']:
        if v in df.columns:
            vals = df[v].dropna()
            non_binary = vals[~vals.isin([0, 1])]
            if len(non_binary) > 0:
                print(f"  WARNING: {v} has {len(non_binary)} non-binary values")
                issues.append(f"{v} has non-binary values")
            else:
                print(f"  {v}: valid binary (N={len(vals)}, mean={vals.mean():.3f})")

    # momXgirl should equal mom * girl
    if all(v in df.columns for v in ['momXgirl', 'mom', 'girl']):
        valid = df[['mom', 'girl', 'momXgirl']].dropna()
        expected = valid['mom'] * valid['girl']
        mismatch = (valid['momXgirl'] != expected).sum()
        print(f"  momXgirl == mom * girl: {len(valid) - mismatch}/{len(valid)} match")
        if mismatch > 0:
            issues.append(f"momXgirl != mom*girl for {mismatch} obs")

    # child goods categorization: childgood should include humcap, enjoyment, toy goods
    child_goods_check = df[df['childgood'] == 1]['good'].unique()
    non_child_goods = df[df['childgood'] == 0]['good'].unique()
    print(f"  Child goods: {sorted(child_goods_check)}")
    print(f"  Non-child goods: {sorted(non_child_goods)}")

    # ================================================================
    # CHECK 4: Missing data patterns
    # ================================================================
    print("\n--- CHECK 4: Missing Data Patterns ---")

    key_vars = ['wtp_std', 'girl', 'mom', 'momXgirl', 'adult_wtp_std', 'strat']
    for v in key_vars:
        n_miss = df[v].isna().sum()
        pct = 100 * n_miss / len(df)
        print(f"  {v:20s}  missing: {n_miss:5d} ({pct:5.1f}%)")

    # Missing by parent gender
    print(f"\n  Missing 'girl' by parent gender:")
    for mom_val in [0, 1]:
        sub = df[df['mom'] == mom_val]
        n_miss = sub['girl'].isna().sum()
        print(f"    mom={mom_val}: {n_miss}/{len(sub)} missing ({100*n_miss/len(sub):.1f}%)")

    # Missing by wave
    print(f"\n  Missing 'girl' by wave:")
    for fu_val in [0, 1]:
        sub = df[df['fu'] == fu_val]
        n_miss = sub['girl'].isna().sum()
        print(f"    fu={fu_val}: {n_miss}/{len(sub)} missing ({100*n_miss/len(sub):.1f}%)")

    # Missing girl by good type
    print(f"\n  Missing 'girl' by good:")
    for g in sorted(df['good'].unique()):
        sub = df[df['good'] == g]
        n_miss = sub['girl'].isna().sum()
        print(f"    {g:15s}: {n_miss}/{len(sub)} ({100*n_miss/len(sub):.1f}%)")

    # ================================================================
    # CHECK 5: Duplicates
    # ================================================================
    print("\n--- CHECK 5: Duplicates ---")

    # Each HH should have at most one obs per good
    dups = df.groupby(['hhid', 'good']).size()
    n_dups = (dups > 1).sum()
    print(f"  Duplicate HH-good combinations: {n_dups}")
    if n_dups > 0:
        issues.append(f"Found {n_dups} duplicate HH-good rows")

    # ================================================================
    # CHECK 6: WTP distributions by group
    # ================================================================
    print("\n--- CHECK 6: WTP by Group ---")

    main = df[(df['childgood'] == 1) & (df['toys_bin'] == 0)].dropna(subset=['girl', 'mom', 'wtp_std'])

    for label, mask in [
        ('Father-Son', (main['mom'] == 0) & (main['girl'] == 0)),
        ('Father-Daughter', (main['mom'] == 0) & (main['girl'] == 1)),
        ('Mother-Son', (main['mom'] == 1) & (main['girl'] == 0)),
        ('Mother-Daughter', (main['mom'] == 1) & (main['girl'] == 1)),
    ]:
        sub = main[mask]
        print(f"  {label:20s}  N={len(sub):5d}  mean_wtp_std={sub['wtp_std'].mean():.3f}  "
              f"sd={sub['wtp_std'].std():.3f}")

    # ================================================================
    # CHECK 7: Stratum balance
    # ================================================================
    print("\n--- CHECK 7: Stratum Balance ---")

    strat_counts = df.groupby('strat')['hhid'].nunique()
    print(f"  Number of strata: {len(strat_counts)}")
    print(f"  HH per stratum: mean={strat_counts.mean():.1f}, min={strat_counts.min()}, max={strat_counts.max()}")

    # Check for singleton strata
    singletons = (strat_counts == 1).sum()
    if singletons > 0:
        print(f"  WARNING: {singletons} strata with only 1 HH")
        issues.append(f"{singletons} singleton strata")

    # ================================================================
    # CHECK 8: Attrition from BL to FU
    # ================================================================
    print("\n--- CHECK 8: Attrition ---")

    bl_hh = df[df['fu'] == 0]['hhid'].unique()
    fu_hh = df[df['fu'] == 1]['hhid'].unique()
    print(f"  BL households: {len(bl_hh)}")
    print(f"  FU households: {len(fu_hh)}")
    attrited = set(bl_hh) - set(fu_hh)
    print(f"  Attrited: {len(attrited)} ({100*len(attrited)/len(bl_hh):.1f}%)")

    # Check if attrition is correlated with parent gender
    bl_data = df[df['fu'] == 0].drop_duplicates('hhid')[['hhid', 'mom', 'girl', 'strat']].copy()
    bl_data['in_fu'] = bl_data['hhid'].isin(fu_hh).astype(int)

    for var in ['mom', 'girl']:
        if var in bl_data.columns:
            sub = bl_data.dropna(subset=[var])
            g0 = sub[sub[var] == 0]['in_fu'].mean()
            g1 = sub[sub[var] == 1]['in_fu'].mean()
            print(f"  FU retention by {var}: {var}=0: {g0:.3f}, {var}=1: {g1:.3f}, diff: {g1-g0:+.3f}")

    # ================================================================
    # CHECK 9: WTP heaping
    # ================================================================
    print("\n--- CHECK 9: WTP Heaping ---")

    # Check for excessive concentration at specific WTP values
    for g in ['test', 'cup', 'Ftest', 'Fjerry']:
        sub = df[df['good'] == g]
        if len(sub) == 0:
            continue
        top5 = sub['wtp_raw'].value_counts(normalize=True).head(5)
        concentration = top5.sum()
        print(f"  {g}: top 5 WTP values account for {100*concentration:.1f}% of obs")
        if concentration > 0.8:
            issues.append(f"Extreme WTP heaping in {g}: top 5 values = {100*concentration:.1f}%")

    # ================================================================
    # CHECK 10: Raw data cross-check
    # ================================================================
    print("\n--- CHECK 10: Raw Data Cross-Check ---")
    print(f"  Raw BL survey: {len(bl_raw)} obs, {bl_raw.shape[1]} vars")
    print(f"  Raw FU survey: {len(fu_raw)} obs, {fu_raw.shape[1]} vars")
    print(f"  Randomized data: {len(rand_raw)} obs, {rand_raw.shape[1]} vars")

    # Check that daughter variable matches raw data
    if 'd04_childgender_' in rand_raw.columns:
        daughter_raw = (rand_raw['d04_childgender_'].astype(str) == '1 female') | rand_raw['d04_childgender_'].isna()
        print(f"  Raw daughter=1: {daughter_raw.sum()} / {len(rand_raw)} "
              f"({100*daughter_raw.mean():.1f}%)")
        # Note: daughter=1 when gender is missing (Stata quirk)
        n_missing_gender = rand_raw['d04_childgender_'].isna().sum()
        n_str_miss = (rand_raw['d04_childgender_'].astype(str) == '').sum()
        print(f"  Raw gender missing (NaN): {n_missing_gender}")
        print(f"  Raw gender empty string: {n_str_miss}")
        if n_str_miss > 0 or n_missing_gender > 0:
            issues.append(f"daughter=1 for {n_missing_gender + n_str_miss} obs with missing child gender")

    # ================================================================
    # SUMMARY
    # ================================================================
    print("\n" + "=" * 60)
    print("DATA AUDIT SUMMARY")
    print("=" * 60)

    if issues:
        print(f"\n  Found {len(issues)} potential issues:")
        for i, issue in enumerate(issues, 1):
            print(f"    {i}. {issue}")
    else:
        print("\n  No major issues found.")

    print("\n  Key observations:")
    print(f"    - Dataset has {len(df)} obs across {df['hhid'].nunique()} HH and {df['good'].nunique()} goods")
    print(f"    - Main sample (child goods, no toys): {len(main)} obs")
    print(f"    - BL-to-FU attrition: {len(attrited)}/{len(bl_hh)} ({100*len(attrited)/len(bl_hh):.1f}%)")
    print(f"    - Girl variable missing for adult goods (by design): "
          f"{df[df['childgood']==0]['girl'].isna().sum()} obs")


if __name__ == '__main__':
    main()
