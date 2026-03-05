"""
Data audit for Bednar & Gicheva (2014) replication.
"Are Female Supervisors More Female-Friendly?"
"""

import numpy as np
import pandas as pd
from utils import load_main_data, load_eada_data, print_separator


def audit_main_data(df):
    """Audit the main panel dataset (p_and_p_data.dta)."""
    print_separator("MAIN DATA AUDIT: p_and_p_data.dta")

    # Basic dimensions
    print(f"Observations: {len(df)}")
    print(f"Variables: {df.shape[1]}")
    print(f"Unique schools: {df['school_id'].nunique()}")
    print(f"Unique ADs (ad_id): {df['ad_id'].nunique()}")

    # ADunique flag
    n_unique = (df['ADunique'] == 1).sum()
    print(f"ADunique==1 observations: {n_unique}")
    print(f"  (Each AD appears once in ADunique; used for cross-sectional regressions)")

    # Female AD share
    print(f"\n--- AD Gender ---")
    print(f"Female AD observations: {(df['female_ad']==1).sum()} ({(df['female_ad']==1).mean()*100:.1f}%)")
    print(f"Male AD observations: {(df['female_ad']==0).sum()} ({(df['female_ad']==0).mean()*100:.1f}%)")

    # Among unique ADs
    unique = df[df['ADunique'] == 1]
    print(f"Among unique ADs: {(unique['female_ad']==1).sum()} female, {(unique['female_ad']==0).sum()} male")

    # Outcome variable
    print(f"\n--- Outcome: fsoc2 (female coach share) ---")
    print(f"Mean: {df['fsoc2'].mean():.4f}")
    print(f"Std: {df['fsoc2'].std():.4f}")
    print(f"Min: {df['fsoc2'].min():.4f}, Max: {df['fsoc2'].max():.4f}")
    print(f"Missing: {df['fsoc2'].isnull().sum()}")
    vc = df['fsoc2'].value_counts().sort_index()
    print(f"Distribution of fsoc2 values:")
    for val in sorted(df['fsoc2'].unique()):
        count = (df['fsoc2'] == val).sum()
        print(f"  {val:.4f}: {count}")

    # Year dummies
    year_cols = [c for c in df.columns if c.startswith('year_dum')]
    print(f"\n--- Year Coverage ---")
    print(f"Number of year dummies: {len(year_cols)}")
    year_counts = {}
    for i, c in enumerate(year_cols):
        n = df[c].sum()
        year_counts[i+1] = n
    print(f"Observations per year:")
    for yr, n in year_counts.items():
        print(f"  Year {yr}: {int(n)}")

    # Interim ADs
    print(f"\n--- Interim ADs ---")
    print(f"Interim observations: {(df['interim']==1).sum()} ({(df['interim']==1).mean()*100:.1f}%)")

    # AD mobility
    print(f"\n--- AD Mobility ---")
    ad_schools = df.groupby('ad_id')['school_id'].nunique()
    print(f"ADs at 1 school: {(ad_schools==1).sum()}")
    print(f"ADs at 2+ schools: {(ad_schools>1).sum()}")
    print(f"ADs at 3+ schools: {(ad_schools>2).sum()}")
    print(f"Max schools for one AD: {ad_schools.max()}")

    # AD dummy structure
    ad_cols = [c for c in df.columns if c.startswith('ad_dum') and not c.startswith('ad_dum_all')]
    ad_all_cols = [c for c in df.columns if c.startswith('ad_dum_all')]
    print(f"\n--- AD Dummy Structure ---")
    print(f"Multi-school AD dummies (ad_dum): {len(ad_cols)}")
    print(f"All AD dummies (ad_dum_all): {len(ad_all_cols)}")

    # Check: each observation belongs to exactly one AD
    ad_assignment = df[ad_all_cols].sum(axis=1)
    print(f"Obs assigned to exactly 1 AD (all): {(ad_assignment==1).sum()}")
    print(f"Obs assigned to 0 ADs (all): {(ad_assignment==0).sum()}")
    print(f"Obs assigned to 2+ ADs (all): {(ad_assignment>1).sum()}")

    ad_multi = df[ad_cols].sum(axis=1)
    print(f"Obs assigned to multi-school AD: {(ad_multi==1).sum()}")
    print(f"Obs not assigned to multi-school AD: {(ad_multi==0).sum()}")

    # Panel balance
    print(f"\n--- Panel Balance ---")
    school_years = df.groupby('school_id').size()
    print(f"Obs per school: mean={school_years.mean():.1f}, min={school_years.min()}, max={school_years.max()}")
    print(f"Schools with full panel (18 years): {(school_years==18).sum()}")

    # Logical checks
    print(f"\n--- Logical Checks ---")
    print(f"fsoc2 in [0,1]: {(df['fsoc2']>=0).all() and (df['fsoc2']<=1).all()}")
    print(f"female_ad in {{0,1}}: {df['female_ad'].isin([0,1]).all()}")

    return df


def audit_eada_data(df):
    """Audit the EADA expenditure dataset."""
    print_separator("EADA DATA AUDIT: p_and_p_EADAdata.dta")

    print(f"Observations: {len(df)}")
    print(f"Variables: {df.shape[1]}")
    print(f"Unique teams: {df['team'].nunique()}")

    # Missing data
    print(f"\n--- Missing Data ---")
    for col in df.columns:
        n_miss = df[col].isnull().sum()
        if n_miss > 0:
            print(f"  {col}: {n_miss} ({n_miss/len(df)*100:.1f}%)")

    # Expenditure variables
    print(f"\n--- Expenditure Variables ---")
    for col in ['exp_rev_w', 'exp_rev_m']:
        sub = df[col].dropna()
        print(f"{col}: N={len(sub)}, mean={sub.mean():.4f}, sd={sub.std():.4f}, "
              f"min={sub.min():.4f}, max={sub.max():.4f}")

    # Female AD in EADA
    print(f"\n--- Female AD in EADA ---")
    print(f"female_ad values: {df['female_ad'].value_counts().sort_index().to_dict()}")
    print(f"  Note: 0.5 values likely indicate mid-year AD changes")

    # FE measures
    print(f"\n--- Female-Friendliness Measures ---")
    for col in ['friendly_all', 'friendly1']:
        sub = df[col].dropna()
        print(f"{col}: N={len(sub)}, mean={sub.mean():.4f}, sd={sub.std():.4f}")

    # Weights
    print(f"\n--- Weights ---")
    for col in ['weight', 'weight_all']:
        sub = df[col].dropna()
        print(f"{col}: N={len(sub)}, mean={sub.mean():.4f}, sd={sub.std():.4f}")

    # Coverage overlap
    print(f"\n--- Coverage ---")
    has_all = df['friendly_all'].notna() & df['exp_rev_w'].notna()
    has_multi = df['friendly1'].notna() & df['exp_rev_w'].notna()
    print(f"Obs with friendly_all + exp_rev_w: {has_all.sum()}")
    print(f"Obs with friendly1 + exp_rev_w: {has_multi.sum()}")
    has_all_m = df['friendly_all'].notna() & df['exp_rev_m'].notna()
    has_multi_m = df['friendly1'].notna() & df['exp_rev_m'].notna()
    print(f"Obs with friendly_all + exp_rev_m: {has_all_m.sum()}")
    print(f"Obs with friendly1 + exp_rev_m: {has_multi_m.sum()}")

    # Year coverage
    year_cols = [c for c in df.columns if c.startswith('year_dum')]
    print(f"\n--- Year Coverage ---")
    print(f"Year dummies: {len(year_cols)}")
    for c in year_cols:
        print(f"  {c}: {int(df[c].sum())}")

    return df


if __name__ == '__main__':
    df_main = load_main_data()
    audit_main_data(df_main)

    df_eada = load_eada_data()
    audit_eada_data(df_eada)

    print_separator("DATA AUDIT COMPLETE")
