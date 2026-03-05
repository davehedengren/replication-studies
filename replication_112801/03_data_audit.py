"""
03_data_audit.py - Data quality audit for Beaudry, Green, Sand (2014).
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import pandas as pd
from utils import (load_working_data, prepare_sample, EDUC_BA, EDUC_POST,
                   OCC_LABELS, print_section)

df = load_working_data()

# ════════════════════════════════════════════════════════════
print_section("1. COVERAGE")
# ════════════════════════════════════════════════════════════

print(f"Total observations: {len(df):,}")
print(f"Year range: {df['year'].min()}-{df['year'].max()}")
print(f"Unique years: {sorted(df['year'].unique())}")
print(f"Age range: {df['age'].min()}-{df['age'].max()}")

print("\nObservations by year:")
yr_counts = df.groupby('year').size()
print(yr_counts.to_string())

print("\nObservations by education:")
for e in sorted(df['educ'].dropna().unique()):
    n = (df['educ'] == e).sum()
    print(f"  educ={int(e)}: {n:,}")
n_miss = df['educ'].isna().sum()
print(f"  educ=missing: {n_miss:,}")

print(f"\nGender: Male={int((df['female']==0).sum()):,}, Female={int((df['female']==1).sum()):,}")

# Analysis samples
for educ, label in [(EDUC_BA, 'BA'), (EDUC_POST, 'Post-College')]:
    s = prepare_sample(df, educ, gender='all', drop_nilf=False)
    print(f"\n{label} analysis sample: {len(s):,} obs")
    print(f"  Year range: {s['year'].min():.0f}-{s['year'].max():.0f}")
    print(f"  Cohorts (jeb): {sorted([int(x) for x in s['jeb'].unique()])}")
    print(f"  Obs per cohort:")
    print(s.groupby('jeb').size().to_string())

# ════════════════════════════════════════════════════════════
print_section("2. VARIABLE COMPLETENESS")
# ════════════════════════════════════════════════════════════

key_vars = ['age', 'year', 'female', 'educ', 'pexp', 'occ4', 'empl',
            'lnrhrw_cpi', 'wgt', 'alloc']
print("Missing values for key variables:")
for v in key_vars:
    n_miss = df[v].isna().sum()
    pct = 100 * n_miss / len(df)
    print(f"  {v:15s}: {n_miss:>10,} missing ({pct:.2f}%)")

# ════════════════════════════════════════════════════════════
print_section("3. DISTRIBUTIONS AND PLAUSIBILITY")
# ════════════════════════════════════════════════════════════

print("Potential experience (pexp):")
pexp = df['pexp'].dropna()
print(f"  Range: {pexp.min():.0f}-{pexp.max():.0f}")
print(f"  Mean: {pexp.mean():.1f}, Median: {pexp.median():.0f}")
neg = (pexp < 0).sum()
print(f"  Negative values: {neg}")

print("\nLog real hourly wage (lnrhrw_cpi):")
wages = df['lnrhrw_cpi'].dropna()
print(f"  N non-missing: {len(wages):,}")
print(f"  Range: {wages.min():.3f} to {wages.max():.3f}")
print(f"  Mean: {wages.mean():.3f}, Median: {wages.median():.3f}")
print(f"  SD: {wages.std():.3f}")
print(f"  p1: {wages.quantile(0.01):.3f}, p99: {wages.quantile(0.99):.3f}")

# Plausibility: log wage of 0 = $1/hr, log wage of 5 = $148/hr
implausible_low = (wages < 0.5).sum()
implausible_high = (wages > 5.0).sum()
print(f"  Below 0.5 (< $1.65/hr): {implausible_low:,}")
print(f"  Above 5.0 (> $148/hr): {implausible_high:,}")

print("\nWeights (wgt):")
w = df['wgt'].dropna()
print(f"  Range: {w.min():.2f} to {w.max():.2f}")
print(f"  Mean: {w.mean():.2f}, Median: {w.median():.2f}")
print(f"  Zero weights: {(w == 0).sum()}")

print("\nAllocation flag:")
print(f"  alloc==0: {(df['alloc']==0).sum():,}")
print(f"  alloc==1: {(df['alloc']==1).sum():,}")
print(f"  alloc missing: {df['alloc'].isna().sum():,}")

# ════════════════════════════════════════════════════════════
print_section("4. OCCUPATION CODING")
# ════════════════════════════════════════════════════════════

print("occ4 distribution (full sample):")
for val in sorted(df['occ4'].dropna().unique()):
    n = (df['occ4'] == val).sum()
    pct = 100 * n / len(df)
    label = OCC_LABELS.get(int(val), 'Unknown')
    print(f"  {int(val)} ({label}): {n:,} ({pct:.1f}%)")
print(f"  Missing: {df['occ4'].isna().sum():,}")

# Check: occ4==5 (NILF) should be non-employed
nilf = df[df['occ4'] == 5]
employed_nilf = nilf[nilf['empl'] == 1]
print(f"\nocc4==5 (NILF) who report employed: {len(employed_nilf):,} / {len(nilf):,}")

# Check: employed should have occ4 in 1-4
employed = df[df['empl'] == 1]
occ_miss_emp = employed['occ4'].isna().sum()
print(f"Employed with missing occ4: {occ_miss_emp:,} / {len(employed):,}")

# ════════════════════════════════════════════════════════════
print_section("5. MISSING DATA PATTERNS")
# ════════════════════════════════════════════════════════════

# Wage missingness by employment status
print("Wage (lnrhrw_cpi) availability by employment:")
for empl_val in [0, 1]:
    sub = df[df['empl'] == empl_val]
    n_wage = sub['lnrhrw_cpi'].notna().sum()
    pct = 100 * n_wage / len(sub) if len(sub) > 0 else 0
    print(f"  empl={empl_val}: {n_wage:,} / {len(sub):,} have wages ({pct:.1f}%)")

# Wage missingness by allocation
print("\nWage availability by allocation flag (employed only):")
emp = df[df['empl'] == 1]
for alloc_val in [0.0, 1.0]:
    sub = emp[emp['alloc'] == alloc_val]
    n_wage = sub['lnrhrw_cpi'].notna().sum()
    pct = 100 * n_wage / len(sub) if len(sub) > 0 else 0
    print(f"  alloc={int(alloc_val)}: {n_wage:,} / {len(sub):,} have wages ({pct:.1f}%)")

# Allocation rate over time
print("\nAllocation rate by year (for BA, employed):")
ba = df[(df['educ'] == EDUC_BA) & (df['empl'] == 1)]
alloc_by_yr = ba.groupby('year')['alloc'].mean()
for yr, rate in alloc_by_yr.items():
    print(f"  {yr}: {rate:.3f}")

# ════════════════════════════════════════════════════════════
print_section("6. PANEL BALANCE / COHORT COVERAGE")
# ════════════════════════════════════════════════════════════

for educ, label in [(EDUC_BA, 'BA'), (EDUC_POST, 'Post-College')]:
    s = prepare_sample(df, educ, gender='all', drop_nilf=False)
    print(f"\n{label} sample - Obs by cohort x experience year:")
    pivot = s.groupby(['jeb', 'myrs']).size().unstack(fill_value=0)
    print(pivot.to_string())

# ════════════════════════════════════════════════════════════
print_section("7. OUTLIERS (IQR + TOP/BOTTOM)")
# ════════════════════════════════════════════════════════════

for educ, label in [(EDUC_BA, 'BA'), (EDUC_POST, 'Post-College')]:
    s = prepare_sample(df, educ, gender='all', drop_nilf=True)
    wages = s['lnrhrw_cpi'].dropna()
    Q1 = wages.quantile(0.25)
    Q3 = wages.quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR
    outliers_low = (wages < lower).sum()
    outliers_high = (wages > upper).sum()
    print(f"\n{label} wages:")
    print(f"  Q1={Q1:.3f}, Q3={Q3:.3f}, IQR={IQR:.3f}")
    print(f"  Lower fence: {lower:.3f}, Upper fence: {upper:.3f}")
    print(f"  Below lower: {outliers_low:,} ({100*outliers_low/len(wages):.1f}%)")
    print(f"  Above upper: {outliers_high:,} ({100*outliers_high/len(wages):.1f}%)")
    print(f"  Bottom 10: {sorted(wages.nsmallest(10).values)}")
    print(f"  Top 10: {sorted(wages.nlargest(10).values, reverse=True)}")

# ════════════════════════════════════════════════════════════
print_section("8. LOGICAL CONSISTENCY CHECKS")
# ════════════════════════════════════════════════════════════

# Potential experience should be >= 0 (already filtered in sample construction)
print(f"pexp < 0 in raw data: {(df['pexp'] < 0).sum()}")
print(f"pexp == 0 in raw data: {(df['pexp'] == 0).sum():,}")

# Employment + occupation consistency
non_employed = df[df['empl'] == 0]
has_occ = non_employed['occ4'].isin([1, 2, 3, 4]).sum()
print(f"Non-employed with occupation 1-4: {has_occ:,} / {len(non_employed):,}")

# Weight should be positive
print(f"Negative weights: {(df['wgt'] < 0).sum()}")
print(f"Zero weights: {(df['wgt'] == 0).sum()}")

# Year and age consistency with pexp
# For educ==4 (BA): pexp = age - 23
ba_check = df[df['educ'] == 4].copy()
expected_pexp = ba_check['age'] - 23
mismatch = (ba_check['pexp'] != expected_pexp).sum()
print(f"\nBA pexp != age-23: {mismatch:,} / {len(ba_check):,}")

# For educ==5 (post-college): pexp = age - 25
post_check = df[df['educ'] == 5].copy()
expected_pexp = post_check['age'] - 25
mismatch = (post_check['pexp'] != expected_pexp).sum()
print(f"Post-college pexp != age-25: {mismatch:,} / {len(post_check):,}")

# ════════════════════════════════════════════════════════════
print_section("9. DUPLICATES")
# ════════════════════════════════════════════════════════════

# Check for exact duplicates across all columns
n_dupes = df.duplicated().sum()
print(f"Exact duplicate rows: {n_dupes:,} / {len(df):,}")

# Check for near-duplicates on key identifiers
key_cols = ['year', 'age', 'female', 'educ', 'pexp', 'occ4', 'lnrhrw_cpi', 'wgt']
n_key_dupes = df[key_cols].dropna().duplicated().sum()
print(f"Duplicates on key variables: {n_key_dupes:,}")

# ════════════════════════════════════════════════════════════
print_section("10. TIME TRENDS IN KEY VARIABLES")
# ════════════════════════════════════════════════════════════

print("BA cognitive share by year (all experience levels, employed):")
ba_all = df[(df['educ'] == EDUC_BA) & (df['occ4'].isin([1, 2, 3, 4]))]
for yr in sorted(ba_all['year'].unique()):
    sub = ba_all[ba_all['year'] == yr]
    cog_share = (sub['occ4'] == 1).mean()
    med_wage = sub['lnrhrw_cpi'].dropna().median()
    print(f"  {yr}: cognitive share={cog_share:.3f}, median wage={med_wage:.3f}, N={len(sub):,}")

print("\n✓ Data audit complete.")
