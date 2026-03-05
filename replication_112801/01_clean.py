"""
01_clean.py - Construct analysis samples and validate against paper.

Replicates Sample.do and initial data construction from Figures-Final.do.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import pandas as pd
from utils import (load_working_data, prepare_sample, collapse_by_cohort,
                   EDUC_BA, EDUC_POST, OCC_LABELS, print_section)

print_section("PHASE 1: Load and inspect raw data")

df = load_working_data()
print(f"Raw data: {len(df):,} obs, {df.shape[1]} variables")
print(f"Year range: {df['year'].min()}-{df['year'].max()}")
print(f"\nEducation distribution:")
print(df['educ'].value_counts().sort_index())
print(f"\nOcc4 distribution:")
for k, v in sorted(OCC_LABELS.items()):
    n = (df['occ4'] == k).sum()
    print(f"  {k} ({v}): {n:,}")

print_section("PHASE 2: Construct BA sample (educ==4)")

ba = prepare_sample(df, EDUC_BA, gender='all', drop_nilf=False)
print(f"BA sample (all gender, with NILF): {len(ba):,} obs")
print(f"Job entry year range: {ba['jeb'].min()}-{ba['jeb'].max()}")
print(f"Experience range (myrs): {ba['myrs'].min()}-{ba['myrs'].max()}")
print(f"\nCohort sizes:")
print(ba.groupby('jeb').size().to_string())

# Employment share version (drop NILF)
ba_emp = prepare_sample(df, EDUC_BA, gender='all', drop_nilf=True)
print(f"\nBA sample (employed only, for employment shares): {len(ba_emp):,} obs")

# Collapse for Figure 1
ba_collapsed = collapse_by_cohort(ba_emp)
print(f"\nCollapsed BA data: {len(ba_collapsed)} cells")

print("\nCognitive employment shares by cohort (first year):")
first_yr = ba_collapsed.groupby('jeb').first().reset_index()
for _, row in first_yr.iterrows():
    print(f"  Cohort {int(row['jeb'])}: Op1={row['Op1']:.3f}, med_wage={row['med']:.3f} "
          f"(N={int(row['obs'])})")

print_section("PHASE 3: Construct Post-College sample (educ==5)")

post = prepare_sample(df, EDUC_POST, gender='all', drop_nilf=False)
print(f"Post-college sample (all gender, with NILF): {len(post):,} obs")

post_emp = prepare_sample(df, EDUC_POST, gender='all', drop_nilf=True)
print(f"Post-college sample (employed only): {len(post_emp):,} obs")

post_collapsed = collapse_by_cohort(post_emp)
print(f"Collapsed post-college data: {len(post_collapsed)} cells")

print("\nCognitive employment shares by cohort (first year):")
first_yr = post_collapsed.groupby('jeb').first().reset_index()
for _, row in first_yr.iterrows():
    print(f"  Cohort {int(row['jeb'])}: Op1={row['Op1']:.3f}, med_wage={row['med']:.3f} "
          f"(N={int(row['obs'])})")

print_section("PHASE 4: Validation checks")

# Check the paper says: cognitive share increases from ~0.52 to ~0.68 across 1990s cohorts for BA
print("Paper claims for BA cognitive employment (Figure 1 Panel A):")
print("  - 1990 cohort starts ~0.52, rises to ~0.55")
print("  - Each successive 1990s cohort starts higher")
print("  - 1998 cohort starts ~0.62-0.65")
print("  - Post-2000 cohorts reverse: lower starting shares")
print("  - 2010 cohort approximately same as 1990 cohort")

# Check key years
for jeb in [1990, 1994, 1998, 2000, 2004, 2010]:
    sub = ba_collapsed[(ba_collapsed['jeb'] == jeb) & (ba_collapsed['myrs'] == 0)]
    if len(sub) > 0:
        print(f"  Cohort {jeb} at entry: cognitive share = {sub['Op1'].values[0]:.3f}")
    else:
        print(f"  Cohort {jeb}: no myrs==0 data")

print("\nPaper claims for BA median wages (Figure 1 Panel B):")
print("  - Entry wage increases ~0.25 log points between 1994-2000 cohorts")
print("  - Falls ~0.1 log points by 2010 cohort")
for jeb in [1990, 1994, 1998, 2000, 2004, 2010]:
    sub = ba_collapsed[(ba_collapsed['jeb'] == jeb) & (ba_collapsed['myrs'] == 0)]
    if len(sub) > 0:
        print(f"  Cohort {jeb} at entry: median ln wage = {sub['med'].values[0]:.3f}")

print_section("PHASE 5: Summary statistics")

for educ, label in [(EDUC_BA, 'BA'), (EDUC_POST, 'Post-College')]:
    sample = prepare_sample(df, educ, gender='all', drop_nilf=False)
    emp = sample[sample['occ4'] != 5]
    print(f"\n{label} workers:")
    print(f"  Total obs (first 5 yrs, jeb 1990-2010): {len(sample):,}")
    print(f"  Employed: {len(emp):,} ({100*len(emp)/len(sample):.1f}%)")
    print(f"  Mean age: {sample['age'].mean():.1f}")
    print(f"  Female share: {sample['female'].mean():.3f}")
    print(f"  Unique cohorts (jeb): {sorted(sample['jeb'].unique())}")

    wages = emp['lnrhrw_cpi'].dropna()
    print(f"  Median ln wage (employed, non-allocated): {wages.median():.3f}")
    print(f"  Mean ln wage: {wages.mean():.3f}")

    print(f"  Occupation distribution (employed):")
    for occ in [1, 2, 3, 4]:
        share = (emp['occ4'] == occ).mean()
        print(f"    {OCC_LABELS[occ]}: {share:.3f}")

print("\n✓ Data cleaning complete.")
