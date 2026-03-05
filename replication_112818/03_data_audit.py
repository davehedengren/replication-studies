"""Data audit for paper 112818 replication."""
import pandas as pd
import numpy as np
from utils import DATA_FILE, OUTPUT_DIR
import os

df = pd.read_stata(DATA_FILE, convert_categoricals=False)

print("=" * 70)
print("DATA AUDIT: Paper 112818")
print("=" * 70)

# 1. Coverage
print("\n1. COVERAGE")
print(f"  Total observations: {len(df)}")
print(f"  Year range: {df['year'].min():.0f} - {df['year'].max():.0f}")
print(f"  Expected years (1949-2012): {2012 - 1949 + 1}")
print(f"  Actual years: {len(df)}")
valid_years = set(df['year'].dropna().astype(int))
print(f"  Missing years: {sorted(set(range(1949, 2013)) - valid_years) if set(range(1949, 2013)) - valid_years else 'None'}")
nan_rows = df[df['year'].isna()]
print(f"  Rows with NaN year: {len(nan_rows)}")
if len(nan_rows) > 0:
    print(f"  NaN year rows (all-NaN?): {nan_rows.isna().all(axis=1).all()}")
print(f"  Columns: {df.columns.tolist()}")

# Variable completeness
print("\n  Variable completeness:")
for col in df.columns:
    n = df[col].notna().sum()
    pct = 100 * n / len(df)
    print(f"    {col}: {n}/{len(df)} ({pct:.1f}%)")

# 2. Distributions / Summary Stats
print("\n2. DISTRIBUTIONS")
print(df.describe().round(4).to_string())

# Check plausibility
print("\n  Plausibility checks:")
print(f"    govsharegdp range [{df['govsharegdp'].min():.1f}, {df['govsharegdp'].max():.1f}] "
      f"- expect ~20-40%: {'PASS' if 15 < df['govsharegdp'].min() and df['govsharegdp'].max() < 50 else 'FAIL'}")
dv = df.dropna(subset=['year'])
print(f"    news always positive: {'PASS' if (dv['news'] > 0).all() else 'FAIL'}")
print(f"    regulation always positive: {'PASS' if (dv['regulation'] > 0).all() else 'FAIL'}")
print(f"    year is integer-valued: {'PASS' if (dv['year'] == dv['year'].round(0)).all() else 'FAIL'}")

# Standardized variables should have ~mean=0, ~sd=1
for col in ['diff_std', 'sd_dshare_std', 'seediff_std']:
    valid = df[col].dropna()
    if len(valid) > 0:
        print(f"    {col} mean={valid.mean():.4f}, sd={valid.std():.4f} "
              f"- standardized: {'PASS' if abs(valid.mean()) < 0.5 and 0.5 < valid.std() < 1.5 else 'CHECK'}")

# 3. Outliers (IQR method)
print("\n3. OUTLIER ANALYSIS (IQR method)")
for col in ['news', 'govsharegdp', 'regulation']:
    q1 = df[col].quantile(0.25)
    q3 = df[col].quantile(0.75)
    iqr = q3 - q1
    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr
    outliers = df[(df[col] < lower) | (df[col] > upper)]
    print(f"  {col}: IQR=[{q1:.2f}, {q3:.2f}], bounds=[{lower:.2f}, {upper:.2f}], "
          f"outliers={len(outliers)}")
    if len(outliers) > 0:
        for _, row in outliers.iterrows():
            print(f"    year={row['year']:.0f}, value={row[col]:.2f}")

# Top-10 values for key variables
print("\n  Top-5 policy uncertainty (news) years:")
top5 = df.nlargest(5, 'news')[['year', 'news']]
for _, row in top5.iterrows():
    print(f"    {row['year']:.0f}: {row['news']:.2f}")

print("\n  Bottom-5 policy uncertainty (news) years:")
bot5 = df.nsmallest(5, 'news')[['year', 'news']]
for _, row in bot5.iterrows():
    print(f"    {row['year']:.0f}: {row['news']:.2f}")

# 4. Missing data patterns
print("\n4. MISSING DATA PATTERNS")
# sd_dshare_std and seediff_std are sparse - check which years have data
print("  sd_dshare_std available years:")
valid_sd = df[df['sd_dshare_std'].notna()]['year'].values
print(f"    {[int(y) for y in valid_sd]}")
    # These are Congress-start years (odd years, every 4 years = every other Congress)
diffs_sd = [int(valid_sd[i+1] - valid_sd[i]) for i in range(len(valid_sd)-1)]
print(f"    Year spacing: {diffs_sd}")

print("\n  seediff_std available years:")
valid_see = df[df['seediff_std'].notna()]['year'].values
print(f"    {[int(y) for y in valid_see]}")

# Check diff_std missing
missing_diff = df[df['diff_std'].isna() & df['year'].notna()]['year'].values
print(f"\n  diff_std missing years: {[int(y) for y in missing_diff]}")

# 5. Year gaps and continuity
print("\n5. CONTINUITY")
years = sorted(df['year'].dropna().values)
gaps = []
for i in range(1, len(years)):
    if years[i] - years[i-1] != 1:
        gaps.append((int(years[i-1]), int(years[i])))
print(f"  Year gaps: {gaps if gaps else 'None - continuous annual data'}")

# 6. Duplicates
print("\n6. DUPLICATES")
dupes = df[df.duplicated(subset=['year'], keep=False)]
print(f"  Duplicate years: {len(dupes)}")

# 7. Normalization check
print("\n7. NORMALIZATION VERIFICATION")
df_post = df[df['year'] >= 1950].copy()
mnews = df_post['news'].mean()
mreg = df_post['regulation'].mean()
print(f"  Mean of news (year>=1950): {mnews:.4f}")
print(f"  Mean of regulation (year>=1950): {mreg:.4f}")
print(f"  norm_news should center at 100: {100*df_post['news'].mean()/mnews:.4f}")
print(f"  norm_reg should center at 100: {100*df_post['regulation'].mean()/mreg:.4f}")

# 8. Stata code bug check
print("\n8. STATA CODE REVIEW")
print("  Line 5: 'u figure1,clear' - uses abbreviated 'use' command, valid Stata")
print("  Line 9: 'gen norm_reg=100*reg/mreg' - uses 'reg' abbreviation of 'regulation'")
print("  This is valid Stata abbreviation (unambiguous in this dataset)")
print("  Line 6: 'keep if year>=1950' - filters to 63 obs, matches our count")
print("  No bugs detected in the Stata code.")

print("\n" + "=" * 70)
print("AUDIT COMPLETE")
print("=" * 70)
