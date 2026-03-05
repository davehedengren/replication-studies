"""
04_data_audit.py - Data quality audit for Altonji, Kahn, Speer (2014).
"""
import numpy as np
import pandas as pd
import sys
sys.path.insert(0, '.')
from utils import load_data, WEIGHT

print("=" * 70)
print("PHASE 3: DATA AUDIT")
print("=" * 70)

df = load_data()
w = df[WEIGHT].values

# =====================================================================
# 1. Coverage
# =====================================================================
print("\n--- 1. Coverage ---")
print(f"  Total observations: {len(df):,d}")
print(f"  Unique majors: {df['major'].nunique()}")
print(f"  Year range: {df['year'].min()}-{df['year'].max()}")
print(f"  Unique years: {sorted(df['year'].unique())}")
print(f"\n  Observations by survey:")
for sv in sorted(df['survey'].unique()):
    n = (df['survey'] == sv).sum()
    sw = df.loc[df['survey'] == sv, 'survey_weight'].sum()
    nw = df.loc[df['survey'] == sv, WEIGHT].sum()
    print(f"    Survey {sv}: {n:>8,d} obs  survey_wt_sum={sw:>12,.0f}  newweight_sum={nw:>12,.0f}")

print(f"\n  Observations by year:")
for yr in sorted(df['year'].unique()):
    n = (df['year'] == yr).sum()
    print(f"    Year {yr}: {n:>8,d} obs")

# Variable completeness
print(f"\n  Variable completeness (missing counts):")
key_vars = ['lnearnings', 'majorbeta_none', 'majorbeta_hgc_wtime',
            'A', 'R', 'M', 'male', 'black', 'hispanic', 'hgc',
            'potexp', WEIGHT, 'survey', 'year', 'fulltime', 'employed']
for var in key_vars:
    if var in df.columns:
        n_miss = df[var].isna().sum()
        print(f"    {var:30s}: {n_miss:>8,d} missing")

# =====================================================================
# 2. Distributions
# =====================================================================
print("\n--- 2. Distributions ---")
print(f"\n  Key continuous variables (weighted):")
for var in ['lnearnings', 'potexp', 'A', 'R', 'M',
            'majorbeta_none', 'majorbeta_hgc_wtime']:
    vals = df[var].values
    wmean = np.average(vals, weights=w)
    wsd = np.sqrt(np.average((vals - wmean)**2, weights=w))
    print(f"    {var:30s}: mean={wmean:9.4f}  sd={wsd:7.4f}  "
          f"min={vals.min():9.4f}  max={vals.max():9.4f}")

print(f"\n  Earnings distribution:")
earn = df['earnings_annual'].astype(float)
earn_valid = earn[earn > 0]
print(f"    Observations with earnings > 0: {len(earn_valid):,d}")
print(f"    Min earnings: ${earn_valid.min():,.0f}")
print(f"    Max earnings: ${earn_valid.max():,.0f}")
print(f"    Median earnings: ${earn_valid.median():,.0f}")
pcts = [1, 5, 10, 25, 50, 75, 90, 95, 99]
for p in pcts:
    print(f"    P{p:2d}: ${np.percentile(earn_valid, p):>12,.0f}")

# Top-coding check (paper says topcoded at $400,000)
n_topcode = (earn >= 400000).sum()
pct_topcode = n_topcode / len(df) * 100
print(f"\n    Earnings >= $400,000 (topcode): {n_topcode:,d} ({pct_topcode:.2f}%)")
# Bottom-coding check (paper says dropped below $500)
n_low = (earn < 500).sum()
print(f"    Earnings < $500: {n_low:,d}")

# Outlier detection using IQR
q1 = np.percentile(df['lnearnings'], 25)
q3 = np.percentile(df['lnearnings'], 75)
iqr = q3 - q1
n_outlier_low = (df['lnearnings'] < q1 - 3*iqr).sum()
n_outlier_high = (df['lnearnings'] > q3 + 3*iqr).sum()
print(f"\n    Log earnings IQR: [{q1:.3f}, {q3:.3f}], IQR={iqr:.3f}")
print(f"    Extreme outliers (3xIQR): {n_outlier_low} low, {n_outlier_high} high")

# =====================================================================
# 3. Logical Consistency
# =====================================================================
print("\n--- 3. Logical Consistency ---")

# Age range
print(f"  Age range: {df['age'].min()}-{df['age'].max()} (paper requires 25-55)")
n_age_violation = ((df['age'] < 25) | (df['age'] > 55)).sum()
print(f"  Age violations: {n_age_violation}")

# Full-time restriction
print(f"  All fulltime==1: {(df['fulltime'] == 1).all()}")
print(f"  All employed==1: {(df['employed'] == 1).all()}")

# Potential experience check
# potexp = age - years_of_schooling - 6
# For BA: potexp = age - 16 - 6 = age - 22
print(f"\n  Potential experience check:")
print(f"    potexp range: {df['potexp'].min():.0f} to {df['potexp'].max():.0f}")
print(f"    Expected: age - yrs_school - 6 (1 to 33 for ages 25-55)")

# HGC distribution
print(f"\n  HGC (education) distribution:")
for h in sorted(df['hgc'].unique()):
    n = (df['hgc'] == h).sum()
    pct = n / len(df) * 100
    labels = {16: 'BA', 18: "Master's", 19: 'Professional', 20: 'Doctorate'}
    print(f"    hgc={h:.0f} ({labels.get(h,'?'):12s}): {n:>8,d} ({pct:5.1f}%)")

# Demographics
print(f"\n  Demographics:")
print(f"    Male: {df['male'].mean():.3f}")
print(f"    Black: {df['black'].mean():.3f}")
print(f"    Hispanic: {df['hispanic'].mean():.3f}")
print(f"    black_x_male = black * male: {(df['black_x_male'] != df['black'] * df['male']).sum()} mismatches")
print(f"    hispanic_x_male = hispanic * male: {(df['hispanic_x_male'] != df['hispanic'] * df['male']).sum()} mismatches")

# Weight checks
print(f"\n  Weight checks:")
print(f"    All newweight > 0: {(df[WEIGHT] > 0).all()}")
print(f"    newweight range: [{df[WEIGHT].min():.4f}, {df[WEIGHT].max():.4f}]")
print(f"    Sum of newweight: {df[WEIGHT].sum():,.2f}")

# Survey weight analysis
for sv in [1993, 2003, 2009]:
    mask = df['survey'] == sv
    sw = df.loc[mask, 'survey_weight'].sum()
    nw = df.loc[mask, WEIGHT].sum()
    ratio = nw / sw if sw > 0 else float('nan')
    print(f"    Survey {sv}: survey_wt_sum={sw:>12,.0f}  newwt_sum={nw:>12,.0f}  ratio={ratio:.4f}")

# =====================================================================
# 4. Missing Data Patterns
# =====================================================================
print("\n--- 4. Missing Data Patterns ---")
print(f"  All key regression variables are non-missing (by construction from keyvars.do)")

# Check task variables
task_vars = ['task_abstract', 'task_routine', 'task_manual']
for var in task_vars:
    if var in df.columns:
        n_miss = df[var].isna().sum()
        print(f"    {var}: {n_miss:,d} missing")

# Check if any occupation is missing
if 'occ1990' in df.columns:
    n_miss_occ = df['occ1990'].isna().sum()
    print(f"    occ1990: {n_miss_occ:,d} missing")
if 'occ1990dd' in df.columns:
    n_miss_occ = df['occ1990dd'].isna().sum()
    print(f"    occ1990dd: {n_miss_occ:,d} missing")

# =====================================================================
# 5. Panel Balance
# =====================================================================
print("\n--- 5. Cross-Section Balance by Major ---")
major_counts = df.groupby(['major', 'survey']).size().unstack(fill_value=0)
print(f"  Majors x Surveys matrix shape: {major_counts.shape}")
print(f"  Majors with 0 obs in a survey: {(major_counts == 0).any(axis=1).sum()}")

# Major size distribution
major_sizes = df.groupby('major').size()
print(f"\n  Major size distribution:")
print(f"    Min obs per major: {major_sizes.min():,d}")
print(f"    Max obs per major: {major_sizes.max():,d}")
print(f"    Median obs per major: {major_sizes.median():,.0f}")

# Smallest majors
print(f"\n  Smallest 5 majors (by unweighted count):")
for major, n in major_sizes.nsmallest(5).items():
    print(f"    {major}: {n:,d} obs")

# Largest majors
print(f"\n  Largest 5 majors (by unweighted count):")
for major, n in major_sizes.nlargest(5).items():
    print(f"    {major}: {n:,d} obs")

# =====================================================================
# 6. Duplicates and Coding Anomalies
# =====================================================================
print("\n--- 6. Duplicates and Coding Anomalies ---")

# Check for duplicate pid within survey
for sv in [1993, 2003]:
    mask = df['survey'] == sv
    n_dup = df.loc[mask, 'pid'].duplicated().sum()
    print(f"  Duplicate pids in survey {sv}: {n_dup:,d}")

# Check major return distribution
print(f"\n  Beta^m (none) top 5 majors:")
major_betas = df.groupby('major')['majorbeta_none'].first().sort_values(ascending=False)
for major, beta in major_betas.head(5).items():
    print(f"    {major}: {beta:.4f}")

print(f"\n  Beta^m (none) bottom 5 majors:")
for major, beta in major_betas.tail(5).items():
    print(f"    {major}: {beta:.4f}")

# Check standardization
print(f"\n  Standardization verification (weighted mean, SD):")
for var in ['majorbeta_none', 'majorbeta_hgc_wtime', 'A', 'R', 'M']:
    wmean = np.average(df[var], weights=w)
    wsd = np.sqrt(np.average((df[var] - wmean)**2, weights=w))
    print(f"    {var:30s}: mean={wmean:12.8f}  sd={wsd:8.6f}")

# Cluster variable check
n_clust = df['clust_var'].nunique()
n_expected = df.groupby(['major', 'year']).ngroups
print(f"\n  Cluster variable: {n_clust} unique values")
print(f"  Expected (major x year groups): {n_expected}")
print(f"  Match: {n_clust == n_expected}")

# Check that major-level variables are constant within major
print(f"\n  Major-level variable consistency:")
for var in ['A', 'R', 'M', 'majorbeta_none', 'majorbeta_hgc_wtime']:
    within_var = df.groupby('major')[var].std()
    max_within = within_var.max()
    print(f"    {var:30s}: max within-major SD = {max_within:.8f} (should be ~0)")

print("\n" + "=" * 70)
print("Data audit complete.")
print("=" * 70)
