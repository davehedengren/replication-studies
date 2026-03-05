"""
04_data_audit.py — Data quality checks for Gobbi, Hannusch & Rossi (2026).
"""

import numpy as np
import pandas as pd
from utils import OUTPUT_DIR, ANALYSIS_DATA_PATH, ANALYSIS_DATA_FIG4_PATH, DATA_DIR
import os

print("=" * 60)
print("04_data_audit.py — Data quality audit")
print("=" * 60)

# ── Load datasets ────────────────────────────────────────────────────

df = pd.read_parquet(ANALYSIS_DATA_PATH)
df_diff = pd.read_parquet(ANALYSIS_DATA_FIG4_PATH)
df_cross = pd.read_parquet(os.path.join(OUTPUT_DIR, 'cross_section.parquet'))

# ══════════════════════════════════════════════════════════════════════
# 1. COVERAGE
# ══════════════════════════════════════════════════════════════════════

print("\n── 1. Coverage ──\n")

print(f"Panel dataset (analysis_data.parquet):")
print(f"  Observations: {len(df)}")
print(f"  Countries: {df['CountryCode'].nunique()}")
print(f"  Year range: {df['Year'].min()} - {df['Year'].max()}")

print(f"\nFigure 4 dataset (analysis_data_fig4.parquet):")
print(f"  Countries: {len(df_diff)}")

print(f"\nCross-section dataset (cross_section.parquet):")
print(f"  Countries: {len(df_cross)}")

# Variable completeness
print("\nVariable completeness (panel dataset):")
for var in ['TFR', 'hc', 'GDPpc', 'chmortality', 'matmortality',
            'sec_enrollment', 'region_wb']:
    n_nonmiss = df[var].notna().sum()
    pct = 100 * n_nonmiss / len(df)
    print(f"  {var:<20}: {n_nonmiss:>6} / {len(df)} ({pct:.1f}%)")

print("\nVariable completeness (Figure 4 dataset):")
for var in ['dtfr', 'dlgdppc', 'dsec', 'dchildmort', 'dmatmort',
            'good_institution_mono', 'good_institution_part', 'region_wb']:
    n_nonmiss = df_diff[var].notna().sum()
    pct = 100 * n_nonmiss / len(df_diff)
    print(f"  {var:<25}: {n_nonmiss:>4} / {len(df_diff)} ({pct:.1f}%)")

# ── Ethnographic Atlas coverage ──────────────────────────────────────

print("\nEthnographic Atlas coverage:")
ethno = pd.read_csv(os.path.join(DATA_DIR, 'EthnographicAtlas.csv'))
print(f"  Countries in Atlas: {len(ethno)}")

n_mono = df_diff['good_institution_mono'].notna().sum()
n_part = df_diff['good_institution_part'].notna().sum()
print(f"  Matched to Figure 4 dataset — mono: {n_mono}, part: {n_part}")

# ══════════════════════════════════════════════════════════════════════
# 2. DISTRIBUTIONS
# ══════════════════════════════════════════════════════════════════════

print("\n── 2. Distributions ──\n")

# Panel variables
panel_vars = ['TFR', 'GDPpc', 'chmortality', 'matmortality', 'hc', 'sec_enrollment']
stats = df[panel_vars].describe().T
stats['missing'] = df[panel_vars].isna().sum()
print("Panel variable summary statistics:")
print(stats[['count', 'missing', 'mean', 'std', 'min', '25%', '50%', '75%', 'max']].to_string())

# Plausibility checks
print("\n\nPlausibility checks:")
issues = []

# TFR should be between 0 and ~10
tfr_out = df[(df['TFR'] < 0.5) | (df['TFR'] > 9)]
if len(tfr_out) > 0:
    issues.append(f"  TFR outside [0.5, 9]: {len(tfr_out)} obs")
    print(f"  TFR outside [0.5, 9]: {len(tfr_out)} obs")
    print(f"    Range: [{df['TFR'].min():.2f}, {df['TFR'].max():.2f}]")
else:
    print("  TFR: all values in [0.5, 9] — OK")

# GDP pc should be positive
gdp_neg = df[df['GDPpc'] <= 0]
if len(gdp_neg) > 0:
    issues.append(f"  GDP pc <= 0: {len(gdp_neg)} obs")
    print(f"  GDP pc <= 0: {len(gdp_neg)} obs")
else:
    print(f"  GDP pc: all positive — OK (range: [{df['GDPpc'].min():.0f}, {df['GDPpc'].max():.0f}])")

# Child mortality should be [0, 1000]
cm_out = df[(df['chmortality'] < 0) | (df['chmortality'] > 600)]
if len(cm_out) > 0:
    issues.append(f"  Child mortality outside [0, 600]: {len(cm_out)} obs")
    print(f"  Child mortality outside [0, 600]: {len(cm_out)} obs")
else:
    print(f"  Child mortality: range [{df['chmortality'].min():.1f}, {df['chmortality'].max():.1f}] — OK")

# Human capital index should be [1, 4.5]
hc_out = df[(df['hc'] < 0.5) | (df['hc'] > 5)]
if len(hc_out) > 0:
    issues.append(f"  HC index outside [0.5, 5]: {len(hc_out)} obs")
    print(f"  HC index outside [0.5, 5]: {len(hc_out)} obs")
else:
    print(f"  HC index: range [{df['hc'].min():.2f}, {df['hc'].max():.2f}] — OK")

# ── Outlier check (IQR method) ───────────────────────────────────────

print("\n\nOutlier detection (>3 IQR from median):")
for var in ['TFR', 'GDPpc', 'chmortality', 'hc']:
    vals = df[var].dropna()
    q1, q3 = vals.quantile(0.25), vals.quantile(0.75)
    iqr = q3 - q1
    lower = q1 - 3 * iqr
    upper = q3 + 3 * iqr
    outliers = vals[(vals < lower) | (vals > upper)]
    print(f"  {var}: {len(outliers)} outliers beyond [{lower:.1f}, {upper:.1f}]")

# ══════════════════════════════════════════════════════════════════════
# 3. LOGICAL CONSISTENCY
# ══════════════════════════════════════════════════════════════════════

print("\n── 3. Logical consistency ──\n")

# Check: mono + non-mono should partition the atlas-matched sample
mono_vals = df_diff['good_institution_mono'].value_counts(dropna=False)
print("good_institution_mono distribution:")
print(mono_vals.to_string())

part_vals = df_diff['good_institution_part'].value_counts(dropna=False)
print("\ngood_institution_part distribution:")
print(part_vals.to_string())

# Check: monogamous countries should not overlap with non-monogamous
n_both_mono = df_diff[(df_diff['good_institution_mono'] == 1) &
                       (df_diff['monogamous'] == 0)].shape[0]
n_both_nonmono = df_diff[(df_diff['good_institution_mono'] == 0) &
                          (df_diff['monogamous'] == 1)].shape[0]
print(f"\nConsistency: mono=1 but monogamous=0: {n_both_mono}")
print(f"Consistency: mono=0 but monogamous=1: {n_both_nonmono}")

# Check region distribution
print("\nRegion distribution (Figure 4 dataset):")
region_counts = df_diff['region_wb'].value_counts()
for region in ['East Asia & Pacific', 'Europe & Central Asia',
               'Latin America & Caribbean', 'Middle East & North Africa',
               'North America', 'South Asia', 'Sub-Saharan Africa']:
    n = region_counts.get(region, 0)
    print(f"  {region:<35}: {n}")
print(f"  {'(no region)':35}: {df_diff['region_wb'].isna().sum()}")

# ══════════════════════════════════════════════════════════════════════
# 4. MISSING DATA PATTERNS
# ══════════════════════════════════════════════════════════════════════

print("\n── 4. Missing data patterns ──\n")

# By year — key analysis years
print("Panel data availability at key analysis years:")
for yr in [1960, 1985, 2010]:
    sub = df[df['Year'] == yr]
    print(f"\n  Year {yr} ({sub['CountryCode'].nunique()} countries):")
    for var in ['TFR', 'hc', 'GDPpc', 'chmortality', 'matmortality']:
        n = sub[var].notna().sum()
        print(f"    {var:<20}: {n}")

# Figure 4 regression sample
print("\n\nFigure 4 regression sample sizes:")
pvars_full = ['dtfr', 'dlgdppc', 'dsec', 'dchildmort', 'dmatmort']

for label, sub in [
    ('All countries', df_diff),
    ('Monogamous', df_diff[df_diff['good_institution_mono'] == 1]),
    ('Non-monogamous', df_diff[df_diff['good_institution_mono'] == 0]),
    ('SSA Partible', df_diff[(df_diff['good_institution_part'] == 1) &
                              (df_diff['region_wb'] == 'Sub-Saharan Africa')]),
    ('SSA Impartible', df_diff[(df_diff['good_institution_part'] == 0) &
                                (df_diff['region_wb'] == 'Sub-Saharan Africa')]),
]:
    n_total = len(sub)
    n_complete = sub.dropna(subset=pvars_full).shape[0]
    print(f"  {label:<25}: {n_complete} / {n_total} complete on all predictors")

# ══════════════════════════════════════════════════════════════════════
# 5. PANEL BALANCE
# ══════════════════════════════════════════════════════════════════════

print("\n── 5. Panel balance ──\n")

# Count years per country for key variables
years_per_country = df.groupby('CountryCode').agg(
    n_years=('Year', 'count'),
    tfr_years=('TFR', lambda x: x.notna().sum()),
    gdp_years=('GDPpc', lambda x: x.notna().sum()),
    hc_years=('hc', lambda x: x.notna().sum()),
).reset_index()

print(f"Total possible years per country: {df['Year'].nunique()}")
print(f"Median TFR years: {years_per_country['tfr_years'].median():.0f}")
print(f"Median GDP years: {years_per_country['gdp_years'].median():.0f}")
print(f"Median HC years: {years_per_country['hc_years'].median():.0f}")

# Countries with very few observations
sparse = years_per_country[years_per_country['tfr_years'] < 10]
print(f"\nCountries with <10 TFR observations: {len(sparse)}")
if len(sparse) > 0 and len(sparse) <= 20:
    print(f"  {', '.join(sparse['CountryCode'].tolist())}")

# ══════════════════════════════════════════════════════════════════════
# 6. DUPLICATES AND CODING ANOMALIES
# ══════════════════════════════════════════════════════════════════════

print("\n── 6. Duplicates and coding anomalies ──\n")

# Check for duplicate country-year
dupes = df.duplicated(subset=['CountryCode', 'Year']).sum()
print(f"Duplicate CountryCode-Year rows in panel: {dupes}")

# Check for duplicate countries in Figure 4
dupes_fig4 = df_diff.duplicated(subset=['CountryCode']).sum()
print(f"Duplicate CountryCode in Figure 4 dataset: {dupes_fig4}")

# Check for impossible TFR changes
extreme_dtfr = df_diff[df_diff['dtfr'].abs() > 6]
print(f"\nExtreme TFR changes (|Δ| > 6): {len(extreme_dtfr)}")
if len(extreme_dtfr) > 0:
    for _, row in extreme_dtfr.iterrows():
        print(f"  {row['CountryCode']}: ΔTFR = {row['dtfr']:.2f}")

# NaN/Inf in computed variables
for var in ['dtfr', 'dlgdppc', 'dsec', 'dchildmort', 'dmatmort']:
    n_inf = np.isinf(df_diff[var]).sum() if df_diff[var].dtype == float else 0
    n_nan = df_diff[var].isna().sum()
    if n_inf > 0 or n_nan > 0:
        print(f"  {var}: {n_nan} NaN, {n_inf} Inf")


print("\n" + "=" * 60)
print("04_data_audit.py — DONE")
print("=" * 60)
