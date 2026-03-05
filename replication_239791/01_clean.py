"""
01_clean.py — Data cleaning for Gobbi, Hannusch & Rossi (2026) replication.

Loads all 7 CSV data files, merges them, computes first differences,
assigns World Bank regions, and saves two analysis datasets:
  1. analysis_data.parquet — for Figures 1 and 2 (annual panel)
  2. analysis_data_fig4.parquet — for Figure 4 and appendix (period averages)
"""

import numpy as np
import pandas as pd
from utils import (DATA_DIR, OUTPUT_DIR, ANALYSIS_DATA_PATH,
                   ANALYSIS_DATA_FIG4_PATH, load_wdi_csv,
                   get_region_mapping_medium, WDI_AGGREGATES, YEAR_COLS)
import os

print("=" * 60)
print("01_clean.py — Building analysis datasets")
print("=" * 60)

# ── 1. Load WDI data ──────────────────────────────────────────────────────

# TFR
tfr_raw = load_wdi_csv(os.path.join(DATA_DIR, 'WDI_TFR.csv'))
tfr_long = tfr_raw.melt(id_vars='CountryCode', var_name='Year',
                        value_name='TFR')
tfr_long['Year'] = tfr_long['Year'].astype(int)

# GDP per capita
gdp_raw = load_wdi_csv(os.path.join(DATA_DIR, 'WDI_GDPpc.csv'))
gdp_long = gdp_raw.melt(id_vars='CountryCode', var_name='Year',
                        value_name='GDPpc')
gdp_long['Year'] = gdp_long['Year'].astype(int)

# Child mortality
chmort_raw = load_wdi_csv(os.path.join(DATA_DIR, 'WDI_ChildMortality.csv'))
chmort_long = chmort_raw.melt(id_vars='CountryCode', var_name='Year',
                              value_name='chmortality')
chmort_long['Year'] = chmort_long['Year'].astype(int)

# Maternal mortality
matmort_raw = load_wdi_csv(os.path.join(DATA_DIR, 'WDI_MaternalMortality.csv'))
matmort_long = matmort_raw.melt(id_vars='CountryCode', var_name='Year',
                                value_name='matmortality')
matmort_long['Year'] = matmort_long['Year'].astype(int)

# Secondary enrollment
sec_raw = load_wdi_csv(os.path.join(DATA_DIR, 'WDI_SecEnrollment.csv'))
sec_long = sec_raw.melt(id_vars='CountryCode', var_name='Year',
                        value_name='sec_enrollment')
sec_long['Year'] = sec_long['Year'].astype(int)

print(f"  TFR: {len(tfr_long)} obs")
print(f"  GDP: {len(gdp_long)} obs")
print(f"  Child mortality: {len(chmort_long)} obs")
print(f"  Maternal mortality: {len(matmort_long)} obs")
print(f"  Secondary enrollment: {len(sec_long)} obs")

# ── 2. Load Barro-Lee ────────────────────────────────────────────────────

bl_raw = pd.read_csv(os.path.join(DATA_DIR, 'BarroLee.csv'))

# Extract hc columns for available years
bl_hc_cols = [c for c in bl_raw.columns if c.startswith('hc') and c[2:].isdigit()]
bl_years = sorted([int(c[2:]) for c in bl_hc_cols])
print(f"  Barro-Lee: {len(bl_raw)} countries, years {bl_years[0]}-{bl_years[-1]}")

# Reshape to long format (only hc variable, years >= 1960)
bl_long_parts = []
for yr in bl_years:
    if yr < 1960:
        continue
    col = f'hc{yr}'
    if col in bl_raw.columns:
        tmp = bl_raw[['WBcode', col]].copy()
        tmp.columns = ['CountryCode', 'hc']
        tmp['Year'] = yr
        bl_long_parts.append(tmp)

bl_long = pd.concat(bl_long_parts, ignore_index=True)
bl_long['hc'] = pd.to_numeric(bl_long['hc'], errors='coerce')

# ── 3. Load Ethnographic Atlas ──────────────────────────────────────────

ethno = pd.read_csv(os.path.join(DATA_DIR, 'EthnographicAtlas.csv'))
ethno = ethno.rename(columns={'isocode': 'CountryCode'})
print(f"  Ethnographic Atlas: {len(ethno)} countries")

# ── 4. Get World Bank region mapping ────────────────────────────────────

print("\n  Getting World Bank region mapping...")
region_map = get_region_mapping_medium()
print(f"  Region mapping: {len(region_map)} countries")

# ── 5. Build Figure 1/2 dataset (annual panel) ─────────────────────────

# Merge all WDI data
df = tfr_long.merge(bl_long, on=['CountryCode', 'Year'], how='left')
df = df.merge(gdp_long, on=['CountryCode', 'Year'], how='left')
df = df.merge(chmort_long, on=['CountryCode', 'Year'], how='left')
df = df.merge(matmort_long, on=['CountryCode', 'Year'], how='left')
df = df.merge(sec_long, on=['CountryCode', 'Year'], how='left')

# Add log GDP
df['log_gdppc'] = np.log(df['GDPpc'].replace(0, np.nan))

# Add regions
df['region_wb'] = df['CountryCode'].map(region_map)

# Filter out WDI aggregates
df = df[~df['CountryCode'].isin(WDI_AGGREGATES)].reset_index(drop=True)

print(f"\n  Merged panel: {len(df)} obs, {df['CountryCode'].nunique()} countries")

# ── 6. Build Figure 4 dataset (period averages) ────────────────────────
# Following R code: averages over 1975-1985 and 2013-2023

df_fig4_base = df.copy()

# Filter to relevant years
mask = ((df_fig4_base['Year'] >= 1975) & (df_fig4_base['Year'] <= 1985)) | \
       ((df_fig4_base['Year'] >= 2013) & (df_fig4_base['Year'] <= 2023))
df_subset = df_fig4_base[mask].copy()

# Period indicator
df_subset['period'] = np.where(df_subset['Year'] <= 1985, 'early', 'late')

# Compute period averages
agg = df_subset.groupby(['CountryCode', 'period']).agg(
    tfr=('TFR', 'mean'),
    gdppc=('GDPpc', 'mean'),
    sec=('sec_enrollment', 'mean'),
    chmortality=('chmortality', 'mean'),
    matmortality=('matmortality', 'mean'),
).reset_index()

# Pivot to wide
early = agg[agg['period'] == 'early'].drop(columns='period').rename(
    columns={c: f'{c}_early' for c in ['tfr', 'gdppc', 'sec', 'chmortality', 'matmortality']}
)
late = agg[agg['period'] == 'late'].drop(columns='period').rename(
    columns={c: f'{c}_late' for c in ['tfr', 'gdppc', 'sec', 'chmortality', 'matmortality']}
)

df_diff = early.merge(late, on='CountryCode', how='inner')

# Compute differences
df_diff['dtfr'] = df_diff['tfr_late'] - df_diff['tfr_early']
# NOTE: The R code computes level differences here (gdppc_avg2010_20 - gdppc_avg1970_80)
# despite naming the variable "dlgdppc" (suggesting log differences).
# Figure 2 correctly uses log(gdp2010) - log(gdp1960).
# We replicate the R code's behavior exactly (level differences) for Figure 4.
df_diff['dlgdppc'] = df_diff['gdppc_late'] - df_diff['gdppc_early']
df_diff['dsec'] = df_diff['sec_late'] - df_diff['sec_early']
df_diff['dchildmort'] = df_diff['chmortality_late'] - df_diff['chmortality_early']
df_diff['dmatmort'] = df_diff['matmortality_late'] - df_diff['matmortality_early']

# Merge ethnographic atlas
df_diff = df_diff.merge(ethno, on='CountryCode', how='left')

# Add regions
df_diff['region_wb'] = df_diff['CountryCode'].map(region_map)

# Filter out aggregates
df_diff = df_diff[~df_diff['CountryCode'].isin(WDI_AGGREGATES)].reset_index(drop=True)

print(f"  Figure 4 dataset: {len(df_diff)} countries with period differences")

# ── 7. Also compute simple 2010-1960 differences for Figure 2 ──────────
# (These are separate from Figure 4's period averages)

# Get single-year values
for yr in [1960, 1985, 2010]:
    yr_data = df[df['Year'] == yr][['CountryCode', 'TFR', 'hc', 'GDPpc',
                                     'chmortality', 'matmortality',
                                     'log_gdppc', 'region_wb']].copy()
    yr_data.columns = ['CountryCode'] + [f'{c}_{yr}' for c in
                       ['TFR', 'hc', 'GDPpc', 'chmortality', 'matmortality',
                        'log_gdppc', 'region_wb']]
    if yr == 1960:
        df_cross = yr_data
    else:
        df_cross = df_cross.merge(yr_data, on='CountryCode', how='outer')

# Use region from 2010 (or 1960 as fallback)
df_cross['region_wb'] = df_cross.get('region_wb_2010',
                                      df_cross.get('region_wb_1960'))

# Compute first differences for Figure 2
df_cross['hc_increase'] = df_cross['hc_2010'] - df_cross['hc_1960']
df_cross['tfr_change'] = df_cross['TFR_2010'] - df_cross['TFR_1960']
df_cross['loggdp_change'] = df_cross['log_gdppc_2010'] - df_cross['log_gdppc_1960']
df_cross['chmort_change'] = df_cross['chmortality_2010'] - df_cross['chmortality_1960']
df_cross['matmort_change'] = df_cross['matmortality_2010'] - df_cross['matmortality_1985']
df_cross['tfr_change_1985'] = df_cross['TFR_2010'] - df_cross['TFR_1985']

# Save cross-section for Figure 2
cross_path = os.path.join(OUTPUT_DIR, 'cross_section.parquet')
df_cross.to_parquet(cross_path, index=False)

# ── 8. Save ─────────────────────────────────────────────────────────────

df.to_parquet(ANALYSIS_DATA_PATH, index=False)
df_diff.to_parquet(ANALYSIS_DATA_FIG4_PATH, index=False)

print(f"\n  Saved: {ANALYSIS_DATA_PATH}")
print(f"  Saved: {ANALYSIS_DATA_FIG4_PATH}")
print(f"  Saved: {cross_path}")

# ── 9. Quick validation ─────────────────────────────────────────────────

# Check Figure 1a R² target: adjusted R² ≈ 0.573 (1960), 0.602 (2010)
from statsmodels.api import OLS, add_constant

for yr, target in [(1960, 0.573), (2010, 0.602)]:
    mask = (df['Year'] == yr) & df['hc'].notna() & df['TFR'].notna()
    sub = df[mask]
    X = add_constant(sub['hc'])
    result = OLS(sub['TFR'], X).fit()
    print(f"\n  Fig1a validation: {yr} adj_R²={result.rsquared_adj:.3f} "
          f"(target={target}, N={len(sub)})")

# Check Figure 2a R² target: R² ≈ 0.009
mask = df_cross['hc_increase'].notna() & df_cross['tfr_change'].notna()
sub = df_cross[mask]
X = add_constant(sub['hc_increase'])
result = OLS(sub['tfr_change'], X).fit()
print(f"\n  Fig2a validation: R²={result.rsquared:.3f} (target=0.009, N={len(sub)})")

print("\n" + "=" * 60)
print("01_clean.py — DONE")
print("=" * 60)
