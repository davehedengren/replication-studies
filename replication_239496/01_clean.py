"""
01_clean.py — Load, validate, and prepare all data sources.

Geruso & Spears (2026) "The Likelihood of Persistently Low Global Fertility"
"""

import numpy as np
import pandas as pd
from utils import (DATA_DIR, OUTPUT_DIR, load_wpp, load_ccf, load_parity,
                   load_romania_ccf, load_births_history, load_births_projections,
                   HFD_COUNTRIES_MAIN, HFD_SUBNATIONAL, HFD_TO_ISO3,
                   NON_UN_TERRITORIES)
import os

print("=" * 60)
print("01_clean.py — Validating and preparing data")
print("=" * 60)

# ══════════════════════════════════════════════════════════════════════
# 1. WPP2024 ESTIMATES
# ══════════════════════════════════════════════════════════════════════

print("\n── WPP2024 Estimates ──")
wpp = load_wpp('Estimates')
print(f"  Rows: {len(wpp)}, Cols: {wpp.shape[1]}")
print(f"  Year range: {wpp['year'].min()}-{wpp['year'].max()}")
print(f"  Types: {wpp['Type'].unique()}")

# Count countries/territories
countries_all = wpp[wpp['Type'] == 'Country/Area']['Region'].nunique()
print(f"  Total countries/territories: {countries_all}")

# Count UN member states
un_members = wpp[(wpp['Type'] == 'Country/Area') &
                 (~wpp['Region'].isin(NON_UN_TERRITORIES))]['Region'].nunique()
print(f"  UN member states: {un_members}")

# World TFR validation
world_tfr_1950 = wpp[(wpp['Region'] == 'World') & (wpp['year'] == 1950)]['tfr'].values[0]
world_tfr_2023 = wpp[(wpp['Region'] == 'World') & (wpp['year'] == 2023)]['tfr'].values[0]
print(f"  World TFR: 1950={world_tfr_1950:.2f}, 2023={world_tfr_2023:.2f}")

# Save (keep only essential columns to avoid mixed-type issues)
wpp_keep = ['Region', 'ISO3', 'year', 'Type', 'pop_thousands', 'tfr', 'births_thousands']
wpp_save = wpp[[c for c in wpp_keep if c in wpp.columns]].copy()
wpp_save.to_parquet(os.path.join(OUTPUT_DIR, 'wpp_estimates.parquet'), index=False)

# WPP Medium variant
wpp_med = load_wpp('Medium variant')
print(f"\n  WPP Medium variant: {len(wpp_med)} rows, year range {wpp_med['year'].min()}-{wpp_med['year'].max()}")
wpp_med_save = wpp_med[[c for c in wpp_keep if c in wpp_med.columns]].copy()
wpp_med_save.to_parquet(os.path.join(OUTPUT_DIR, 'wpp_medium.parquet'), index=False)


# ══════════════════════════════════════════════════════════════════════
# 2. CCF (COMPLETED COHORT FERTILITY)
# ══════════════════════════════════════════════════════════════════════

print("\n── CCF (HFD) ──")
ccf = load_ccf()
print(f"  Total rows (long): {len(ccf)}")
print(f"  Countries: {ccf['country'].nunique()}")
print(f"  Cohort range: {ccf['cohort'].min()}-{ccf['cohort'].max()}")

# Excluding sub-national
ccf_main = ccf[~ccf['country'].isin(HFD_SUBNATIONAL)].copy()
n_countries_main = ccf_main[ccf_main['ccf'].notna()]['country'].nunique()
print(f"  Countries (excl. sub-national): {n_countries_main}")

ccf.to_parquet(os.path.join(OUTPUT_DIR, 'ccf.parquet'), index=False)


# ══════════════════════════════════════════════════════════════════════
# 3. PARITY (CHILDLESSNESS)
# ══════════════════════════════════════════════════════════════════════

print("\n── PARITY (HFD) ──")
parity = load_parity()
print(f"  Total rows (long): {len(parity)}")
print(f"  Countries: {parity['country'].nunique()}")

parity.to_parquet(os.path.join(OUTPUT_DIR, 'parity.parquet'), index=False)


# ══════════════════════════════════════════════════════════════════════
# 4. ROMANIA CCF
# ══════════════════════════════════════════════════════════════════════

print("\n── Romania CCF (Ghetau 1997) ──")
rom = load_romania_ccf()
print(f"  Observations: {len(rom)}")
print(f"  Cohort range: {rom['cohort'].min()}-{rom['cohort'].max()}")
print(f"  CCF range: {rom['fd'].min():.3f}-{rom['fd'].max():.3f}")


# ══════════════════════════════════════════════════════════════════════
# 5. BIRTHS DATA
# ══════════════════════════════════════════════════════════════════════

print("\n── Historical births ──")
hist = load_births_history()
print(f"  Rows: {len(hist)}, Year range: {hist['year'].min()} to {hist['year'].max()}")

print("\n── Projected births ──")
proj = load_births_projections()
print(f"  Rows: {len(proj)}, Year range: {proj['year'].min()} to {proj['year'].max()}")
print(f"  TFR scenarios: {sorted(proj['tfr_scenario'].dropna().astype(str).unique())}")


# ══════════════════════════════════════════════════════════════════════
# 6. MERGED HFD + WPP (for Figures 3, 6)
# ══════════════════════════════════════════════════════════════════════

print("\n── Merged HFD + WPP ──")

# Get WPP TFR by ISO3 and year
wpp_tfr = wpp[wpp['Type'] == 'Country/Area'][['ISO3', 'year', 'tfr', 'Region']].copy()
wpp_tfr = wpp_tfr.dropna(subset=['ISO3'])
wpp_tfr['cohort'] = wpp_tfr['year'] - 30  # Align cohort to year when cohort turned 30

# CCF data (main countries only)
ccf_main = ccf[~ccf['country'].isin(HFD_SUBNATIONAL)].copy()
ccf_main['ISO3'] = ccf_main['country'].map(HFD_TO_ISO3)

# Merge on ISO3 + cohort
merged = wpp_tfr.merge(ccf_main[['country', 'cohort', 'ccf', 'ISO3']],
                       on=['ISO3', 'cohort'], how='left')

# Keep only countries that have HFD data
has_hfd = merged.groupby('ISO3')['ccf'].apply(lambda x: x.notna().any())
hfd_isos = has_hfd[has_hfd].index.tolist()
merged_hfd = merged[merged['ISO3'].isin(hfd_isos)].copy()

print(f"  Countries with both WPP and HFD: {len(hfd_isos)}")
print(f"  Merged rows: {len(merged_hfd)}")

merged_hfd.to_parquet(os.path.join(OUTPUT_DIR, 'merged_wpp_ccf.parquet'), index=False)


# ══════════════════════════════════════════════════════════════════════
# QUICK VALIDATION
# ══════════════════════════════════════════════════════════════════════

print("\n\n── Quick validation ──")

# Sub-two population fraction in 2023
ca = wpp[wpp['Type'] == 'Country/Area'].copy()
ca['subtwo'] = ca['tfr'] < 2
total_pop_2023 = ca[ca['year'] == 2023]['pop_thousands'].sum()
sub2_pop_2023 = ca[(ca['year'] == 2023) & (ca['subtwo'])]['pop_thousands'].sum()
frac_sub2 = sub2_pop_2023 / total_pop_2023
print(f"  Fraction of world pop in TFR<2 countries (2023): {frac_sub2:.3f}")
print(f"  (Paper says ~two-thirds)")

# Regional TFR 1950 vs 2023
for region in ['Europe', 'Northern America', 'South America', 'China', 'India', 'Sub-Saharan Africa']:
    t50 = wpp[(wpp['Region'] == region) & (wpp['year'] == 1950)]['tfr'].values
    t23 = wpp[(wpp['Region'] == region) & (wpp['year'] == 2023)]['tfr'].values
    if len(t50) > 0 and len(t23) > 0:
        print(f"  {region}: {t50[0]:.2f} (1950) → {t23[0]:.2f} (2023)")


print("\n" + "=" * 60)
print("01_clean.py — DONE")
print("=" * 60)
