"""
04_data_audit.py — Data quality checks for Geruso & Spears (2026).
"""

import numpy as np
import pandas as pd
from utils import (OUTPUT_DIR, load_wpp, load_ccf, load_parity,
                   load_romania_ccf, HFD_SUBNATIONAL, HFD_TO_ISO3,
                   NON_UN_TERRITORIES)
import os

print("=" * 60)
print("04_data_audit.py — Data quality audit")
print("=" * 60)


# ══════════════════════════════════════════════════════════════════════
# 1. COVERAGE
# ══════════════════════════════════════════════════════════════════════

print("\n── 1. Coverage ──\n")

wpp = load_wpp('Estimates')
ca = wpp[wpp['Type'] == 'Country/Area'].copy()

print(f"  WPP countries/territories: {ca['Region'].nunique()}")
print(f"  WPP year range: {ca['year'].min():.0f}-{ca['year'].max():.0f}")

# UN member states
un = ca[~ca['Region'].isin(NON_UN_TERRITORIES)]
print(f"  UN member states: {un['Region'].nunique()}")

# TFR coverage
tfr_miss = ca['tfr'].isna().sum()
print(f"  TFR missing: {tfr_miss}/{len(ca)} ({100*tfr_miss/len(ca):.1f}%)")

# Pop coverage
pop_miss = ca['pop_thousands'].isna().sum()
print(f"  Pop missing: {pop_miss}/{len(ca)} ({100*pop_miss/len(ca):.1f}%)")

# CCF coverage
ccf = load_ccf()
ccf_main = ccf[~ccf['country'].isin(HFD_SUBNATIONAL)]
ccf_valid = ccf_main.dropna(subset=['ccf'])
print(f"\n  HFD countries (excl sub-national): {ccf_main['country'].nunique()}")
print(f"  HFD country-cohort obs with CCF: {len(ccf_valid)}")
print(f"  Cohort range: {ccf_valid['cohort'].min()}-{ccf_valid['cohort'].max()}")

# Parity coverage
parity = load_parity()
par_main = parity[~parity['country'].isin(HFD_SUBNATIONAL)]
par_valid = par_main.dropna(subset=['childlessness'])
print(f"\n  PARITY country-cohort obs: {len(par_valid)}")

# Romania
rom = load_romania_ccf()
print(f"\n  Romania CCF: {len(rom)} obs, cohorts {rom['cohort'].min()}-{rom['cohort'].max()}")


# ══════════════════════════════════════════════════════════════════════
# 2. DISTRIBUTIONS
# ══════════════════════════════════════════════════════════════════════

print("\n── 2. Distributions ──\n")

# TFR distribution in 2023
tfr_2023 = ca[ca['year'] == 2023]['tfr']
print(f"  TFR 2023: mean={tfr_2023.mean():.2f}, median={tfr_2023.median():.2f}, "
      f"range=[{tfr_2023.min():.2f}, {tfr_2023.max():.2f}]")

# Fraction sub-replacement by year
for year in [1950, 1975, 2000, 2023]:
    sub = ca[(ca['year'] == year) & (ca['tfr'] < 2.1)]
    frac = len(sub) / len(ca[ca['year'] == year])
    print(f"  Fraction countries TFR<2.1 in {year}: {frac:.3f} ({len(sub)} countries)")

# CCF distribution
print(f"\n  CCF: mean={ccf_valid['ccf'].mean():.3f}, "
      f"median={ccf_valid['ccf'].median():.3f}, "
      f"range=[{ccf_valid['ccf'].min():.3f}, {ccf_valid['ccf'].max():.3f}]")

# Childlessness
print(f"  Childlessness: mean={par_valid['childlessness'].mean():.1f}%, "
      f"range=[{par_valid['childlessness'].min():.1f}%, {par_valid['childlessness'].max():.1f}%]")


# ══════════════════════════════════════════════════════════════════════
# 3. LOGICAL CONSISTENCY
# ══════════════════════════════════════════════════════════════════════

print("\n── 3. Logical consistency ──\n")

# CCF should be > 0 where non-missing
print(f"  CCF <= 0: {(ccf_valid['ccf'] <= 0).sum()}")

# Childlessness should be 0-100
print(f"  Childlessness outside [0,100]: {((par_valid['childlessness'] < 0) | (par_valid['childlessness'] > 100)).sum()}")

# CEB among parous should be >= CCF (since parous = removing childless)
merged = ccf.merge(parity, on=['country', 'cohort'])
merged = merged[~merged['country'].isin(HFD_SUBNATIONAL)]
merged = merged.dropna(subset=['ccf', 'childlessness'])
merged['ceb_if_any'] = merged['ccf'] / ((100 - merged['childlessness']) / 100)
bad = (merged['ceb_if_any'] < merged['ccf']).sum()
print(f"  CEB_parous < CCF: {bad} (should be 0)")

# Romania CCF should be positive
print(f"  Romania CCF <= 0: {(rom['fd'] <= 0).sum()}")
print(f"  Romania CCF monotone decreasing: {(rom['fd'].diff().dropna() <= 0).all()}")


# ══════════════════════════════════════════════════════════════════════
# 4. MISSING DATA PATTERNS
# ══════════════════════════════════════════════════════════════════════

print("\n── 4. Missing data patterns ──\n")

# CCF by cohort decade
for start in range(1930, 1980, 10):
    end = start + 9
    mask = (ccf_valid['cohort'] >= start) & (ccf_valid['cohort'] <= end)
    n = mask.sum()
    n_countries = ccf_valid.loc[mask, 'country'].nunique()
    print(f"  CCF {start}-{end}: {n} obs across {n_countries} countries")

# Which countries have the longest CCF series?
series_len = ccf_valid.groupby('country').size().sort_values(ascending=False)
print(f"\n  Longest CCF series: {series_len.head(5).to_dict()}")
print(f"  Shortest CCF series: {series_len.tail(5).to_dict()}")


# ══════════════════════════════════════════════════════════════════════
# 5. DUPLICATES
# ══════════════════════════════════════════════════════════════════════

print("\n── 5. Duplicates ──\n")

# WPP duplicates
wpp_dupes = ca.duplicated(subset=['Region', 'year']).sum()
print(f"  WPP duplicates: {wpp_dupes}")

# CCF duplicates
ccf_dupes = ccf_valid.duplicated(subset=['country', 'cohort']).sum()
print(f"  CCF duplicates: {ccf_dupes}")

# Parity duplicates
par_dupes = par_valid.duplicated(subset=['country', 'cohort']).sum()
print(f"  Parity duplicates: {par_dupes}")


# ══════════════════════════════════════════════════════════════════════
# 6. KEY CLAIM VERIFICATION
# ══════════════════════════════════════════════════════════════════════

print("\n── 6. Key claim verification ──\n")

# "0-for-24" or "0-for-26" claim
ccf_main = ccf[~ccf['country'].isin(HFD_SUBNATIONAL)].dropna(subset=['ccf'])

# Version 1: Excluding sub-national (main text: 0-for-24)
ever_sub19 = ccf_main.groupby('country').apply(lambda g: (g['ccf'] < 1.9).any()).sum()

def check_rebound(g):
    sub19 = g[g['ccf'] < 1.9]
    if len(sub19) == 0:
        return False
    first = sub19['cohort'].min()
    return ((g['cohort'] > first) & (g['ccf'] > 2.1)).any()

rebounds = ccf_main.groupby('country').apply(check_rebound).sum()
print(f"  Countries with CCF ever below 1.9 (main): {ever_sub19}")
print(f"  Of those, subsequent cohort above 2.1: {rebounds}")
print(f"  → 0-for-{ever_sub19} claim: {'CONFIRMED' if rebounds == 0 else 'NOT CONFIRMED'}")

# Version 2: Including sub-national but excluding whole country (footnote: 0-for-26)
ccf_fn = ccf[~ccf['country'].isin(['DEUTNP', 'GBR_NP'])].dropna(subset=['ccf'])
ever_sub19_fn = ccf_fn.groupby('country').apply(lambda g: (g['ccf'] < 1.9).any()).sum()
rebounds_fn = ccf_fn.groupby('country').apply(check_rebound).sum()
print(f"\n  Countries with CCF ever below 1.9 (footnote version): {ever_sub19_fn}")
print(f"  Of those, subsequent cohort above 2.1: {rebounds_fn}")
print(f"  → 0-for-{ever_sub19_fn} claim: {'CONFIRMED' if rebounds_fn == 0 else 'NOT CONFIRMED'}")


print("\n" + "=" * 60)
print("04_data_audit.py — DONE")
print("=" * 60)
