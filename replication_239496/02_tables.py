"""
02_tables.py — Reproduce Table 1 and Table A2 (Kitagawa decomposition).

Geruso & Spears (2026) "The Likelihood of Persistently Low Global Fertility"

Note: Table A1 requires India DHS data (IAIR7EFL.DTA) not available.
"""

import numpy as np
import pandas as pd
from utils import (OUTPUT_DIR, load_ccf, load_parity,
                   HFD_SUBNATIONAL, HFD_NAMES)
import os

print("=" * 60)
print("02_tables.py — Reproducing tables")
print("=" * 60)


# ══════════════════════════════════════════════════════════════════════
# LOAD AND PREPARE HFD DATA
# ══════════════════════════════════════════════════════════════════════

ccf = load_ccf()
parity = load_parity()

# Merge
hfd = ccf.merge(parity, on=['country', 'cohort'], how='outer')
# Drop sub-national and German national total for this analysis
hfd = hfd[~hfd['country'].isin(['DEUTNP', 'DEUTE', 'GBRTENW', 'GBR_NIR', 'GBR_SCO'])]

# Compute CEB among parous
hfd['ceb_if_any'] = hfd['ccf'] / ((100 - hfd['childlessness']) / 100)

# Country code
hfd['countrycode'] = hfd['country'].astype('category').cat.codes

# Lowest CCF cohort per country
min_ccf = hfd.dropna(subset=['ccf']).groupby('country')['ccf'].min().rename('min_ccf')
hfd = hfd.merge(min_ccf, on='country', how='left')
hfd['ismin'] = (hfd['ccf'] - hfd['min_ccf']).abs() < 0.0001

# Cohort group flags
hfd['c_35_59'] = (hfd['cohort'] >= 1935) & (hfd['cohort'] <= 1959)
hfd['c_60_79'] = (hfd['cohort'] >= 1960) & (hfd['cohort'] <= 1979)


# ══════════════════════════════════════════════════════════════════════
# TABLE 1: KITAGAWA DECOMPOSITION OF 20-YEAR DECLINE
# ══════════════════════════════════════════════════════════════════════

print("\n── TABLE 1: Kitagawa decomposition of 20-year decline ──\n")

# Identify latest cohort with data for each country
in_sample = hfd.dropna(subset=['ccf', 'ceb_if_any'])
latest = in_sample.groupby('country')['cohort'].max().rename('latest_cohort')
earliest = in_sample.groupby('country')['cohort'].min().rename('earliest_cohort')
hfd = hfd.merge(latest, on='country', how='left')
hfd = hfd.merge(earliest, on='country', how='left')

# "Twenty before" = latest_cohort - 20
hfd['is_twenty_before'] = (hfd['cohort'] == hfd['latest_cohort'] - 20)
# Only if this cohort has data
hfd.loc[hfd['ccf'].isna() | hfd['ceb_if_any'].isna(), 'is_twenty_before'] = False

# Check if country has a valid 20-year-prior cohort
has_20 = hfd[hfd['is_twenty_before']].groupby('country').size()
hfd['has_twenty_before'] = hfd['country'].isin(has_20.index)
hfd['is_latest'] = (hfd['cohort'] == hfd['latest_cohort']) & hfd['has_twenty_before']

# Panel A: Later cohort
later = hfd[hfd['is_latest']].copy()
earlier = hfd[hfd['is_twenty_before']].copy()

print("  Panel A: Comparing later cohort to 20-year-earlier cohort\n")
print(f"  {'':30s} {'Later cohort':>15s} {'Earlier cohort':>15s}")

n_later = len(later)
n_earlier = len(earlier)
print(f"  {'Cohorts in sample':<30s} {n_later:>15d} {n_earlier:>15d}")

mean_cohort_later = later['cohort'].mean()
mean_cohort_earlier = earlier['cohort'].mean()
print(f"  {'Average cohort year':<30s} {mean_cohort_later:>15.1f} {mean_cohort_earlier:>15.1f}")

mean_ccf_later = later['ccf'].mean()
mean_ccf_earlier = earlier['ccf'].mean()
print(f"  {'Avg completed fertility':<30s} {mean_ccf_later:>15.3f} {mean_ccf_earlier:>15.3f}")

mean_ceb_later = later['ceb_if_any'].mean()
mean_ceb_earlier = earlier['ceb_if_any'].mean()
print(f"  {'Avg CEB among parous':<30s} {mean_ceb_later:>15.3f} {mean_ceb_earlier:>15.3f}")

mean_cl_later = later['childlessness'].mean() / 100
mean_cl_earlier = earlier['childlessness'].mean() / 100
print(f"  {'Avg childlessness':<30s} {mean_cl_later:>15.3f} {mean_cl_earlier:>15.3f}")

# Kitagawa decomposition
ccf_d = mean_ccf_earlier - mean_ccf_later  # Earlier is higher
ceb_d = mean_ceb_earlier - mean_ceb_later
cl_d = -(mean_cl_earlier - mean_cl_later)  # Sign convention
cl_a = (mean_cl_earlier + mean_cl_later) / 2
ceb_a = (mean_ceb_earlier + mean_ceb_later) / 2

# Decomposition following Stata code
due_to_childlessness = ccf_d - ceb_d * (1 - cl_a)
frac_childlessness = due_to_childlessness / ccf_d
due_to_parous = ceb_d * (1 - cl_a)
frac_parous = due_to_parous / ccf_d

print(f"\n  Panel B: Kitagawa decomposition\n")
print(f"  Total difference in CCF: {ccf_d:.3f}")
print(f"  Due to childlessness: {due_to_childlessness:.3f} ({100*frac_childlessness:.1f}%)")
print(f"  Due to parous fertility: {due_to_parous:.3f} ({100*frac_parous:.1f}%)")
print(f"  (Paper reports ~38% due to childlessness, ~62% due to parous fertility)")


# ══════════════════════════════════════════════════════════════════════
# TABLE A2: COUNTRY-COHORT DETAILS
# ══════════════════════════════════════════════════════════════════════

print("\n\n── TABLE A2: Country-cohort details ──\n")

# List countries with both latest and 20-before cohorts
table_a2 = pd.concat([
    earlier[['country', 'cohort', 'ccf', 'ceb_if_any', 'childlessness']].assign(period='earlier'),
    later[['country', 'cohort', 'ccf', 'ceb_if_any', 'childlessness']].assign(period='later')
]).sort_values(['country', 'cohort'])

print(f"  {'Country':<10s} {'Cohort':>8s} {'CCF':>8s} {'CEB parous':>12s} {'Childless%':>12s}")
print("  " + "-" * 55)
for _, row in table_a2.iterrows():
    print(f"  {row['country']:<10s} {row['cohort']:>8.0f} {row['ccf']:>8.3f} "
          f"{row['ceb_if_any']:>12.3f} {row['childlessness']:>12.1f}")


# ══════════════════════════════════════════════════════════════════════
# ADDITIONAL IN-TEXT STATISTICS
# ══════════════════════════════════════════════════════════════════════

print("\n\n── In-text statistics ──\n")

# "In X of Y country-cohorts with CCF<2.1, CEB among parous was below 2.1"
sub21 = hfd[(hfd['ccf'] < 2.1) & hfd['ccf'].notna() & hfd['ceb_if_any'].notna()]
ceb_sub21 = (sub21['ceb_if_any'] < 2.1).sum()
print(f"  Country-cohorts with CCF<2.1: {len(sub21)}")
print(f"  Of those, CEB_parous < 2.1: {ceb_sub21}")

# Mean childlessness where CCF < 2.1
mean_cl_sub21 = sub21['childlessness'].mean()
print(f"  Mean childlessness (%) where CCF<2.1: {mean_cl_sub21:.1f}")

# Countries where all available cohorts have ceb_parous < 2.1
by_country = hfd.dropna(subset=['ceb_if_any']).groupby('country').apply(
    lambda g: (g['ceb_if_any'] < 2.1).all())
always_below = by_country[by_country].index.tolist()
print(f"  Countries where ALL cohorts have CEB_parous < 2.1: {always_below}")


print("\n" + "=" * 60)
print("02_tables.py — DONE")
print("=" * 60)
