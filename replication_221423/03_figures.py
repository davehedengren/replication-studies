"""
03_figures.py — Replicate Figure 2 and Figure 3 for 221423-V1.

Paper: "Income Inequality in the Nordic Countries"
Figure 2: Participation tax rate vs employment subsidies (Kleven 2014)
Figure 3a: Union density by region (1980-2019)
Figure 3b: Bargaining coverage by region (1980-2019)
"""

import sys, os
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from utils import *

print('=' * 70)
print('03_figures.py — Figure replication')
print('=' * 70)

# ══════════════════════════════════════════════════════════════════════
# Figure 2: Tax rate vs Employment subsidies (Kleven 2014)
# ══════════════════════════════════════════════════════════════════════
print('\nFIGURE 2: Kleven tax/subsidy scatter')
print('=' * 70)

DATA_FILE = os.path.join(FIGURE2_DIR, 'kleven_jep2014_exceldata.xlsx')

# Load tax rate data (Figure 4a)
tax = pd.read_excel(DATA_FILE, sheet_name='Figure 4a', header=1)
# First column is source info, second should be country names
# From Stata: cellrange("A3"), keep Countryname and participationtaxrate
# The actual columns need inspection
tax_cols = tax.columns.tolist()
print(f'  Figure 4a columns: {tax_cols[:6]}')

# Read with header at row 3 (0-indexed row 2)
tax = pd.read_excel(DATA_FILE, sheet_name='Figure 4a', header=2)
tax_cols = tax.columns.tolist()
print(f'  Figure 4a columns: {tax_cols}')

# Stata: keep Countryname participationtaxrate
# Columns: 'Country name', '3-letter ISO...', 'Employment rate...', '1 - participation tax rate', ...
country_col = 'Country name'
tax_col = '1 - participation tax rate'
print(f'  Using country={country_col}, tax_rate={tax_col}')
tax_data = tax[[country_col, tax_col]].dropna().copy()
tax_data.columns = ['country', 'one_minus_parti']
tax_data['one_minus_parti'] = pd.to_numeric(tax_data['one_minus_parti'], errors='coerce')
tax_data = tax_data.dropna()

# Load subsidy data (Figure 5a)
sub = pd.read_excel(DATA_FILE, sheet_name='Figure 5a', header=1)
sub_cols = sub.columns.tolist()
print(f'  Figure 5a (header=1) columns: {sub_cols}')

# Stata: keep Countryname Employmentsubsidiesshareofl
sub_country_col = 'Country name'
sub_val_col = 'Employment subsidies (share of labor income)'
print(f'  Using country={sub_country_col}, subsidy={sub_val_col}')
sub_data = sub[[sub_country_col, sub_val_col]].dropna().copy()
sub_data.columns = ['country', 'subsidy']
sub_data['subsidy'] = pd.to_numeric(sub_data['subsidy'], errors='coerce')
sub_data = sub_data.dropna()

# Merge
scatter = tax_data.merge(sub_data, on='country')
print(f'  Merged: {len(scatter)} countries')

# Mark highlight countries
highlight = ['Denmark', 'Finland', 'Norway', 'Sweden', 'United Kingdom', 'United States']
scatter['mark'] = scatter.country.isin(highlight)

# Plot
fig, ax = plt.subplots(figsize=(10, 8))
unmarked = scatter[~scatter.mark]
marked = scatter[scatter.mark]

ax.scatter(unmarked.subsidy, unmarked.one_minus_parti, s=20, c='gray', alpha=0.6)
ax.scatter(marked.subsidy, marked.one_minus_parti, s=30, c='steelblue', zorder=5)

for _, row in marked.iterrows():
    offset = (5, 5) if row.country != 'Norway' else (-60, 5)
    ax.annotate(row.country, (row.subsidy, row.one_minus_parti),
                textcoords='offset points', xytext=offset, fontsize=9)

ax.set_xlabel('Participation subsidies (share of labor income)')
ax.set_ylabel('1 - participation tax rate')
ax.set_xlim(0, 0.07)
ax.set_ylim(0, 0.7)
ax.set_xticks(np.arange(0, 0.08, 0.01))
ax.set_yticks(np.arange(0, 0.8, 0.1))
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'figure2.png'), dpi=150)
plt.close()
print('  Saved: figure2.png')


# ══════════════════════════════════════════════════════════════════════
# Figure 3: Union density & Bargaining coverage
# ══════════════════════════════════════════════════════════════════════
print('\nFIGURE 3: Union density & Bargaining coverage')
print('=' * 70)

# Load data
adt = pd.read_csv(os.path.join(FIGURE3_DIR, 'OECD-AIAS-ICTWSS-CSV.csv'))
lbf = pd.read_csv(os.path.join(FIGURE3_DIR, 'OECD-Labor-Force.csv'))
lbf = lbf[['Country', 'TIME', 'Value']].rename(
    columns={'Country': 'country', 'TIME': 'year', 'Value': 'workers'})

# Country groups
eulist = ['Norway', 'Germany', 'France', 'Spain', 'Portugal',
          'Sweden', 'Denmark', 'Finland', 'United Kingdom']

# Fix country names
adt.loc[adt.country == 'United States of America', 'country'] = 'United States'

# Set Spain 1977-1979 UD to NA
adt.loc[(adt.country == 'Spain') & (adt.year.isin([1977, 1978, 1979])), 'UD'] = np.nan
adt.loc[(adt.country == 'Spain') & (adt.year.isin([1977, 1978, 1979])), 'UD_hist'] = np.nan

# Subset to relevant countries
countries_use = eulist + ['United States']
snt = adt[adt.country.isin(countries_use)][['country', 'year', 'UD', 'UD_s', 'UD_hist',
                                              'WCB', 'WCB_s', 'ExtE',
                                              'UnadjCov', 'UnadjCov_s', 'AdjCov_hist']].copy()

# Replace negatives with NA
for col in snt.columns[2:]:
    snt.loc[snt[col] < 0, col] = np.nan

# Effective UD: prefer UD over UD_s
snt['UD_eff'] = snt['UD'].where(snt['UD'].notna() & (snt['UD'] > 0), snt['UD_s'])

# Effective coverage: prefer UnadjCov_s over UnadjCov
snt['UnadjCov_eff'] = snt['UnadjCov_s'].where(
    snt['UnadjCov_s'].notna() & (snt['UnadjCov_s'] > 0), snt['UnadjCov'])

# Merge with labor force
snt = snt.merge(lbf, on=['country', 'year'], how='left')

# Sort for interpolation
snt = snt.sort_values(['country', 'year']).reset_index(drop=True)

# Interpolation per country
for c in snt.country.unique():
    mask = snt.country == c
    for col in ['UD_hist', 'UnadjCov_eff', 'AdjCov_hist']:
        vals = snt.loc[mask, col].copy()
        snt.loc[mask, col + '_int'] = vals.interpolate(method='linear')

# Country-specific fills (matching R code)
# France UD: locf for 2010-2019
mask_fr = (snt.country == 'France') & (snt.year >= 2010) & (snt.year <= 2019)
snt.loc[mask_fr, 'UD_hist_int'] = snt.loc[snt.country == 'France', 'UD_hist_int'].ffill()

# France UnadjCov: locf 2010-2018, nocb <=2010 from 1980
mask_fr2 = (snt.country == 'France') & (snt.year >= 2010) & (snt.year <= 2018)
snt.loc[mask_fr2, 'UnadjCov_eff_int'] = snt.loc[snt.country == 'France', 'UnadjCov_eff_int'].ffill()
mask_fr3 = (snt.country == 'France') & (snt.year <= 2010) & (snt.year >= 1980)
snt.loc[mask_fr3, 'UnadjCov_eff_int'] = snt.loc[snt.country == 'France', 'UnadjCov_eff_int'].bfill()

# Norway and Finland coverage fills
for c in ['Norway', 'Finland']:
    mask_c = (snt.country == c) & (snt.year >= 2010) & (snt.year <= 2018)
    snt.loc[mask_c, 'UnadjCov_eff_int'] = snt.loc[snt.country == c, 'UnadjCov_eff_int'].ffill()
    snt.loc[mask_c, 'AdjCov_hist_int'] = snt.loc[snt.country == c, 'UnadjCov_eff_int'].ffill()

# Country labels
def assign_label(c):
    if c == 'United States': return 'A1'
    if c == 'United Kingdom': return 'A2'
    if c in ['France', 'Germany', 'Italy', 'Spain', 'Portugal']: return 'B'
    if c in ['Norway', 'Sweden', 'Denmark', 'Finland']: return 'C'
    return None

snt['country_label'] = snt.country.map(assign_label)
snt = snt[snt.country_label.notna()].copy()

# Compute weighted means for Figure 3a (union density)
plot1 = snt.dropna(subset=['UD_hist_int', 'workers']).copy()
plot1 = plot1[(plot1.UD_hist_int > 0) & (plot1.year >= 1980) & (plot1.year < 2020)]
plot1_agg = plot1.groupby(['year', 'country_label']).apply(
    lambda g: np.average(g['UD_hist_int'], weights=g['workers']),
    include_groups=False
).reset_index(name='UD_mean')

# Compute weighted means for Figure 3b (bargaining coverage)
plot2 = snt.dropna(subset=['UnadjCov_eff_int', 'workers']).copy()
plot2 = plot2[(plot2.UnadjCov_eff_int > 0) & (plot2.year >= 1980) & (plot2.year < 2020)]
plot2_agg = plot2.groupby(['year', 'country_label']).apply(
    lambda g: np.average(g['UnadjCov_eff_int'], weights=g['workers']),
    include_groups=False
).reset_index(name='AdjCov_mean')

# Plot settings
namelist = {'A1': 'United States', 'A2': 'United Kingdom',
            'B': 'Continental Europe', 'C': 'Nordic Countries'}
colors = {'A1': 'goldenrod', 'A2': 'red', 'B': 'dodgerblue', 'C': 'mediumseagreen'}
markers = {'A1': 'o', 'A2': 's', 'B': 'D', 'C': '^'}

# Figure 3a: Union Density
fig, ax = plt.subplots(figsize=(10, 8))
for label in ['A1', 'A2', 'B', 'C']:
    d = plot1_agg[plot1_agg.country_label == label].sort_values('year')
    if len(d) == 0:
        continue
    ax.plot(d.year, d.UD_mean, color=colors[label], linewidth=1.5,
            marker=markers[label], markersize=3, label=namelist[label])
    # Larger markers at start and end
    for yr in [d.year.min(), d.year.max()]:
        row = d[d.year == yr]
        if len(row) > 0:
            ax.plot(row.year, row.UD_mean, color=colors[label],
                    marker=markers[label], markersize=8)

ax.set_xlim(1980, 2020)
ax.set_ylim(0, 100)
ax.set_yticks(range(0, 110, 10))
ax.set_xticks(range(1980, 2030, 10))
ax.set_ylabel('Union Density (%)', fontsize=14)
ax.legend(loc='lower left', ncol=2, fontsize=11)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'figure3a.png'), dpi=150)
plt.close()
print('  Saved: figure3a.png')

# Figure 3b: Bargaining Coverage
fig, ax = plt.subplots(figsize=(10, 8))
for label in ['A1', 'A2', 'B', 'C']:
    d = plot2_agg[plot2_agg.country_label == label].sort_values('year')
    if len(d) == 0:
        continue
    ax.plot(d.year, d.AdjCov_mean, color=colors[label], linewidth=1.5,
            marker=markers[label], markersize=3, label=namelist[label])
    for yr in [d.year.min(), d.year.max()]:
        row = d[d.year == yr]
        if len(row) > 0:
            ax.plot(row.year, row.AdjCov_mean, color=colors[label],
                    marker=markers[label], markersize=8)

ax.set_xlim(1980, 2020)
ax.set_ylim(0, 100)
ax.set_yticks(range(0, 110, 10))
ax.set_xticks(range(1980, 2030, 10))
ax.set_ylabel('Bargaining Coverage (%)', fontsize=14)
ax.legend(loc='lower left', ncol=2, fontsize=11)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'figure3b.png'), dpi=150)
plt.close()
print('  Saved: figure3b.png')

# Save data
plot1_agg.to_parquet(os.path.join(OUTPUT_DIR, 'figure3a_data.parquet'), index=False)
plot2_agg.to_parquet(os.path.join(OUTPUT_DIR, 'figure3b_data.parquet'), index=False)

print('\n' + '=' * 70)
print('03_figures.py — DONE')
print('=' * 70)
