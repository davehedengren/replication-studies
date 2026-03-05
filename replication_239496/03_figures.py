"""
03_figures.py — Reproduce Figures 1-7 (except 5b which needs India DHS).

Geruso & Spears (2026) "The Likelihood of Persistently Low Global Fertility"
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from utils import (OUTPUT_DIR, load_wpp, load_ccf, load_parity,
                   load_romania_ccf, load_births_history, load_births_projections,
                   HFD_COUNTRIES_MAIN, HFD_SUBNATIONAL, HFD_TO_ISO3, HFD_NAMES,
                   NON_UN_TERRITORIES)
import os

print("=" * 60)
print("03_figures.py — Reproducing all figures")
print("=" * 60)


# ══════════════════════════════════════════════════════════════════════
# FIGURE 1: World births spike (1700-2300)
# ══════════════════════════════════════════════════════════════════════

print("\n── Figure 1: World births spike ──")

# Historical births before 1950
hist = load_births_history()
hist = hist[hist['year'] < 1950][['year', 'births']].copy()
hist['source'] = 'history'

# WPP estimates 1950-2023
wpp = load_wpp('Estimates')
world_est = wpp[wpp['Region'] == 'World'][['year', 'births_thousands']].copy()
world_est['births'] = world_est['births_thousands'] * 1000
world_est['source'] = 'WPP_est'
world_est = world_est[['year', 'births', 'source']]

# WPP medium projections 2024-2100
wpp_med = load_wpp('Medium variant')
world_med = wpp_med[wpp_med['Region'] == 'World'][['year', 'births_thousands']].copy()
world_med['births'] = world_med['births_thousands'] * 1000
world_med['source'] = 'WPP_med'
world_med = world_med[['year', 'births', 'source']]

# Spears et al. projections 2101-2300 (TFR→1.5)
proj = load_births_projections()
proj['tfr_scenario'] = proj['tfr_scenario'].astype(str)
spears = proj[(proj['age'] == 'all') & (proj['tfr_scenario'] == '1.5')].copy()
spears['births'] = spears['births'] / 5  # 5-year aggregations to single years
spears = spears[spears['year'] > 2100][['year', 'births']].copy()
spears['source'] = 'spears'

all_births = pd.concat([hist, world_est, world_med, spears], ignore_index=True)
all_births = all_births.sort_values('year')
all_births['million_births'] = all_births['births'] / 1e6

fig, ax = plt.subplots(figsize=(10, 6.5))

# Observed 1700-2023
mask1 = (all_births['year'] >= 1700) & (all_births['year'] < 2024)
d1 = all_births[mask1]
ax.plot(d1['year'], d1['million_births'], 'k-', linewidth=2)

# UN Medium 2024-2100
mask2 = (all_births['year'] >= 2024) & (all_births['year'] < 2101)
d2 = all_births[mask2]
ax.plot(d2['year'], d2['million_births'], '-', color='gray', linewidth=2)

# Spears et al. 2101-2300
mask3 = (all_births['year'] >= 2101) & (all_births['year'] < 2300)
d3 = all_births[mask3]
ax.plot(d3['year'], d3['million_births'], '--', color='gray', linewidth=2)

ax.axvline(2023, color='lightgray', linewidth=0.8)
ax.axvline(2100, color='lightgray', linewidth=0.8)
ax.set_xlabel('Year', fontsize=12)
ax.set_ylabel('Births per year (millions)', fontsize=12)
ax.set_xlim(1700, 2300)
ax.set_ylim(0, 150)
ax.set_xticks(range(1700, 2301, 100))
ax.text(2030, 5, 'UN Medium\nProjections\n2024-2100', fontsize=8, color='gray')
ax.text(2110, 5, 'Spears et al. (2024)\nProjection if TFR → 1.5', fontsize=8, color='gray')

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'Figure1_births_spike.png'), dpi=150)
plt.close()
print("  Figure1_births_spike.png saved")


# ══════════════════════════════════════════════════════════════════════
# FIGURE 2: World TFR and fraction of population below replacement
# ══════════════════════════════════════════════════════════════════════

print("\n── Figure 2: World TFR and fraction sub-2 ──")

wpp = load_wpp('Estimates')
ca = wpp[wpp['Type'] == 'Country/Area'].copy()
world = wpp[wpp['Region'] == 'World'].copy()

# Compute fraction of population in TFR<2 countries by year
ca['subtwo'] = ca['tfr'] < 2
total_pop = ca.groupby('year')['pop_thousands'].sum()
sub2_pop = ca[ca['subtwo']].groupby('year')['pop_thousands'].sum()
frac_sub2 = (sub2_pop / total_pop).fillna(0)

fig, ax1 = plt.subplots(figsize=(10, 6.5))
ax2 = ax1.twinx()

# World TFR on left axis
ax1.plot(world['year'], world['tfr'], 'k-', linewidth=2)
ax1.set_ylabel('Total Fertility Rate', fontsize=12)
ax1.set_ylim(2, 5)
ax1.set_xlabel('Year', fontsize=12)

# Fraction below 2 on right axis
ax2.plot(frac_sub2.index, frac_sub2.values, '-', color='gray', linewidth=1.5)
ax2.set_ylabel('Fraction Below 2', fontsize=12, color='gray')
ax2.set_ylim(0, 1)
ax2.tick_params(axis='y', colors='gray')

ax1.set_xlim(1950, 2023)
ax1.text(1952, 4.6, 'World total\nfertility rate', fontsize=9, color='black')
ax1.text(1952, 2.3, 'Fraction of world\npopulation in a country\nwith TFR below 2', fontsize=9, color='gray')

# Annotate key events
ax2.annotate('China falls\nbelow 2 →', xy=(1991, frac_sub2.get(1991, 0.28)),
             fontsize=8, color='gray', ha='right')
ax2.annotate('India falls\nbelow 2 →', xy=(2020, frac_sub2.get(2020, 0.55)),
             fontsize=8, color='gray', ha='right')

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'Figure2_TFR_sub2.png'), dpi=150)
plt.close()
print("  Figure2_TFR_sub2.png saved")

# Print key statistics
for year in [1950, 1975, 2000, 2023]:
    val = frac_sub2.get(year, 0)
    print(f"    Fraction pop sub-2 in {year}: {val:.3f}")


# ══════════════════════════════════════════════════════════════════════
# FIGURE 3: CCF vs TFR by country
# ══════════════════════════════════════════════════════════════════════

print("\n── Figure 3: CCF vs TFR by country ──")

wpp = load_wpp('Estimates')
wpp_tfr = wpp[wpp['Type'] == 'Country/Area'][['ISO3', 'year', 'tfr']].copy()
wpp_tfr = wpp_tfr.dropna(subset=['ISO3'])
wpp_tfr['cohort'] = wpp_tfr['year'] - 30

ccf = load_ccf()
ccf_main = ccf[~ccf['country'].isin(HFD_SUBNATIONAL)].copy()
ccf_main['ISO3'] = ccf_main['country'].map(HFD_TO_ISO3)

# Merge
merged = wpp_tfr.merge(ccf_main[['country', 'cohort', 'ccf', 'ISO3']],
                       on=['ISO3', 'cohort'], how='left')
has_hfd = merged.groupby('ISO3')['ccf'].apply(lambda x: x.notna().any())
hfd_isos = has_hfd[has_hfd].index.tolist()
merged = merged[merged['ISO3'].isin(hfd_isos)]

countries = sorted(merged['country'].dropna().unique())
n_countries = len(countries)
ncols = 6
nrows = (n_countries + ncols - 1) // ncols

fig, axes = plt.subplots(nrows, ncols, figsize=(20, 3.5 * nrows))
axes = axes.flatten()

for i, country in enumerate(countries):
    ax = axes[i]
    sub = merged[(merged['country'] == country) & (merged['year'] >= 1960)]
    ax.plot(sub['year'], sub['tfr'], 'k-', linewidth=1.5)
    sub_ccf = sub.dropna(subset=['ccf'])
    ax.plot(sub_ccf['year'], sub_ccf['ccf'], '--', color='gray', linewidth=1.5)
    name = HFD_NAMES.get(country, country)
    ax.set_title(name, fontsize=9)
    ax.set_ylim(1.0, 3.0)
    ax.set_xlim(1960, 2020)
    ax.set_xticks([1960, 1980, 2000, 2020])

# Hide empty axes
for i in range(n_countries, len(axes)):
    axes[i].set_visible(False)

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'Figure3_CCF_vs_TFR.png'), dpi=150)
plt.close()
print(f"  Figure3_CCF_vs_TFR.png saved ({n_countries} countries)")


# ══════════════════════════════════════════════════════════════════════
# FIGURE 5a: Children ever born among parous (HFD)
# ══════════════════════════════════════════════════════════════════════

print("\n── Figure 5a: CEB among parous (HFD) ──")

ccf = load_ccf()
parity = load_parity()

# Merge CCF and parity
hfd = ccf.merge(parity, on=['country', 'cohort'], how='outer')
hfd = hfd[~hfd['country'].isin(['DEUTNP', 'DEUTE', 'GBRTENW', 'GBR_NIR', 'GBR_SCO'])]

hfd['ceb_if_any'] = hfd['ccf'] / ((100 - hfd['childlessness']) / 100)

fig, ax = plt.subplots(figsize=(6, 6))

mask = hfd['ceb_if_any'].notna() & hfd['ccf'].notna()
ax.scatter(hfd.loc[mask, 'ccf'], hfd.loc[mask, 'ceb_if_any'],
           marker='x', color='gray', s=20, alpha=0.7)

# Fit line
from numpy.polynomial.polynomial import polyfit
x = hfd.loc[mask, 'ccf'].values
y = hfd.loc[mask, 'ceb_if_any'].values
b, m = polyfit(x, y, 1)
x_line = np.linspace(x.min(), x.max(), 100)
ax.plot(x_line, b + m * x_line, 'k-', linewidth=0.8)

# R² from OLS
import statsmodels.api as sm
res = sm.OLS(y, sm.add_constant(x)).fit()
r2 = res.rsquared
print(f"  R² (CEB_parous ~ CCF): {r2:.2f}")

# Count below 2.1 among parous
below21 = (hfd['ceb_if_any'] < 2.1) & hfd['ceb_if_any'].notna()
n_below = below21.sum()
n_total = hfd['ceb_if_any'].notna().sum()
print(f"  Country-cohorts with CEB_parous < 2.1: {n_below} of {n_total}")

ax.axhline(2.1, color='lightgray', linewidth=0.8)
ax.set_xlabel('Mean children ever born', fontsize=11)
ax.set_ylabel('Mean children ever born among parous', fontsize=11)
ax.text(1.8, 3.0, f'R² = {r2:.2f}', fontsize=10)
ax.set_aspect('equal')

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'Figure5a_CEB_parous.png'), dpi=150)
plt.close()
print("  Figure5a_CEB_parous.png saved")


# ══════════════════════════════════════════════════════════════════════
# FIGURE 6: Latest CCF vs Lowest CCF
# ══════════════════════════════════════════════════════════════════════

print("\n── Figure 6: Latest vs Lowest CCF ──")

ccf = load_ccf()
ccf_main = ccf[~ccf['country'].isin(HFD_SUBNATIONAL)].copy()
ccf_main = ccf_main.dropna(subset=['ccf'])

# Lowest and latest CCF per country
stats = ccf_main.groupby('country').agg(
    lowest_ccf=('ccf', 'min'),
    latest_cohort=('cohort', 'max')
).reset_index()

# Get latest CCF value
latest_vals = ccf_main.merge(stats[['country', 'latest_cohort']],
                              on='country')
latest_vals = latest_vals[latest_vals['cohort'] == latest_vals['latest_cohort']]
latest_vals = latest_vals[['country', 'ccf']].rename(columns={'ccf': 'latest_ccf'})

stats = stats.merge(latest_vals, on='country')

# Relabel for plotting
stats['label'] = stats['country'].replace({
    'FRATNP': 'FRA', 'DEUTNP': 'DEU', 'GBR_NP': 'GBR'
})

fig, ax = plt.subplots(figsize=(7, 7))

for _, row in stats.iterrows():
    ax.scatter(row['lowest_ccf'], row['latest_ccf'], marker='o',
               facecolors='none', edgecolors='black', s=60)
    ax.annotate(row['label'], (row['lowest_ccf'], row['latest_ccf']),
                fontsize=6, ha='left', va='bottom')

# 45-degree line
lims = [1.35, 2.25]
ax.plot(lims, lims, '-', linewidth=0.5, color='lightgray')

ax.set_xlabel('Completed Fertility of Lowest-Fertility Cohort', fontsize=11)
ax.set_ylabel('Completed Fertility of Latest Cohort', fontsize=11)
ax.set_aspect('equal')

# R²
from scipy import stats as sp_stats
slope, intercept, r_val, p_val, se = sp_stats.linregress(
    stats['lowest_ccf'], stats['latest_ccf'])
print(f"  R² (latest ~ lowest): {r_val**2:.4f}")

# Count countries where latest = lowest
same = (stats['latest_ccf'] == stats['lowest_ccf']).sum()
print(f"  Countries where latest = lowest: {same}/{len(stats)}")

# 0-for-24 statistic
ever_sub19 = ccf_main.groupby('country').apply(
    lambda g: (g['ccf'] < 1.9).any()).sum()
print(f"  Countries with cohorts ever below 1.9: {ever_sub19}")

# Check for rebounds
def check_rebound(g):
    sub19 = g[g['ccf'] < 1.9]
    if len(sub19) == 0:
        return False
    first_sub19 = sub19['cohort'].min()
    later = g[(g['cohort'] > first_sub19) & (g['ccf'] > 2.1)]
    return len(later) > 0

rebounds = ccf_main.groupby('country').apply(check_rebound).sum()
print(f"  Countries with rebound to >2.1: {rebounds}")

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'Figure6_latest_vs_lowest.png'), dpi=150)
plt.close()
print("  Figure6_latest_vs_lowest.png saved")


# ══════════════════════════════════════════════════════════════════════
# FIGURE 7: Romania vs Bulgaria (Decree 770)
# ══════════════════════════════════════════════════════════════════════

print("\n── Figure 7: Romania vs Bulgaria ──")

# Romania TFR and Bulgaria TFR from WPP
wpp = load_wpp('Estimates')
rom_tfr = wpp[(wpp['Region'] == 'Romania')][['year', 'tfr']].copy()
bgr_tfr = wpp[(wpp['Region'] == 'Bulgaria')][['year', 'tfr']].copy()

# Romania CCF from Ghetau
rom_ccf = load_romania_ccf()
rom_ccf['year'] = rom_ccf['cohort'] + 30  # Align cohort to year turned 30

# Bulgaria CCF from HFD
ccf = load_ccf()
bgr_ccf = ccf[ccf['country'] == 'BGR'].copy()
bgr_ccf['year'] = bgr_ccf['cohort'] + 30

# Panel B: Period TFR
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

# Shading for Decree 770 (1966-1989)
for ax in [ax1, ax2]:
    ax.axvspan(1966, 1989, alpha=0.1, color='gray')

mask_yr = (rom_tfr['year'] >= 1960) & (rom_tfr['year'] <= 2000)
ax1.plot(rom_tfr.loc[mask_yr, 'year'], rom_tfr.loc[mask_yr, 'tfr'],
         'k-', linewidth=2, label='Romania')
mask_yr = (bgr_tfr['year'] >= 1960) & (bgr_tfr['year'] <= 2000)
ax1.plot(bgr_tfr.loc[mask_yr, 'year'], bgr_tfr.loc[mask_yr, 'tfr'],
         '--', color='gray', linewidth=2, label='Bulgaria')
ax1.set_title('Period TFR', fontsize=12)
ax1.set_ylabel('Total Fertility Rate', fontsize=11)
ax1.set_xlabel('Year', fontsize=11)
ax1.set_ylim(1.5, 3.5)
ax1.text(1977, 3.4, 'Decree 770 in force\nin Romania', fontsize=8, ha='center')
ax1.legend(fontsize=9)

# Panel C: CCF
rom_plot = rom_ccf[(rom_ccf['year'] >= 1960) & (rom_ccf['year'] <= 2000)]
bgr_plot = bgr_ccf[(bgr_ccf['year'] >= 1960) & (bgr_ccf['year'] <= 2000)]
ax2.plot(rom_plot['year'], rom_plot['fd'], 'k-', linewidth=2, label='Romania')
ax2.plot(bgr_plot['year'], bgr_plot['ccf'], '--', color='gray', linewidth=2, label='Bulgaria')
ax2.set_title('Completed Cohort Fertility', fontsize=12)
ax2.set_ylabel('Completed Cohort Fertility', fontsize=11)
ax2.set_xlabel('Year', fontsize=11)
ax2.set_ylim(1.5, 3.5)
ax2.text(1977, 3.4, 'Decree 770 in force\nin Romania', fontsize=8, ha='center')
ax2.legend(fontsize=9)

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'Figure7_Romania_Bulgaria.png'), dpi=150)
plt.close()
print("  Figure7_Romania_Bulgaria.png saved")


print("\n" + "=" * 60)
print("03_figures.py — DONE")
print("=" * 60)
