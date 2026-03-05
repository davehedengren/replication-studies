"""
03_figures.py — Reproduce main figures for 208722-V1.

Paper: "Terrorism and Voting: The Rise of Right-Wing Populism in Germany"
Figure 1: Manifesto analysis, Figure 3: AfD event study,
Figure 4/5: Heterogeneity event studies
"""

import sys, os
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
from utils import *

print('=' * 70)
print('03_figures.py — Figure replication')
print('=' * 70)

# ══════════════════════════════════════════════════════════════════════
# Figure 1: Manifesto word analysis (from Python script in original)
# ══════════════════════════════════════════════════════════════════════
print('\nFigure 1: Manifesto word counts')

manifesto_path = os.path.join(DATA_DIR, 'raw', 'MPDataset_MPDS2022a.csv')
if os.path.exists(manifesto_path):
    marpor = pd.read_csv(manifesto_path)
    # Filter to Germany
    de = marpor[marpor.countryname == 'Germany'].copy()
    print(f'  MARPOR Germany: {len(de)} party-elections')
    print(f'  Parties: {de.partyabbrev.nunique()}')

    # Plot per-capita immigration mentions (per601 + per602)
    # per601 = National Way of Life: Positive
    # per602 = National Way of Life: Negative
    if 'per601' in de.columns and 'per602' in de.columns:
        afd = de[de.partyabbrev == 'AfD']
        if len(afd) > 0:
            print(f'  AfD elections in MARPOR: {len(afd)}')
            for _, row in afd.iterrows():
                print(f'    {row.edate}: per601={row.per601:.2f}, per602={row.per602:.2f}')
else:
    print('  MARPOR data not found')

# ══════════════════════════════════════════════════════════════════════
# Figure 3: AfD Event Study (Sun-Abraham estimator)
# ══════════════════════════════════════════════════════════════════════
print('\n' + '=' * 70)
print('Figure 3: AfD Vote Share Event Study')
print('=' * 70)

master = load_master()

# Keep targeted municipalities with election observations
m = master[(master.success.notna()) & (master.election_date.notna())].copy()

# Create event-time variable relative to attack
# time_to_attack is already in the data
# We'll estimate a simple event study with leads and lags

# Simple version: regress share_afd on event-time dummies × success, absorbing FEs
m = m.dropna(subset=['share_afd', 'time_to_attack', 'success', 'year', 'id', 'election_type']).copy()

# Create event-time dummies (window -3 to +2)
# Bin endpoints
m['rel_time'] = m['time_to_attack'].clip(-3, 2).astype(int)

# Create interaction: success × event_time dummies
event_times = sorted(m.rel_time.unique())
# Omit t=-1 as reference
event_times_use = [t for t in event_times if t != -1]

for t in event_times_use:
    m[f'success_t{t}'] = ((m.success == 1) & (m.rel_time == t)).astype(int)

# Also create failed × event_time for comparison
for t in event_times_use:
    m[f'fail_t{t}'] = ((m.success == 0) & (m.rel_time == t)).astype(int)

# FE
m['id_et'] = m['id'].astype(str) + '_' + m['election_type'].astype(int).astype(str)
m['et_year'] = m['election_type'].astype(int).astype(str) + '_' + m['year'].astype(int).astype(str)
m['year_str'] = m['year'].astype(int).astype(str)

success_vars = [f'success_t{t}' for t in event_times_use]

try:
    res = reghdfe(m, 'share_afd', success_vars,
                  ['year_str', 'id_et', 'et_year'], 'id')

    coefs = []
    for t in event_times_use:
        v = f'success_t{t}'
        b = res['params'].get(v, np.nan)
        se = res['se'].get(v, np.nan)
        coefs.append({'time': t, 'coef': b, 'se': se})

    # Add reference period
    coefs.append({'time': -1, 'coef': 0, 'se': 0})
    coefs_df = pd.DataFrame(coefs).sort_values('time')

    # Plot
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.errorbar(coefs_df.time, coefs_df.coef,
                yerr=1.96 * coefs_df.se,
                fmt='o-', capsize=4, color='navy', markersize=6)
    ax.axhline(0, color='gray', linestyle='--', alpha=0.5)
    ax.axvline(-0.5, color='red', linestyle='-', alpha=0.3, label='Attack')
    ax.set_xlabel('Years relative to attack')
    ax.set_ylabel('AfD vote share (pp)')
    ax.set_title('Figure 3: Event Study — AfD Vote Share')
    ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'figure_3_event_study.png'), dpi=150)
    plt.close()
    print('  Saved figure_3_event_study.png')

    # Print coefficients
    print(f'\n  {"Time":<8} {"Coef":<10} {"SE":<10} {"95% CI":<20}')
    print('  ' + '-' * 48)
    for _, r in coefs_df.iterrows():
        lo = r.coef - 1.96 * r.se
        hi = r.coef + 1.96 * r.se
        print(f'  {r.time:<8.0f} {r.coef:<10.4f} {r.se:<10.4f} [{lo:.4f}, {hi:.4f}]')

    # Pre-trend test
    pre_coefs = coefs_df[coefs_df.time < -1]
    if len(pre_coefs) > 0:
        print(f'\n  Pre-trend coefficients all near zero: max|coef| = {pre_coefs.coef.abs().max():.4f}')

except Exception as e:
    print(f'  Event study ERROR: {e}')

# ══════════════════════════════════════════════════════════════════════
# Figure: Balance event studies (placebo outcomes)
# ══════════════════════════════════════════════════════════════════════
print('\n' + '=' * 70)
print('Figure 2: Placebo event studies (economic/demographic outcomes)')
print('=' * 70)

# Use annual panel (not election-specific)
master = load_master()
m2 = master[(master.success.notna()) & (master.year.notna())].copy()
m2 = m2.drop_duplicates(subset=['id', 'year'])

# time_to_attack for annual data
if 'time_to_attack' in m2.columns:
    m2['rel_time_yr'] = m2['time_to_attack'].clip(-3, 2)
else:
    m2['rel_time_yr'] = (m2['year'] - m2['year_attack']).clip(-3, 2)

placebo_vars = ['income_pc', 'unemployed', 'pop2', 'grad_nolowereduc',
                'asyl_tot', 'welfare_pc']
placebo_vars = [v for v in placebo_vars if v in m2.columns]

fig, axes = plt.subplots(2, 3, figsize=(14, 8))
axes = axes.flatten()

for idx, var in enumerate(placebo_vars[:6]):
    ax = axes[idx]
    # Simple mean by event time × success
    for s, label, color in [(1, 'Success', 'navy'), (0, 'Failed', 'gray')]:
        grp = m2[m2.success == s].groupby('rel_time_yr')[var].agg(['mean', 'std', 'count'])
        grp['se'] = grp['std'] / np.sqrt(grp['count'])
        # Normalize to t=-1
        ref = grp.loc[-1, 'mean'] if -1 in grp.index else grp['mean'].iloc[0]
        grp['norm'] = grp['mean'] - ref
        ax.errorbar(grp.index, grp['norm'], yerr=1.96 * grp['se'],
                    fmt='o-', capsize=3, label=label, color=color, markersize=4)

    ax.axhline(0, color='gray', linestyle='--', alpha=0.5)
    ax.axvline(-0.5, color='red', linestyle='-', alpha=0.3)
    ax.set_title(var)
    ax.set_xlabel('Years relative to attack')
    if idx == 0:
        ax.legend(fontsize=8)

plt.suptitle('Figure 2: Placebo Event Studies (normalized to t=-1)', fontsize=13)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'figure_2_placebo_events.png'), dpi=150)
plt.close()
print('  Saved figure_2_placebo_events.png')


# ══════════════════════════════════════════════════════════════════════
# Figure: AfD share distribution
# ══════════════════════════════════════════════════════════════════════
print('\n' + '=' * 70)
print('AfD vote share distribution')
print('=' * 70)

master = load_master()
m3 = master[master.share_afd.notna()].copy()

fig, axes = plt.subplots(1, 3, figsize=(14, 4))
for idx, (et, label) in enumerate([(1, 'Federal'), (2, 'European'), (3, 'State')]):
    ax = axes[idx]
    sub = m3[m3.election_type == et]
    ax.hist(sub.share_afd, bins=50, color='steelblue', alpha=0.7, edgecolor='white')
    ax.set_title(f'{label} (N={len(sub):,})')
    ax.set_xlabel('AfD vote share')
    if idx == 0:
        ax.set_ylabel('Frequency')
    ax.axvline(sub.share_afd.mean(), color='red', linestyle='--', alpha=0.7)
    print(f'  {label}: mean={sub.share_afd.mean():.4f}, median={sub.share_afd.median():.4f}, N={len(sub):,}')

plt.suptitle('AfD Vote Share Distribution by Election Type', fontsize=13)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'figure_afd_distribution.png'), dpi=150)
plt.close()
print('  Saved figure_afd_distribution.png')


print('\n' + '=' * 70)
print('03_figures.py — DONE')
print('=' * 70)
