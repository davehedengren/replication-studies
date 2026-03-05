"""
04_data_audit.py — Data coverage, distributions, and consistency checks.

Paper: "Terrorism and Voting: The Rise of Right-Wing Populism in Germany"
"""

import sys, os
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import pandas as pd
from utils import *

print('=' * 70)
print('04_data_audit.py — Data audit')
print('=' * 70)

master = load_master()
terror = load_terror()

# ══════════════════════════════════════════════════════════════════════
# 1. Overall Dataset Structure
# ══════════════════════════════════════════════════════════════════════
print('\n' + '=' * 70)
print('1. Dataset Structure')
print('=' * 70)

print(f'  master.dta: {master.shape[0]:,} rows × {master.shape[1]} cols')
print(f'  Unique municipalities: {master.id.nunique():,}')
print(f'  Years: {sorted(master.year.dropna().unique().astype(int))}')
print(f'  Year range: {master.year.min():.0f}–{master.year.max():.0f}')

# Panel structure
yr_counts = master.groupby('id')['year'].nunique()
print(f'\n  Panel balance:')
for n in sorted(yr_counts.unique()):
    ct = (yr_counts == n).sum()
    if ct > 100:
        print(f'    {n} years: {ct:,} municipalities')

# ══════════════════════════════════════════════════════════════════════
# 2. Treatment Variable
# ══════════════════════════════════════════════════════════════════════
print('\n' + '=' * 70)
print('2. Treatment Variable: success')
print('=' * 70)

# success is only non-missing for targeted municipalities
targeted = master[master.success.notna()]
print(f'  Targeted observations: {len(targeted):,}')
print(f'  Targeted municipalities: {targeted.id.nunique()}')
print(f'  Success=1: {(targeted.success == 1).sum():,} obs from {targeted[targeted.success == 1].id.nunique()} municipalities')
print(f'  Success=0: {(targeted.success == 0).sum():,} obs from {targeted[targeted.success == 0].id.nunique()} municipalities')

# Attack timing
if 'year_attack' in targeted.columns:
    atk_yr = targeted.drop_duplicates('id')['year_attack'].dropna()
    print(f'\n  Attack year distribution:')
    for yr in sorted(atk_yr.unique()):
        print(f'    {yr:.0f}: {(atk_yr == yr).sum()} municipalities')

# ══════════════════════════════════════════════════════════════════════
# 3. Outcome Variable: AfD Vote Share
# ══════════════════════════════════════════════════════════════════════
print('\n' + '=' * 70)
print('3. Outcome: AfD Vote Share')
print('=' * 70)

afd = master[master.share_afd.notna()]
print(f'  Non-missing: {len(afd):,}')
print(f'  Mean: {afd.share_afd.mean():.4f}')
print(f'  Std: {afd.share_afd.std():.4f}')
print(f'  Min: {afd.share_afd.min():.4f}, Max: {afd.share_afd.max():.4f}')
print(f'  Percentiles: p10={afd.share_afd.quantile(0.1):.4f}, p50={afd.share_afd.median():.4f}, p90={afd.share_afd.quantile(0.9):.4f}')

# By election type
print(f'\n  By election type:')
for et in [1, 2, 3]:
    sub = afd[afd.election_type == et]
    label = {1: 'Federal', 2: 'European', 3: 'State'}[et]
    print(f'    {label}: mean={sub.share_afd.mean():.4f}, N={len(sub):,}')

# ══════════════════════════════════════════════════════════════════════
# 4. Key Covariates
# ══════════════════════════════════════════════════════════════════════
print('\n' + '=' * 70)
print('4. Key Covariate Distributions')
print('=' * 70)

key_vars = {
    'income_pc': 'Per capita income (000s)',
    'unemployed': 'Unemployed',
    'pop2': 'Population (000s)',
    'avg_age': 'Average age',
    'share_men': 'Share male',
    'asyl_tot_tsd': 'Asylum seekers (000s)',
    'grad_abitur': 'University eligible',
    'welfare_pc': 'Welfare recipients PC',
    'east': 'East Germany',
    'share_nsdap33': '1933 NSDAP share',
}

print(f'\n  {"Variable":<25} {"N":>8} {"Mean":>10} {"Std":>10} {"Min":>10} {"Max":>10}')
print('  ' + '-' * 73)
for var, label in key_vars.items():
    if var in master.columns:
        s = master[var].dropna()
        print(f'  {label:<25} {len(s):>8,} {s.mean():>10.3f} {s.std():>10.3f} '
              f'{s.min():>10.3f} {s.max():>10.3f}')

# ══════════════════════════════════════════════════════════════════════
# 5. Terror Attack Characteristics
# ══════════════════════════════════════════════════════════════════════
print('\n' + '=' * 70)
print('5. Terror Attack Characteristics (N=232)')
print('=' * 70)

print(f'  Total attacks: {len(terror)}')
print(f'  Successful: {(terror.success == 1).sum()} ({(terror.success == 1).mean()*100:.1f}%)')
print(f'  Failed: {(terror.success == 0).sum()} ({(terror.success == 0).mean()*100:.1f}%)')

for var in ['nkill', 'nwound']:
    if var in terror.columns:
        s = terror[var].dropna()
        print(f'  {var}: mean={s.mean():.2f}, max={s.max():.0f}, total={s.sum():.0f}')

print(f'\n  Weapon types:')
for var in ['explosives', 'firearms', 'melee']:
    if var in terror.columns:
        print(f'    {var}: {terror[var].sum():.0f} ({terror[var].mean()*100:.1f}%)')

print(f'\n  Motivations:')
for var in ['right_wing', 'nazi', 'left_wing']:
    if var in terror.columns:
        print(f'    {var}: {terror[var].sum():.0f} ({terror[var].mean()*100:.1f}%)')
if 'islamistic' in terror.columns:
    print(f'    islamist: {terror.islamistic.sum():.0f} ({terror.islamistic.mean()*100:.1f}%)')

# ══════════════════════════════════════════════════════════════════════
# 6. Election Coverage
# ══════════════════════════════════════════════════════════════════════
print('\n' + '=' * 70)
print('6. Election Coverage')
print('=' * 70)

elections = master[master.election_date.notna()]
print(f'  Election observations: {len(elections):,}')

# Unique election dates
edates = elections.drop_duplicates('election_date')[['election_date', 'election_type']].sort_values('election_date')
print(f'\n  Unique election dates: {len(edates)}')
for _, row in edates.iterrows():
    et_label = {1: 'Federal', 2: 'European', 3: 'State'}[int(row.election_type)]
    n = (elections.election_date == row.election_date).sum()
    print(f'    {row.election_date}: {et_label}, N={n:,}')

# ══════════════════════════════════════════════════════════════════════
# 7. Missing Data Patterns
# ══════════════════════════════════════════════════════════════════════
print('\n' + '=' * 70)
print('7. Missing Data')
print('=' * 70)

important_vars = ['share_afd', 'success', 'post_election', 'income_pc',
                  'pop2', 'unemployed', 'share_nsdap33', 'east']
for var in important_vars:
    if var in master.columns:
        n_miss = master[var].isna().sum()
        pct = n_miss / len(master) * 100
        print(f'  {var}: {n_miss:,} missing ({pct:.1f}%)')

# ══════════════════════════════════════════════════════════════════════
# 8. Source Data Files
# ══════════════════════════════════════════════════════════════════════
print('\n' + '=' * 70)
print('8. Source Data Files')
print('=' * 70)

for name, path in [('derived', DERIVED_DIR), ('raw', RAW_DIR)]:
    if os.path.exists(path):
        files = []
        for root, dirs, fnames in os.walk(path):
            for f in fnames:
                if not f.startswith('.'):
                    files.append(f)
        print(f'  {name}/: {len(files)} files')
    else:
        print(f'  {name}/: NOT FOUND')

print('\n' + '=' * 70)
print('04_data_audit.py — DONE')
print('=' * 70)
