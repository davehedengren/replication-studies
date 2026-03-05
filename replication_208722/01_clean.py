"""
01_clean.py — Load and validate the analysis datasets for 208722-V1.

Paper: "Terrorism and Voting: The Rise of Right-Wing Populism in Germany"

The original code builds master.dta from raw data in 7 Stata scripts.
We start from the pre-built derived datasets since all are provided.
"""

import sys, os
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import pandas as pd
from utils import *

print('=' * 70)
print('01_clean.py — Data loading and validation')
print('=' * 70)

# ══════════════════════════════════════════════════════════════════════
# Load datasets
# ══════════════════════════════════════════════════════════════════════
master = load_master()
terror = load_terror()
master_ps = load_master_propensity()

print(f'\nmaster.dta: {master.shape[0]:,} rows × {master.shape[1]} cols')
print(f'  Unique municipalities: {master.id.nunique():,}')
print(f'  Years: {sorted(master.year.dropna().unique().astype(int))}')
print(f'  Election types: {sorted(master.election_type.dropna().unique().astype(int))}')
print(f'    1=Federal, 2=European, 3=State')

print(f'\nterror.dta: {terror.shape[0]} attacks')
print(f'  Successful: {(terror.success == 1).sum()}')
print(f'  Failed: {(terror.success == 0).sum()}')

print(f'\nmaster_propensity.dta: {master_ps.shape[0]:,} rows × {master_ps.shape[1]} cols')

# ══════════════════════════════════════════════════════════════════════
# Validate key variables
# ══════════════════════════════════════════════════════════════════════
print('\n' + '=' * 70)
print('Key variable coverage in master.dta')
print('=' * 70)

# Targeted municipalities = those with success not missing
targeted = master[master.success.notna()]
print(f'\nTargeted municipalities (success not missing): {len(targeted):,} obs')
print(f'  Unique targeted IDs: {targeted.id.nunique()}')
print(f'  Success=1 (attack succeeded): {(targeted.success == 1).sum():,}')
print(f'  Success=0 (attack failed): {(targeted.success == 0).sum():,}')

# Election observations
elections = master[master.election_date.notna()]
print(f'\nElection observations: {len(elections):,}')
for et in [1, 2, 3]:
    n = (elections.election_type == et).sum()
    label = {1: 'Federal', 2: 'European', 3: 'State'}[et]
    print(f'  Type {et} ({label}): {n:,}')

# AfD vote share coverage
afd = master.share_afd.dropna()
print(f'\nAfD vote share: {len(afd):,} non-missing')
print(f'  Mean: {afd.mean():.4f}')
print(f'  Std: {afd.std():.4f}')
print(f'  Min: {afd.min():.4f}, Max: {afd.max():.4f}')

# Estimation sample: targeted + election obs + AfD non-missing
est_sample = master[(master.success.notna()) & (master.election_date.notna()) & (master.share_afd.notna())]
print(f'\nEstimation sample (targeted × election × AfD non-miss): {len(est_sample):,}')
print(f'  Unique municipalities: {est_sample.id.nunique()}')

# Terror data summary
print('\n' + '=' * 70)
print('Terror attack characteristics')
print('=' * 70)

for var in ['nkill', 'nwound', 'explosives', 'firearms', 'melee',
            'right_wing', 'nazi', 'left_wing', 'islamistic']:
    if var in terror.columns:
        s = terror[var].dropna()
        print(f'  {var}: mean={s.mean():.3f}, sum={s.sum():.0f}')

# ══════════════════════════════════════════════════════════════════════
# Save analysis-ready copy
# ══════════════════════════════════════════════════════════════════════
master.to_parquet(os.path.join(OUTPUT_DIR, 'master.parquet'))
terror.to_parquet(os.path.join(OUTPUT_DIR, 'terror.parquet'))
master_ps.to_parquet(os.path.join(OUTPUT_DIR, 'master_propensity.parquet'))

print(f'\nSaved: master.parquet, terror.parquet, master_propensity.parquet')

print('\n' + '=' * 70)
print('01_clean.py — DONE')
print('=' * 70)
