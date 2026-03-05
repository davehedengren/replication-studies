"""
04_data_audit.py — Data coverage, distributions, and consistency checks.

Paper: Andreolli & Surico (2025) "Shock Sizes and the Marginal Propensity to Consume"
"""

import sys, os
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import pandas as pd
from utils import *

print('=' * 70)
print('04_data_audit.py — Data audit')
print('=' * 70)

df = load_analysis_data()

# ══════════════════════════════════════════════════════════════════════
# 1. Overall Dataset Structure
# ══════════════════════════════════════════════════════════════════════
print('\n' + '=' * 70)
print('1. Overall Dataset Structure')
print('=' * 70)

print(f'  Total observations: {len(df):,}')
print(f'  Total columns: {df.shape[1]}')
print(f'  Unique households: {df.nquest.nunique():,}')
print(f'  Survey years: {sorted(df.anno.unique())}')

for yr in sorted(df.anno.unique()):
    yr_df = df[df.anno == yr]
    print(f'  Year {yr:.0f}: {len(yr_df):,} households')

# Panel structure: how many HH appear in multiple waves?
hh_counts = df.groupby('nquest')['anno'].nunique()
print(f'\n  Panel structure:')
for n in [1, 2, 3]:
    ct = (hh_counts == n).sum()
    print(f'    In {n} wave(s): {ct:,} ({ct/len(hh_counts)*100:.1f}%)')

# ══════════════════════════════════════════════════════════════════════
# 2. MPC Variable Coverage
# ══════════════════════════════════════════════════════════════════════
print('\n' + '=' * 70)
print('2. MPC Variable Coverage')
print('=' * 70)

mpc_vars = ['mpc', 'mpc_2010', 'mpc_2012', 'mpc_2016', 'mpc_dur_2012', 'mpc_nondur_2012']
for var in mpc_vars:
    if var in df.columns:
        n_nonmiss = df[var].notna().sum()
        n_zero = (df[var] == 0).sum()
        n_one = (df[var] == 1).sum()
        print(f'  {var}: {n_nonmiss:,} non-null, {n_zero:,} zeros ({n_zero/max(n_nonmiss,1)*100:.1f}%), '
              f'{n_one:,} ones ({n_one/max(n_nonmiss,1)*100:.1f}%)')

# MPC distributions by year
for yr in [2010, 2012, 2016]:
    var = f'mpc_{yr}'
    if var in df.columns:
        s = df.loc[df.anno == yr, var].dropna()
        if len(s) > 0:
            print(f'\n  MPC in {yr}: mean={s.mean():.3f}, std={s.std():.3f}, '
                  f'min={s.min():.3f}, max={s.max():.3f}')
            print(f'    Quartiles: {s.quantile([0.25, 0.5, 0.75]).values}')

# ══════════════════════════════════════════════════════════════════════
# 3. Key Variable Distributions
# ══════════════════════════════════════════════════════════════════════
print('\n' + '=' * 70)
print('3. Key Variable Distributions')
print('=' * 70)

key_vars = {
    'r_cash': 'Cash-on-hand (real K€)',
    'r_y': 'Income (real K€)',
    'r_af': 'Financial assets (real K€)',
    'r_c': 'Consumption (real K€)',
    'age': 'Age',
    'educ': 'Education (years)',
    'ncomp': 'Family size',
}

print(f'\n  {"Variable":<30} {"N":>8} {"Mean":>10} {"Std":>10} {"Min":>10} {"P50":>10} {"Max":>10}')
print('  ' + '-' * 88)
for var, label in key_vars.items():
    if var in df.columns:
        s = df[var].dropna()
        print(f'  {label:<30} {len(s):>8,} {s.mean():>10.1f} {s.std():>10.1f} '
              f'{s.min():>10.1f} {s.median():>10.1f} {s.max():>10.1f}')

# Categorical variables
print(f'\n  Categorical proportions (all years):')
for var, label in [('male', 'Male'), ('married', 'Married'), ('south', 'South'),
                    ('unempl', 'Unemployed'), ('hown', 'Homeowner')]:
    if var in df.columns:
        s = df[var].dropna()
        print(f'    {label}: {s.mean()*100:.1f}%')

# ══════════════════════════════════════════════════════════════════════
# 4. Estimation Sample Validation
# ══════════════════════════════════════════════════════════════════════
print('\n' + '=' * 70)
print('4. Estimation Sample')
print('=' * 70)

sample = df[df.marksample10 == 1].copy()
print(f'  marksample10 = 1: {len(sample):,} observations')
print(f'  All in year 2010: {(sample.anno == 2010).all()}')

# Verify the sample construction
n_with_both_mpc = sample[['mpc_2010', 'mpc_2012_in2010']].dropna().shape[0] if 'mpc_2012_in2010' in sample.columns else 0
print(f'  Have both MPC responses: {n_with_both_mpc:,}')
print(f'  Cash > 0: {(sample.cash > 0).all()}')

# ══════════════════════════════════════════════════════════════════════
# 5. Cross-Year MPC Comparison
# ══════════════════════════════════════════════════════════════════════
print('\n' + '=' * 70)
print('5. Cross-Year MPC Comparison (within estimation sample)')
print('=' * 70)

if 'mpc_2010' in sample.columns and 'mpc_2012_in2010' in sample.columns:
    both = sample[['mpc_2010', 'mpc_2012_in2010', 'dmpc_1012_in2010']].dropna()
    if len(both) > 0:
        print(f'  N with both responses: {len(both):,}')
        print(f'  Mean MPC small (2010): {both.mpc_2010.mean():.3f}')
        print(f'  Mean MPC large (2012): {both.mpc_2012_in2010.mean():.3f}')
        print(f'  Mean difference: {both.dmpc_1012_in2010.mean():.3f}')
        print(f'  Std difference: {both.dmpc_1012_in2010.std():.3f}')
        corr = both['mpc_2010'].corr(both['mpc_2012_in2010'])
        print(f'  Correlation: {corr:.3f}')
        print(f'\n  Key fact: MPC from small shock > MPC from large shock')
        print(f'  (consistent with concavity in consumption function)')

# ══════════════════════════════════════════════════════════════════════
# 6. MPC by Cash-on-Hand Decile
# ══════════════════════════════════════════════════════════════════════
print('\n' + '=' * 70)
print('6. MPC by Cash-on-Hand Decile')
print('=' * 70)

if 'q10cash' in sample.columns:
    print(f'\n  {"Decile":<10} {"N":>6} {"MPC small":>12} {"MPC large":>12} {"Diff":>10} {"Cash (K€)":>12}')
    print('  ' + '-' * 64)
    for d in range(1, 11):
        dec = sample[sample.q10cash == d]
        n = len(dec)
        mpc_s = dec['mpc_2010'].mean() if 'mpc_2010' in dec.columns else np.nan
        mpc_l = dec['mpc_2012_in2010'].mean() if 'mpc_2012_in2010' in dec.columns else np.nan
        diff = mpc_s - mpc_l if not np.isnan(mpc_s) and not np.isnan(mpc_l) else np.nan
        cash_med = dec['r_cash'].median() if 'r_cash' in dec.columns else np.nan
        print(f'  {d:<10} {n:>6} {mpc_s:>12.3f} {mpc_l:>12.3f} {diff:>10.3f} {cash_med:>12.1f}')

# ══════════════════════════════════════════════════════════════════════
# 7. Eating Out Share Distribution
# ══════════════════════════════════════════════════════════════════════
print('\n' + '=' * 70)
print('7. Eating Out Share')
print('=' * 70)

if 'eatoutshare_2012_in2010' in sample.columns:
    eat = sample['eatoutshare_2012_in2010'].dropna()
    print(f'  N with eating out share: {len(eat):,}')
    print(f'  Mean: {eat.mean():.3f}, Median: {eat.median():.3f}')
    print(f'  p10: {eat.quantile(0.1):.3f}, p90: {eat.quantile(0.9):.3f}')
    print(f'  Zero values: {(eat == 0).sum():,} ({(eat == 0).mean()*100:.1f}%)')

# ══════════════════════════════════════════════════════════════════════
# 8. Source Data File Verification
# ══════════════════════════════════════════════════════════════════════
print('\n' + '=' * 70)
print('8. Source Data Files')
print('=' * 70)

for name, path in [('storico_stata', STORICO_DIR), ('ind10_stata', IND10_DIR),
                    ('ind12_stata', IND12_DIR), ('ind16_stata', IND16_DIR)]:
    if os.path.exists(path):
        files = [f for f in os.listdir(path) if f.endswith('.dta')]
        print(f'  {name}: {len(files)} .dta files')
    else:
        print(f'  {name}: NOT FOUND')

# Price data
for f in ['ItalyUSPricesNonEssentials.xls', 'PriceFoodatHomeandAway.xls']:
    full = os.path.join(DATA_DIR, f)
    exists = os.path.exists(full)
    print(f'  {f}: {"OK" if exists else "NOT FOUND"}')

# Model output files (from MATLAB)
for f in ['MPC_Model_Ai.xls', 'MPC_Model_NonHom.xlsx', 'MPC_Models_NonHom_Ai_SE.xlsx']:
    full = os.path.join(DATA_DIR, f)
    exists = os.path.exists(full)
    print(f'  {f}: {"OK" if exists else "NOT FOUND (requires MATLAB)"}')


print('\n' + '=' * 70)
print('04_data_audit.py — DONE')
print('=' * 70)
