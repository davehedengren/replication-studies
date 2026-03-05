"""
01_clean.py — Build analysis dataset for Andreolli & Surico (2025).

Replicates f01_MergeDatasets.do + f02_ProcessRawData.do:
- Merges SHIW storico panel with MPC supplements (2010, 2012, 2016)
- Constructs variables: cash-on-hand, percentiles, demographics, MPC
- Creates panel lead/lag structure for within-household MPC comparisons
"""

import sys, os
sys.path.insert(0, os.path.dirname(__file__))

from utils import *

print('=' * 70)
print('01_clean.py — Build analysis dataset')
print('=' * 70)

# ── Step 1: Build and process the main panel ──────────────────────────────
df = build_main_panel()

# ── Step 2: Create panel lead/lag variables ───────────────────────────────
df = create_panel_variables(df)

# ── Step 3: Validation ────────────────────────────────────────────────────
print('\n-- Validation --')
print(f'  Total observations: {len(df):,}')
print(f'  Unique households: {df.nquest.nunique():,}')

for yr in sorted(df['anno'].unique()):
    yr_df = df[df['anno'] == yr]
    n_mpc = yr_df['mpc'].notna().sum()
    mean_mpc = yr_df['mpc'].mean()
    print(f'  Year {yr:.0f}: N={len(yr_df):,}, MPC non-null={n_mpc:,}, mean MPC={mean_mpc:.3f}' if n_mpc > 0 else f'  Year {yr:.0f}: N={len(yr_df):,}, no MPC data')

# Key sample sizes
n_sample10 = (df['marksample10'] == 1).sum()
print(f'\n  Estimation sample (marksample10=1): {n_sample10:,}')

# MPC distributions
sample = df[df['marksample10'] == 1]
if len(sample) > 0:
    print(f'\n  MPC (small shock, 2010):')
    print(f'    mean={sample["mpc_2010"].mean():.3f}, median={sample["mpc_2010"].median():.3f}')
    print(f'    p10={sample["mpc_2010"].quantile(0.1):.3f}, p90={sample["mpc_2010"].quantile(0.9):.3f}')
    if 'mpc_2012_in2010' in sample.columns:
        valid = sample['mpc_2012_in2010'].dropna()
        if len(valid) > 0:
            print(f'  MPC (large shock, 2012 in 2010):')
            print(f'    mean={valid.mean():.3f}, median={valid.median():.3f}')
            print(f'    p10={valid.quantile(0.1):.3f}, p90={valid.quantile(0.9):.3f}')
    if 'dmpc_1012_in2010' in sample.columns:
        valid = sample['dmpc_1012_in2010'].dropna()
        if len(valid) > 0:
            print(f'  MPC difference (small - large):')
            print(f'    mean={valid.mean():.3f}, median={valid.median():.3f}')

# Cash-on-hand distribution
if 'r_cash' in sample.columns:
    print(f'\n  Cash-on-hand (real, thousands EUR):')
    print(f'    mean={sample["r_cash"].mean():.1f}, median={sample["r_cash"].median():.1f}')
    print(f'    p10={sample["r_cash"].quantile(0.1):.1f}, p90={sample["r_cash"].quantile(0.9):.1f}')

# ── Step 4: Save ──────────────────────────────────────────────────────────
out_path = os.path.join(OUTPUT_DIR, 'analysis_data.parquet')
df.to_parquet(out_path, index=False)
print(f'\n  Saved: {out_path}')
print(f'  Columns: {df.shape[1]}')

print('\n' + '=' * 70)
print('01_clean.py — DONE')
print('=' * 70)
