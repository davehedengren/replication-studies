"""
04_data_audit.py — Data coverage, distributions, and consistency checks for GHT (2025).

Paper: "Temporary Layoffs, Loss-of-Recall, and Cyclical Unemployment Dynamics"
"""

import sys, os
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import pandas as pd
from utils import *

print('=' * 70)
print('04_data_audit.py — Data audit')
print('=' * 70)

# ══════════════════════════════════════════════════════════════════════
# 1. CPS Monthly Transition Probabilities
# ══════════════════════════════════════════════════════════════════════
print('\n' + '=' * 70)
print('1. CPS Monthly Transition Probabilities')
print('=' * 70)

tp = load_transition_probabilities()
tp_pre = tp[(tp.year >= 1978) & (tp.year <= 2019)].copy()

print(f'  Total rows: {len(tp)}')
print(f'  Pre-COVID rows (1978-2019): {len(tp_pre)}')
print(f'  Year range: {tp.year.min()}-{tp.year.max()}')

# Check completeness
expected_months = (tp.year.max() - tp.year.min() + 1) * 12
print(f'  Expected months: ~{expected_months}, Actual: {len(tp)}')

# Flow column summary
flow_cols = [c for c in tp.columns if c.startswith('ctflow_')]
print(f'\n  Flow columns ({len(flow_cols)}):')
for col in flow_cols:
    vals = tp_pre[col]
    print(f'    {col}: mean={vals.mean():.4f}, std={vals.std():.4f}, '
          f'min={vals.min():.4f}, max={vals.max():.4f}, '
          f'missing={vals.isnull().sum()}')

# Check rows sum to ~1
print('\n  Row-sum checks (should be ~1.0):')
for state, code in [('E', 'e'), ('TL', 't'), ('JL', 'p'), ('N', 'n')]:
    state_cols = [f'ctflow_{code}{c}' for c in ['e', 't', 'p', 'n']]
    existing = [c for c in state_cols if c in tp_pre.columns]
    row_sums = tp_pre[existing].sum(axis=1)
    print(f'    From {state}: mean={row_sums.mean():.6f}, min={row_sums.min():.6f}, max={row_sums.max():.6f}')

# Unemployment rates
if 'trate' in tp_pre.columns:
    print(f'\n  TL rate (trate): mean={tp_pre.trate.mean()*100:.2f}%, '
          f'min={tp_pre.trate.min()*100:.2f}%, max={tp_pre.trate.max()*100:.2f}%')
if 'prate' in tp_pre.columns:
    print(f'  JL rate (prate): mean={tp_pre.prate.mean()*100:.2f}%, '
          f'min={tp_pre.prate.min()*100:.2f}%, max={tp_pre.prate.max()*100:.2f}%')

# ══════════════════════════════════════════════════════════════════════
# 2. Quarterly Transition Probabilities
# ══════════════════════════════════════════════════════════════════════
print('\n' + '=' * 70)
print('2. Quarterly Transition Probabilities')
print('=' * 70)

tpq = load_quarterly_transitions()
tpq_pre = tpq[(tpq.year >= 1978) & (tpq.year <= 2019)].copy()

print(f'  Total rows: {len(tpq)}')
print(f'  Pre-COVID rows: {len(tpq_pre)}')

# JLfromTL availability
jl = tpq['JLfromTL']
print(f'\n  JLfromTL: {jl.notna().sum()}/{len(tpq)} non-null ({100*jl.notna().sum()/len(tpq):.0f}%)')
print(f'    First available: year={tpq.loc[jl.first_valid_index(), "year"]}, '
      f'month={tpq.loc[jl.first_valid_index(), "month"]}')

# fracRecall_noI
if 'fracRecall_noI' in tpq.columns:
    fr = tpq_pre['fracRecall_noI'].dropna()
    print(f'\n  fracRecall_noI: mean={fr.mean():.3f}, std={fr.std():.3f}, n={len(fr)}')

# ══════════════════════════════════════════════════════════════════════
# 3. GHT Summary Statistics
# ══════════════════════════════════════════════════════════════════════
print('\n' + '=' * 70)
print('3. GHT Summary Statistics')
print('=' * 70)

ght = load_ght_stats()
print(f'  Shape: {ght.shape}')
print(f'  Columns: {list(ght.columns)}')
print(f'\n  Row structure (4 rows):')
row_labels = ['Means', 'Rel. Std. Dev.', 'Corr. with GDP', 'Autocorrelation']
for i, lbl in enumerate(row_labels):
    if i < len(ght):
        row = ght.iloc[i]
        non_null = row.notna().sum()
        print(f'    Row {i} ({lbl}): {non_null} non-null values')

# ══════════════════════════════════════════════════════════════════════
# 4. SIPP Hazard Data
# ══════════════════════════════════════════════════════════════════════
print('\n' + '=' * 70)
print('4. SIPP Hazard Data')
print('=' * 70)

haz = load_sipp_hazard()
print(f'  Rows: {len(haz)}')
print(f'  Columns: {list(haz.columns)}')

tl = haz[haz.rwkesr2 == 3]
ps = haz[haz.rwkesr2 == 4]
print(f'  TL separators (rwkesr2=3): {len(tl)} rows')
print(f'  PS separators (rwkesr2=4): {len(ps)} rows')

print(f'\n  TL hazard by duration:')
print(f'    {"Dur":>4} {"Pr(E)":>8} {"Pr(N)":>8} {"Pr(R)":>8}')
for _, row in tl.iterrows():
    print(f'    {row.duration:>4.0f} {row.pE:>8.3f} {row.pN:>8.3f} {row.pR:>8.3f}')

print(f'\n  PS hazard by duration:')
print(f'    {"Dur":>4} {"Pr(E)":>8} {"Pr(N)":>8} {"Pr(R)":>8}')
for _, row in ps.iterrows():
    print(f'    {row.duration:>4.0f} {row.pE:>8.3f} {row.pN:>8.3f} {row.pR:>8.3f}')

# Verify: pE should be close to pN + pR for each row
print('\n  pE ≈ pN + pR check:')
for _, row in haz.iterrows():
    diff = abs(row.pE - (row.pN + row.pR))
    typ = 'TL' if row.rwkesr2 == 3 else 'PS'
    status = 'OK' if diff < 0.001 else f'DIFF={diff:.4f}'
    print(f'    {typ} dur={row.duration:.0f}: pE={row.pE:.4f}, pN+pR={row.pN+row.pR:.4f} [{status}]')

# ══════════════════════════════════════════════════════════════════════
# 5. SIPP Hazard Panel
# ══════════════════════════════════════════════════════════════════════
print('\n' + '=' * 70)
print('5. SIPP Hazard Panel')
print('=' * 70)

haz_panel = load_sipp_hazard_panel()
print(f'  Rows: {len(haz_panel)}, Columns: {haz_panel.shape[1]}')
print(f'  Columns: {list(haz_panel.columns)}')

if 'sipp_panel' in haz_panel.columns:
    print(f'  SIPP panels: {sorted(haz_panel.sipp_panel.unique())}')

# ══════════════════════════════════════════════════════════════════════
# 6. COVID-Period Data
# ══════════════════════════════════════════════════════════════════════
print('\n' + '=' * 70)
print('6. COVID-Period Data (forMatlabAdj)')
print('=' * 70)

matlab = load_for_matlab()
print(f'  Rows: {len(matlab)}, Columns: {matlab.shape[1]}')
print(f'  Date range: {matlab.year.min()}/{matlab.month.min()} to {matlab.year.max()}/{matlab.month.max()}')

# Peak unemployment
if 'Urate_adj' in matlab.columns:
    peak_idx = matlab['Urate_adj'].idxmax()
    peak = matlab.loc[peak_idx]
    print(f'\n  Peak unemployment: {peak["Urate_adj"]*100:.1f}% '
          f'({peak["year"]:.0f}/{peak["month"]:.0f})')

if 'Trate_adj' in matlab.columns:
    peak_idx = matlab['Trate_adj'].idxmax()
    peak = matlab.loc[peak_idx]
    print(f'  Peak TL unemployment: {peak["Trate_adj"]*100:.1f}% '
          f'({peak["year"]:.0f}/{peak["month"]:.0f})')

# COVID shock magnitude
covid_apr = matlab[(matlab.year == 2020) & (matlab.month == 4)]
if len(covid_apr) > 0:
    apr = covid_apr.iloc[0]
    print(f'\n  April 2020 snapshot:')
    print(f'    pET (E->TL): {apr["pET"]:.4f} (normal ~0.005)')
    print(f'    pTE (TL->E): {apr["pTE"]:.4f} (normal ~0.44)')
    print(f'    Trate: {apr["Trate_adj"]*100:.1f}% (normal ~0.8%)')
    print(f'    Urate: {apr["Urate_adj"]*100:.1f}% (normal ~6%)')

# ══════════════════════════════════════════════════════════════════════
# 7. Published Statistics Completeness
# ══════════════════════════════════════════════════════════════════════
print('\n' + '=' * 70)
print('7. Published Statistics Coverage')
print('=' * 70)

pub = load_published_stats()

categories = {
    'Unemployment stocks (avg/std/corr)': [k for k in pub if any(k.startswith(p) for p in ['avg', 'std', 'rhogdp', 'ac']) and 'RATEQ' in k or 'FROMTL' in k],
    'Transition matrix': [k for k in pub if k.startswith('avg') and 'full' in k],
    'Cyclical properties': [k for k in pub if (k.startswith('std') or k.startswith('rhogdp')) and 'full' in k],
    'Employment flows': [k for k in pub if k.startswith('flow_')],
    'Recession decomp': [k for k in pub if k.startswith('frac') or k.startswith('IoverD')],
    'Recall shares': [k for k in pub if 'recall' in k or 'rShare' in k],
    'Flow regressions': [k for k in pub if k.startswith('FR_')],
    'Model parameters': [k for k in pub if 'IS0_NC0' in k],
    'Industry/demographics': [k for k in pub if '_ind_' in k or '_age_' in k or '_sex_' in k or '_educ_' in k],
    'SIPP reference months': [k for k in pub if 'srefmon' in k],
}

total = 0
for cat, keys in categories.items():
    n = len(keys)
    total += n
    print(f'  {cat:<40} {n:>4} statistics')

print(f'  {"TOTAL":<40} {total:>4} statistics')
print(f'  (Full dict has {len(pub)} entries)')

# ══════════════════════════════════════════════════════════════════════
# 8. Cross-Validation: CSV vs Published Stats
# ══════════════════════════════════════════════════════════════════════
print('\n' + '=' * 70)
print('8. Cross-Validation: CSV data vs Published Statistics')
print('=' * 70)

tp_pre = tp[(tp.year >= 1978) & (tp.year <= 2019)]

# Compare transition matrix means
print('\n  Transition matrix averages (CSV vs Published):')
checks = []
for fr, fr_code in [('E', 'e'), ('TL', 't'), ('JL', 'p'), ('N', 'n')]:
    for to, to_code in [('E', 'e'), ('TL', 't'), ('JL', 'p'), ('N', 'n')]:
        col = f'ctflow_{fr_code}{to_code}'
        pub_key = f'avg{fr_code.upper()}{to_code.upper()}full'
        if col in tp_pre.columns and pub_key in pub:
            csv_val = tp_pre[col].mean()
            pub_val = sed_val(pub, pub_key)
            if pub_val is not None:
                diff = abs(csv_val - pub_val)
                status = 'MATCH' if diff < 0.002 else 'CHECK'
                checks.append((f'{fr}->{to}', csv_val, pub_val, diff, status))
                print(f'    p({fr}->{to}): CSV={csv_val:.4f}, Pub={pub_val:.3f}, diff={diff:.4f} [{status}]')

n_match = sum(1 for c in checks if c[4] == 'MATCH')
print(f'\n  Summary: {n_match}/{len(checks)} transition probabilities match within 0.002')

# Note on why CSV means may differ slightly from published:
# The published values come from seasonally adjusted, time-aggregation-corrected
# transition probabilities. The raw CSV may be pre- or post-adjustment.

print('\n' + '=' * 70)
print('04_data_audit.py — DONE')
print('=' * 70)
