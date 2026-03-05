"""
04_data_audit.py — Data coverage, distributions, and consistency checks.

Paper: "Labor Market Power, Self-Employment, and Development"
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
# 1. Dataset Inventory
# ══════════════════════════════════════════════════════════════════════
print('\n' + '=' * 70)
print('1. Dataset Inventory')
print('=' * 70)

dta_files = [f for f in os.listdir(DATA_DIR) if f.endswith('.dta')]
print(f'  Total .dta files: {len(dta_files)}')
for f in sorted(dta_files):
    size_mb = os.path.getsize(os.path.join(DATA_DIR, f)) / 1e6
    print(f'    {f}: {size_mb:.1f} MB')

# ══════════════════════════════════════════════════════════════════════
# 2. Firm Market Data (baseline_firm_llmy)
# ══════════════════════════════════════════════════════════════════════
print('\n' + '=' * 70)
print('2. Firm Market Data')
print('=' * 70)

firm_llmy = load_firm_llmy()
print(f'  Shape: {firm_llmy.shape}')
print(f'  LLM × industry markets: {firm_llmy.llm_ind.nunique():,}')
print(f'  Locations: {firm_llmy.id_llm.nunique()}')
print(f'  Industries: {firm_llmy.ind2d.nunique()} (codes: {sorted(firm_llmy.ind2d.unique())})')
print(f'  Years: {sorted(firm_llmy.year.unique())}')

# Balance check
yr_counts = firm_llmy.groupby('llm_ind')['year'].nunique()
print(f'\n  Panel balance (LLM-industry markets):')
for n in sorted(yr_counts.unique()):
    ct = (yr_counts == n).sum()
    print(f'    {n} years: {ct:,} markets')

# HHI distribution
print(f'\n  HHI (wage-bill) distribution:')
hhi = firm_llmy['hhi_wbill'].dropna()
print(f'    N={len(hhi):,}, mean={hhi.mean():.3f}, median={hhi.median():.3f}')
print(f'    p10={hhi.quantile(0.1):.3f}, p90={hhi.quantile(0.9):.3f}')
print(f'    Min={hhi.min():.3f}, Max={hhi.max():.3f}')
print(f'    HHI > 0.25: {(hhi > 0.25).sum():,} ({(hhi > 0.25).mean()*100:.1f}%)')
print(f'    HHI = 1 (single firm): {(hhi == 1).sum():,} ({(hhi == 1).mean()*100:.1f}%)')

# ══════════════════════════════════════════════════════════════════════
# 3. Worker Cross-Section (baseline_workers)
# ══════════════════════════════════════════════════════════════════════
print('\n' + '=' * 70)
print('3. Worker Cross-Section')
print('=' * 70)

w = load_workers()
print(f'  Shape: {w.shape}')
print(f'  Years: {sorted(w.year.dropna().unique().astype(int))}')

# Employment status
emp = w[w.employed == 1]
print(f'  Employed: {len(emp):,}')
manuf = emp[emp.manuf == 1]
print(f'  Manufacturing employed: {len(manuf):,}')

# Weighted employment shares
if 'fac500' in manuf.columns:
    wt = manuf['fac500']
    for var in ['empdep', 'empindep']:
        m, s = weighted_stats(manuf[var].values, wt.values)
        print(f'  {var}: weighted mean={m:.3f}, sd={s:.3f}')

# Earnings distributions
for var, label in [('wmdep_tr', 'Daily wage'), ('wmindep_tr', 'Daily SE earnings')]:
    if var in manuf.columns:
        s = manuf[var].dropna()
        print(f'\n  {label}:')
        print(f'    N={len(s):,}, mean={s.mean():.2f}, median={s.median():.2f}')
        print(f'    p10={s.quantile(0.1):.2f}, p90={s.quantile(0.9):.2f}')
        print(f'    Min={s.min():.2f}, Max={s.max():.2f}')

del w

# ══════════════════════════════════════════════════════════════════════
# 4. Merged Dataset (merged_dataset_emp)
# ══════════════════════════════════════════════════════════════════════
print('\n' + '=' * 70)
print('4. Merged Worker × Firm Dataset')
print('=' * 70)

merged = load_merged()
print(f'  Shape: {merged.shape}')
print(f'  LLM × industry markets with data: {merged.llm_ind.nunique():,}')
print(f'  Locations: {merged.id_llm.nunique()}')
print(f'  Years: {sorted(merged.year.dropna().unique().astype(int))}')

# HHI coverage in worker data
hhi_available = merged['hhi_wbill'].notna().sum()
print(f'  HHI available for: {hhi_available:,}/{len(merged):,} workers ({hhi_available/len(merged)*100:.1f}%)')

# Key variable distributions
for var in ['hhi_wbill', 'hhi_nl', 'n_firms', 'lwmdep_tr', 'lwmindep_tr']:
    if var in merged.columns:
        s = merged[var].dropna()
        print(f'\n  {var}: N={len(s):,}, mean={s.mean():.3f}, sd={s.std():.3f}')

del merged

# ══════════════════════════════════════════════════════════════════════
# 5. Firm Panel (baseline_firm)
# ══════════════════════════════════════════════════════════════════════
print('\n' + '=' * 70)
print('5. Firm Panel')
print('=' * 70)

firm = load_firm()
print(f'  Shape: {firm.shape}')
print(f'  Unique firms: {firm.iruc.nunique():,}')
print(f'  Years: {sorted(firm.year.dropna().unique().astype(int))}')

manuf = firm[(firm.ciiu_n >= 10) & (firm.ciiu_n <= 33)]
print(f'  Manufacturing: {len(manuf):,} obs, {manuf.iruc.nunique():,} firms')

# Panel balance
yr_per_firm = manuf.groupby('iruc')['year'].nunique()
print(f'  Years per firm: mean={yr_per_firm.mean():.1f}, median={yr_per_firm.median():.0f}')

# Key variables for IV
for var in ['n_labor', 'x_labor_twagessalaries', 'net_sales', 'x_energy_water']:
    if var in manuf.columns:
        s = manuf[var].dropna()
        print(f'  {var}: N={len(s):,}, mean={s.mean():.0f}, median={s.median():.0f}')

del firm

# ══════════════════════════════════════════════════════════════════════
# 6. Electrification Data
# ══════════════════════════════════════════════════════════════════════
print('\n' + '=' * 70)
print('6. Electrification (PER) Data')
print('=' * 70)

elec = load_electrification()
print(f'  Shape: {elec.shape}')
print(f'  Years: {sorted(elec.year.unique())}')
print(f'  LLMs with projects: {elec.id_llm.nunique()}')
print(f'  Total projects: {elec.elec_projects.sum():.0f}')
print(f'  Projects per LLM-year: mean={elec.elec_projects.mean():.1f}')

# ══════════════════════════════════════════════════════════════════════
# 7. Census Data
# ══════════════════════════════════════════════════════════════════════
print('\n' + '=' * 70)
print('7. Economic Census (2007)')
print('=' * 70)

try:
    censo = load_census()
    print(f'  Shape: {censo.shape}')
    if 'manuf' in censo.columns:
        manuf_c = censo[censo.manuf == 1]
        print(f'  Manufacturing firms: {len(manuf_c):,}')
        if 'n_labor' in manuf_c.columns:
            print(f'  Total manuf employment: {manuf_c.n_labor.sum():,.0f}')
    del censo
except Exception as e:
    print(f'  Error: {e}')

# ══════════════════════════════════════════════════════════════════════
# 8. Missing Data Patterns
# ══════════════════════════════════════════════════════════════════════
print('\n' + '=' * 70)
print('8. Missing Data Summary')
print('=' * 70)

firm_llmy = load_firm_llmy()
important_vars = ['hhi_wbill', 'hhi_nl', 'n_firms', 'hhi_wbill_w', 'hhi_nl_wte',
                  'pm1firm', 'pm1firm_wbill', 'pm1firm_emp']
for var in important_vars:
    if var in firm_llmy.columns:
        miss = firm_llmy[var].isna().sum()
        pct = miss / len(firm_llmy) * 100
        print(f'  {var}: {miss:,} missing ({pct:.1f}%)')

print('\n' + '=' * 70)
print('04_data_audit.py — DONE')
print('=' * 70)
