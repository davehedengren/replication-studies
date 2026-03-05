"""
01_clean.py — Load and validate datasets for 219907-V1.

Paper: "Labor Market Power, Self-Employment, and Development"
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
# 1. Firm-level market data (baseline_firm_llmy.dta)
# ══════════════════════════════════════════════════════════════════════
print('\n1. Loading baseline_firm_llmy.dta ...')
firm_llmy = load_firm_llmy()
print(f'   Shape: {firm_llmy.shape}')
print(f'   Unique LLM-industry markets: {firm_llmy.llm_ind.nunique()}')
print(f'   Unique LLMs (locations): {firm_llmy.id_llm.nunique()}')
print(f'   Unique industries: {firm_llmy.ind2d.nunique()}')
print(f'   Years: {sorted(firm_llmy.year.unique())}')
print(f'   Key vars: {[c for c in firm_llmy.columns if "hhi" in c.lower() or "firm" in c.lower()][:10]}')
firm_llmy.to_parquet(os.path.join(OUTPUT_DIR, 'firm_llmy.parquet'), index=False)

# ══════════════════════════════════════════════════════════════════════
# 2. Worker cross-section (baseline_workers.dta)
# ══════════════════════════════════════════════════════════════════════
print('\n2. Loading baseline_workers.dta ...')
workers = load_workers()
print(f'   Shape: {workers.shape}')
print(f'   Years: {sorted(workers.year.dropna().unique().astype(int))}')

# Manufacturing filter
manuf = workers[workers.manuf == 1] if 'manuf' in workers.columns else workers
emp = manuf[manuf.employed == 1] if 'employed' in manuf.columns else manuf
print(f'   Manufacturing employed: {len(emp):,}')

# Key vars
for var in ['empdep', 'empindep', 'wmdep_tr', 'wmindep_tr', 'fac500']:
    if var in workers.columns:
        s = workers[var].dropna()
        print(f'   {var}: N={len(s):,}, mean={s.mean():.4f}')

workers.to_parquet(os.path.join(OUTPUT_DIR, 'workers.parquet'), index=False)
del workers  # free memory

# ══════════════════════════════════════════════════════════════════════
# 3. Worker panel (baseline_workerpanel.dta)
# ══════════════════════════════════════════════════════════════════════
print('\n3. Loading baseline_workerpanel.dta ...')
panel = load_workerpanel()
print(f'   Shape: {panel.shape}')
print(f'   Unique individuals: {panel.num_per.nunique():,}')
print(f'   Years: {sorted(panel.year.dropna().unique().astype(int))}')
panel.to_parquet(os.path.join(OUTPUT_DIR, 'workerpanel.parquet'), index=False)
del panel

# ══════════════════════════════════════════════════════════════════════
# 4. Merged dataset (merged_dataset_emp.dta)
# ══════════════════════════════════════════════════════════════════════
print('\n4. Loading merged_dataset_emp.dta ...')
merged = load_merged()
print(f'   Shape: {merged.shape}')
print(f'   Unique LLM-industry markets: {merged.llm_ind.nunique():,}')
for var in ['hhi_wbill', 'hhi_nl', 'n_firms', 'lwmdep', 'lwmindep']:
    if var in merged.columns:
        s = merged[var].dropna()
        print(f'   {var}: N={len(s):,}, mean={s.mean():.4f}')
merged.to_parquet(os.path.join(OUTPUT_DIR, 'merged.parquet'), index=False)
del merged

# ══════════════════════════════════════════════════════════════════════
# 5. Firm-level panel (baseline_firm.dta)
# ══════════════════════════════════════════════════════════════════════
print('\n5. Loading baseline_firm.dta ...')
firm = load_firm()
print(f'   Shape: {firm.shape}')
print(f'   Unique firms (iruc): {firm.iruc.nunique():,}')
print(f'   Years: {sorted(firm.year.dropna().unique().astype(int))}')
manuf_firm = firm[(firm.ciiu_n >= 10) & (firm.ciiu_n <= 33)]
print(f'   Manufacturing firms: {len(manuf_firm):,} obs, {manuf_firm.iruc.nunique():,} unique')
firm.to_parquet(os.path.join(OUTPUT_DIR, 'firm.parquet'), index=False)
del firm

# ══════════════════════════════════════════════════════════════════════
# 6. Electrification data
# ══════════════════════════════════════════════════════════════════════
print('\n6. Loading electrification data ...')
elec = load_electrification()
print(f'   Shape: {elec.shape}')
print(f'   Years with projects: {sorted(elec.year.unique())}')
print(f'   Total projects: {elec.elec_projects.sum():.0f}')
elec.to_parquet(os.path.join(OUTPUT_DIR, 'electrification.parquet'), index=False)

# ══════════════════════════════════════════════════════════════════════
# 7. Census (censo_dataset.dta)
# ══════════════════════════════════════════════════════════════════════
print('\n7. Loading censo_dataset.dta ...')
try:
    censo = load_census()
    print(f'   Shape: {censo.shape}')
    if 'manuf' in censo.columns:
        print(f'   Manufacturing firms: {censo.manuf.sum():,.0f}')
    censo.to_parquet(os.path.join(OUTPUT_DIR, 'census.parquet'), index=False)
    del censo
except Exception as e:
    print(f'   Error loading census: {e}')

print('\n' + '=' * 70)
print('01_clean.py — DONE')
print('=' * 70)
