"""
01_clean.py — Load and validate all pre-computed datasets for GHT (2025).

Paper: "Temporary Layoffs, Loss-of-Recall, and Cyclical Unemployment Dynamics"
"""

import sys, os
sys.path.insert(0, os.path.dirname(__file__))

from utils import *

print('=' * 70)
print('01_clean.py — Data loading and validation')
print('=' * 70)

# ──────────────────────────────────────────────────────────────────────
# 1. CPS Transition Probabilities (monthly)
# ──────────────────────────────────────────────────────────────────────
print('\n-- CPS Monthly Transition Probabilities --')
tp = load_transition_probabilities()
print(f'  Rows: {len(tp)}, Columns: {tp.shape[1]}')
print(f'  Date range: {tp.year.min()}/{tp.month.min()} to {tp.year.max()}/{tp.month.max()}')
print(f'  Years: {tp.year.nunique()}')
print(f'  Key columns: {list(tp.columns[:6])}...')

# Check no missing values in key flow columns
flow_cols = [c for c in tp.columns if c.startswith('ctflow_')]
missing = tp[flow_cols].isnull().sum().sum()
print(f'  Missing values in flow columns: {missing}')

# ──────────────────────────────────────────────────────────────────────
# 2. CPS Quarterly Transition Probabilities
# ──────────────────────────────────────────────────────────────────────
print('\n-- CPS Quarterly Transition Probabilities --')
tpq = load_quarterly_transitions()
print(f'  Rows: {len(tpq)}, Columns: {tpq.shape[1]}')
print(f'  Date range: {tpq.year.min()}Q to {tpq.year.max()}Q')

# Check JLfromTL availability
jl_avail = tpq['JLfromTL'].notna().sum()
print(f'  JLfromTL non-null: {jl_avail}/{len(tpq)} ({100*jl_avail/len(tpq):.0f}%)')

# ──────────────────────────────────────────────────────────────────────
# 3. GHT Summary Statistics
# ──────────────────────────────────────────────────────────────────────
print('\n-- GHT Summary Statistics --')
ght = load_ght_stats()
print(f'  Rows: {len(ght)}, Columns: {ght.shape[1]}')
print(f'  Row labels (from structure): means, rel_std, corr_gdp, autocorr')

# ──────────────────────────────────────────────────────────────────────
# 4. MATLAB-formatted Data (COVID period)
# ──────────────────────────────────────────────────────────────────────
print('\n-- MATLAB-formatted data (COVID period) --')
matlab = load_for_matlab()
print(f'  Rows: {len(matlab)}, Columns: {matlab.shape[1]}')
print(f'  Date range: {matlab.year.min()}/{matlab.month.min()} to {matlab.year.max()}/{matlab.month.max()}')
print(f'  Columns: {list(matlab.columns)}')

# ──────────────────────────────────────────────────────────────────────
# 5. SIPP Hazard Data
# ──────────────────────────────────────────────────────────────────────
print('\n-- SIPP Hazard Data --')
haz = load_sipp_hazard()
print(f'  Rows: {len(haz)}, Columns: {haz.shape[1]}')
print(f'  Columns: {list(haz.columns)}')
print(f'  Duration range: {haz.duration.min()} to {haz.duration.max()} months')
print(f'  RWKESR2 values: {sorted(haz.rwkesr2.unique())}')

# TL (rwkesr2=3) and PS (rwkesr2=4) separation
tl_haz = haz[haz.rwkesr2 == 3]
ps_haz = haz[haz.rwkesr2 == 4]
print(f'  TL hazard rows: {len(tl_haz)}, PS hazard rows: {len(ps_haz)}')

# ──────────────────────────────────────────────────────────────────────
# 6. SIPP Hazard Panel
# ──────────────────────────────────────────────────────────────────────
print('\n-- SIPP Hazard Panel --')
haz_panel = load_sipp_hazard_panel()
print(f'  Rows: {len(haz_panel)}, Columns: {haz_panel.shape[1]}')
print(f'  Columns: {list(haz_panel.columns)}')

# ──────────────────────────────────────────────────────────────────────
# 7. Published Statistics (all sed files)
# ──────────────────────────────────────────────────────────────────────
print('\n-- Published Statistics (from sed files) --')
pub = load_published_stats()
print(f'  Total statistics parsed: {len(pub)}')

# Count by source
cps_keys = [k for k in pub if k.startswith('avg') or k.startswith('std') or k.startswith('rho') or k.startswith('ac')]
recall_keys = [k for k in pub if 'recall' in k or 'rShare' in k or 'srefmon' in k]
flow_keys = [k for k in pub if k.startswith('flow_') or k.startswith('FR_')]
model_keys = [k for k in pub if '_mIS0_NC0' in k or '_m_IS0_NC0' in k]
recession_keys = [k for k in pub if 'frac' in k or 'IoverD' in k]

print(f'  CPS aggregate stats: ~{len(cps_keys)}')
print(f'  Recall/SIPP stats: ~{len(recall_keys)}')
print(f'  Flow regression stats: ~{len(flow_keys)}')
print(f'  Model parameter stats: ~{len(model_keys)}')
print(f'  Recession decomposition: ~{len(recession_keys)}')

# ──────────────────────────────────────────────────────────────────────
# 8. Plot Data (raw and adjusted)
# ──────────────────────────────────────────────────────────────────────
print('\n-- Plot Data --')
raw_plot = load_for_raw_plot()
adj_plot = load_for_adjusted_plot()
print(f'  Raw plot data: {len(raw_plot)} rows, {raw_plot.shape[1]} cols')
print(f'  Adjusted plot data: {len(adj_plot)} rows, {adj_plot.shape[1]} cols')

# ──────────────────────────────────────────────────────────────────────
# 9. Validation Summary
# ──────────────────────────────────────────────────────────────────────
print('\n' + '=' * 70)
print('VALIDATION SUMMARY')
print('=' * 70)

checks = [
    ('CPS monthly transitions', len(tp) > 400, f'{len(tp)} rows'),
    ('CPS quarterly transitions', len(tpq) > 150, f'{len(tpq)} rows'),
    ('GHT stats', len(ght) > 0, f'{len(ght)} rows'),
    ('MATLAB data', len(matlab) > 20, f'{len(matlab)} rows'),
    ('SIPP hazard', len(haz) == 16, f'{len(haz)} rows (8 durations × 2 types)'),
    ('Published stats', len(pub) > 300, f'{len(pub)} statistics'),
    ('No missing flows', missing == 0, f'{missing} missing'),
]

for name, passed, detail in checks:
    status = 'PASS' if passed else 'FAIL'
    print(f'  [{status}] {name}: {detail}')

print('\n' + '=' * 70)
print('01_clean.py — DONE')
print('=' * 70)
