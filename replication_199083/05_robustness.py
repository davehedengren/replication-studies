"""
05_robustness.py - Robustness checks for paper 199083
Kantor & Whalley - Moonshot: Public R&D and Growth
"""
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

PROJ = '/Users/davehedengren/code/replication_studies/199083-V1'
PROC = f'{PROJ}/data/process'

from utils import run_reghdfe, print_reg

df = pd.read_pickle(f'{PROC}/final/3.compile.pkl')

# Setup: treatment vars and controls
keep_vars = ['amspost_sv2_stemmed_sr', 'amspost_sv2_stemmed_psr',
             'namspost_sv2_stemmed_sr', 'namspost_sv2_stemmed_psr']

if 'mstot_patent58' not in df.columns:
    for c in ['mstot_patent58_x', 'mstot_patent58_y']:
        if c in df.columns:
            df['mstot_patent58'] = df[c]
            break
df['mstot_patent58'] = df['mstot_patent58'].fillna(0)

year_dummies = pd.get_dummies(df['year'], prefix='yr', drop_first=True)
patent_year_controls = []
for col in year_dummies.columns:
    cname = f'patent_x_{col}'
    df[cname] = year_dummies[col].values * df['mstot_patent58'].values
    patent_year_controls.append(cname)

xvars = keep_vars + patent_year_controls

def run_and_print(title, yvar, xvars, fe_vars, cluster_vars, data=None):
    """Run regression and print summary of treatment effects."""
    d = data if data is not None else df
    res = run_reghdfe(d, yvar, xvars, fe_vars, cluster_vars)
    print(f"\n{title}")
    print(f"  N={res['n']}, R2={res.get('r2', 0):.4f}")
    for v in keep_vars:
        if v in res['xvars']:
            idx = res['xvars'].index(v)
            t = abs(res['t'][idx])
            sig = '***' if t > 2.576 else '**' if t > 1.96 else '*' if t > 1.645 else ''
            print(f"  {v:<35} {res['b'][idx]:>8.4f} ({res['se'][idx]:.4f}) {sig}")
    return res


# ========================================================================
# 1. ALTERNATIVE SAMPLES
# ========================================================================
print("=" * 70)
print("ROBUSTNESS 1: ALTERNATIVE SAMPLES")
print("=" * 70)

# Baseline
run_and_print("Baseline: lva, County+Industry+Year FE", 'lva', xvars, ['year', 'fips', 'sic'], ['msa_code', 'sic'])

# Drop MSA fill-in years (keep only years with strong county data)
for subset_label, year_list in [
    ("Drop 1947", [y for y in df['year'].unique() if y != 1947]),
    ("Drop 1947+1954", [y for y in df['year'].unique() if y not in [1947, 1954]]),
    ("Post-1958 only", [y for y in df['year'].unique() if y >= 1958]),
]:
    sub = df[df['year'].isin(year_list)].copy()
    run_and_print(f"Sample: {subset_label}", 'lva', xvars, ['year', 'fips', 'sic'], ['msa_code', 'sic'], data=sub)

# ========================================================================
# 2. ALTERNATIVE OUTCOMES
# ========================================================================
print("\n" + "=" * 70)
print("ROBUSTNESS 2: ALTERNATIVE OUTCOMES")
print("=" * 70)

for yvar, label in [('lemp', 'Log(Employment)'), ('lcapital', 'Log(Capital)'),
                     ('ltfp', 'Log(TFP)'), ('npw_share', 'Non-Prod Worker Share')]:
    if yvar in df.columns:
        run_and_print(f"Outcome: {label}", yvar, xvars, ['year', 'fips', 'sic'], ['msa_code', 'sic'])

# ========================================================================
# 3. ALTERNATIVE CLUSTERING
# ========================================================================
print("\n" + "=" * 70)
print("ROBUSTNESS 3: ALTERNATIVE CLUSTERING")
print("=" * 70)

run_and_print("Cluster: MSA only", 'lva', xvars, ['year', 'fips', 'sic'], ['msa_code'])
run_and_print("Cluster: SIC only", 'lva', xvars, ['year', 'fips', 'sic'], ['sic'])
run_and_print("Cluster: County (fips)", 'lva', xvars, ['year', 'fips', 'sic'], ['fips'])

# ========================================================================
# 4. ALTERNATIVE FIXED EFFECTS
# ========================================================================
print("\n" + "=" * 70)
print("ROBUSTNESS 4: ALTERNATIVE FIXED EFFECTS")
print("=" * 70)

run_and_print("FE: County + Year only", 'lva', xvars, ['year', 'fips'], ['msa_code', 'sic'])
run_and_print("FE: Industry + Year only", 'lva', xvars, ['year', 'sic'], ['msa_code', 'sic'])

# ========================================================================
# 5. PLACEBO: PRE-TREATMENT TRENDS
# ========================================================================
print("\n" + "=" * 70)
print("ROBUSTNESS 5: PRE-TREATMENT PERIOD (1947-1958)")
print("=" * 70)

pre = df[df['year'] <= 1958].copy()
if len(pre) > 0:
    run_and_print("Pre-treatment: lva (1947-1958)", 'lva', xvars, ['year', 'fips', 'sic'], ['msa_code', 'sic'], data=pre)
    run_and_print("Pre-treatment: lemp (1947-1958)", 'lemp', xvars, ['year', 'fips', 'sic'], ['msa_code', 'sic'], data=pre)

print("\n" + "=" * 70)
print("ROBUSTNESS CHECKS COMPLETE")
print("=" * 70)
