"""
02_tables.py - Replicate Tables 1-3 for paper 199083
Kantor & Whalley - Moonshot: Public R&D and Growth
"""
import pandas as pd
import numpy as np
import statsmodels.api as sm
import warnings
warnings.filterwarnings('ignore')

PROJ = '/Users/davehedengren/code/replication_studies/199083-V1'
PROC = f'{PROJ}/data/process'

from utils import run_reghdfe, print_reg

# Load final dataset
df = pd.read_pickle(f'{PROC}/final/3.compile.pkl')
print(f"Dataset: {len(df)} obs, {df['fips'].nunique()} counties, {df['sic'].nunique()} SIC codes")
print(f"Years: {sorted(df['year'].unique())}")

# ========================================================================
# TABLE 1: Descriptive Statistics (1958 cross-section)
# ========================================================================
print("\n" + "=" * 70)
print("TABLE 1: DESCRIPTIVE STATISTICS (1958)")
print("=" * 70)

d58 = df[df['year'] == 1958].copy()
d58 = d58.dropna(subset=['amspost_sv2_stemmed'])

# Create sub-columns
d58['mval_add_m'] = d58['mval_add'] / 1e6
d58['minvest_t'] = d58['minvest'] / 1000
d58['income'] = d58['mpayroll'] / d58['memp']
d58['nasa_ind'] = d58['sic'].isin([36, 37]).astype(int)

# Column layout:
# [1] Full sample, [2] High Space Cap (amspost=1), [3] Low Space Cap (amspost=0)
# [4] Diff p-value [2]-[3]
# [5] NASA industry (SIC 36,37), [6] Non-NASA
# [7] Diff p-value [5]-[6]
# [8] DDD p-value

vars_tab1 = {
    'mval_add_m': 'Value Added ($M)',
    'memp': 'Employment',
    'income': 'Labor Income',
    'minvest_t': 'Capital Investment ($K)',
}

print(f"\nN = {len(d58)} (1958 cross-section)")
print(f"{'Variable':<25} {'Full':>10} {'High SC':>10} {'Low SC':>10} {'p(diff)':>8} {'NASA':>10} {'Non-NASA':>10} {'p(diff)':>8}")
print("-" * 95)

for v, label in vars_tab1.items():
    if v not in d58.columns:
        continue
    full_mean = d58[v].mean()
    high = d58[d58['amspost_sv2_stemmed'] == 1][v].mean()
    low = d58[d58['amspost_sv2_stemmed'] == 0][v].mean()
    nasa = d58[d58['nasa_ind'] == 1][v].mean()
    non_nasa = d58[d58['nasa_ind'] == 0][v].mean()

    # P-value for diff High-Low
    try:
        sub = d58.dropna(subset=[v])
        X = sm.add_constant(sub['amspost_sv2_stemmed'])
        res = sm.OLS(sub[v], X).fit(cov_type='cluster', cov_kwds={'groups': sub['fips']})
        p_hl = res.pvalues.iloc[1]
    except:
        p_hl = np.nan

    # P-value for NASA vs non-NASA
    try:
        X = sm.add_constant(sub['nasa_ind'])
        res = sm.OLS(sub[v], X).fit(cov_type='cluster', cov_kwds={'groups': sub['fips']})
        p_nn = res.pvalues.iloc[1]
    except:
        p_nn = np.nan

    print(f"{label:<25} {full_mean:>10.1f} {high:>10.1f} {low:>10.1f} {p_hl:>8.3f} {nasa:>10.1f} {non_nasa:>10.1f} {p_nn:>8.3f}")


# ========================================================================
# TABLE 3: Space Capability and Manufacturing (Core DID results)
# ========================================================================
print("\n" + "=" * 70)
print("TABLE 3: SPACE CAPABILITY AND MANUFACTURING")
print("=" * 70)

# Treatment variables
keep_vars = ['amspost_sv2_stemmed_sr', 'amspost_sv2_stemmed_psr',
             'namspost_sv2_stemmed_sr', 'namspost_sv2_stemmed_psr']

# Control: year dummies x mstot_patent58
# Handle duplicate mstot_patent58 columns from merges
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

# Outcome variables
outcomes = [('lva', 'Log(Value Added)'), ('lemp', 'Log(Employment)'),
            ('lcapital', 'Log(Capital)'), ('ltfp', 'Log(TFP)')]

results_table3 = {}
for yvar, ylabel in outcomes:
    if yvar not in df.columns:
        print(f"\n  {ylabel}: variable not found, skipping")
        continue

    # Model 1: absorb(year fips sic), cluster(msa_code sic)
    res1 = run_reghdfe(df, yvar, xvars, ['year', 'fips', 'sic'], ['msa_code', 'sic'])
    print_reg(res1, f"\nTable 3: {ylabel} - County+Industry+Year FE")

    # Model 2: absorb(fips sic msa_code_year), cluster(msa_code sic)
    df['msa_year'] = df['msa_code'].astype(str) + '_' + df['year'].astype(str)
    res2 = run_reghdfe(df, yvar, xvars, ['fips', 'sic', 'msa_year'], ['msa_code', 'sic'])
    print_reg(res2, f"\nTable 3: {ylabel} - County+Industry+MSAxYear FE")

    results_table3[yvar] = (res1, res2)

# ========================================================================
# TABLE 3 SUMMARY (Treatment coefficients only)
# ========================================================================
print("\n" + "=" * 70)
print("TABLE 3 SUMMARY: KEY COEFFICIENTS")
print("=" * 70)
print(f"\n{'Outcome':<20} {'Variable':<30} {'Model 1':>10} {'SE':>10} {'Model 2':>10} {'SE':>10}")
print("-" * 90)

for yvar, ylabel in outcomes:
    if yvar not in results_table3:
        continue
    res1, res2 = results_table3[yvar]
    for i, v in enumerate(keep_vars):
        if i >= len(res1['b']):
            continue
        idx1 = res1['xvars'].index(v) if v in res1['xvars'] else None
        idx2 = res2['xvars'].index(v) if v in res2['xvars'] else None
        b1 = res1['b'][idx1] if idx1 is not None else np.nan
        se1 = res1['se'][idx1] if idx1 is not None else np.nan
        b2 = res2['b'][idx2] if idx2 is not None else np.nan
        se2 = res2['se'][idx2] if idx2 is not None else np.nan
        sig1 = '***' if abs(b1/se1) > 2.576 else '**' if abs(b1/se1) > 1.96 else '*' if abs(b1/se1) > 1.645 else ''
        sig2 = '***' if abs(b2/se2) > 2.576 else '**' if abs(b2/se2) > 1.96 else '*' if abs(b2/se2) > 1.645 else ''
        lbl = ylabel if i == 0 else ''
        print(f"{lbl:<20} {v:<30} {b1:>9.4f}{sig1:<1} ({se1:.4f})  {b2:>9.4f}{sig2:<1} ({se2:.4f})")
    print()

# ========================================================================
# TABLE 2: Space Capability, NASA Spending, and NASA Patents (Panel A only)
# ========================================================================
print("\n" + "=" * 70)
print("TABLE 2 PANEL A: NASA SPENDING & PATENTS (Extensive Margin)")
print("=" * 70)

# Indicators
if 'ar58_s2_nasa_contractor_spend' not in df.columns:
    df['ar58_s2_nasa_contractor_spend'] = (df['r58_s2_nasa_contractor_spend'] > 0).astype(int)
if 'asmsfnasa_patent5' not in df.columns:
    df['asmsfnasa_patent5'] = (df['smsfnasa_patent5'] > 0).astype(int) if 'smsfnasa_patent5' in df.columns else 0

tab2_outcomes = [
    ('ar58_s2_nasa_contractor_spend', 'Any NASA Spending'),
    ('asmsfnasa_patent5', 'Any NASA Patents'),
]

for yvar, ylabel in tab2_outcomes:
    if yvar not in df.columns:
        continue
    res = run_reghdfe(df, yvar, xvars, ['year', 'fips', 'sic'], ['msa_code', 'sic'])
    print_reg(res, f"\nTable 2A: {ylabel} - County+Industry+Year FE")

print("\n" + "=" * 70)
print("REPLICATION COMPLETE")
print("=" * 70)
