"""
02_tables.py — Reproduce main tables for 219907-V1.

Paper: "Labor Market Power, Self-Employment, and Development"
Table 1 (summary stats), Table A2 (OLS), Table 2 (IV estimation)
"""

import sys, os
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import pandas as pd
import statsmodels.api as sm
from utils import *

print('=' * 70)
print('02_tables.py — Table replication')
print('=' * 70)

# ══════════════════════════════════════════════════════════════════════
# TABLE 1, PANEL I: Manufacturing Local Labor Markets
# ══════════════════════════════════════════════════════════════════════
print('\n' + '=' * 70)
print('Table 1, Panel I: Manufacturing Local Labor Markets')
print('=' * 70)

firm_llmy = load_firm_llmy()

vars_panel1 = {
    'n_firms': 'Number of Firms',
    'hhi_wbill': 'Wage-bill HHI',
    'hhi_wbill_w': 'Wage-bill HHI (Weighted)',
    'hhi_nl': 'Employment HHI',
    'hhi_nl_wte': 'Employment HHI (Empl. Weighted)',
    'pm1firm': '% LLMs with 1 firm',
    'pm1firm_wbill': '% payroll in 1-firm LLMs',
    'pm1firm_emp': '% employment in 1-firm LLMs',
}

print(f'\n  {"Variable":<45} {"Mean":>8} {"SD":>8}')
print('  ' + '-' * 61)
for var, label in vars_panel1.items():
    if var in firm_llmy.columns:
        s = firm_llmy[var].dropna()
        print(f'  {label:<45} {s.mean():>8.2f} {s.std():>8.2f}')

print(f'\n  Unique LLM × industry markets: {firm_llmy.llm_ind.nunique():,}')
print(f'  Unique locations: {firm_llmy.id_llm.nunique()}')
print(f'  Industries: {firm_llmy.ind2d.nunique()}')

# ══════════════════════════════════════════════════════════════════════
# TABLE 1, PANEL II: Manufacturing Workers
# ══════════════════════════════════════════════════════════════════════
print('\n' + '=' * 70)
print('Table 1, Panel II: Manufacturing Workers')
print('=' * 70)

# Load worker cross-section
w = load_workers()
w = w[(w.employed == 1) & (w.manuf == 1)].copy()

# Weighted stats
wt = w['fac500'].values

print(f'\n  {"Variable":<45} {"Mean":>8} {"SD":>8}')
print('  ' + '-' * 61)

for var, label in [('empdep', 'Wage Worker'), ('wmdep_tr', 'Daily Wage'),
                   ('empindep', 'Self-Employed'), ('wmindep_tr', 'Daily SE Earnings')]:
    if var in w.columns:
        valid = w[[var, 'fac500']].dropna()
        if len(valid) > 0:
            m, s = weighted_stats(valid[var].values, valid['fac500'].values)
            print(f'  {label:<45} {m:>8.2f} {s:>8.2f}')

del w

# Transitions from panel
print('\n  Transitions (manufacturing panel):')
panel = load_workerpanel()
panel = panel.sort_values(['num_per', 'year'])
panel = panel[(panel.employed == 1) & (panel.unemp != 1)].copy()

# Number of observations per person
nobs = panel.groupby('num_per')['year'].transform('nunique')
panel = panel[nobs > 1].copy()

# Lag variables
panel['l_empdep'] = panel.groupby('num_per')['empdep'].shift(1)
panel['l_empindep'] = panel.groupby('num_per')['empindep'].shift(1)
panel['l_manuf'] = panel.groupby('num_per')['manuf'].shift(1)

# W→S transition: self-employed now, was wage worker in manuf last period
ws = panel[(panel.l_empdep == 1) & (panel.l_manuf == 1) & (panel.manuf == 1)]
ws_rate = ws['empindep'].mean() if len(ws) > 0 else np.nan

# S→W transition: wage worker now, was self-employed in manuf last period
sw = panel[(panel.l_empindep == 1) & (panel.l_manuf == 1) & (panel.manuf == 1)]
sw_rate = sw['empdep'].mean() if len(sw) > 0 else np.nan

print(f'  W→S Transition rate: {ws_rate:.2f}')
print(f'  S→W Transition rate: {sw_rate:.2f}')

del panel


# ══════════════════════════════════════════════════════════════════════
# TABLE A2: OLS — Self-employment rate, wages, SE earnings on HHI
# ══════════════════════════════════════════════════════════════════════
print('\n' + '=' * 70)
print('Table A2: OLS Regressions — Concentration and Labor Market Outcomes')
print('=' * 70)

merged = load_merged()

# Create key variables (merged uses _tr suffix on earnings)
merged['empindep_w'] = merged['lwmindep_tr'].notna().astype(int)
merged['empdep_w'] = merged['lwmdep_tr'].notna().astype(int)
merged['lhhi_wbill'] = np.log(merged['hhi_wbill'])

# Prepare FE variables — drop rows with missing FE variables first
merged = merged.copy()  # defrag
merged = merged.dropna(subset=['id_llm', 'ind2d', 'year']).copy()
merged['id_llm_str'] = merged['id_llm'].astype(int).astype(str)
merged['ind2d_str'] = merged['ind2d'].astype(int).astype(str)
merged['year_str'] = merged['year'].astype(int).astype(str)

# Controls
std_ctrl = ['female', 'age', 'age2', 'edu_level']
std_ctrl = [c for c in std_ctrl if c in merged.columns]

outcomes = [
    ('empindep_w', 'Self-Employment Rate'),
    ('lwmdep_tr', 'Log Wage'),
    ('lwmindep_tr', 'Log SE Earnings'),
]

for outcome_var, outcome_label in outcomes:
    print(f'\n  Outcome: {outcome_label}')
    sub = merged.dropna(subset=[outcome_var, 'lhhi_wbill'] + std_ctrl).copy()

    # Spec 1: No FE
    try:
        X = sub[['lhhi_wbill'] + std_ctrl]
        y = sub[outcome_var]
        model = sm.OLS(y, sm.add_constant(X)).fit(
            cov_type='cluster', cov_kwds={'groups': sub['llm_ind']})
        b = model.params['lhhi_wbill']
        se = model.bse['lhhi_wbill']
        p = model.pvalues['lhhi_wbill']
        stars = '***' if p < 0.01 else ('**' if p < 0.05 else ('*' if p < 0.1 else ''))
        print(f'    No FE:        β={b:.4f}{stars} (se={se:.4f}), N={int(model.nobs):,}')
    except Exception as e:
        print(f'    No FE: ERROR — {e}')

    # Spec 2: Industry + Year FE
    try:
        res = reghdfe(sub, outcome_var, ['lhhi_wbill'] + std_ctrl,
                      ['ind2d_str', 'year_str'], 'llm_ind')
        b = res['params']['lhhi_wbill']
        se = res['se']['lhhi_wbill']
        p = res['pvalues']['lhhi_wbill']
        stars = '***' if p < 0.01 else ('**' if p < 0.05 else ('*' if p < 0.1 else ''))
        print(f'    Ind+Yr FE:    β={b:.4f}{stars} (se={se:.4f}), N={res["nobs"]:,}')
    except Exception as e:
        print(f'    Ind+Yr FE: ERROR — {e}')

    # Spec 3: LLM + Industry + Year FE (main spec)
    try:
        res = reghdfe(sub, outcome_var, ['lhhi_wbill'] + std_ctrl,
                      ['id_llm_str', 'ind2d_str', 'year_str'], 'llm_ind')
        b = res['params']['lhhi_wbill']
        se = res['se']['lhhi_wbill']
        p = res['pvalues']['lhhi_wbill']
        stars = '***' if p < 0.01 else ('**' if p < 0.05 else ('*' if p < 0.1 else ''))
        print(f'    LLM+Ind+Yr:   β={b:.4f}{stars} (se={se:.4f}), N={res["nobs"]:,}')
    except Exception as e:
        print(f'    LLM+Ind+Yr: ERROR — {e}')

del merged


# ══════════════════════════════════════════════════════════════════════
# TABLE 2: IV Estimation — Labor Market Power
# ══════════════════════════════════════════════════════════════════════
print('\n' + '=' * 70)
print('Table 2: IV Estimation — Inverse Labor Supply Elasticity')
print('=' * 70)

# Build the IV dataset following 02d_iv_estimation.do
firm = load_firm()
firm = firm[(firm.ciiu_n >= 10) & (firm.ciiu_n <= 33)].copy()

# Merge electrification data
elec = load_electrification()
firm = firm.merge(elec, on=['id_llm', 'year'], how='left')
firm['elec_projects'] = firm['elec_projects'].fillna(0)
firm['projvalue'] = firm['projvalue'].fillna(0)

# Cumulative electrification projects per LLM
firm = firm.sort_values(['id_llm', 'year'])
first_in_llm_yr = ~firm.duplicated(subset=['id_llm', 'year'])
firm['etemp'] = firm['elec_projects'].where(first_in_llm_yr, 0)
firm['cumelec'] = firm.groupby('id_llm')['etemp'].cumsum()

# Key outcome and endogenous variables
firm['lnaw'] = np.log(firm['x_labor_twagessalaries'] / firm['n_labor'])
firm['lnemp'] = np.log(firm['n_labor'])

# Electricity intensity: residualize investment share on sales shares within industry
firm['linvelshare'] = np.log(firm['net_sales'] / firm['x_energy_water'])
firm = firm.replace([np.inf, -np.inf], np.nan)

# Sales shares for controls
firm['totsales'] = firm.groupby(['ind2d', 'id_llm', 'year'])['net_sales'].transform('sum')
firm['totsalesnat'] = firm.groupby(['ind2d', 'year'])['net_sales'].transform('sum')
firm['sh_sales'] = firm['net_sales'] / firm['totsales']
firm['sh_sales2'] = firm['sh_sales'] ** 2
firm['sh_salesnat'] = firm['net_sales'] / firm['totsalesnat']
firm['sh_salesnat2'] = firm['sh_salesnat'] ** 2

# Residualize electricity intensity on sales shares + industry FE
emd_valid = firm.dropna(subset=['linvelshare', 'sh_sales', 'sh_sales2', 'sh_salesnat', 'sh_salesnat2']).copy()
if len(emd_valid) > 100:
    emd_valid['ciiu_str'] = emd_valid['ciiu_n'].astype(int).astype(str)
    try:
        res_emd = reghdfe(emd_valid, 'linvelshare',
                          ['sh_sales', 'sh_sales2', 'sh_salesnat', 'sh_salesnat2'],
                          ['ciiu_str'], 'id_llm')
        emd_valid['emd'] = res_emd['result'].resids
    except Exception:
        # Fallback: simple OLS with industry dummies
        ind_dums = pd.get_dummies(emd_valid['ciiu_n'], prefix='ind', drop_first=True)
        X_emd = pd.concat([emd_valid[['sh_sales', 'sh_sales2', 'sh_salesnat', 'sh_salesnat2']], ind_dums], axis=1)
        model_emd = sm.OLS(emd_valid['linvelshare'], sm.add_constant(X_emd)).fit()
        emd_valid['emd'] = model_emd.resid

    # First observation per firm
    emd_valid = emd_valid.sort_values(['iruc', 'year'])
    first_per_firm = emd_valid.drop_duplicates('iruc', keep='first')[['iruc', 'emd']]
    first_per_firm.rename(columns={'emd': 'iemd'}, inplace=True)

    # Median split for high electricity intensity
    med = first_per_firm['iemd'].median()
    first_per_firm['hiemd'] = (first_per_firm['iemd'] > med).astype(int)

    firm = firm.merge(first_per_firm[['iruc', 'hiemd']], on='iruc', how='left')
    firm['intelcum'] = firm['hiemd'] * firm['cumelec']

    # Merge self-employment shares
    llm_se = load_llm_hhi_se()
    firm = firm.merge(llm_se[['ind2d', 'id_llm', 'year', 'empindep_w', 'se_sample']],
                      on=['ind2d', 'id_llm', 'year'], how='left')

    # Merge HHI
    firm_llmy = load_firm_llmy()
    firm = firm.merge(firm_llmy[['ind2d', 'id_llm', 'year', 'hhi_nl', 'hhi_wbill', 'n_firms']],
                      on=['ind2d', 'id_llm', 'year'], how='left', suffixes=('', '_llmy'))

    firm = firm[firm['hhi_nl'].notna()].copy()

    # Create FE variables
    firm['iruc_str'] = firm['iruc'].astype(str)
    firm['idllmt'] = (firm['ind2d'].astype(int).astype(str) + '_' +
                      firm['id_llm'].astype(str) + '_' +
                      firm['year'].astype(int).astype(str))

    # ── Column 1: Average labor market power ──
    print('\nColumn 1: Average Labor Market Power')
    iv_data = firm.dropna(subset=['lnaw', 'lnemp', 'intelcum', 'empindep_w',
                                   'iruc_str', 'idllmt', 'id_llm']).copy()
    iv_data = iv_data[iv_data['empindep_w'].notna()].copy()
    print(f'  IV estimation sample: {len(iv_data):,} firm-year obs')

    try:
        res_iv = ivreghdfe(iv_data, 'lnaw', 'lnemp', 'intelcum',
                           ['iruc_str', 'idllmt'], 'id_llm')
        stars = '***' if res_iv['pval'] < 0.01 else ('**' if res_iv['pval'] < 0.05 else ('*' if res_iv['pval'] < 0.1 else ''))
        print(f'  β(lnemp) = {res_iv["beta"]:.3f}{stars} (se={res_iv["se"]:.3f})')
        print(f'  First-stage F = {res_iv["first_f"]:.2f}')
        print(f'  N = {res_iv["nobs"]:,}, Clusters = {res_iv["n_clust"]}')
    except Exception as e:
        print(f'  ERROR: {e}')

    # ── Column 2: Heterogeneity by HHI terciles ──
    print('\nColumn 2: Heterogeneity by HHI Tercile')
    iv_data['h3con'] = 0
    iv_data.loc[(iv_data['hhi_wbill'] > 0.18) & (iv_data['hhi_wbill'] <= 0.25), 'h3con'] = 1
    iv_data.loc[(iv_data['hhi_wbill'] > 0.25), 'h3con'] = 2

    for h, label in [(0, 'HHI 0-0.18'), (1, 'HHI 0.18-0.25'), (2, 'HHI 0.25-1')]:
        iv_data[f'lnemp_h{h}'] = iv_data['lnemp'] * (iv_data['h3con'] == h).astype(float)
        iv_data[f'intelcum_h{h}'] = iv_data['intelcum'] * (iv_data['h3con'] == h).astype(float)

    endog = [f'lnemp_h{h}' for h in [0, 1, 2]]
    instr = [f'intelcum_h{h}' for h in [0, 1, 2]]

    try:
        res_het = ivreghdfe_multi(iv_data, 'lnaw', endog, instr,
                                  ['iruc_str', 'idllmt'], 'id_llm')
        for h, label in [(0, 'HHI 0-0.18'), (1, 'HHI 0.18-0.25'), (2, 'HHI 0.25-1')]:
            v = f'lnemp_h{h}'
            b = res_het['beta'][v]
            se = res_het['se'][v]
            p = res_het['pvals'][v]
            stars = '***' if p < 0.01 else ('**' if p < 0.05 else ('*' if p < 0.1 else ''))
            print(f'  {label}: β={b:.3f}{stars} (se={se:.3f})')
        print(f'  N = {res_het["nobs"]:,}, Clusters = {res_het["n_clust"]}')
    except Exception as e:
        print(f'  ERROR: {e}')

    # ── Columns 3-4: Further split by self-employment rate ──
    print('\nColumns 3-4: Heterogeneity by HHI × Self-Employment Rate')
    se_mean = iv_data.loc[iv_data['se_sample'].notna(), 'empindep_w'].mean()
    iv_data['hse'] = (iv_data['empindep_w'] > se_mean).astype(int)
    iv_data.loc[iv_data['empindep_w'].isna(), 'hse'] = np.nan

    iv_data['hhcon'] = (iv_data['hhi_wbill'] > 0.25).astype(int)

    for hh in [0, 1]:
        for hs in [0, 1]:
            iv_data[f'lnemp_hh{hh}_hs{hs}'] = (iv_data['lnemp'] *
                                                  (iv_data['hhcon'] == hh).astype(float) *
                                                  (iv_data['hse'] == hs).astype(float))
            iv_data[f'intelcum_hh{hh}_hs{hs}'] = (iv_data['intelcum'] *
                                                     (iv_data['hhcon'] == hh).astype(float) *
                                                     (iv_data['hse'] == hs).astype(float))

    endog4 = [f'lnemp_hh{hh}_hs{hs}' for hh in [0, 1] for hs in [0, 1]]
    instr4 = [f'intelcum_hh{hh}_hs{hs}' for hh in [0, 1] for hs in [0, 1]]

    try:
        res_het2 = ivreghdfe_multi(iv_data.dropna(subset=['hse']), 'lnaw', endog4, instr4,
                                    ['iruc_str', 'idllmt'], 'id_llm')
        for hh, hh_label in [(0, 'HHI≤0.25'), (1, 'HHI>0.25')]:
            for hs, hs_label in [(0, 'Low SE'), (1, 'High SE')]:
                v = f'lnemp_hh{hh}_hs{hs}'
                b = res_het2['beta'][v]
                se = res_het2['se'][v]
                p = res_het2['pvals'][v]
                stars = '***' if p < 0.01 else ('**' if p < 0.05 else ('*' if p < 0.1 else ''))
                print(f'  {hh_label}, {hs_label}: β={b:.3f}{stars} (se={se:.3f})')
        print(f'  N = {res_het2["nobs"]:,}, Clusters = {res_het2["n_clust"]}')
    except Exception as e:
        print(f'  ERROR: {e}')

else:
    print('  Insufficient data for electricity intensity estimation')

del firm

print('\n' + '=' * 70)
print('02_tables.py — DONE')
print('=' * 70)
