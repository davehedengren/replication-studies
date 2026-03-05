"""
02_tables.py — Reproduce main tables for 208722-V1.

Paper: "Terrorism and Voting: The Rise of Right-Wing Populism in Germany"
Tables 1 (balance), 2 (baseline DiD), 3 (propensity matching)
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
# TABLE 1: Balance Table
# ══════════════════════════════════════════════════════════════════════
print('\n' + '=' * 70)
print('Table 1A: Balance across municipalities (pre-attack)')
print('=' * 70)

master = load_master()

# Keep targeted municipalities only
targeted = master[master.success.notna()].copy()

# Cross-section: one obs per municipality-year (first election type)
targeted_sorted = targeted.sort_values(['id', 'year', 'election_type'])
targeted_sorted['counter'] = targeted_sorted.groupby(['id', 'year']).cumcount() + 1

# Pre-attack period: time_to_attack == -1
pre_attack = targeted_sorted[(targeted_sorted.time_to_attack == -1) & (targeted_sorted.counter == 1)]

covariates = [
    'income_pc', 'unemployed', 'empl_tot', 'tax_pc',
    'pop2', 'avg_age', 'share_men',
    'influx_pc', 'outflux_pc', 'asyl_tot_tsd', 'asyl_syr_tsd',
    'grad_abitur', 'grad_nolowereduc',
    'area', 'area_forest', 'area_agr', 'east',
    'welfare_pc', 'welfare_pc_for',
    'total_accidents', 'fatal_accidents',
    'hotels_open', 'guests_night_tsd',
    'hospitals', 'beds_avail',
]

# Filter to variables that exist
covariates = [c for c in covariates if c in pre_attack.columns]

bal = balance_test(pre_attack, 'success', covariates, cluster_var='id')

print(f'\n  {"Variable":<25} {"Mean(0)":<10} {"Mean(1)":<10} {"Diff":<10} {"p-val":<8} {"N":<6}')
print('  ' + '-' * 69)
for _, row in bal.iterrows():
    sig = '*' if row.pval < 0.05 else ''
    print(f'  {row["var"]:<25} {row.mean_0:<10.2f} {row.mean_1:<10.2f} '
          f'{row["diff"]:<10.2f} {row.pval:<8.3f} {row.n:<6.0f}{sig}')

n_sig = (bal.pval < 0.05).sum()
print(f'\n  {n_sig}/{len(bal)} variables significant at 5%')


# Table 1B: Balance across attacks
print('\n' + '=' * 70)
print('Table 1B: Balance across attacks (terror.dta)')
print('=' * 70)

terror = load_terror()
# Rename islamistic to islamist for display
if 'islamistic' in terror.columns:
    terror = terror.rename(columns={'islamistic': 'islamist'})

attack_vars = ['explosives', 'firearms', 'melee', 'nkill', 'nwound',
               'right_wing', 'nazi', 'left_wing', 'islamist']
attack_vars = [c for c in attack_vars if c in terror.columns]

bal_attack = balance_test(terror, 'success', attack_vars)

print(f'\n  {"Variable":<20} {"Mean(0)":<10} {"Mean(1)":<10} {"Diff":<10} {"p-val":<8} {"N":<6}')
print('  ' + '-' * 64)
for _, row in bal_attack.iterrows():
    sig = '*' if row.pval < 0.05 else ''
    print(f'  {row["var"]:<20} {row.mean_0:<10.3f} {row.mean_1:<10.3f} '
          f'{row["diff"]:<10.3f} {row.pval:<8.3f} {row.n:<6.0f}{sig}')


# ══════════════════════════════════════════════════════════════════════
# TABLE 2: Baseline DiD — AfD vote share
# ══════════════════════════════════════════════════════════════════════
print('\n' + '=' * 70)
print('Table 2: Baseline Results — AfD Vote Share')
print('=' * 70)

master = load_master()

# Create interaction terms for the DiD specification
# The key regressors: success × post × election_type interactions
# Stata: i.success##i.post_election#ib2.election_type
# We need: success × post × Federal, success × post × European, success × post × State

m = master.dropna(subset=['success', 'share_afd', 'election_type', 'post_election', 'year', 'id']).copy()

# Create interaction dummies
for et in [1, 2, 3]:
    m[f'success_post_et{et}'] = (
        (m.success == 1).astype(int) *
        (m.post_election == 1).astype(int) *
        (m.election_type == et).astype(int)
    )

# Also need lower-order interactions for proper specification
m['success_post'] = (m.success == 1).astype(int) * (m.post_election == 1).astype(int)
for et in [1, 2, 3]:
    m[f'post_et{et}'] = (m.post_election == 1).astype(int) * (m.election_type == et).astype(int)
    m[f'success_et{et}'] = (m.success == 1).astype(int) * (m.election_type == et).astype(int)
m['post'] = (m.post_election == 1).astype(int)
m['success_d'] = (m.success == 1).astype(int)

# Build FE variables
# Stata: a(i.year i.id#i.election_type i.election_type#i.year)
m['id_et'] = m['id'].astype(str) + '_' + m['election_type'].astype(int).astype(str)
m['et_year'] = m['election_type'].astype(int).astype(str) + '_' + m['year'].astype(int).astype(str)
m['year_str'] = m['year'].astype(int).astype(str)

# Full interaction set for proper DD specification
xvars_report = ['success_post_et1', 'success_post_et2', 'success_post_et3']
xvars_all = xvars_report + ['success_d', 'post'] + \
            [f'post_et{et}' for et in [1, 2, 3]] + \
            [f'success_et{et}' for et in [1, 2, 3]] + \
            ['success_post']

fe_vars = ['year_str', 'id_et', 'et_year']

# Filter to non-constant columns
xvars_use = []
for v in xvars_all:
    if v in m.columns and m[v].std() > 0:
        xvars_use.append(v)

print('\nColumn 1: Baseline Model')
try:
    res1 = reghdfe(m, 'share_afd', xvars_report, fe_vars, 'id')
    for v in xvars_report:
        label = {
            'success_post_et1': 'Success × Post × Federal',
            'success_post_et2': 'Success × Post × European',
            'success_post_et3': 'Success × Post × State',
        }[v]
        b = res1['params'].get(v, np.nan)
        se = res1['se'].get(v, np.nan)
        p = res1['pvalues'].get(v, np.nan)
        stars = '***' if p < 0.01 else ('**' if p < 0.05 else ('*' if p < 0.1 else ''))
        print(f'  {label:<35} {b:>10.4f}{stars}')
        print(f'  {"":35} ({se:>8.4f})')
    print(f'  N = {res1["nobs"]:,}, Clusters = {res1["n_clust"]}')

    # Mean and SD of AfD for state elections
    state_afd = m.loc[(m.election_type == 3) & m.share_afd.notna(), 'share_afd']
    print(f'  Mean Y (state): {state_afd.mean():.4f}, SD: {state_afd.std():.4f}')
except Exception as e:
    print(f'  ERROR: {e}')

# Column 2: East × Year
print('\nColumn 2: East × Year FE')
try:
    m['east_year'] = m['east'].astype(int).astype(str) + '_' + m['year'].astype(int).astype(str)
    fe_vars2 = fe_vars + ['east_year']
    res2 = reghdfe(m, 'share_afd', xvars_report, fe_vars2, 'id')
    for v in xvars_report:
        b = res2['params'].get(v, np.nan)
        se = res2['se'].get(v, np.nan)
        p = res2['pvalues'].get(v, np.nan)
        stars = '***' if p < 0.01 else ('**' if p < 0.05 else ('*' if p < 0.1 else ''))
        print(f'  {v:<35} {b:>10.4f}{stars} ({se:.4f})')
    print(f'  N = {res2["nobs"]:,}, Clusters = {res2["n_clust"]}')
except Exception as e:
    print(f'  ERROR: {e}')

# Column 3: Omit Berlin
print('\nColumn 3: Omit Berlin (land != 11)')
try:
    res3 = reghdfe(m[m.land != 11], 'share_afd', xvars_report, fe_vars, 'id')
    for v in xvars_report:
        b = res3['params'].get(v, np.nan)
        se = res3['se'].get(v, np.nan)
        p = res3['pvalues'].get(v, np.nan)
        stars = '***' if p < 0.01 else ('**' if p < 0.05 else ('*' if p < 0.1 else ''))
        print(f'  {v:<35} {b:>10.4f}{stars} ({se:.4f})')
    print(f'  N = {res3["nobs"]:,}, Clusters = {res3["n_clust"]}')
except Exception as e:
    print(f'  ERROR: {e}')


# ══════════════════════════════════════════════════════════════════════
# TABLE 3: Propensity Score Matching
# ══════════════════════════════════════════════════════════════════════
print('\n' + '=' * 70)
print('Table 3: Propensity Score Matching')
print('=' * 70)

mp = load_master_propensity()

# Columns 1-2: 1933 NSDAP vote share (cross-section)
mp_sorted = mp.sort_values(['id', 'year', 'election_type'])
mp_sorted['xsec_counter'] = mp_sorted.groupby(['id', 'year', 'election_type']).cumcount() + 1

xsec = mp_sorted[mp_sorted.xsec_counter == 1].copy()

print('\nPanel A: 1933 NSDAP Vote Share')
for treat_var, label in [('new_success', 'Success vs Placebo'), ('new_fail', 'Failed vs Placebo')]:
    if treat_var in xsec.columns and 'share_nsdap33' in xsec.columns:
        valid = xsec[[treat_var, 'share_nsdap33', 'id']].dropna()
        if len(valid) > 10:
            y = valid['share_nsdap33'].values
            X = sm.add_constant(valid[treat_var].values)
            model = sm.OLS(y, X).fit(cov_type='cluster',
                                      cov_kwds={'groups': valid['id'].values})
            b = model.params[1]
            se = model.bse[1]
            p = model.pvalues[1]
            stars = '***' if p < 0.01 else ('**' if p < 0.05 else ('*' if p < 0.1 else ''))
            print(f'  {label}: beta={b:.4f}{stars} (se={se:.4f}), N={int(model.nobs)}, Clusters={valid.id.nunique()}')

# Columns 3-4: AfD DiD with matched sample
print('\nPanel B: AfD Vote Share (matched sample)')
mp2 = mp.dropna(subset=['share_afd', 'election_type', 'post_election', 'year', 'id']).copy()

for treat_var, match_var, label in [
    ('new_success', 'matched_sample_success', 'Success matched'),
    ('new_fail', 'matched_sample_fail', 'Failed matched'),
]:
    if treat_var in mp2.columns and match_var in mp2.columns:
        sub = mp2[(mp2[match_var] == 1)].copy()
        if len(sub) < 50:
            print(f'  {label}: insufficient data ({len(sub)} obs)')
            continue

        for et in [1, 2, 3]:
            sub[f'{treat_var}_post_et{et}'] = (
                (sub[treat_var] == 1).astype(int) *
                (sub.post_election == 1).astype(int) *
                (sub.election_type == et).astype(int)
            )

        sub['id_et'] = sub['id'].astype(str) + '_' + sub['election_type'].astype(int).astype(str)
        sub['et_year'] = sub['election_type'].astype(int).astype(str) + '_' + sub['year'].astype(int).astype(str)
        sub['year_str'] = sub['year'].astype(int).astype(str)

        report_vars = [f'{treat_var}_post_et{et}' for et in [1, 2, 3]]

        try:
            res = reghdfe(sub, 'share_afd', report_vars,
                          ['year_str', 'id_et', 'et_year'], 'id')
            print(f'\n  {label}:')
            for v in report_vars:
                b = res['params'].get(v, np.nan)
                se = res['se'].get(v, np.nan)
                p = res['pvalues'].get(v, np.nan)
                stars = '***' if p < 0.01 else ('**' if p < 0.05 else ('*' if p < 0.1 else ''))
                et_label = v.split('_et')[1]
                et_name = {1: 'Federal', 2: 'European', 3: 'State'}[int(et_label)]
                print(f'    {treat_var} × Post × {et_name}: {b:.4f}{stars} ({se:.4f})')
            print(f'    N = {res["nobs"]:,}, Clusters = {res["n_clust"]}')
        except Exception as e:
            print(f'  {label}: ERROR — {e}')


print('\n' + '=' * 70)
print('02_tables.py — DONE')
print('=' * 70)
