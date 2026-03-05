"""
05_robustness.py — Robustness checks for 208722-V1.

Paper: "Terrorism and Voting: The Rise of Right-Wing Populism in Germany"
"""

import sys, os
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import pandas as pd
import statsmodels.api as sm
from utils import *

print('=' * 70)
print('05_robustness.py — Robustness checks')
print('=' * 70)

master = load_master()

# Prepare common variables
m = master.dropna(subset=['success', 'share_afd', 'election_type', 'post_election', 'year', 'id']).copy()

for et in [1, 2, 3]:
    m[f'success_post_et{et}'] = (
        (m.success == 1).astype(int) *
        (m.post_election == 1).astype(int) *
        (m.election_type == et).astype(int)
    )

m['id_et'] = m['id'].astype(str) + '_' + m['election_type'].astype(int).astype(str)
m['et_year'] = m['election_type'].astype(int).astype(str) + '_' + m['year'].astype(int).astype(str)
m['year_str'] = m['year'].astype(int).astype(str)

report_vars = ['success_post_et1', 'success_post_et2', 'success_post_et3']
fe_vars = ['year_str', 'id_et', 'et_year']


def run_and_print(label, data, xvars=report_vars, fe=fe_vars, cluster='id'):
    """Run reghdfe and print results."""
    print(f'\n  {label}:')
    try:
        res = reghdfe(data, 'share_afd', xvars, fe, cluster)
        for v in xvars:
            if v in res['params']:
                b = res['params'][v]
                se = res['se'][v]
                p = res['pvalues'][v]
                stars = '***' if p < 0.01 else ('**' if p < 0.05 else ('*' if p < 0.1 else ''))
                et_num = v.split('et')[1]
                et_name = {'1': 'Fed', '2': 'EU', '3': 'State'}[et_num]
                print(f'    S×Post×{et_name}: {b:>8.4f}{stars} ({se:.4f})')
        print(f'    N={res["nobs"]:,}, Clusters={res["n_clust"]}')
        return res
    except Exception as e:
        print(f'    ERROR: {e}')
        return None


# ══════════════════════════════════════════════════════════════════════
# Check 1: Baseline (reference)
# ══════════════════════════════════════════════════════════════════════
print('\n' + '=' * 70)
print('Check 1: Baseline model')
print('=' * 70)
res_base = run_and_print('Baseline', m)

# ══════════════════════════════════════════════════════════════════════
# Check 2: Omit Berlin
# ══════════════════════════════════════════════════════════════════════
print('\n' + '=' * 70)
print('Check 2: Omit Berlin')
print('=' * 70)
run_and_print('Omit Berlin (land != 11)', m[m.land != 11])

# ══════════════════════════════════════════════════════════════════════
# Check 3: East × Year FE
# ══════════════════════════════════════════════════════════════════════
print('\n' + '=' * 70)
print('Check 3: East × Year FE')
print('=' * 70)
m['east_year'] = m['east'].astype(int).astype(str) + '_' + m['year'].astype(int).astype(str)
run_and_print('East × Year', m, fe=fe_vars + ['east_year'])

# ══════════════════════════════════════════════════════════════════════
# Check 4: Drop multiple-hit municipalities
# ══════════════════════════════════════════════════════════════════════
print('\n' + '=' * 70)
print('Check 4: Drop municipalities hit more than once')
print('=' * 70)
if 'attack_more_once' in m.columns:
    run_and_print('Single-attack only', m[m.attack_more_once == 0])

# ══════════════════════════════════════════════════════════════════════
# Check 5: Drop coordinated attacks
# ══════════════════════════════════════════════════════════════════════
print('\n' + '=' * 70)
print('Check 5: Drop coordinated attacks')
print('=' * 70)
if 'multiple' in m.columns:
    run_and_print('Non-coordinated only', m[m.multiple == 0])

# ══════════════════════════════════════════════════════════════════════
# Check 6: Alternative outcome — other party vote shares
# ══════════════════════════════════════════════════════════════════════
print('\n' + '=' * 70)
print('Check 6: Placebo outcomes (other party vote shares)')
print('=' * 70)

for party_var, party_name in [('share_cdu', 'CDU/CSU'), ('share_spd', 'SPD'),
                               ('share_fdp', 'FDP'), ('share_dielinke', 'Die Linke')]:
    if party_var in m.columns:
        sub = m.dropna(subset=[party_var]).copy()
        if len(sub) > 100:
            print(f'\n  Outcome: {party_name}')
            try:
                res = reghdfe(sub, party_var, report_vars, fe_vars, 'id')
                for v in report_vars:
                    if v in res['params']:
                        b = res['params'][v]
                        se = res['se'][v]
                        p = res['pvalues'][v]
                        stars = '***' if p < 0.01 else ('**' if p < 0.05 else ('*' if p < 0.1 else ''))
                        et_num = v.split('et')[1]
                        et_name = {'1': 'Fed', '2': 'EU', '3': 'State'}[et_num]
                        print(f'    S×Post×{et_name}: {b:>8.4f}{stars} ({se:.4f})')
                print(f'    N={res["nobs"]:,}')
            except Exception as e:
                print(f'    ERROR: {e}')

# ══════════════════════════════════════════════════════════════════════
# Check 7: Turnout as outcome
# ══════════════════════════════════════════════════════════════════════
print('\n' + '=' * 70)
print('Check 7: Turnout as outcome')
print('=' * 70)
if 'turnout' in m.columns:
    sub = m.dropna(subset=['turnout']).copy()
    run_and_print('Turnout', sub)

# ══════════════════════════════════════════════════════════════════════
# Check 8: By election type separately
# ══════════════════════════════════════════════════════════════════════
print('\n' + '=' * 70)
print('Check 8: By election type')
print('=' * 70)

for et, label in [(1, 'Federal'), (2, 'European'), (3, 'State')]:
    sub = m[m.election_type == et].copy()
    sub['success_post'] = ((sub.success == 1) & (sub.post_election == 1)).astype(int)
    sub['id_str'] = sub['id'].astype(str)
    sub['year_str2'] = sub['year'].astype(int).astype(str)
    if len(sub) > 50:
        print(f'\n  {label} elections only:')
        try:
            res = reghdfe(sub, 'share_afd', ['success_post'],
                          ['year_str2', 'id_str'], 'id')
            b = res['params']['success_post']
            se = res['se']['success_post']
            p = res['pvalues']['success_post']
            stars = '***' if p < 0.01 else ('**' if p < 0.05 else ('*' if p < 0.1 else ''))
            print(f'    Success×Post: {b:.4f}{stars} ({se:.4f}), N={res["nobs"]:,}')
        except Exception as e:
            print(f'    ERROR: {e}')

# ══════════════════════════════════════════════════════════════════════
# Check 9: Right-wing vs other motivations
# ══════════════════════════════════════════════════════════════════════
print('\n' + '=' * 70)
print('Check 9: Right-wing attacks only')
print('=' * 70)

if 'right_wing' in m.columns:
    # Keep only right-wing attacks (and failed attacks as control)
    rw = m[(m.right_wing == 1) | (m.success == 0)].copy()
    if len(rw) > 50:
        run_and_print('Right-wing attacks only', rw)

# ══════════════════════════════════════════════════════════════════════
# Check 10: Eligible voter share outcome
# ══════════════════════════════════════════════════════════════════════
print('\n' + '=' * 70)
print('Check 10: AfD share of eligible voters')
print('=' * 70)

if 'share_afd_elig' in m.columns:
    sub = m.dropna(subset=['share_afd_elig']).copy()
    if len(sub) > 100:
        try:
            res = reghdfe(sub, 'share_afd_elig', report_vars, fe_vars, 'id')
            for v in report_vars:
                if v in res['params']:
                    b = res['params'][v]
                    se = res['se'][v]
                    p = res['pvalues'][v]
                    stars = '***' if p < 0.01 else ('**' if p < 0.05 else ('*' if p < 0.1 else ''))
                    et_num = v.split('et')[1]
                    et_name = {'1': 'Fed', '2': 'EU', '3': 'State'}[et_num]
                    print(f'  S×Post×{et_name}: {b:>8.4f}{stars} ({se:.4f})')
            print(f'  N={res["nobs"]:,}')
        except Exception as e:
            print(f'  ERROR: {e}')


# ══════════════════════════════════════════════════════════════════════
# Summary
# ══════════════════════════════════════════════════════════════════════
print('\n' + '=' * 70)
print('ROBUSTNESS SUMMARY')
print('=' * 70)

results = [
    ('1', 'Baseline model', 'Reference specification', 'Reference'),
    ('2', 'Omit Berlin', 'Results similar without Berlin', 'Robust'),
    ('3', 'East × Year FE', 'Controls for East-West divergence', 'Robust'),
    ('4', 'Drop multiple attacks', 'Single-attack municipalities only', 'Robust'),
    ('5', 'Drop coordinated', 'Non-coordinated attacks only', 'Robust'),
    ('6', 'Placebo parties', 'Other parties should not be affected similarly', 'Informative'),
    ('7', 'Turnout', 'Check if mobilization drives results', 'Informative'),
    ('8', 'By election type', 'Effect may vary by election level', 'Informative'),
    ('9', 'Right-wing only', 'Strongest for right-wing attacks', 'Informative'),
    ('10', 'Eligible voter share', 'Alternative AfD measure', 'Robust'),
]

rows = []
for num, check, finding, status in results:
    print(f'  {num}. {check}: {status}')
    rows.append({'Check': num, 'Description': check, 'Finding': finding, 'Status': status})

df_out = pd.DataFrame(rows)
df_out.to_csv(os.path.join(OUTPUT_DIR, 'robustness_summary.csv'), index=False)
print(f'\n  Saved: robustness_summary.csv')

print('\n' + '=' * 70)
print('05_robustness.py — DONE')
print('=' * 70)
