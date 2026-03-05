"""
02_tables.py — Reproduce all main tables from pre-computed statistics for GHT (2025).

Paper: "Temporary Layoffs, Loss-of-Recall, and Cyclical Unemployment Dynamics"
"""

import sys, os
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import pandas as pd
from utils import *

print('=' * 70)
print('02_tables.py — Table replication')
print('=' * 70)

pub = load_published_stats()

# ══════════════════════════════════════════════════════════════════════
# TABLE 1: Unemployment stocks statistics (1978-2019)
# ══════════════════════════════════════════════════════════════════════
print('\n' + '=' * 70)
print('TABLE 1: Total, jobless, and TL unemployment, 1978-2019')
print('=' * 70)

tab1_vars = ['URATEQ', 'PRATEQ', 'TRATEQ', 'JLFROMTL']
tab1_labels = ['u = u^JL + u^TL', 'u^JL', 'u^TL', 'u^JL-from-TL']

print(f'\n  {"Statistic":<20} ', end='')
for lbl in tab1_labels:
    print(f'{lbl:>16}', end='')
print()
print('  ' + '-' * 84)

for stat, sfmt in [('avg', 'mean(x)'), ('std', 'std(x)/std(Y)'), ('rhogdp', 'corr(x,Y)')]:
    print(f'  {sfmt:<20} ', end='')
    for v in tab1_vars:
        key = f'{stat}{v}'
        val = sed_val(pub, key)
        if val is not None:
            print(f'{val:>16.1f}' if stat == 'avg' else f'{val:>16.1f}' if stat == 'std' else f'{val:>16.2f}', end='')
        else:
            print(f'{"---":>16}', end='')
    print()

# Now verify from CSV data
print('\n  -- Independent verification from CSV data --')
tpq = load_quarterly_transitions()
# The quarterly data has trateQ, prateQ columns - compute means
# The paper uses 1978-2019 and reports rates as percentages (e.g., 6.2% unemployment)
tpq_pre = tpq[(tpq.year >= 1978) & (tpq.year <= 2019)].copy()

# Transition probabilities need to be converted to unemployment rates
# The trate/prate columns are rates not percentages
# GHT_stats.csv row 1 has the means
ght = load_ght_stats()
ght_means = ght.iloc[0]  # First row = means
ght_std = ght.iloc[1]    # Second row = relative std
ght_corr = ght.iloc[2]   # Third row = correlations
ght_ac = ght.iloc[3]     # Fourth row = autocorrelations

print(f'  From GHT_stats.csv:')
print(f'    mean(URATE)  = {ght_means["URATEQ"]:.1f}  (published: {sed_val(pub, "avgURATEQ"):.1f})')
print(f'    mean(PRATE)  = {ght_means["PRATEQ"]:.1f}  (published: {sed_val(pub, "avgPRATEQ"):.1f})')
print(f'    mean(TRATE)  = {ght_means["TRATEQ"]:.1f}  (published: {sed_val(pub, "avgTRATEQ"):.1f})')
print(f'    mean(JLFROMTL)={ght_means["JLFROMTL"]:.1f}  (published: {sed_val(pub, "avgJLFROMTL"):.1f})')

# ══════════════════════════════════════════════════════════════════════
# TABLE 2: Transition matrix (4-state, 1978-2019)
# ══════════════════════════════════════════════════════════════════════
print('\n' + '=' * 70)
print('TABLE 2: Transition matrix, gross worker flows, 1978-2019')
print('=' * 70)

states = ['E', 'TL', 'JL', 'N']
flow_map = {'E': 'E', 'TL': 'T', 'JL': 'P', 'N': 'N'}

print(f'\n  {"From\\To":<8}', end='')
for s in states:
    print(f'{s:>10}', end='')
print()
print('  ' + '-' * 48)

for fr in states:
    print(f'  {fr:<8}', end='')
    for to in states:
        key = f'avg{flow_map[fr]}{flow_map[to]}full'
        val = sed_val(pub, key)
        if val is not None:
            print(f'{val:>10.3f}', end='')
        else:
            print(f'{"---":>10}', end='')
    print()

# Verify from monthly CSV
print('\n  -- Independent verification from CSV --')
tp = load_transition_probabilities()
tp_pre = tp[(tp.year >= 1978) & (tp.year <= 2019)].copy()

# Compute means of transition probabilities
for fr_lbl, fr_code in [('E', 'e'), ('TL', 't'), ('JL', 'p')]:
    for to_lbl, to_code in [('E', 'e'), ('TL', 't'), ('JL', 'p'), ('N', 'n')]:
        col = f'ctflow_{fr_code}{to_code}'
        if col in tp_pre.columns:
            csv_mean = tp_pre[col].mean()
            pub_key = f'avg{fr_code.upper()}{to_code.upper()}full'
            pub_val = sed_val(pub, pub_key)
            match = 'MATCH' if pub_val is not None and abs(csv_mean - pub_val) < 0.002 else 'CHECK'
            if pub_val is not None:
                print(f'    p({fr_lbl}->{to_lbl}): CSV={csv_mean:.3f}, Pub={pub_val:.3f} [{match}]')

# ══════════════════════════════════════════════════════════════════════
# TABLE 3: Employment probabilities by unemployment subgroup
# ══════════════════════════════════════════════════════════════════════
print('\n' + '=' * 70)
print('TABLE 3: Employment probabilities by unemployment subgroup')
print('=' * 70)

tab3 = [
    ('JL -> E', 'flow_pe'),
    ('TL -> E', 'flow_te'),
    ('TL-JL diff', 'flow_tpe'),
    ('JL, TL comp (demo)', 'flow_pe_tcomp_I'),
    ('JL, TL comp (industry)', 'flow_pe_tcomp_II'),
]

print(f'\n  {"Category":<28} {"Pr(X to E)":>12}')
print('  ' + '-' * 42)
for lbl, key in tab3:
    val = sed_val(pub, key)
    if val is not None:
        print(f'  {lbl:<28} {val:>12.3f}')

# ══════════════════════════════════════════════════════════════════════
# TABLE 4: Employment probs by duration and unemployment subgroup
# ══════════════════════════════════════════════════════════════════════
print('\n' + '=' * 70)
print('TABLE 4: Employment probs by duration and unemployment subgroup')
print('=' * 70)

tab4 = [
    ('E-JL-JL -> E', 'flow_eppe'),
    ('E-TL-TL -> E', 'flow_ette'),
    ('E-TL-JL -> E', 'flow_etpe'),
    ('E-JL-JL, E-TL-TL comp (demo)', 'flow_eppe_ettcomp_I'),
    ('E-JL-JL, E-TL-TL comp (ind)', 'flow_eppe_ettcomp_II'),
]

print(f'\n  {"Category":<34} {"Pr(X to E)":>12}')
print('  ' + '-' * 48)
for lbl, key in tab4:
    val = sed_val(pub, key)
    if val is not None:
        print(f'  {lbl:<34} {val:>12.3f}')

# ══════════════════════════════════════════════════════════════════════
# TABLE 5: SIPP Recall Shares
# ══════════════════════════════════════════════════════════════════════
print('\n' + '=' * 70)
print('TABLE 5: Recall shares from unemployment by reason for job loss')
print('=' * 70)

panels = ['All', '1996', '2001', '2004', '2008']
print(f'\n  {"Reason":<25}', end='')
for p in panels:
    print(f'{p:>10}', end='')
print()
print('  ' + '-' * 75)

# TL row
print(f'  {"Temporary layoff":<25}', end='')
for p in panels:
    key = 'recall_TL' if p == 'All' else f'recall_TL_{p}'
    val = sed_val(pub, key)
    print(f'{val:>10.3f}' if val else f'{"---":>10}', end='')
print()

# PS row
print(f'  {"Permanent separation":<25}', end='')
for p in panels:
    key = 'recall_PS' if p == 'All' else f'recall_PS_{p}'
    val = sed_val(pub, key)
    print(f'{val:>10.3f}' if val else f'{"---":>10}', end='')
print()

# ══════════════════════════════════════════════════════════════════════
# TABLE 6: Cyclical properties, gross worker flows
# ══════════════════════════════════════════════════════════════════════
print('\n' + '=' * 70)
print('TABLE 6: Cyclical properties, gross worker flows, 1978-2019')
print('=' * 70)

cyc_flows = [
    ('p^{E,TL}', 'ET'),
    ('p^{E,JL}', 'EP'),
    ('p^{TL,E}', 'TE'),
    ('p^{JL,E}', 'PE'),
    ('p^{TL,JL}', 'TP'),
]

print(f'\n  {"Statistic":<20}', end='')
for lbl, _ in cyc_flows:
    print(f'{lbl:>12}', end='')
print()
print('  ' + '-' * 80)

for stat, sfmt in [('std', 'std(x)/std(Y)'), ('rhogdp', 'corr(x,Y)')]:
    print(f'  {sfmt:<20}', end='')
    for _, code in cyc_flows:
        key = f'{stat}{code}full'
        val = sed_val(pub, key)
        print(f'{val:>12.3f}' if val else f'{"---":>12}', end='')
    print()

# ══════════════════════════════════════════════════════════════════════
# TABLE 7: Recession decompositions
# ══════════════════════════════════════════════════════════════════════
print('\n' + '=' * 70)
print('TABLE 7: Decomposition of unemployment increases by recession')
print('=' * 70)

recessions = [('1980/81', '80s'), ('1990', '90s'), ('2001', '01'), ('2007', '08'), ('2020', '20')]

print(f'\n  {"Statistic":<30}', end='')
for lbl, _ in recessions:
    print(f'{lbl:>10}', end='')
print()
print('  ' + '-' * 80)

print(f'  {"From TL, direct + indirect":<30}', end='')
for _, code in recessions:
    key = f'fracT{code}'
    val = sed_val(pub, key)
    print(f'{val:>9.1f}%' if val else f'{"---":>10}', end='')
print()

print(f'  {"Ratio indirect/direct":<30}', end='')
for _, code in recessions:
    key = f'IoverD{code}'
    val = sed_val(pub, key)
    print(f'{val:>10.2f}' if val else f'{"---":>10}', end='')
print()

# ══════════════════════════════════════════════════════════════════════
# TABLE 9: Estimated Parameters (Inner Loop)
# ══════════════════════════════════════════════════════════════════════
print('\n' + '=' * 70)
print('TABLE 9: Calibration — Estimated Parameters (Inner Loop)')
print('=' * 70)

inner_params = [
    ('chi', 'chi_mIS0_NC0', 'Scale, hiring costs'),
    ('varsig_gam', 'varsig_gam_mIS0_NC0', 'Scale, overhead costs (worker)'),
    ('varsig_thet', 'varsig_thet_mIS0_NC0', 'Scale, overhead costs (firm)'),
    ('1-rho_r', 'one_minus_rhor_mIS0_NC0', 'Loss of recall rate'),
    ('b', 'ben_mIS0_NC0', 'Flow value of unemployment'),
]

print(f'\n  {"Param":<15} {"Description":<35} {"Value":>10}')
print('  ' + '-' * 62)
for pname, key, desc in inner_params:
    val = sed_val(pub, key)
    print(f'  {pname:<15} {desc:<35} {val:>10.3f}' if val else f'  {pname:<15} {desc:<35} {"---":>10}')

# ══════════════════════════════════════════════════════════════════════
# TABLE 10: Estimated Parameters (Outer Loop) + Moment Targets
# ══════════════════════════════════════════════════════════════════════
print('\n' + '=' * 70)
print('TABLE 10: Calibration — Outer Loop Parameters + Targets')
print('=' * 70)

outer_params = [
    ('h_elas', 'h_elas_m_IS0_NC0', 'Hiring elasticity, new hires'),
    ('r_elas', 'r_elas_m_IS0_NC0', 'Hiring elasticity, recalls'),
    ('sigma_theta', 'std_theta_m_IS0_NC0', 'Param lognormal F'),
    ('sigma_gamma', 'std_gam_m_IS0_NC0', 'Param lognormal G'),
]

print(f'\n  {"Param":<15} {"Description":<35} {"Value":>10}')
print('  ' + '-' * 62)
for pname, key, desc in outer_params:
    val = sed_val(pub, key)
    print(f'  {pname:<15} {desc:<35} {val:>10.2f}' if val else f'  {pname:<15} {desc:<35} {"---":>10}')

print(f'\n  {"Moment":<40} {"Target":>10} {"Model":>10}')
print('  ' + '-' * 62)
moments = [
    ('SD of hiring rate', 'std_xtot_d', 'std_xtot_m_IS0_NC0'),
    ('SD of total separation rate', 'std_pE_TU_d', 'std_pE_TU_m_IS0_NC0'),
    ('SD of TL unemployment', 'std_ur_d', 'std_ur_m_IS0_NC0'),
    ('SD of JL unemployment', 'std_uu_d', 'std_uu_m_IS0_NC0'),
    ('SD hire from JL / hire from TL', 'std_xx_xr_d', 'std_xx_xr_m_IS0_NC0'),
]
for mlbl, tkey, mkey in moments:
    tval = sed_val(pub, tkey)
    mval = sed_val(pub, mkey)
    ts = f'{tval:.2f}' if tval else '---'
    ms = f'{mval:.2f}' if mval else '---'
    print(f'  {mlbl:<40} {ts:>10} {ms:>10}')

# ══════════════════════════════════════════════════════════════════════
# TABLE C1-C2: Pandemic Parameters
# ══════════════════════════════════════════════════════════════════════
print('\n' + '=' * 70)
print('TABLE C1-C2: Pandemic experiment parameters and shocks')
print('=' * 70)

pandemic_params = [
    ('rho_z', 'rhoz_m_IS0_NC0', 'AR coeff, persistent utilization'),
    ('xi', 'xiR_m_IS0_NC0', 'Adjustment costs for lockdown'),
    ('1-rho_r_phi', 'rhorPHI_m_IS0_NC0', 'Exogenous loss of recall prob'),
]

print(f'\n  {"Param":<15} {"Description":<40} {"Value":>10}')
print('  ' + '-' * 67)
for pname, key, desc in pandemic_params:
    val = sed_val(pub, key)
    print(f'  {pname:<15} {desc:<40} {val:>10.3f}' if val else f'  {pname:<15} {desc:<40} {"---":>10}')

shocks = [
    ('April 2020', 'zshock0_m_IS0_NC0'),
    ('September 2020', 'zshock1_m_IS0_NC0'),
    ('January 2021', 'zshock2_m_IS0_NC0'),
]
print(f'\n  {"Shock timing":<25} {"Value (%)":>10}')
print('  ' + '-' * 37)
for slbl, skey in shocks:
    val = sed_val(pub, skey)
    print(f'  {slbl:<25} {val:>9.2f}%' if val else f'  {slbl:<25} {"---":>10}')

# ══════════════════════════════════════════════════════════════════════
# APPENDIX TABLE A.2: Flow regressions (3-month)
# ══════════════════════════════════════════════════════════════════════
print('\n' + '=' * 70)
print('TABLE A.2: Re-employment rates, 3-month transitions')
print('=' * 70)

print(f'\n  {"Variable":<15}', end='')
for col in range(1, 5):
    print(f'{"(" + str(col) + ")":>12}', end='')
print()
print('  ' + '-' * 63)

for var, vname in [('TU', 'TL-JL'), ('U', 'JL'), ('cons', 'Constant')]:
    # Coefficients
    print(f'  {vname:<15}', end='')
    for col in range(1, 5):
        val = sed_val(pub, f'FR_3m_{col}_{var}BB')
        print(f'{val:>12.3f}' if val else f'{"---":>12}', end='')
    print()
    # SEs
    print(f'  {"":15}', end='')
    for col in range(1, 5):
        raw = pub.get(f'FR_3m_{col}_{var}SE', '')
        print(f'{raw:>12}', end='')
    print()

# N row
print(f'  {"N":<15}', end='')
for col in range(1, 5):
    raw = pub.get(f'FR_3m_{col}_N', '---')
    print(f'{raw:>12}', end='')
print()

# ══════════════════════════════════════════════════════════════════════
# APPENDIX TABLE A.3: Flow regressions (4-month)
# ══════════════════════════════════════════════════════════════════════
print('\n' + '=' * 70)
print('TABLE A.3: Re-employment rates, 4-month transitions')
print('=' * 70)

print(f'\n  {"Variable":<15}', end='')
for col in range(1, 5):
    print(f'{"(" + str(col) + ")":>12}', end='')
print()
print('  ' + '-' * 63)

for var, vname in [('ETU', 'E-TL-JL'), ('EUU', 'E-JL-JL'), ('cons', 'Constant')]:
    print(f'  {vname:<15}', end='')
    for col in range(1, 5):
        val = sed_val(pub, f'FR_4m_{col}_{var}BB')
        print(f'{val:>12.3f}' if val else f'{"---":>12}', end='')
    print()
    print(f'  {"":15}', end='')
    for col in range(1, 5):
        raw = pub.get(f'FR_4m_{col}_{var}SE', '---')
        print(f'{raw:>12}', end='')
    print()

print(f'  {"N":<15}', end='')
for col in range(1, 5):
    raw = pub.get(f'FR_4m_{col}_N', '---')
    print(f'{raw:>12}', end='')
print()

# ══════════════════════════════════════════════════════════════════════
# APPENDIX TABLE A.4: Recall shares by duration
# ══════════════════════════════════════════════════════════════════════
print('\n' + '=' * 70)
print('TABLE A.4: Recall shares by duration')
print('=' * 70)

durations = range(2, 9)
print(f'\n  {"Reason":<25}', end='')
for d in durations:
    print(f'{"<=" + str(d):>8}', end='')
print()
print('  ' + '-' * 81)

print(f'  {"TL":<25}', end='')
for d in durations:
    val = sed_val(pub, f'rShare_TL_{d}')
    print(f'{val:>8.3f}' if val else f'{"---":>8}', end='')
print()

print(f'  {"PS (w/ corrections)":<25}', end='')
for d in durations:
    key = f'rShare_PS_jinfo_{d}'
    val = sed_val(pub, key)
    print(f'{val:>8.3f}' if val else f'{"---":>8}', end='')
print()

print(f'  {"PS (no corrections)":<25}', end='')
for d in durations:
    val = sed_val(pub, f'rShare_PS_{d}')
    print(f'{val:>8.3f}' if val else f'{"---":>8}', end='')
print()

print('\n' + '=' * 70)
print('02_tables.py — DONE')
print('=' * 70)
