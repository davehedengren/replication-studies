"""
02_tables.py — Reproduce main empirical tables for 226781-V1.

Paper: "Trade, Value Added, and Productivity Linkages"

Tables replicated:
  Table 4: Trade Proximity and GDP Correlation (panel FE regressions)
  Table 6: Trade, SR, and Profit (NOS) Comovement
  Table 9: GDP Correlations and EM/IM Margins
  Table 10: Markups-GDP-ToT Correlation
"""

import sys, os
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import pandas as pd
from utils import *

print('=' * 70)
print('02_tables.py — Table replication')
print('=' * 70)

# ══════════════════════════════════════════════════════════════════════
# TABLE 4: Trade Proximity and GDP Correlation
# ══════════════════════════════════════════════════════════════════════
print('\n' + '=' * 70)
print('Table 4: Trade Proximity and GDP Correlation')
print('=' * 70)

tcp = load_tcp_10()
tcp = tcp[(tcp.country1.isin(COUNTRY_LIST)) & (tcp.country2.isin(COUNTRY_LIST))].copy()

# Control variables used in the R code
eu_vars = ['EURO', 'EU_1970', 'EU_1980', 'EU_1990', 'EU_2000', 'URSS']
eu_vars = [v for v in eu_vars if v in tcp.columns]

# Model 1: HP filter, no time window FE
# Model 2: HP filter, + time window FE
# Model 3: HP filter, + controls (SITC sector similarity, third country index)
# Models 4-6: Same for FD filter

specs = []
for filt, dep in [('HP', 'corr_GDP_HP'), ('FD', 'corr_GDP_FD')]:
    for model_num, extra_controls in [
        (1, []),
        (2, []),  # time window FE handled separately
        (3, []),  # + sector + third country
    ]:
        xvars = ['log_inx_int_trade', 'log_inx_fin_trade'] + eu_vars

        if model_num >= 2:
            # Add time window dummies (factor(tw1))
            tcp_sub = tcp.copy()
            for tw in sorted(tcp_sub.tw1.unique()):
                if tw != sorted(tcp_sub.tw1.unique())[0]:  # drop first for identification
                    tcp_sub[f'tw_{tw}'] = (tcp_sub.tw1 == tw).astype(float)
                    xvars.append(f'tw_{tw}')
        else:
            tcp_sub = tcp.copy()

        if model_num == 3:
            for ctrl in ['SITC_sector', 'third_tot']:
                if ctrl in tcp_sub.columns:
                    xvars.append(ctrl)

        try:
            res = feols(tcp_sub, dep, xvars, 'country_pair', 'country_pair')
            specs.append({
                'filter': filt, 'model': model_num,
                'dep': dep,
                'b_int': res['params']['log_inx_int_trade'],
                'se_int': res['se']['log_inx_int_trade'],
                'p_int': res['pvalues']['log_inx_int_trade'],
                'b_fin': res['params']['log_inx_fin_trade'],
                'se_fin': res['se']['log_inx_fin_trade'],
                'p_fin': res['pvalues']['log_inx_fin_trade'],
                'nobs': res['nobs'],
                'r2': res['r2'],
            })
        except Exception as e:
            print(f'  Model {filt}-{model_num}: ERROR — {e}')
            specs.append({'filter': filt, 'model': model_num, 'error': str(e)})

print(f'\n  {"Filter":>6} {"Model":>5} {"β(int)":>16} {"β(fin)":>16} {"N":>6} {"R²":>6}')
print('  ' + '-' * 60)
for s in specs:
    if 'error' not in s:
        int_str = format_coef(s['b_int'], s['se_int'], s['p_int'])
        fin_str = format_coef(s['b_fin'], s['se_fin'], s['p_fin'])
        print(f'  {s["filter"]:>6} {s["model"]:>5} {int_str:>16} {fin_str:>16} {s["nobs"]:>6} {s["r2"]:>6.3f}')

# Paper targets
print('\n  Paper Table 4 targets:')
print('  HP-2: β(int)=0.0662** (0.0323), β(fin)=-0.0140 (0.0317), N=1,740')
print('  FD-2: β(int)=0.0653** (0.0297), β(fin)=-0.0279 (0.0293), N=1,740')


# ══════════════════════════════════════════════════════════════════════
# TABLE 6: Trade, SR (Solow Residual), and Profit (NOS) Comovement
# ══════════════════════════════════════════════════════════════════════
print('\n' + '=' * 70)
print('Table 6: Trade, SR and NOS Comovement')
print('=' * 70)

# Merge SR correlations into TCP
sr = load_sr_data()
sr['country_pair'] = sr['country1'] + '_' + sr['country2']
sr = sr[sr.year.isin([1970, 1980, 1990, 2000])].copy()
sr['tw1'] = sr['year'].map({1970: 1, 1980: 2, 1990: 3, 2000: 4})
# Keep relevant columns
sr_cols = [c for c in sr.columns if 'country_pair' in c or 'tw1' in c or 'corr_SR' in c]
sr_sub = sr[sr_cols].copy()
sr_sub['tw1'] = sr_sub['tw1'].astype(float)
tcp['tw1'] = tcp['tw1'].astype(float)
tcp_sr = tcp.merge(sr_sub, on=['tw1', 'country_pair'], how='left', suffixes=('', '_sr'))

# Time window dummies
tw_vals = sorted(tcp_sr.tw1.dropna().unique())
for tw in tw_vals[1:]:
    tcp_sr[f'tw_{int(tw)}'] = (tcp_sr.tw1 == tw).astype(float)
tw_dummies = [f'tw_{int(tw)}' for tw in tw_vals[1:]]

# SR regressions (HP and FD)
print('\n  Solow Residual (SR3) Correlations:')
sr_dep_vars = [
    ('corr_SR3_HP_10_start', 'HP'),
    ('corr_SR3_FD_10_start', 'FD'),
]
for dep, filt in sr_dep_vars:
    if dep not in tcp_sr.columns:
        print(f'    {filt}: {dep} not found')
        continue
    for model_num, tw_fe in [(1, False), (2, True)]:
        xvars = ['log_inx_int_trade', 'log_inx_fin_trade'] + eu_vars
        if tw_fe:
            xvars += tw_dummies
        try:
            res = feols(tcp_sr, dep, xvars, 'country_pair', 'country_pair')
            b_int = res['params']['log_inx_int_trade']
            se_int = res['se']['log_inx_int_trade']
            p_int = res['pvalues']['log_inx_int_trade']
            b_fin = res['params']['log_inx_fin_trade']
            se_fin = res['se']['log_inx_fin_trade']
            p_fin = res['pvalues']['log_inx_fin_trade']
            print(f'    SR-{filt}-{model_num}: β(int)={format_coef(b_int, se_int, p_int)}, '
                  f'β(fin)={format_coef(b_fin, se_fin, p_fin)}, N={res["nobs"]}')
        except Exception as e:
            print(f'    SR-{filt}-{model_num}: ERROR — {e}')

# Profit (NOS) regressions
print('\n  Net Operating Surplus (NOS) Correlations:')
profit = load_tcp_profit()
profit['log_inx_int_trade'] = np.log(profit['int_trade_index'] + profit['cap_trade_index'])
profit['log_inx_fin_trade'] = np.log(profit['fin_trade_index'])

# Need EU variables for profit data — check availability
profit_eu_vars = [v for v in ['EURO', 'EU_1970', 'EU_1980', 'EU_1990', 'EU_2000', 'URSS']
                  if v in profit.columns]

# Time window dummies for profit data
if 'time_window' in profit.columns:
    tw_vals_p = sorted(profit.time_window.dropna().unique())
    for tw in tw_vals_p[1:]:
        profit[f'tw_{tw}'] = (profit.time_window == tw).astype(float)
    tw_dummies_p = [f'tw_{tw}' for tw in tw_vals_p[1:]]
else:
    tw_dummies_p = []

for dep, filt in [('corrPROFITS_HP', 'HP'), ('corrPROFITS_FD', 'FD')]:
    if dep not in profit.columns:
        print(f'    {dep} not found')
        continue
    for model_num, tw_fe in [(1, False), (2, True)]:
        xvars = ['log_inx_int_trade', 'log_inx_fin_trade'] + profit_eu_vars
        if tw_fe:
            xvars += tw_dummies_p
        try:
            res = feols(profit, dep, xvars, 'country_pair', 'country_pair')
            b_int = res['params']['log_inx_int_trade']
            se_int = res['se']['log_inx_int_trade']
            p_int = res['pvalues']['log_inx_int_trade']
            b_fin = res['params']['log_inx_fin_trade']
            se_fin = res['se']['log_inx_fin_trade']
            p_fin = res['pvalues']['log_inx_fin_trade']
            print(f'    NOS-{filt}-{model_num}: β(int)={format_coef(b_int, se_int, p_int)}, '
                  f'β(fin)={format_coef(b_fin, se_fin, p_fin)}, N={res["nobs"]}')
        except Exception as e:
            print(f'    NOS-{filt}-{model_num}: ERROR — {e}')

print('\n  Paper Table 6 targets:')
print('  SR-HP-2: β(int)=0.0653** (0.0302), N=1,740')
print('  NOS-HP-2: β(int)=0.2387** (0.1186), N=312')


# ══════════════════════════════════════════════════════════════════════
# TABLE 9: GDP Correlations and EM/IM Margins
# ══════════════════════════════════════════════════════════════════════
print('\n' + '=' * 70)
print('Table 9: EM/IM Margins')
print('=' * 70)

eim = load_eim()
eim = eim[(eim.country1.isin(COUNTRY_LIST)) & (eim.country2.isin(COUNTRY_LIST))].copy()

# Create variables as in R code
eim['avg_em_int'] = np.log(eim['sym_value_em_int_norm'])
eim['avg_im_int'] = np.log(eim['sym_value_im_int_norm'])
eim['sqrt_em_int'] = np.log(np.sqrt(eim['var_sym_value_em_int_norm']))
eim['sqrt_im_int'] = np.log(np.sqrt(eim['var_sym_value_im_int_norm']))

# EU and time window controls
eim_eu = [v for v in ['EU'] if v in eim.columns]
if 'time_window' in eim.columns:
    tw_eim = sorted(eim.time_window.dropna().unique())
    for tw in tw_eim[1:]:
        eim[f'tw_{tw}'] = (eim.time_window == tw).astype(float)
    tw_dummies_eim = [f'tw_{tw}' for tw in tw_eim[1:]]
else:
    tw_dummies_eim = []

eim_specs = [
    ('corr_GDP_HP_10_start', ['avg_em_int', 'avg_im_int'], 'HP-avg'),
    ('corr_GDP_HP_10_start', ['sqrt_em_int', 'sqrt_im_int'], 'HP-sqrt'),
    ('corr_GDP_FD_10_start', ['avg_em_int', 'avg_im_int'], 'FD-avg'),
    ('corr_GDP_FD_10_start', ['sqrt_em_int', 'sqrt_im_int'], 'FD-sqrt'),
]

for dep, margin_vars, label in eim_specs:
    if dep not in eim.columns:
        print(f'  {label}: {dep} not found')
        continue
    xvars = margin_vars + eim_eu + tw_dummies_eim
    try:
        res = feols(eim, dep, xvars, 'country_pair', 'country_pair')
        em_var = margin_vars[0]
        im_var = margin_vars[1]
        print(f'  {label}: β(EM)={format_coef(res["params"][em_var], res["se"][em_var], res["pvalues"][em_var])}, '
              f'β(IM)={format_coef(res["params"][im_var], res["se"][im_var], res["pvalues"][im_var])}, '
              f'N={res["nobs"]}')
    except Exception as e:
        print(f'  {label}: ERROR — {e}')

print('\n  Paper Table 9 targets:')
print('  HP-avg: β(EM)=0.0861** (0.0382), β(IM)=-0.0022 (0.0228), N=1,739')
print('  HP-sqrt: β(EM)=0.0854*** (0.0253), β(IM)=-0.0264* (0.0140), N=1,736')


# ══════════════════════════════════════════════════════════════════════
# TABLE 10: Markups and GDP-ToT Correlation
# ══════════════════════════════════════════════════════════════════════
print('\n' + '=' * 70)
print('Table 10: Markups-GDP-ToT Correlation')
print('=' * 70)

mktot = load_markup_tot()
mktot = mktot[mktot.country.isin(COUNTRY_LIST)].copy()

print(f'  Data shape: {mktot.shape}')
print(f'  Columns: {list(mktot.columns)}')

# Regression 1: No time window FE
if 'corr_GDP_TOT_HP' in mktot.columns and 'mean_markup' in mktot.columns:
    try:
        res1 = feols(mktot, 'corr_GDP_TOT_HP', ['mean_markup'], 'country', 'country')
        print(f'  Reg 1: β(markup)={format_coef(res1["params"]["mean_markup"], res1["se"]["mean_markup"], res1["pvalues"]["mean_markup"])}, '
              f'N={res1["nobs"]}, R²={res1["r2"]:.3f}')
    except Exception as e:
        print(f'  Reg 1: ERROR — {e}')

    # Regression 2: + time window FE
    if 'tw' in mktot.columns:
        tw_vals_mk = sorted(mktot.tw.dropna().unique())
        for tw in tw_vals_mk[1:]:
            mktot[f'tw_{int(tw)}'] = (mktot.tw == tw).astype(float)
        tw_mk = [f'tw_{int(tw)}' for tw in tw_vals_mk[1:]]
        try:
            res2 = feols(mktot, 'corr_GDP_TOT_HP', ['mean_markup'] + tw_mk, 'country', 'country')
            print(f'  Reg 2: β(markup)={format_coef(res2["params"]["mean_markup"], res2["se"]["mean_markup"], res2["pvalues"]["mean_markup"])}, '
                  f'N={res2["nobs"]}, R²={res2["r2"]:.3f}')
        except Exception as e:
            print(f'  Reg 2: ERROR — {e}')

    print('\n  Paper Table 10 targets:')
    print('  Reg 1: β=-0.761*** (0.19), N=73, R²=0.155')
    print('  Reg 2: β=-0.536* (0.289), N=73, R²=0.241')


# ══════════════════════════════════════════════════════════════════════
# Summary
# ══════════════════════════════════════════════════════════════════════
print('\n' + '=' * 70)
print('REPLICATION SUMMARY')
print('=' * 70)

summary_rows = []
for s in specs:
    if 'error' not in s:
        summary_rows.append({
            'Table': 4,
            'Spec': f'{s["filter"]}-{s["model"]}',
            'beta_int': round(s['b_int'], 4),
            'se_int': round(s['se_int'], 4),
            'beta_fin': round(s['b_fin'], 4),
            'se_fin': round(s['se_fin'], 4),
            'N': s['nobs'],
        })

pd.DataFrame(summary_rows).to_csv(os.path.join(OUTPUT_DIR, 'table4_results.csv'), index=False)
print('  Saved table4_results.csv')

print('\n' + '=' * 70)
print('02_tables.py — DONE')
print('=' * 70)
