"""
04_data_audit.py — Data coverage, distributions, and consistency checks.

Paper: "Trade, Value Added, and Productivity Linkages"
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
# 1. Trade-Comovement Dataset Coverage
# ══════════════════════════════════════════════════════════════════════
print('\n' + '=' * 70)
print('1. Trade-Comovement Dataset (TCP_10_third)')
print('=' * 70)

tcp = load_tcp_10()
tcp_30 = tcp[(tcp.country1.isin(COUNTRY_LIST)) & (tcp.country2.isin(COUNTRY_LIST))].copy()

print(f'  Total rows: {len(tcp)}, 30-country: {len(tcp_30)}')
print(f'  Country pairs: {tcp_30.country_pair.nunique()}')
print(f'  Expected pairs: {30*29//2} = C(30,2)')
print(f'  Time windows: {sorted(tcp_30.tw1.unique())}')

# Balance
pairs_per_tw = tcp_30.groupby('tw1')['country_pair'].nunique()
print(f'\n  Pairs per time window:')
for tw, n in pairs_per_tw.items():
    print(f'    tw={tw}: {n} pairs')

# Key variable distributions
print(f'\n  Key Variable Distributions:')
for var in ['corr_GDP_HP', 'corr_GDP_FD', 'log_inx_int_trade', 'log_inx_fin_trade',
            'inx_int_trade', 'inx_fin_trade']:
    if var in tcp_30.columns:
        s = tcp_30[var].dropna()
        print(f'    {var}: N={len(s)}, mean={s.mean():.4f}, sd={s.std():.4f}, '
              f'min={s.min():.4f}, max={s.max():.4f}')

# Missing data
print(f'\n  Missing Data:')
for var in tcp_30.columns[:30]:
    miss = tcp_30[var].isna().sum()
    if miss > 0:
        print(f'    {var}: {miss} ({miss/len(tcp_30)*100:.1f}%)')

# ══════════════════════════════════════════════════════════════════════
# 2. Trade Data Coverage
# ══════════════════════════════════════════════════════════════════════
print('\n' + '=' * 70)
print('2. Johnson-Noguera Trade Data')
print('=' * 70)

trade = load_trade_va()
trade = trade[trade.year.between(1970, 2009)]
print(f'  Shape: {trade.shape}')
print(f'  Years: {int(trade.year.min())}–{int(trade.year.max())} ({trade.year.nunique()} years)')
print(f'  Country codes: {sorted(trade.ecode.unique())}')

# Check for zeros/negatives
for var in ['exports', 'intermedexports', 'finalexports']:
    if var in trade.columns:
        zeros = (trade[var] == 0).sum()
        negs = (trade[var] < 0).sum()
        print(f'  {var}: {zeros} zeros, {negs} negatives')

# ══════════════════════════════════════════════════════════════════════
# 3. GDP Correlation Distributions
# ══════════════════════════════════════════════════════════════════════
print('\n' + '=' * 70)
print('3. GDP Correlation Plausibility')
print('=' * 70)

for var in ['corr_GDP_HP', 'corr_GDP_FD']:
    s = tcp_30[var].dropna()
    print(f'\n  {var}:')
    print(f'    Range: [{s.min():.3f}, {s.max():.3f}]')
    print(f'    In [-1,1]: {((s >= -1) & (s <= 1)).all()}')
    print(f'    Negative: {(s < 0).sum()} ({(s < 0).mean()*100:.1f}%)')
    print(f'    p25={s.quantile(0.25):.3f}, median={s.median():.3f}, p75={s.quantile(0.75):.3f}')

# ══════════════════════════════════════════════════════════════════════
# 4. Profit Dataset
# ══════════════════════════════════════════════════════════════════════
print('\n' + '=' * 70)
print('4. Profit (NOS) Dataset')
print('=' * 70)

profit = load_tcp_profit()
print(f'  Shape: {profit.shape}')
print(f'  Countries: {sorted(set(profit.country1) | set(profit.country2))}')
print(f'  Time windows: {sorted(profit.time_window.unique()) if "time_window" in profit.columns else "N/A"}')

for var in ['corrPROFITS_HP', 'corrPROFITS_FD']:
    if var in profit.columns:
        s = profit[var].dropna()
        print(f'  {var}: N={len(s)}, mean={s.mean():.3f}, range=[{s.min():.3f}, {s.max():.3f}]')

# ══════════════════════════════════════════════════════════════════════
# 5. EM/IM Margin Data
# ══════════════════════════════════════════════════════════════════════
print('\n' + '=' * 70)
print('5. Extensive/Intensive Margin Data')
print('=' * 70)

eim = load_eim()
eim_30 = eim[(eim.country1.isin(COUNTRY_LIST)) & (eim.country2.isin(COUNTRY_LIST))]
print(f'  Shape: {eim.shape}, 30-country: {eim_30.shape}')

for var in ['sym_value_em_int_norm', 'sym_value_im_int_norm',
            'var_sym_value_em_int_norm', 'var_sym_value_im_int_norm']:
    if var in eim_30.columns:
        s = eim_30[var].dropna()
        print(f'  {var}: N={len(s)}, mean={s.mean():.4f}, zeros={(s==0).sum()}, negatives={(s<0).sum()}')

# ══════════════════════════════════════════════════════════════════════
# 6. Markup-ToT-GDP Data
# ══════════════════════════════════════════════════════════════════════
print('\n' + '=' * 70)
print('6. Markup-ToT-GDP Data')
print('=' * 70)

mktot = load_markup_tot()
mktot_30 = mktot[mktot.country.isin(COUNTRY_LIST)]
print(f'  Shape: {mktot.shape}, 30-country: {mktot_30.shape}')
print(f'  Countries: {sorted(mktot_30.country.unique())}')

for var in mktot_30.columns:
    if mktot_30[var].dtype in [np.float64, np.float32, np.int64]:
        s = mktot_30[var].dropna()
        print(f'  {var}: N={len(s)}, mean={s.mean():.3f}, sd={s.std():.3f}')

# ══════════════════════════════════════════════════════════════════════
# 7. Penn World Table Validation
# ══════════════════════════════════════════════════════════════════════
print('\n' + '=' * 70)
print('7. PWT Coverage Check')
print('=' * 70)

pwt = load_pwt()
pwt_30 = pwt[pwt.countrycode.isin(COUNTRY_LIST) & pwt.year.between(1970, 2009)]
print(f'  30-country, 1970-2009: {pwt_30.shape}')
print(f'  Countries with data: {pwt_30.countrycode.nunique()}/30')

# Check for missing years
for c in COUNTRY_LIST:
    yrs = pwt_30[pwt_30.countrycode == c].year.nunique()
    if yrs < 40:
        print(f'  {c}: only {yrs}/40 years')

print('\n' + '=' * 70)
print('04_data_audit.py — DONE')
print('=' * 70)
