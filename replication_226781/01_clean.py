"""
01_clean.py — Load and validate all datasets for 226781-V1.

Paper: "Trade, Value Added, and Productivity Linkages"
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
# 1. Main Trade-Comovement Dataset (TCP_10_third)
# ══════════════════════════════════════════════════════════════════════
print('\n1. Trade-Comovement Dataset (10-year windows)')
print('=' * 70)

tcp = load_tcp_10()
print(f'  Shape: {tcp.shape}')
print(f'  Country pairs: {tcp.country_pair.nunique()}')
print(f'  Time windows: {sorted(tcp.tw1.unique())}')
print(f'  Countries: {sorted(set(tcp.country1.unique()) | set(tcp.country2.unique()))}')

# Filter to 30-country sample
tcp = tcp[(tcp.country1.isin(COUNTRY_LIST)) & (tcp.country2.isin(COUNTRY_LIST))].copy()
print(f'  After country filter: {tcp.shape}')

# Key variables
for var in ['corr_GDP_HP', 'corr_GDP_FD', 'log_inx_int_trade', 'log_inx_fin_trade']:
    if var in tcp.columns:
        s = tcp[var].dropna()
        print(f'  {var}: N={len(s)}, mean={s.mean():.4f}, sd={s.std():.4f}')

tcp.to_parquet(os.path.join(OUTPUT_DIR, 'tcp_10.parquet'), index=False)

# ══════════════════════════════════════════════════════════════════════
# 2. Value-Added Trade Data
# ══════════════════════════════════════════════════════════════════════
print('\n2. Johnson-Noguera Trade Data')
print('=' * 70)

trade = load_trade_va()
trade = trade[trade.year.between(1970, 2009)]
print(f'  Shape: {trade.shape}')
print(f'  Years: {int(trade.year.min())}–{int(trade.year.max())}')
print(f'  Exporters: {trade.ecode.nunique()}, Importers: {trade.icode.nunique()}')

# 30-country sample trade share
trade_bilateral = trade[trade.ecode != trade.icode]
trade_30 = trade_bilateral[
    (trade_bilateral.ecode.isin(COUNTRY_LIST)) &
    (trade_bilateral.icode.isin(COUNTRY_LIST))
]
pct = trade_30.exports.sum() / trade_bilateral.exports.sum() * 100
print(f'  30-country share of bilateral exports: {pct:.0f}%')
int_pct = trade_30.intermedexports.sum() / trade_bilateral.intermedexports.sum() * 100
print(f'  30-country share of intermediate exports: {int_pct:.0f}%')
int_share = trade_30.intermedexports.sum() / trade_30.exports.sum() * 100
print(f'  Intermediate share of 30-country exports: {int_share:.0f}%')

# ══════════════════════════════════════════════════════════════════════
# 3. Penn World Table
# ══════════════════════════════════════════════════════════════════════
print('\n3. Penn World Table 10.0')
print('=' * 70)

pwt = load_pwt()
print(f'  Shape: {pwt.shape}')
pwt_sample = pwt[pwt.countrycode.isin(COUNTRY_LIST) & pwt.year.between(1970, 2009)]
gdp_share = pwt_sample.rgdpo.sum() / pwt[pwt.year.between(1970, 2009)].rgdpo.sum()
print(f'  30-country GDP share (1970-2009): {gdp_share:.1%}')

# ══════════════════════════════════════════════════════════════════════
# 4. Solow Residual Correlations
# ══════════════════════════════════════════════════════════════════════
print('\n4. Solow Residual Correlations')
print('=' * 70)

sr = load_sr_data()
print(f'  Shape: {sr.shape}')
print(f'  Columns: {list(sr.columns)}')

# ══════════════════════════════════════════════════════════════════════
# 5. Profit Comovement
# ══════════════════════════════════════════════════════════════════════
print('\n5. Profit (NOS) Comovement')
print('=' * 70)

profit = load_tcp_profit()
print(f'  Shape: {profit.shape}')
print(f'  Country pairs: {profit.country_pair.nunique()}')
countries_nos = sorted(set(profit.country1.unique()) | set(profit.country2.unique()))
print(f'  Countries: {len(countries_nos)} — {countries_nos}')

# ══════════════════════════════════════════════════════════════════════
# 6. EM/IM Margins
# ══════════════════════════════════════════════════════════════════════
print('\n6. Extensive/Intensive Margin Dataset')
print('=' * 70)

eim = load_eim()
eim = eim[(eim.country1.isin(COUNTRY_LIST)) & (eim.country2.isin(COUNTRY_LIST))].copy()
print(f'  Shape: {eim.shape}')
print(f'  Country pairs: {eim.country_pair.nunique() if "country_pair" in eim.columns else "N/A"}')

# ══════════════════════════════════════════════════════════════════════
# 7. Markup-ToT-GDP Dataset
# ══════════════════════════════════════════════════════════════════════
print('\n7. Markup-ToT-GDP Dataset')
print('=' * 70)

mktot = load_markup_tot()
mktot = mktot[mktot.country.isin(COUNTRY_LIST)].copy()
print(f'  Shape: {mktot.shape}')
print(f'  Countries: {sorted(mktot.country.unique())}')
print(f'  Columns: {list(mktot.columns)}')

print('\n' + '=' * 70)
print('01_clean.py — DONE')
print('=' * 70)
