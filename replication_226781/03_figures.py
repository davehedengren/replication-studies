"""
03_figures.py — Reproduce main figures for 226781-V1.

Paper: "Trade, Value Added, and Productivity Linkages"

Note: Figures 1 & 2 in the paper use model-simulated data (MATLAB/Dynare).
We reproduce partial-residual scatter plots from the empirical data instead,
showing the trade-comovement relationship after partialing out FEs.
"""

import sys, os
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
from utils import *

print('=' * 70)
print('03_figures.py — Figure replication')
print('=' * 70)

# ══════════════════════════════════════════════════════════════════════
# Empirical Figure: Trade-GDP Comovement Scatter (partial residuals)
# ══════════════════════════════════════════════════════════════════════
print('\nFigure: Trade-GDP Comovement (Empirical Partial Residuals)')
print('=' * 70)

tcp = load_tcp_10()
tcp = tcp[(tcp.country1.isin(COUNTRY_LIST)) & (tcp.country2.isin(COUNTRY_LIST))].copy()

# Use the main specification (HP filter, time window FE, country pair FE)
# Partial out country_pair FE and log_fin from both Y and log_int
# Then plot residuals

tcp_clean = tcp.dropna(subset=['corr_GDP_HP', 'log_inx_int_trade', 'log_inx_fin_trade']).copy()

# Create country pair dummies via demeaning
for var in ['corr_GDP_HP', 'log_inx_int_trade', 'log_inx_fin_trade']:
    tcp_clean[f'{var}_dm'] = tcp_clean.groupby('country_pair')[var].transform(
        lambda x: x - x.mean())

# Partial out log_fin from both Y and log_int (Frisch-Waugh)
res_y = sm.OLS(tcp_clean['corr_GDP_HP_dm'],
               sm.add_constant(tcp_clean['log_inx_fin_trade_dm'])).fit()
res_x = sm.OLS(tcp_clean['log_inx_int_trade_dm'],
               sm.add_constant(tcp_clean['log_inx_fin_trade_dm'])).fit()

resid_y_int = res_y.resid
resid_x_int = res_x.resid

# Partial out log_int from both Y and log_fin
res_y2 = sm.OLS(tcp_clean['corr_GDP_HP_dm'],
                sm.add_constant(tcp_clean['log_inx_int_trade_dm'])).fit()
res_x2 = sm.OLS(tcp_clean['log_inx_fin_trade_dm'],
                sm.add_constant(tcp_clean['log_inx_int_trade_dm'])).fit()

resid_y_fin = res_y2.resid
resid_x_fin = res_x2.resid

# Highlight specific country pairs
show_pairs = ['CHN_USA', 'DEU_FRA', 'JPN_MEX', 'GBR_USA']
colors = {'CHN_USA': '#c83434', 'DEU_FRA': '#3d85c6',
          'JPN_MEX': '#674ea7', 'GBR_USA': '#e69138'}

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Left: Intermediate goods
ax = axes[0]
mask_other = ~tcp_clean.country_pair.isin(show_pairs)
ax.scatter(resid_x_int[mask_other], resid_y_int[mask_other],
           alpha=0.15, s=10, c='gray', label='All pairs')
for cp in show_pairs:
    mask = tcp_clean.country_pair == cp
    if mask.any():
        ax.scatter(resid_x_int[mask], resid_y_int[mask],
                   s=30, c=colors[cp], label=cp, zorder=5)

# Fit line
z = np.polyfit(resid_x_int, resid_y_int, 1)
x_line = np.linspace(resid_x_int.min(), resid_x_int.max(), 100)
ax.plot(x_line, np.polyval(z, x_line), 'k-', lw=2)
ax.set_xlabel('Log Intermediate Trade (residual)')
ax.set_ylabel('GDP Correlation (residual)')
ax.set_title('Intermediate Inputs')
ax.legend(fontsize=7, loc='upper left')
print(f'  Intermediate goods slope: {z[0]:.4f}')

# Right: Final goods
ax = axes[1]
ax.scatter(resid_x_fin[mask_other], resid_y_fin[mask_other],
           alpha=0.15, s=10, c='gray', label='All pairs')
for cp in show_pairs:
    mask = tcp_clean.country_pair == cp
    if mask.any():
        ax.scatter(resid_x_fin[mask], resid_y_fin[mask],
                   s=30, c=colors[cp], label=cp, zorder=5)

z2 = np.polyfit(resid_x_fin, resid_y_fin, 1)
x_line2 = np.linspace(resid_x_fin.min(), resid_x_fin.max(), 100)
ax.plot(x_line2, np.polyval(z2, x_line2), 'k-', lw=2)
ax.set_xlabel('Log Final Trade (residual)')
ax.set_ylabel('GDP Correlation (residual)')
ax.set_title('Final Goods')
ax.legend(fontsize=7, loc='upper left')
print(f'  Final goods slope: {z2[0]:.4f}')

plt.suptitle('Trade-GDP Comovement (Empirical, HP Filter)', fontsize=13)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'figure_trade_comovement.png'), dpi=150)
plt.close()
print('  Saved figure_trade_comovement.png')


# ══════════════════════════════════════════════════════════════════════
# Figure: Trade Intensity Over Time
# ══════════════════════════════════════════════════════════════════════
print('\n' + '=' * 70)
print('Figure: Trade Intensity Trends')
print('=' * 70)

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Intermediate trade by time window
tw_labels = {1: '1970s', 2: '1980s', 3: '1990s', 4: '2000s'}
tcp['tw_label'] = tcp.tw1.astype(float).map(tw_labels)

for var, label, ax in [('inx_int_trade', 'Intermediate', axes[0]),
                        ('inx_fin_trade', 'Final', axes[1])]:
    if var in tcp.columns:
        medians = tcp.groupby('tw1')[var].median()
        ax.bar(range(len(medians)), medians.values, color='steelblue')
        ax.set_xticks(range(len(medians)))
        ax.set_xticklabels([tw_labels.get(tw, str(tw)) for tw in medians.index])
        ax.set_ylabel('Median Trade Intensity Index')
        ax.set_title(f'{label} Trade')

plt.suptitle('Median Bilateral Trade Intensity by Decade', fontsize=13)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'figure_trade_trends.png'), dpi=150)
plt.close()
print('  Saved figure_trade_trends.png')


# ══════════════════════════════════════════════════════════════════════
# Figure: GDP Correlation Distribution
# ══════════════════════════════════════════════════════════════════════
print('\n' + '=' * 70)
print('Figure: GDP Correlation Distribution')
print('=' * 70)

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

for dep, filt, ax in [('corr_GDP_HP', 'HP', axes[0]),
                       ('corr_GDP_FD', 'FD', axes[1])]:
    if dep in tcp.columns:
        vals = tcp[dep].dropna()
        ax.hist(vals, bins=30, color='steelblue', alpha=0.7, edgecolor='white')
        ax.axvline(vals.median(), color='red', ls='--', lw=2, label=f'Median={vals.median():.3f}')
        ax.set_xlabel(f'GDP Correlation ({filt})')
        ax.set_ylabel('Frequency')
        ax.set_title(f'{filt}-Filtered GDP Correlation')
        ax.legend()
        print(f'  {filt}: median={vals.median():.3f}, mean={vals.mean():.3f}')

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'figure_gdp_corr_dist.png'), dpi=150)
plt.close()
print('  Saved figure_gdp_corr_dist.png')


print('\n' + '=' * 70)
print('03_figures.py — DONE')
print('=' * 70)
