"""
03_figures.py — Reproduce main figures for 219907-V1.

Paper: "Labor Market Power, Self-Employment, and Development"
Figure 1: Earnings transitions, Figure 2: Binscatter HHI vs outcomes
"""

import sys, os
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from utils import *

print('=' * 70)
print('03_figures.py — Figure replication')
print('=' * 70)

# ══════════════════════════════════════════════════════════════════════
# FIGURE 1: Earnings Transitions
# Left: P(wage worker | SE earnings decile)
# Right: P(self-employed | wage earnings decile)
# ══════════════════════════════════════════════════════════════════════
print('\nFigure 1: Earnings and Employment Transitions')
print('=' * 70)

panel = load_workerpanel()
panel = panel.sort_values(['num_per', 'year'])

# Keep workers who stay in same LLM×industry
panel['l_llm_ind'] = panel.groupby('num_per')['llm_ind'].shift(1)
panel = panel[panel.llm_ind == panel.l_llm_ind].copy()
panel = panel[panel.employed == 1].copy()

# Create earnings indicator variables
panel['empindep_w'] = panel['lwmindep_tr'].notna().astype(int)
panel['empdep_w'] = panel['lwmdep_tr'].notna().astype(int)

# Lag variables
for var in ['empindep_w', 'empdep_w']:
    panel[f'l_{var}'] = panel.groupby('num_per')[var].shift(1)

# Rank workers by earnings within LLM×industry×year into deciles
for var, suffix in [('lwmindep_tr', 'indep'), ('lwmdep_tr', 'dep')]:
    if var in panel.columns:
        valid = panel[var].notna()
        panel.loc[valid, f'xwm{suffix}'] = panel.loc[valid].groupby(
            ['year', 'llm_ind'])[var].transform(
            lambda x: pd.qcut(x, 10, labels=False, duplicates='drop') + 1
            if len(x) >= 10 else np.nan)

# Lag the decile ranks
for var in ['xwmindep', 'xwmdep']:
    if var in panel.columns:
        panel[f'l_{var}'] = panel.groupby('num_per')[var].shift(1)

# LEFT PANEL: Among those who were self-employed at t-1 (in decile of SE earnings),
# what fraction are wage workers at t?
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Left: P(wage worker at t | SE earnings decile at t-1)
ax = axes[0]
if 'l_xwmindep' in panel.columns:
    # Forward-looking: SE decile at t-1 → P(wage worker at t)
    data_left_fwd = panel.dropna(subset=['l_xwmindep', 'empdep_w']).copy()
    left_fwd = data_left_fwd.groupby('l_xwmindep').agg(
        y=('empdep_w', 'mean'), n=('empdep_w', 'count')).reset_index()

    # Backward-looking: SE decile at t → P(wage worker at t-1)
    data_left_bwd = panel.dropna(subset=['xwmindep', 'l_empdep_w']).copy()
    left_bwd = data_left_bwd.groupby('xwmindep').agg(
        y=('l_empdep_w', 'mean'), n=('l_empdep_w', 'count')).reset_index()

    ax.scatter(left_fwd['l_xwmindep'], left_fwd['y'], c='blue', marker='o',
               label='Wage Workers in Following Year', zorder=5)
    z_fwd = np.polyfit(left_fwd['l_xwmindep'], left_fwd['y'], 1)
    ax.plot(left_fwd['l_xwmindep'], np.polyval(z_fwd, left_fwd['l_xwmindep']),
            'b-', lw=2)

    ax.scatter(left_bwd['xwmindep'], left_bwd['y'], c='red', marker='^',
               label='Wage Workers in Past Year', zorder=5)
    z_bwd = np.polyfit(left_bwd['xwmindep'], left_bwd['y'], 1)
    ax.plot(left_bwd['xwmindep'], np.polyval(z_bwd, left_bwd['xwmindep']),
            'r--', lw=2)

    ax.set_xlabel('Earnings from Self-employment (Decile)')
    ax.set_ylabel('Share')
    ax.legend(fontsize=8)
    ax.set_title('Self-Employment Earnings → Wage Work')

    print('  Left panel (SE earnings → wage work):')
    for _, r in left_fwd.iterrows():
        print(f'    Decile {r.l_xwmindep:.0f}: P(WW next)={r.y:.3f} (N={r.n:.0f})')

# RIGHT PANEL: Among those who were wage workers at t-1 (in decile of wages),
# what fraction are self-employed at t?
ax = axes[1]
if 'l_xwmdep' in panel.columns:
    # Forward-looking: wage decile at t-1 → P(SE at t)
    data_right_fwd = panel.dropna(subset=['l_xwmdep', 'empindep_w']).copy()
    right_fwd = data_right_fwd.groupby('l_xwmdep').agg(
        y=('empindep_w', 'mean'), n=('empindep_w', 'count')).reset_index()

    # Backward-looking: wage decile at t → P(SE at t-1)
    data_right_bwd = panel.dropna(subset=['xwmdep', 'l_empindep_w']).copy()
    right_bwd = data_right_bwd.groupby('xwmdep').agg(
        y=('l_empindep_w', 'mean'), n=('l_empindep_w', 'count')).reset_index()

    ax.scatter(right_fwd['l_xwmdep'], right_fwd['y'], c='blue', marker='o',
               label='Self-Employed in Following Year', zorder=5)
    z_fwd = np.polyfit(right_fwd['l_xwmdep'], right_fwd['y'], 1)
    ax.plot(right_fwd['l_xwmdep'], np.polyval(z_fwd, right_fwd['l_xwmdep']),
            'b-', lw=2)

    ax.scatter(right_bwd['xwmdep'], right_bwd['y'], c='red', marker='^',
               label='Self-Employed in Past Year', zorder=5)
    z_bwd = np.polyfit(right_bwd['xwmdep'], right_bwd['y'], 1)
    ax.plot(right_bwd['xwmdep'], np.polyval(z_bwd, right_bwd['xwmdep']),
            'r--', lw=2)

    ax.set_xlabel('Earnings from Wage Work (Decile)')
    ax.set_ylabel('Share')
    ax.legend(fontsize=8)
    ax.set_title('Wage Earnings → Self-Employment')

    print('\n  Right panel (wage earnings → SE):')
    for _, r in right_fwd.iterrows():
        print(f'    Decile {r.l_xwmdep:.0f}: P(SE next)={r.y:.3f} (N={r.n:.0f})')

plt.suptitle('Figure 1: Earnings and Employment Transitions', fontsize=13)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'figure_1_transitions.png'), dpi=150)
plt.close()
print('  Saved figure_1_transitions.png')

del panel


# ══════════════════════════════════════════════════════════════════════
# FIGURE 2: Binscatter — Concentration vs SE rate and Earnings
# ══════════════════════════════════════════════════════════════════════
print('\n' + '=' * 70)
print('Figure 2: Concentration and Self-Employment / Earnings')
print('=' * 70)

merged = load_merged()
merged['empindep_w'] = merged['lwmindep_tr'].notna().astype(int)

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Left: SE rate vs HHI
ax = axes[0]
try:
    x_bins, y_bins = binscatter(merged, 'empindep_w', 'hhi_wbill', nbins=10)
    ax.scatter(x_bins, y_bins, c='blue', marker='^', zorder=5)
    z = np.polyfit(x_bins, y_bins, 1)
    ax.plot(x_bins, np.polyval(z, x_bins), 'b--', lw=2)
    ax.set_xlabel('HHI (Wage Bill)')
    ax.set_ylabel('Self-Employment Rate')
    ax.set_title('Concentration and Self-Employment')
    print(f'  SE rate vs HHI: slope={z[0]:.4f}')
except Exception as e:
    print(f'  SE rate binscatter ERROR: {e}')

# Right: Earnings vs HHI (wage workers + self-employed)
ax = axes[1]
try:
    x1, y1 = binscatter(merged.dropna(subset=['lwmdep_tr']), 'lwmdep_tr', 'hhi_wbill', nbins=10)
    ax.scatter(x1, y1, c='blue', marker='o', label='Wage Workers', zorder=5)
    z1 = np.polyfit(x1, y1, 1)
    ax.plot(x1, np.polyval(z1, x1), 'b-', lw=2)

    x2, y2 = binscatter(merged.dropna(subset=['lwmindep_tr']), 'lwmindep_tr', 'hhi_wbill', nbins=10)
    ax.scatter(x2, y2, c='red', marker='^', label='Self-Employed', zorder=5)
    z2 = np.polyfit(x2, y2, 1)
    ax.plot(x2, np.polyval(z2, x2), 'r--', lw=2)

    ax.set_xlabel('HHI (Wage Bill)')
    ax.set_ylabel('Log of Earnings')
    ax.set_title('Concentration and Earnings')
    ax.legend()
    print(f'  Wages vs HHI: slope={z1[0]:.4f}')
    print(f'  SE earnings vs HHI: slope={z2[0]:.4f}')
except Exception as e:
    print(f'  Earnings binscatter ERROR: {e}')

plt.suptitle('Figure 2: Employer Concentration and Labor Market Outcomes', fontsize=13)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'figure_2_binscatter.png'), dpi=150)
plt.close()
print('  Saved figure_2_binscatter.png')


# ══════════════════════════════════════════════════════════════════════
# FIGURE B3: Scatter — HHI correlations
# ══════════════════════════════════════════════════════════════════════
print('\n' + '=' * 70)
print('Figure B3: HHI Correlations')
print('=' * 70)

firm_llmy = load_firm_llmy()

fig, axes = plt.subplots(1, 3, figsize=(15, 4))

for idx, (xvar, yvar, title) in enumerate([
    ('hhi_wbill', 'hhi_nl', 'Employment vs Wage-bill HHI'),
    ('hhi_wbill', 'n_firms', 'Firms vs Wage-bill HHI'),
    ('hhi_nl', 'n_firms', 'Firms vs Employment HHI'),
]):
    ax = axes[idx]
    valid = firm_llmy[[xvar, yvar]].dropna()
    ax.scatter(valid[xvar], valid[yvar], alpha=0.2, s=5, c='navy')
    ax.set_xlabel(xvar)
    ax.set_ylabel(yvar)
    ax.set_title(title)
    corr = valid[xvar].corr(valid[yvar])
    print(f'  Corr({xvar}, {yvar}) = {corr:.3f}')

plt.suptitle('Figure B3: HHI Correlations', fontsize=13)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'figure_b3_hhi_correlations.png'), dpi=150)
plt.close()
print('  Saved figure_b3_hhi_correlations.png')


del merged

print('\n' + '=' * 70)
print('03_figures.py — DONE')
print('=' * 70)
