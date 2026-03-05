"""
03_figures.py — Reproduce main figures for Andreolli & Surico (2025).

Paper: "Shock Sizes and the Marginal Propensity to Consume"
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

df = load_analysis_data()
sample = df[(df['marksample10'] == 1)].copy()

# ══════════════════════════════════════════════════════════════════════
# FIGURE 1: MPC Distribution Histograms
# ══════════════════════════════════════════════════════════════════════
print('\n-- Figure 1: MPC distribution histograms --')

fig, axes = plt.subplots(1, 3, figsize=(15, 4))

# Small shock (2010 MPC)
axes[0].hist(sample['mpc_2010'].dropna(), bins=20, color='tab:blue', alpha=0.7,
             edgecolor='white', density=True)
axes[0].set_title('Small')
axes[0].set_xlabel('MPC')
axes[0].set_ylabel('Density')

# Large shock (2012 MPC in 2010 sample)
if 'mpc_2012_in2010' in sample.columns:
    axes[1].hist(sample['mpc_2012_in2010'].dropna(), bins=20, color='tab:red', alpha=0.7,
                 edgecolor='white', density=True)
    axes[1].set_title('Large')
    axes[1].set_xlabel('MPC')

# Difference
if 'dmpc_1012_in2010' in sample.columns:
    axes[2].hist(sample['dmpc_1012_in2010'].dropna(), bins=20, color='tab:green', alpha=0.7,
                 edgecolor='white', density=True)
    axes[2].set_title('Difference')
    axes[2].set_xlabel('Change in MPC')

fig.suptitle('Figure 1: MPC distributions', y=1.02)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'figure_1_mpc_histograms.png'), dpi=150, bbox_inches='tight')
plt.close()
print('  Saved figure_1_mpc_histograms.png')


# ══════════════════════════════════════════════════════════════════════
# FIGURE 2: MPC by Cash-on-Hand Percentiles
# ══════════════════════════════════════════════════════════════════════
print('\n-- Figure 2: MPC by cash-on-hand percentiles --')

# Compute MPC by percentile (OLS of MPC on percentile dummies, no constant)
pct_cols = [f'q100cash_d{i}' for i in range(1, 101)]
available_pcts = [c for c in pct_cols if c in sample.columns and sample[c].notna().any()]

if len(available_pcts) > 10:
    # Small shock
    mask = sample['mpc_2010'].notna()
    for c in available_pcts:
        mask = mask & sample[c].notna()
    s = sample[mask]
    X = s[available_pcts].values
    y_small = s['mpc_2010'].values
    model_small = sm.OLS(y_small, X).fit(cov_type='HC1')
    b_small = model_small.params
    se_small = model_small.bse

    # Large shock
    if 'mpc_2012_in2010' in sample.columns:
        mask2 = sample['mpc_2012_in2010'].notna()
        for c in available_pcts:
            mask2 = mask2 & sample[c].notna()
        s2 = sample[mask2]
        X2 = s2[available_pcts].values
        y_large = s2['mpc_2012_in2010'].values
        model_large = sm.OLS(y_large, X2).fit(cov_type='HC1')
        b_large = model_large.params
        se_large = model_large.bse

    percentiles = np.arange(1, len(available_pcts) + 1)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Panel 1: Small shock
    axes[0].scatter(percentiles, b_small, alpha=0.3, color='blue', s=15)
    # Lowess fit
    lowess = sm.nonparametric.lowess(b_small, percentiles, frac=0.3)
    axes[0].plot(lowess[:, 0], lowess[:, 1], color='blue', linewidth=2)
    axes[0].fill_between(percentiles, b_small - 1.96 * se_small, b_small + 1.96 * se_small,
                         alpha=0.15, color='blue')
    axes[0].set_ylabel('MPC')
    axes[0].set_xlabel('Cash-on-hand percentiles')
    axes[0].set_title('Small')
    axes[0].set_ylim(0.2, 0.8)

    # Panel 2: Large shock
    if 'mpc_2012_in2010' in sample.columns:
        axes[1].scatter(percentiles, b_large, alpha=0.3, color='red', s=15)
        lowess2 = sm.nonparametric.lowess(b_large, percentiles, frac=0.3)
        axes[1].plot(lowess2[:, 0], lowess2[:, 1], color='red', linewidth=2)
        axes[1].fill_between(percentiles, b_large - 1.96 * se_large, b_large + 1.96 * se_large,
                             alpha=0.15, color='red')
        axes[1].set_xlabel('Cash-on-hand percentiles')
        axes[1].set_title('Large')
        axes[1].set_ylim(0.2, 0.8)

    # Panel 3: Both overlaid
    lowess_s = sm.nonparametric.lowess(b_small, percentiles, frac=0.3)
    axes[2].plot(lowess_s[:, 0], lowess_s[:, 1], color='blue', linewidth=2, label='Small')
    axes[2].fill_between(percentiles, b_small - 1.96 * se_small, b_small + 1.96 * se_small,
                         alpha=0.1, color='blue')
    if 'mpc_2012_in2010' in sample.columns:
        lowess_l = sm.nonparametric.lowess(b_large, percentiles, frac=0.3)
        axes[2].plot(lowess_l[:, 0], lowess_l[:, 1], color='red', linewidth=2,
                     linestyle='--', label='Large')
        axes[2].fill_between(percentiles, b_large - 1.96 * se_large, b_large + 1.96 * se_large,
                             alpha=0.1, color='red')
    axes[2].set_xlabel('Cash-on-hand percentiles')
    axes[2].set_title('Both')
    axes[2].set_ylim(0.2, 0.8)
    axes[2].legend()

    fig.suptitle('Figure 2: MPC by cash-on-hand percentiles', y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'figure_2_mpc_percentiles.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print('  Saved figure_2_mpc_percentiles.png')
else:
    print('  Insufficient percentile dummies for Figure 2')


# ══════════════════════════════════════════════════════════════════════
# FIGURE 3: Non-essential Spending and Cash-on-Hand
# ══════════════════════════════════════════════════════════════════════
print('\n-- Figure 3: Eating out share by cash-on-hand --')

if 'eatoutshare_2012_in2010' in sample.columns:
    # Create 50 bins of cash-on-hand
    sample_fig3 = sample[sample['eatoutshare_2012_in2010'].notna()].copy()
    sample_fig3['q50cash'] = pd.qcut(sample_fig3['cash'], 50, labels=False, duplicates='drop') + 1

    # Median eating out share by bin
    medians = sample_fig3.groupby('q50cash')['eatoutshare_2012_in2010'].median()

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.scatter(medians.index * 2, medians.values, color='green', s=30)
    ax.set_xlabel('Percentiles of cash-on-hand')
    ax.set_ylabel('Eating out share')
    ax.set_title('Figure 3: Non-essential spending and cash-on-hand')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'figure_3_eatout_cash.png'), dpi=150)
    plt.close()
    print('  Saved figure_3_eatout_cash.png')
else:
    print('  eatoutshare_2012_in2010 not available')


# ══════════════════════════════════════════════════════════════════════
# FIGURE 4: Binscatter MPC vs Eating Out Share
# ══════════════════════════════════════════════════════════════════════
print('\n-- Figure 4: Binscatter MPC vs eating out share --')

if 'eatoutshare_2012_in2010' in sample.columns:
    def binscatter(x, y, nbins=10):
        """Simple binscatter: bin x, compute mean y in each bin."""
        valid = pd.DataFrame({'x': x, 'y': y}).dropna()
        valid['bin'] = pd.qcut(valid['x'], nbins, labels=False, duplicates='drop')
        return valid.groupby('bin').agg({'x': 'mean', 'y': 'mean'})

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # Small, no controls
    bs = binscatter(sample['eatoutshare_2012_in2010'], sample['mpc_2010'])
    axes[0, 0].scatter(bs['x'], bs['y'], color='blue', s=50)
    z = np.polyfit(bs['x'], bs['y'], 1)
    x_line = np.linspace(bs['x'].min(), bs['x'].max(), 100)
    axes[0, 0].plot(x_line, np.polyval(z, x_line), 'b--', alpha=0.7)
    axes[0, 0].set_title('Small income gain - no controls')
    axes[0, 0].set_xlabel('Eating out share')
    axes[0, 0].set_ylabel('MPC')

    # Large, no controls
    if 'mpc_2012_in2010' in sample.columns:
        bs2 = binscatter(sample['eatoutshare_2012_in2010'], sample['mpc_2012_in2010'])
        axes[0, 1].scatter(bs2['x'], bs2['y'], color='red', s=50)
        z2 = np.polyfit(bs2['x'], bs2['y'], 1)
        axes[0, 1].plot(x_line, np.polyval(z2, x_line), 'r--', alpha=0.7)
        axes[0, 1].set_title('Large income gain - no controls')
        axes[0, 1].set_xlabel('Eating out share')

    # Small, with controls (residualize on decile dummies + demographics)
    decile_cols = [f'q10cash_d{i}' for i in range(1, 11)]
    control_cols = decile_cols + ['age30', 'age45', 'age60', 'male', 'married',
                                   'educ', 'ncomp', 'south', 'unempl']
    available_controls = [c for c in control_cols if c in sample.columns]

    valid_mask = sample['eatoutshare_2012_in2010'].notna() & sample['mpc_2010'].notna()
    for c in available_controls:
        valid_mask = valid_mask & sample[c].notna()
    s_ctrl = sample[valid_mask].copy()

    if len(s_ctrl) > 50:
        X_ctrl = s_ctrl[available_controls].values
        # Residualize MPC and eatout
        resid_mpc = sm.OLS(s_ctrl['mpc_2010'].values, X_ctrl).fit().resid + s_ctrl['mpc_2010'].mean()
        resid_eat = sm.OLS(s_ctrl['eatoutshare_2012_in2010'].values, X_ctrl).fit().resid + s_ctrl['eatoutshare_2012_in2010'].mean()
        bs3 = binscatter(resid_eat, resid_mpc)
        axes[1, 0].scatter(bs3['x'], bs3['y'], color='blue', s=50)
        z3 = np.polyfit(bs3['x'], bs3['y'], 1)
        x_line3 = np.linspace(bs3['x'].min(), bs3['x'].max(), 100)
        axes[1, 0].plot(x_line3, np.polyval(z3, x_line3), 'b--', alpha=0.7)
        axes[1, 0].set_title('Small income gain - with controls')
        axes[1, 0].set_xlabel('Eating out share')
        axes[1, 0].set_ylabel('MPC')

        # Large, with controls
        if 'mpc_2012_in2010' in s_ctrl.columns:
            valid_mask2 = s_ctrl['mpc_2012_in2010'].notna()
            s_ctrl2 = s_ctrl[valid_mask2]
            if len(s_ctrl2) > 50:
                X_ctrl2 = s_ctrl2[available_controls].values
                resid_mpc2 = sm.OLS(s_ctrl2['mpc_2012_in2010'].values, X_ctrl2).fit().resid + s_ctrl2['mpc_2012_in2010'].mean()
                resid_eat2 = sm.OLS(s_ctrl2['eatoutshare_2012_in2010'].values, X_ctrl2).fit().resid + s_ctrl2['eatoutshare_2012_in2010'].mean()
                bs4 = binscatter(resid_eat2, resid_mpc2)
                axes[1, 1].scatter(bs4['x'], bs4['y'], color='red', s=50)
                z4 = np.polyfit(bs4['x'], bs4['y'], 1)
                x_line4 = np.linspace(bs4['x'].min(), bs4['x'].max(), 100)
                axes[1, 1].plot(x_line4, np.polyval(z4, x_line4), 'r--', alpha=0.7)
                axes[1, 1].set_title('Large income gain - with controls')
                axes[1, 1].set_xlabel('Eating out share')

    fig.suptitle('Figure 4: MPC and non-essential spending', y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'figure_4_binscatter_mpc_eatout.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print('  Saved figure_4_binscatter_mpc_eatout.png')
else:
    print('  eatoutshare not available for Figure 4')


# ══════════════════════════════════════════════════════════════════════
# FIGURE: MPC by Cash-on-Hand Deciles (Tobit coefficients)
# ══════════════════════════════════════════════════════════════════════
print('\n-- Figure: MPC by cash-on-hand deciles (Tobit) --')

decile_cols = [f'q10cash_d{i}' for i in range(1, 11)]
avail = [c for c in decile_cols if c in sample.columns]

if len(avail) == 10:
    # OLS on deciles (approximating Tobit pattern)
    mask = sample['mpc_2010'].notna()
    for c in avail:
        mask = mask & sample[c].notna()
    s = sample[mask]

    X = s[avail].values
    model_s = sm.OLS(s['mpc_2010'].values, X).fit(cov_type='HC1')
    model_l = sm.OLS(s['mpc_2012_in2010'].dropna().reindex(s.index).values,
                     X) if 'mpc_2012_in2010' in s.columns else None

    # Refit with valid observations for large
    if 'mpc_2012_in2010' in sample.columns:
        mask_l = mask & sample['mpc_2012_in2010'].notna()
        s_l = sample[mask_l]
        model_l = sm.OLS(s_l['mpc_2012_in2010'].values, s_l[avail].values).fit(cov_type='HC1')

    fig, ax = plt.subplots(figsize=(8, 5))
    deciles = np.arange(1, 11)

    ax.plot(deciles, model_s.params, 'bo-', label='Small (1m income)', linewidth=2)
    ax.fill_between(deciles, model_s.params - 1.96 * model_s.bse,
                    model_s.params + 1.96 * model_s.bse, alpha=0.15, color='blue')

    if model_l is not None:
        ax.plot(deciles, model_l.params, 'rs--', label='Large (1y income)', linewidth=2)
        ax.fill_between(deciles, model_l.params - 1.96 * model_l.bse,
                        model_l.params + 1.96 * model_l.bse, alpha=0.15, color='red')

    ax.set_xlabel('Cash-on-hand decile')
    ax.set_ylabel('MPC')
    ax.set_title('MPC by Cash-on-Hand Decile')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xticks(deciles)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'figure_mpc_deciles.png'), dpi=150)
    plt.close()
    print('  Saved figure_mpc_deciles.png')


print('\n' + '=' * 70)
print('03_figures.py — DONE')
print('=' * 70)
