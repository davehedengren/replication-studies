"""
Replicate Figure 1 from Bednar & Gicheva (2014).
"Are Female Supervisors More Female-Friendly?"

Figure 1: Histogram of the normalized AD fixed effects (friendly0 and friendly_all0).
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
from utils import (load_main_data, get_ad_dum_cols, get_ad_dum_all_cols,
                   get_year_dum_cols, get_school_dummies, OUTPUT_DIR,
                   print_separator)
import os


def drop_collinear(X):
    """Drop perfectly collinear columns from design matrix."""
    _, R = np.linalg.qr(X.values.astype(float))
    independent = np.abs(np.diag(R)) > 1e-10
    return X[X.columns[independent]]


def compute_fes(df):
    """Run the regressions and extract normalized FEs."""
    y = df['fsoc2'].values.astype(float)
    year_cols = get_year_dum_cols(df)
    school_dums = get_school_dummies(df)
    ad_cols = get_ad_dum_cols(df)
    ad_all_cols = get_ad_dum_all_cols(df)

    # Model 2: 137 multi-school ADs
    X2 = pd.concat([df[ad_cols].astype(float), df[year_cols].astype(float), school_dums], axis=1)
    X2 = sm.add_constant(X2)
    X2 = drop_collinear(X2)
    model2 = sm.OLS(y, X2).fit()

    fem_friendly = np.zeros(len(df))
    for col in ad_cols:
        if col in model2.params.index:
            fem_friendly += model2.params[col] * df[col].values.astype(float)
    fem_friendly[fem_friendly == 0] = np.nan
    df['fem_friendly'] = fem_friendly

    # Model 3: All 433 ADs
    X3 = pd.concat([df[ad_all_cols].astype(float), df[year_cols].astype(float), school_dums], axis=1)
    X3 = sm.add_constant(X3)
    X3 = drop_collinear(X3)
    model3 = sm.OLS(y, X3).fit()

    fe_all = np.zeros(len(df))
    for col in ad_all_cols:
        if col in model3.params.index:
            fe_all += model3.params[col] * df[col].values.astype(float)
    fe_all[fe_all == 0] = np.nan
    df['fe_all'] = fe_all

    # Normalize
    mask = df['ADunique'] == 1
    df['friendly0'] = df['fem_friendly'] - df.loc[mask, 'fem_friendly'].mean()
    df['friendly_all0'] = df['fe_all'] - df.loc[mask, 'fe_all'].mean()

    return df


def plot_figure1(df):
    """Plot histogram of normalized AD fixed effects."""
    mask = df['ADunique'] == 1

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Panel A: Multi-school ADs (friendly0)
    vals_a = df.loc[mask, 'friendly0'].dropna()
    axes[0].hist(vals_a, bins=20, edgecolor='black', alpha=0.7, color='steelblue')
    axes[0].set_xlabel('Female-Friendliness (Normalized FE)')
    axes[0].set_ylabel('Frequency')
    axes[0].set_title(f'Panel A: Multi-School ADs (N={len(vals_a)})')
    axes[0].axvline(x=0, color='red', linestyle='--', alpha=0.5)

    # Panel B: All ADs (friendly_all0)
    vals_b = df.loc[mask, 'friendly_all0'].dropna()
    axes[1].hist(vals_b, bins=20, edgecolor='black', alpha=0.7, color='coral')
    axes[1].set_xlabel('Female-Friendliness (Normalized FE)')
    axes[1].set_ylabel('Frequency')
    axes[1].set_title(f'Panel B: All ADs (N={len(vals_b)})')
    axes[1].axvline(x=0, color='red', linestyle='--', alpha=0.5)

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'figure1_histogram.png'), dpi=150)
    plt.close()
    print(f"Figure 1 saved to {os.path.join(OUTPUT_DIR, 'figure1_histogram.png')}")

    print(f"\nfriendly0 (multi-school): mean={vals_a.mean():.4f}, sd={vals_a.std():.4f}, N={len(vals_a)}")
    print(f"friendly_all0 (all ADs): mean={vals_b.mean():.4f}, sd={vals_b.std():.4f}, N={len(vals_b)}")


if __name__ == '__main__':
    print_separator("FIGURE 1: Distribution of AD Female-Friendliness")
    df = load_main_data()
    df = compute_fes(df)
    plot_figure1(df)
