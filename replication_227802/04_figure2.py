"""
04_figure2.py — Figure 2: Event Study (Stacked DiD)

Dynamic effects of childcare law enactment on FLFP (25-54).
Uses stacked difference-in-differences with country-clustered SEs.
"""

import sys, os
import numpy as np
import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')

sys.path.insert(0, os.path.dirname(__file__))
from utils import (CLEAN_DATA_PATH, OUTPUT_DIR, create_stack, weights_stacked_did)


def run_stacked_event_study(df, y_var='lfpr_f_25_54', treat_time='enactment',
                            kappa_pre=6, kappa_post=7):
    """
    Run stacked DiD event study regression.
    y = sum_k(beta_k * 1{event_time==k} * treat) + alpha_treat + gamma_event_time + eps
    Reference period: event_time = -1
    """
    # Filter to non-missing outcome
    df_clean = df[df[y_var].notna()].copy()

    # Create stacked dataset
    print("Creating stacked dataset...")
    df_stack = create_stack(df_clean, 'year', 'id', treat_time, kappa_pre, kappa_post)
    print(f"  Stacked dataset: {len(df_stack)} rows")

    # Add weights
    df_stack = weights_stacked_did(df_stack, 'treat', 'event_time', 'sub_exp')
    df_stack['event_time'] = df_stack['event_time'].astype(int)

    # Create interaction dummies: event_time * treat (excluding ref period -1)
    event_times = sorted(df_stack['event_time'].unique())
    ref_time = -1

    # Build design matrix
    y = df_stack[y_var].values
    weights = df_stack['stack_weight'].values

    # Create event_time x treat dummies (excluding reference)
    et_treat_cols = {}
    for et in event_times:
        if et == ref_time:
            continue
        col_name = f'et_{et}_treat'
        et_treat_cols[col_name] = ((df_stack['event_time'] == et) &
                                    (df_stack['treat'] == 1)).astype(float).values

    # Treat and event_time fixed effects
    treat_fe = df_stack['treat'].astype(float).values
    et_dummies = pd.get_dummies(df_stack['event_time'], prefix='et', drop_first=True, dtype=float)

    X_df = pd.DataFrame(et_treat_cols)
    X_df['treat_fe'] = treat_fe
    for col in et_dummies.columns:
        X_df[col] = et_dummies[col].values
    X_df = sm.add_constant(X_df)

    # Weighted OLS with clustering by country
    model = sm.WLS(y, X_df, weights=weights)
    groups = df_stack['id'].values

    try:
        results = model.fit(cov_type='cluster', cov_kwds={'groups': groups})
    except Exception:
        results = model.fit(cov_type='HC1')

    # Extract event-time coefficients
    coefs = []
    for et in event_times:
        if et == ref_time:
            coefs.append({'event_time': et, 'estimate': 0.0, 'se': 0.0, 'pval': 1.0})
            continue
        col_name = f'et_{et}_treat'
        if col_name in results.params.index:
            coefs.append({
                'event_time': et,
                'estimate': results.params[col_name],
                'se': results.bse[col_name],
                'pval': results.pvalues[col_name],
            })

    coef_df = pd.DataFrame(coefs)
    coef_df['ci_lo'] = coef_df['estimate'] - 1.96 * coef_df['se']
    coef_df['ci_hi'] = coef_df['estimate'] + 1.96 * coef_df['se']

    # Compute ATT (average of post-treatment coefficients)
    post_coefs = coef_df[coef_df['event_time'] >= 0]
    att = post_coefs['estimate'].mean()

    # ATT SE via delta method (average of post-treatment coefficients)
    from scipy import stats as sp_stats

    post_et_names = [f'et_{et}_treat' for et in range(0, kappa_post + 1)
                     if f'et_{et}_treat' in results.params.index]
    if post_et_names:
        n_post = len(post_et_names)
        vcov = results.cov_params()
        att_var = 0
        for name1 in post_et_names:
            for name2 in post_et_names:
                if name1 in vcov.index and name2 in vcov.columns:
                    att_var += vcov.loc[name1, name2]
        att_se = np.sqrt(max(att_var, 0)) / n_post
        att_pval = 2 * (1 - sp_stats.norm.cdf(abs(att / att_se))) if att_se > 0 else 1.0
    else:
        att_se = np.nan
        att_pval = np.nan

    return coef_df, att, att_se, att_pval, results


def plot_event_study(coef_df, att, att_se, att_pval):
    """Create event study plot matching Figure 2 from the paper."""
    fig, ax = plt.subplots(figsize=(8, 5))

    pre = coef_df[coef_df['event_time'] < 0]
    post = coef_df[coef_df['event_time'] >= 0]

    color_pre = '#F98400'
    color_post = '#00A08A'

    # Pre-treatment
    ax.errorbar(pre['event_time'], pre['estimate'],
                yerr=[pre['estimate'] - pre['ci_lo'], pre['ci_hi'] - pre['estimate']],
                fmt='o', color=color_pre, capsize=3, markersize=6, label='Pre')

    # Post-treatment
    ax.errorbar(post['event_time'], post['estimate'],
                yerr=[post['estimate'] - post['ci_lo'], post['ci_hi'] - post['estimate']],
                fmt='o', color=color_post, capsize=3, markersize=6, label='Post')

    ax.axhline(y=0, color='black', linewidth=0.5)
    ax.axvline(x=-0.5, color='black', linestyle='--', linewidth=0.5)

    ax.set_xlabel('Years from childcare law enactment', fontsize=12)
    ax.set_ylabel('Estimate', fontsize=12)
    ax.set_xticks(range(-6, 8))

    # ATT annotation
    ax.annotate(f'ATT: {att:.2f} ({att_se:.3f}). p-val. {att_pval:.2f}',
                xy=(0.95, 0.95), xycoords='axes fraction',
                ha='right', va='top', fontsize=9,
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'figure2.png'), dpi=150)
    plt.savefig(os.path.join(OUTPUT_DIR, 'figure2.pdf'))
    plt.close()


def main():
    print("=" * 60)
    print("04_figure2.py: Event Study (Stacked DiD)")
    print("=" * 60)

    df = pd.read_parquet(CLEAN_DATA_PATH)
    print(f"Data loaded: {len(df)} observations")

    coef_df, att, att_se, att_pval, results = run_stacked_event_study(df)

    print("\nEvent Study Coefficients:")
    print(f"{'Event Time':>12s} {'Estimate':>10s} {'SE':>10s} {'CI Low':>10s} {'CI High':>10s}")
    print("-" * 55)
    for _, row in coef_df.iterrows():
        print(f"{int(row['event_time']):12d} {row['estimate']:10.3f} {row['se']:10.3f} "
              f"{row['ci_lo']:10.3f} {row['ci_hi']:10.3f}")

    print(f"\nATT (avg post-treatment): {att:.3f} (SE={att_se:.3f}), p={att_pval:.3f}")
    print(f"Published ATT from paper: ~1.41 (from synthetic DiD)")

    plot_event_study(coef_df, att, att_se, att_pval)
    print("\nFigure 2 saved")

    coef_df.to_csv(os.path.join(OUTPUT_DIR, 'figure2_coefficients.csv'), index=False)


if __name__ == '__main__':
    main()
