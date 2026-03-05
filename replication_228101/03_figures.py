"""
03_figures.py — Figure 1a and Figure 1b

Figure 1a: Network Size by Gender (Bar Chart)
  - For each network variable, regress on female with robust SE
  - Male mean = intercept; Female mean = intercept + coefficient
  - 95% CI for female based on ±1.96*SE of female coefficient

Figure 1b: Network Composition Gender Differences (Coefficient Plot)
  - For female share and friends/relatives share variables
  - Plot regression coefficients of female dummy with 95% CIs
"""

import sys
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')

sys.path.insert(0, os.path.dirname(__file__))
from utils import (run_ols, FIGURE1A_VARS, FIGURE1A_LABELS,
                   FEMALE_SHARE_VARS, FRIENDREL_SHARE_VARS,
                   FIGURE1B_LABELS, NETWORK_SIZE_VARS, CLEAN_DATA_PATH)


def compute_gender_means(df, var):
    """
    Regress var on female with robust SE.
    Returns male_mean, female_mean, female_ci_lo, female_ci_hi, coef, se.
    """
    res = run_ols(df, var, ['female'])
    male_mean = res.params['const']
    coef = res.params['female']
    se = res.bse['female']
    female_mean = male_mean + coef
    ci_lo = female_mean - 1.96 * se
    ci_hi = female_mean + 1.96 * se
    return {
        'male_mean': male_mean,
        'female_mean': female_mean,
        'ci_lo': ci_lo,
        'ci_hi': ci_hi,
        'coef': coef,
        'se': se,
        'pval': res.pvalues['female'],
        'n': int(res.nobs),
    }


def figure1a(df):
    """
    Figure 1a: Network size bar chart.
    Stata code plots indices 1,3,5,6,7 of the 7 network variables:
    business_help_n, meet_other_business_n, total_collab, suppliers_n, clients_n
    """
    results = []
    for var in FIGURE1A_VARS:
        r = compute_gender_means(df, var)
        r['var'] = var
        results.append(r)

    # Create bar chart
    fig, ax = plt.subplots(figsize=(10, 6))

    x = np.arange(len(results))
    width = 0.35

    male_means = [r['male_mean'] for r in results]
    female_means = [r['female_mean'] for r in results]
    female_ci_lo = [r['ci_lo'] for r in results]
    female_ci_hi = [r['ci_hi'] for r in results]
    female_err = [[f - lo for f, lo in zip(female_means, female_ci_lo)],
                  [hi - f for f, hi in zip(female_means, female_ci_hi)]]

    bars_male = ax.bar(x - width/2, male_means, width, label='Male',
                       color='navy', alpha=0.75)
    bars_female = ax.bar(x + width/2, female_means, width, label='Female',
                         color='maroon', alpha=0.75,
                         yerr=female_err, capsize=4,
                         error_kw={'color': 'gray', 'capthick': 1.5})

    # Add value labels on bars
    for bar in bars_male:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.2f}', ha='center', va='bottom', fontsize=8)
    for bar in bars_female:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.05,
                f'{height:.2f}', ha='center', va='bottom', fontsize=8)

    ax.set_xticks(x)
    ax.set_xticklabels(FIGURE1A_LABELS, fontsize=9)
    ax.set_ylabel('')
    ax.legend(loc='upper right')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(axis='y', linestyle='--', alpha=0.5)

    plt.tight_layout()
    plt.savefig('replication/output/Figure1a.png', dpi=150, bbox_inches='tight')
    plt.savefig('replication/output/Figure1a.pdf', bbox_inches='tight')
    plt.close()

    print("\nFigure 1a: Network Size by Gender")
    print("-" * 70)
    print(f"{'Variable':35s} {'Male':>8s} {'Female':>8s} {'Diff':>8s} {'SE':>8s} {'p':>8s}")
    print("-" * 70)
    for r, label in zip(results, FIGURE1A_LABELS):
        label_clean = label.replace('\n', ' ')
        print(f"{label_clean:35s} {r['male_mean']:8.3f} {r['female_mean']:8.3f} "
              f"{r['coef']:8.3f} {r['se']:8.3f} {r['pval']:8.3f}")

    return results


def figure1b(df):
    """
    Figure 1b: Network composition coefficient plot.
    Plots coefficients of female dummy for share variables.
    """
    # Female share regressions
    fem_results = []
    for var in FEMALE_SHARE_VARS:
        r = compute_gender_means(df, var)
        r['var'] = var
        fem_results.append(r)

    # Friends/relatives share regressions
    fr_results = []
    for var in FRIENDREL_SHARE_VARS:
        r = compute_gender_means(df, var)
        r['var'] = var
        fr_results.append(r)

    # Create coefficient plot
    fig, ax = plt.subplots(figsize=(10, 6))

    y = np.arange(len(FIGURE1B_LABELS))
    offset = 0.15

    # Female share coefficients
    fem_coefs = [r['coef'] for r in fem_results]
    fem_ci = [[r['coef'] - (r['coef'] - 1.96 * r['se']) for r in fem_results],
              [(r['coef'] + 1.96 * r['se']) - r['coef'] for r in fem_results]]

    ax.errorbar(fem_coefs, y + offset, xerr=fem_ci, fmt='o',
                color='navy', capsize=4, capthick=1.5, label='Female Share',
                markersize=6)

    # Friends/relatives share coefficients
    fr_coefs = [r['coef'] for r in fr_results]
    fr_ci = [[r['coef'] - (r['coef'] - 1.96 * r['se']) for r in fr_results],
             [(r['coef'] + 1.96 * r['se']) - r['coef'] for r in fr_results]]

    ax.errorbar(fr_coefs, y - offset, xerr=fr_ci, fmt='o',
                color='navy', alpha=0.5, capsize=4, capthick=1.5,
                label='Friends or Relatives Share', markersize=6)

    ax.axvline(x=0, color='gray', linewidth=0.8)
    ax.set_yticks(y)
    ax.set_yticklabels(FIGURE1B_LABELS, fontsize=10)
    ax.set_xlabel('')
    ax.legend(loc='lower right')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(axis='x', linestyle='--', alpha=0.5)
    ax.set_xlim(-0.25, 0.65)

    plt.tight_layout()
    plt.savefig('replication/output/Figure1b.png', dpi=150, bbox_inches='tight')
    plt.savefig('replication/output/Figure1b.pdf', bbox_inches='tight')
    plt.close()

    print("\nFigure 1b: Network Composition Gender Differences (Female coefficient)")
    print("-" * 70)
    print(f"{'Variable':35s} {'Coef':>8s} {'SE':>8s} {'p':>8s} {'N':>6s}")
    print("-" * 70)
    print("Female Share:")
    for r, label in zip(fem_results, FIGURE1B_LABELS):
        label_clean = label.replace('\n', ' ')
        print(f"  {label_clean:33s} {r['coef']:8.3f} {r['se']:8.3f} {r['pval']:8.3f} {r['n']:6d}")
    print("Friends/Relatives Share:")
    for r, label in zip(fr_results, FIGURE1B_LABELS):
        label_clean = label.replace('\n', ' ')
        print(f"  {label_clean:33s} {r['coef']:8.3f} {r['se']:8.3f} {r['pval']:8.3f} {r['n']:6d}")

    return fem_results, fr_results


def main():
    print("=" * 60)
    print("03_figures.py: Figures 1a and 1b")
    print("=" * 60)

    df = pd.read_parquet(CLEAN_DATA_PATH)
    print(f"Data loaded: {len(df)} observations")

    # Also compute all 7 network variables for completeness
    print("\nAll 7 Network Size Variables (regression of var on female, robust SE):")
    print("-" * 70)
    print(f"{'Variable':35s} {'Male':>8s} {'Female':>8s} {'Diff':>8s} {'SE':>8s} {'p':>8s}")
    print("-" * 70)
    for var in NETWORK_SIZE_VARS:
        r = compute_gender_means(df, var)
        print(f"{var:35s} {r['male_mean']:8.3f} {r['female_mean']:8.3f} "
              f"{r['coef']:8.3f} {r['se']:8.3f} {r['pval']:8.3f}")

    fig1a_results = figure1a(df)
    fig1b_fem, fig1b_fr = figure1b(df)

    print("\nFigures saved to replication/output/")
    return fig1a_results, fig1b_fem, fig1b_fr


if __name__ == '__main__':
    main()
