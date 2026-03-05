"""
03_figures.py — Figures 2, 3, and 4 from Dizon-Ross & Jayachandran (2023)

Figure 2: Bar chart of predicted WTP by parent × child gender (all goods)
Figure 3: Same split by human capital vs enjoyment goods
Figure 4: Same split by whether household says mother loves children more
"""

import sys, os
import numpy as np
import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(__file__))
from utils import ANALYSIS_DATA_PATH, OUTPUT_DIR, _demean_by_group


def run_bar_regression(df, condition):
    """Run the Col 2 regression on a subset, return predicted bar values."""
    data = df[condition].copy()
    good_fe_cols = [c for c in data.columns if c.startswith('_G')]
    x_vars = ['girl', 'momXgirl', 'mom', 'adult_wtp_std', 'adult_wtp_stdXfu',
              'fu', 'young'] + good_fe_cols
    y_var = 'wtp_std'
    absorb_var = 'strat'

    x_vars = [v for v in x_vars if v in data.columns]
    data = data.dropna(subset=[y_var] + x_vars + [absorb_var, 'hhid']).copy()
    data = data.reset_index(drop=True)

    if len(data) == 0:
        return None

    groups_fe = data[absorb_var].values
    n_groups = data[absorb_var].nunique()

    y_dm = _demean_by_group(data[y_var].values.astype(float), groups_fe)
    X_dm = pd.DataFrame(index=data.index)
    for v in x_vars:
        X_dm[v] = _demean_by_group(data[v].values.astype(float), groups_fe)

    model = sm.OLS(y_dm, X_dm)
    try:
        results = model.fit(cov_type='cluster',
                           cov_kwds={'groups': data['hhid'].values, 'df_correction': True})
    except Exception:
        results = model.fit(cov_type='HC1')

    # Father-son mean
    fs_mask = (data['mom'] == 0) & (data['girl'] == 0)
    b1 = data.loc[fs_mask, y_var].mean()

    girl_coef = results.params.get('girl', 0)
    mom_coef = results.params.get('mom', 0)
    momXgirl_coef = results.params.get('momXgirl', 0)

    b2 = b1 + girl_coef                         # Father-Daughter
    b3 = b1 + mom_coef                          # Mother-Son
    b4 = b3 + momXgirl_coef + girl_coef         # Mother-Daughter

    # P-values for annotations
    girl_pval = results.pvalues.get('girl', 1.0)
    # Test girl + momXgirl = 0 for mothers' daughter gap
    try:
        t = results.t_test('girl + momXgirl = 0')
        mother_daughter_pval = float(t.pvalue)
    except Exception:
        mother_daughter_pval = 1.0

    # momXgirl p-value (test of gender gap difference)
    momXgirl_pval = results.pvalues.get('momXgirl', 1.0)

    return {
        'father_son': b1, 'father_daughter': b2,
        'mother_son': b3, 'mother_daughter': b4,
        'girl_coef': girl_coef, 'girl_pval': girl_pval,
        'momXgirl_coef': momXgirl_coef, 'momXgirl_pval': momXgirl_pval,
        'mother_girl_coef': girl_coef + momXgirl_coef,
        'mother_girl_pval': mother_daughter_pval,
        'N': len(data),
    }


def stars(pval):
    if pval < 0.01: return '***'
    if pval < 0.05: return '**'
    if pval < 0.1: return '*'
    return ''


def make_bar_chart(vals, title, filename, ylim=None):
    """Create a 2-panel bar chart (Fathers / Mothers) with Sons vs Daughters."""
    fig, axes = plt.subplots(1, 2, figsize=(8, 5), sharey=True)

    colors_father = ['#4878CF', '#4878CF80']  # blue, lighter blue
    colors_mother = ['#4C004C80', '#270027']   # purple shades

    for ax, parent, bars, colors, coef, pval in [
        (axes[0], 'Fathers',
         [vals['father_son'], vals['father_daughter']],
         colors_father, vals['girl_coef'], vals['girl_pval']),
        (axes[1], 'Mothers',
         [vals['mother_son'], vals['mother_daughter']],
         colors_mother, vals['mother_girl_coef'], vals['mother_girl_pval']),
    ]:
        x = [0, 1]
        ax.bar(x, bars, color=colors, width=0.6, edgecolor='none')

        # Value labels
        for i, v in enumerate(bars):
            ax.text(i, v - 0.05, f'{v:.2f}', ha='center', va='top', fontsize=12, fontweight='bold')

        # Bracket with coefficient
        ymax = max(bars) + 0.15
        ax.plot([0, 0, 1, 1], [ymax-0.05, ymax, ymax, ymax-0.05], 'k-', lw=1.5)
        ax.text(0.5, ymax + 0.02, f'{coef:.2f}{stars(pval)}', ha='center', va='bottom', fontsize=13)

        ax.set_xticks([0, 1])
        ax.set_xticklabels(['Sons', 'Daughters'], fontsize=12)
        ax.set_title(parent, fontsize=14, fontweight='bold')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

    axes[0].set_ylabel('WTP (SD)', fontsize=12)

    if ylim:
        axes[0].set_ylim(ylim)
        axes[1].set_ylim(ylim)

    # Add momXgirl bracket across panels
    fig.suptitle(title, fontsize=14, y=1.02)
    fig.tight_layout()

    outpath = os.path.join(OUTPUT_DIR, filename)
    fig.savefig(outpath, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {outpath}")


def main():
    print("=" * 60)
    print("03_figures.py: Figures 2, 3, 4")
    print("=" * 60)

    df = pd.read_parquet(ANALYSIS_DATA_PATH)
    base = (df['childgood'] == 1) & (df['toys_bin'] == 0)

    # --- Figure 2: All child goods ---
    print("\nFigure 2: All child goods")
    vals = run_bar_regression(df, base)
    if vals:
        print(f"  Father-Son: {vals['father_son']:.3f}")
        print(f"  Father-Daughter: {vals['father_daughter']:.3f}")
        print(f"  Mother-Son: {vals['mother_son']:.3f}")
        print(f"  Mother-Daughter: {vals['mother_daughter']:.3f}")
        print(f"  girl coef: {vals['girl_coef']:.3f}{stars(vals['girl_pval'])}")
        print(f"  momXgirl coef: {vals['momXgirl_coef']:.3f}{stars(vals['momXgirl_pval'])}")
        print(f"  N: {vals['N']}")
        make_bar_chart(vals, 'Figure 2: WTP by Parent and Child Gender',
                       'figure2.png', ylim=(1.4, 2.7))

    # --- Figure 3a: Human capital goods ---
    print("\nFigure 3a: Human capital goods")
    vals_hc = run_bar_regression(df, base & (df['humcap'] == 1))
    if vals_hc:
        print(f"  Father-Son: {vals_hc['father_son']:.3f}")
        print(f"  Father-Daughter: {vals_hc['father_daughter']:.3f}")
        print(f"  Mother-Son: {vals_hc['mother_son']:.3f}")
        print(f"  Mother-Daughter: {vals_hc['mother_daughter']:.3f}")
        print(f"  girl coef: {vals_hc['girl_coef']:.3f}{stars(vals_hc['girl_pval'])}")
        print(f"  momXgirl coef: {vals_hc['momXgirl_coef']:.3f}{stars(vals_hc['momXgirl_pval'])}")
        print(f"  N: {vals_hc['N']}")
        make_bar_chart(vals_hc, 'Figure 3a: Human Capital Goods',
                       'figure3a.png', ylim=(1.4, 2.7))

    # --- Figure 3b: Non-human capital (enjoyment) goods ---
    print("\nFigure 3b: Enjoyment goods")
    vals_nhc = run_bar_regression(df, base & (df['humcap'] == 0))
    if vals_nhc:
        print(f"  Father-Son: {vals_nhc['father_son']:.3f}")
        print(f"  Father-Daughter: {vals_nhc['father_daughter']:.3f}")
        print(f"  Mother-Son: {vals_nhc['mother_son']:.3f}")
        print(f"  Mother-Daughter: {vals_nhc['mother_daughter']:.3f}")
        print(f"  girl coef: {vals_nhc['girl_coef']:.3f}{stars(vals_nhc['girl_pval'])}")
        print(f"  momXgirl coef: {vals_nhc['momXgirl_coef']:.3f}{stars(vals_nhc['momXgirl_pval'])}")
        print(f"  N: {vals_nhc['N']}")
        make_bar_chart(vals_nhc, 'Figure 3b: Enjoyment Goods',
                       'figure3b.png', ylim=(1.1, 2.3))

    # --- Figure 4a: HC goods, lovemore_mom_HHmed == 0 ---
    print("\nFigure 4a: HC goods, does NOT say mom loves more")
    if 'lovemore_mom_HHmed' in df.columns:
        vals_4a = run_bar_regression(df, base & (df['humcap'] == 1) & (df['lovemore_mom_HHmed'] == 0))
        if vals_4a:
            print(f"  Father-Son: {vals_4a['father_son']:.3f}")
            print(f"  Father-Daughter: {vals_4a['father_daughter']:.3f}")
            print(f"  girl coef: {vals_4a['girl_coef']:.3f}{stars(vals_4a['girl_pval'])}")
            print(f"  momXgirl coef: {vals_4a['momXgirl_coef']:.3f}{stars(vals_4a['momXgirl_pval'])}")
            print(f"  N: {vals_4a['N']}")
            make_bar_chart(vals_4a, "Figure 4a: HC Goods — HH Does Not Say Mom Loves More",
                           'figure4a.png', ylim=(1.4, 2.7))

    # --- Figure 4b: HC goods, lovemore_mom_HHmed == 1 ---
    print("\nFigure 4b: HC goods, says mom loves more")
    if 'lovemore_mom_HHmed' in df.columns:
        vals_4b = run_bar_regression(df, base & (df['humcap'] == 1) & (df['lovemore_mom_HHmed'] == 1))
        if vals_4b:
            print(f"  Father-Son: {vals_4b['father_son']:.3f}")
            print(f"  Father-Daughter: {vals_4b['father_daughter']:.3f}")
            print(f"  girl coef: {vals_4b['girl_coef']:.3f}{stars(vals_4b['girl_pval'])}")
            print(f"  momXgirl coef: {vals_4b['momXgirl_coef']:.3f}{stars(vals_4b['momXgirl_pval'])}")
            print(f"  N: {vals_4b['N']}")
            make_bar_chart(vals_4b, "Figure 4b: HC Goods — HH Says Mom Loves More",
                           'figure4b.png', ylim=(1.4, 2.7))


if __name__ == '__main__':
    main()
