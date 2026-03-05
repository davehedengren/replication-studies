"""
Phase 2: Replicate Figures A1 and A2 for 173341.
"""
import sys
sys.path.insert(0, '.')
from replication_173341.utils import *
import matplotlib.pyplot as plt


def figure_a1():
    """Figure A1: Requests for Private Goods and Rainfall Shocks."""
    print("\n" + "=" * 70)
    print("FIGURE A1: Requests for Private Goods and Rainfall Shocks")
    print("=" * 70)

    stacked = load_stacked()

    # Collapse to municipality-year level (matching Stata collapse)
    collapsed = stacked.groupby(['mun_id', 'year2012']).agg({
        'ask_private_stacked': 'mean',
        'rainfall_std_stacked': 'mean',
        'rainfall_std_stacked_og': 'mean',
    }).reset_index()

    yr2012 = collapsed[collapsed['year2012'] == 1]
    yr2013 = collapsed[collapsed['year2012'] == 0]

    fig, ax = plt.subplots(figsize=(9.5, 5.5))
    ax.scatter(yr2012['rainfall_std_stacked_og'], yr2012['ask_private_stacked'],
               s=40, marker='o', color='C0', label='2012 observations', zorder=3)
    ax.scatter(yr2013['rainfall_std_stacked_og'], yr2013['ask_private_stacked'],
               s=40, marker='^', facecolors='white', edgecolors='C1',
               label='2013 observations', zorder=3)

    ax.set_xlabel('Deviation from long-term average rainfall in municipality', fontsize=9)
    ax.set_ylabel('Proportion of respondents who requested a private good', fontsize=9)
    ax.set_xlim(-0.7, 0.7)
    ax.set_xticks([-0.7, -0.35, 0, 0.35, 0.7])
    ax.legend(fontsize=9)
    ax.tick_params(labelsize=9)
    fig.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, 'figure_a1.png'), dpi=150)
    plt.close()

    print(f"  2012 points: {len(yr2012)}, 2013 points: {len(yr2013)}")
    print("  Saved to output/figure_a1.png")


def figure_a2():
    """Figure A2: Frequency of Conversations before 2012 Election Campaign, by Treatment Status."""
    print("\n" + "=" * 70)
    print("FIGURE A2: Frequency of Conversations by Treatment Status")
    print("=" * 70)

    indiv = load_individual()

    # Create never-talked variable
    indiv['talk_pol_never_2012_bef'] = np.where(
        indiv['talk_pol_atleastonce_2012_bef'] == 0, 1.0,
        np.where(indiv['talk_pol_atleastonce_2012_bef'] == 1, 0.0, np.nan)
    )

    depvar_list = [
        'talk_pol_atleastdaily_2012_bef',
        'talk_pol_atleastweekly_2012_bef',
        'talk_pol_atleastmonthly_2012_bef',
        'talk_pol_atleastonce_2012_bef',
        'talk_pol_never_2012_bef'
    ]
    labels = ['At least Daily', 'At least Weekly', 'At least Monthly',
              'At least Occasionally', 'Never']

    results = []
    for v in depvar_list:
        mask = indiv[v].notna() & indiv['treatment'].notna() & indiv['b_clusters'].notna()
        y = indiv.loc[mask, v].astype(float)
        X = indiv.loc[mask, [['treatment']]].astype(float) if False else indiv.loc[mask, ['treatment']].astype(float)
        cl = indiv.loc[mask, 'b_clusters']
        model = sm.OLS(y, sm.add_constant(X)).fit(cov_type='cluster', cov_kwds={'groups': cl})

        # Control group: intercept
        ctrl_mean = model.params['const']
        ctrl_ci_lo = model.conf_int().loc['const', 0]
        ctrl_ci_hi = model.conf_int().loc['const', 1]

        # Treatment group: intercept + treatment
        from scipy import stats as sp_stats
        treat_mean = model.params['const'] + model.params['treatment']
        # Use lincom for CI
        cov = model.cov_params()
        se_lc = np.sqrt(cov.loc['const', 'const'] + cov.loc['treatment', 'treatment'] + 2 * cov.loc['const', 'treatment'])
        t_crit = sp_stats.t.ppf(0.975, model.df_resid)
        treat_ci_lo = treat_mean - t_crit * se_lc
        treat_ci_hi = treat_mean + t_crit * se_lc

        results.append({
            'var': v,
            'ctrl_mean': ctrl_mean, 'ctrl_lo': ctrl_ci_lo, 'ctrl_hi': ctrl_ci_hi,
            'treat_mean': treat_mean, 'treat_lo': treat_ci_lo, 'treat_hi': treat_ci_hi,
        })

    # Plot grouped bar chart
    fig, ax = plt.subplots(figsize=(9.5, 5.5))
    x_positions = [0, 5, 10, 15, 20]
    width = 0.8

    for i, (r, label, xpos) in enumerate(zip(results, labels, x_positions)):
        # Treatment bar
        ax.bar(xpos + 1, r['treat_mean'], width, color='gray', edgecolor='gray', label='Treatment' if i == 0 else '')
        ax.errorbar(xpos + 1, r['treat_mean'], yerr=[[r['treat_mean'] - r['treat_lo']], [r['treat_hi'] - r['treat_mean']]],
                    color='black', capsize=3, fmt='none')
        # Control bar
        ax.bar(xpos, r['ctrl_mean'], width, color='lightgray', edgecolor='lightgray', label='Control' if i == 0 else '')
        ax.errorbar(xpos, r['ctrl_mean'], yerr=[[r['ctrl_mean'] - r['ctrl_lo']], [r['ctrl_hi'] - r['ctrl_mean']]],
                    color='black', capsize=3, fmt='none')

    ax.set_xticks([0.5, 5.5, 10.5, 15.5, 20.5])
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel('Share of Respondents', fontsize=10)
    ax.legend(fontsize=9)
    ax.tick_params(labelsize=9)
    fig.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, 'figure_a2.png'), dpi=150)
    plt.close()

    print("  Results:")
    for r, label in zip(results, labels):
        print(f"    {label:<25} Control: {r['ctrl_mean']:.3f}  Treatment: {r['treat_mean']:.3f}")
    print("  Saved to output/figure_a2.png")


def main():
    figure_a1()
    figure_a2()


if __name__ == '__main__':
    main()
