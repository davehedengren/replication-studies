"""
03_tables.py — Tables 1, 2, and A1

Table 1: Effects of law enactment/enforcement on women's labor outcomes
Table 2: Effects of law dimensions (availability, affordability, quality)
Table A1: Summary statistics

NOTE: The original paper uses synthetic DiD (ssynthdid R package). Since no
Python equivalent exists, we use two-way fixed effects (TWFE) DiD as the
closest Python-native estimator. We document the comparison to published
synthetic DiD estimates. The TWFE results serve as a useful reference point
but may differ from the published synthetic DiD results.
"""

import sys, os
import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats

sys.path.insert(0, os.path.dirname(__file__))
from utils import (CLEAN_DATA_PATH, OUTPUT_DIR, OUTCOME_VARS, OUTCOME_LABELS,
                   TREATMENT_VARS, TREATMENT_LABELS, prepare_estimation_sample)

# Published synthetic DiD estimates from Table 1 (for comparison)
PUBLISHED_TABLE1 = {
    ('enactment', 'lfpr_f_25_54'): (1.41, 0.060),
    ('enactment', 'lfpr_hhchild_f'): (1.584, 0.018),
    ('enactment', 'emp2pop_f_yge25'): (1.071, 0.111),
    ('enactment', 'unemp_f_yge25'): (-0.35, 0.408),
    ('enforcement', 'lfpr_f_25_54'): (1.384, 0.064),
    ('enforcement', 'lfpr_hhchild_f'): (1.333, 0.041),
    ('enforcement', 'emp2pop_f_yge25'): (0.989, 0.145),
    ('enforcement', 'unemp_f_yge25'): (-0.314, 0.462),
}

PUBLISHED_TABLE2 = {
    ('enactment_av', 'lfpr_f_25_54'): (1.410, 0.060),
    ('enactment_af', 'lfpr_f_25_54'): (2.690, 0.001),
    ('enactment_qua', 'lfpr_f_25_54'): (1.848, 0.038),
    ('enactment_av', 'lfpr_hhchild_f'): (1.584, 0.018),
    ('enactment_af', 'lfpr_hhchild_f'): (2.245, 0.007),
    ('enactment_qua', 'lfpr_hhchild_f'): (0.579, 0.519),
    ('enactment_av', 'emp2pop_f_yge25'): (1.071, 0.111),
    ('enactment_af', 'emp2pop_f_yge25'): (1.541, 0.008),
    ('enactment_qua', 'emp2pop_f_yge25'): (1.614, 0.021),
    ('enactment_av', 'unemp_f_yge25'): (-0.350, 0.408),
    ('enactment_af', 'unemp_f_yge25'): (-0.214, 0.722),
    ('enactment_qua', 'unemp_f_yge25'): (-0.513, 0.388),
}


def run_twfe_did(df, y_var, treat_time):
    """
    Run TWFE DiD regression with country and year fixed effects.
    y = alpha_c + beta_t + tau * treat + epsilon
    Returns estimate, SE, p-value, N, control mean.
    """
    foo = prepare_estimation_sample(df, y_var, treat_time).reset_index(drop=True)

    if len(foo) == 0 or foo['treat'].sum() == 0:
        return None

    # Create fixed effects via dummies (demeaning approach)
    y = foo[y_var].values
    treat = foo['treat'].values.astype(float)

    # Country dummies
    country_dummies = pd.get_dummies(foo['id'], prefix='c', drop_first=True, dtype=float)
    # Year dummies
    year_dummies = pd.get_dummies(foo['year'], prefix='y', drop_first=True, dtype=float)

    X = pd.concat([
        pd.DataFrame({'treat': treat}),
        country_dummies,
        year_dummies
    ], axis=1)

    # OLS with clustered standard errors by country
    model = sm.OLS(y, sm.add_constant(X))

    # Cluster by country
    groups = foo['id'].values
    try:
        results = model.fit(cov_type='cluster', cov_kwds={'groups': groups})
    except Exception:
        results = model.fit(cov_type='HC1')

    estimate = results.params['treat']
    se = results.bse['treat']
    pval = results.pvalues['treat']

    # Control means (never-treated countries)
    control_mean = foo[foo[treat_time].isna()][y_var].mean()

    return {
        'estimate': estimate,
        'se': se,
        'pval': pval,
        'n': len(foo),
        'control_mean': control_mean,
    }


def run_table(df, treatments, published_ref, table_name):
    """Run regressions for a set of treatments and display results."""
    print(f"\n{'='*120}")
    print(f"  {table_name}")
    print(f"{'='*120}")

    header = f"{'Treatment':20s}"
    for y_var in OUTCOME_VARS:
        header += f" {OUTCOME_LABELS[y_var]:>20s}"
    print(header)
    print("-" * 120)

    all_results = {}
    for treat in treatments:
        row_est = f"{TREATMENT_LABELS[treat]:20s}"
        row_pval = f"{'':20s}"
        row_pub = f"{'  (published)':20s}"

        for y_var in OUTCOME_VARS:
            res = run_twfe_did(df, y_var, treat)
            if res:
                stars = '***' if res['pval'] < 0.01 else '**' if res['pval'] < 0.05 else '*' if res['pval'] < 0.1 else ''
                row_est += f" {res['estimate']:17.3f}{stars:3s}"
                pval_str = f"[{res['pval']:.3f}]"
                row_pval += f" {pval_str:>20s}"
                all_results[(treat, y_var)] = res

                # Published comparison
                pub = published_ref.get((treat, y_var))
                if pub:
                    row_pub += f" {pub[0]:17.3f}   "
                else:
                    row_pub += f" {'':>20s}"
            else:
                row_est += f" {'N/A':>20s}"
                row_pval += f" {'':>20s}"
                row_pub += f" {'':>20s}"

        print(row_est)
        print(row_pval)
        print(row_pub)
        print()

    # Observations and control means
    print("-" * 120)
    row_n = f"{'Observations':20s}"
    row_mean = f"{'Never-treated mean':20s}"
    for y_var in OUTCOME_VARS:
        for treat in treatments:
            res = all_results.get((treat, y_var))
            if res:
                row_n = f"{'Observations':20s}"
                break
    # Just print for first treatment per outcome
    for treat in treatments[:1]:
        n_row = f"{'Observations':20s}"
        m_row = f"{'Never-treated mean':20s}"
        for y_var in OUTCOME_VARS:
            res = all_results.get((treat, y_var))
            if res:
                n_row += f" {res['n']:>20,d}"
                m_row += f" {res['control_mean']:20.2f}"
            else:
                n_row += f" {'':>20s}"
                m_row += f" {'':>20s}"
        print(n_row)
        print(m_row)

    return all_results


def table_a1(df):
    """Summary statistics (Table A1)."""
    print(f"\n{'='*100}")
    print(f"  TABLE A1: Summary Statistics")
    print(f"{'='*100}")

    # Use the estimation sample for LFPR 25-54 enactment
    foo = prepare_estimation_sample(df, 'lfpr_f_25_54', 'enactment')

    vars_to_summarize = [
        ('has_childcare_law', 'Has a childcare law'),
        ('availability', 'Law regulates availability'),
        ('affordability', 'Law regulates affordability'),
        ('quality', 'Law regulates quality'),
        ('lfpr_f_25_54', 'FLFP 25-54'),
        ('lfpr_hhchild_f', 'FLFP 25-54 w. children <6'),
        ('emp2pop_f_yge25', '% of women working (25+)'),
        ('unemp_f_yge25', 'Female unemployment rate (25+)'),
    ]

    print(f"{'Variable':45s} {'Obs':>8s} {'Mean':>10s} {'SD':>10s} "
          f"{'Median':>10s} {'Min':>10s} {'Max':>10s}")
    print("-" * 100)

    rows = []
    for var, label in vars_to_summarize:
        if var not in foo.columns:
            continue
        col = foo[var].dropna()
        row = {
            'variable': label,
            'obs': len(col),
            'mean': col.mean(),
            'sd': col.std(),
            'median': col.median(),
            'min': col.min(),
            'max': col.max(),
        }
        rows.append(row)
        print(f"{label:45s} {len(col):8d} {col.mean():10.2f} {col.std():10.2f} "
              f"{col.median():10.2f} {col.min():10.2f} {col.max():10.2f}")

    return pd.DataFrame(rows)


def main():
    print("=" * 60)
    print("03_tables.py: Tables 1, 2, and A1")
    print("=" * 60)

    df = pd.read_parquet(CLEAN_DATA_PATH)
    print(f"Data loaded: {len(df)} observations, {df['id'].nunique()} countries")

    print("\nNOTE: We use TWFE DiD (not synthetic DiD as in the published paper).")
    print("Results may differ from published estimates due to methodological differences.")
    print("Published synthetic DiD estimates shown in parentheses for comparison.\n")

    # Table 1
    t1_results = run_table(
        df, ['enactment', 'enforcement'], PUBLISHED_TABLE1,
        "TABLE 1: Childcare Laws and Women's Labor Market Outcomes (TWFE DiD)"
    )

    # Table 2
    t2_results = run_table(
        df, ['enactment_av', 'enactment_af', 'enactment_qua'], PUBLISHED_TABLE2,
        "TABLE 2: Childcare Law Attributes (TWFE DiD)"
    )

    # Table A1
    stats_df = table_a1(df)

    # Save
    with open(os.path.join(OUTPUT_DIR, 'tables.txt'), 'w') as f:
        f.write("See console output for formatted tables\n")
        f.write("TWFE DiD results - comparison with published synthetic DiD\n")
    stats_df.to_csv(os.path.join(OUTPUT_DIR, 'table_a1.csv'), index=False)
    print("\nSaved: tables.txt, table_a1.csv")

    return t1_results, t2_results


if __name__ == '__main__':
    main()
