"""
02_tables.py: Reproduce Tables 1-4 for replication of
Bryan, Karlan & Osman (AER, 2024) "Big Loans to Small Businesses"
"""

import os, sys
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
from utils import (MERGED_PATH, PENALTY_PATH, ML_GROUP_PATH, CLEAN_DATA_DIR,
                   OUTPUT_DIR, run_areg, winsor, stars)


def load_data():
    """Load merged analysis data and penalty data."""
    merged = pd.read_parquet(MERGED_PATH)
    penalty = pd.read_parquet(PENALTY_PATH)
    return merged, penalty


def load_stata_data():
    """Load Stata-cleaned data for comparison."""
    stata = pd.read_stata(os.path.join(CLEAN_DATA_DIR, '16-ABA-All-Merged.dta'),
                          convert_categoricals=False)
    stata_pen = pd.read_stata(os.path.join(CLEAN_DATA_DIR, '17-ABA-penalty-clean.dta'),
                              convert_categoricals=False)
    return stata, stata_pen


# ================================================================
# TABLE 1: Baseline Balance
# ================================================================
def table1(df):
    """Reproduce Table 1: Baseline balance test."""
    print("\n" + "=" * 70)
    print("TABLE 1: Baseline Balance Test")
    print("=" * 70)

    d = df[df['unique'] == 1].copy()

    dep_vars = [
        ('b_age', 'Age'),
        ('b_gender', 'Female'),
        ('years_of_ed', 'Years of education'),
        ('registration', 'Registered business'),
        ('b_experience', 'Years of experience'),
        ('b_monthly_profits_w', 'Monthly profits (w)'),
        ('b_monthly_revenue_w', 'Monthly revenue (w)'),
        ('b_monthly_expenses_w', 'Monthly expenses (w)'),
        ('b_wagebill_all_w', 'Wage bill (w)'),
        ('b_staff_yesno', 'Has employees'),
        ('b_num_employ_full', 'Number employees'),
        ('b_size_lastloan', 'Size of last loan'),
        ('b_has_other_loan', 'Has other loan'),
        ('b_othercredit', 'Other credit'),
    ]

    rows = []
    for yvar, label in dep_vars:
        if yvar not in d.columns:
            print(f"  WARNING: {yvar} not in data")
            continue

        # Control mean and SD
        ctrl = d.loc[d['treatment'] == 0, yvar].dropna()
        ctrl_mean = ctrl.mean()
        ctrl_sd = ctrl.std()

        # areg Y treatment, a(loid_baseline) ro
        res = run_areg(d, yvar, ['treatment'], 'loid_baseline')
        if res is None:
            continue
        results, data = res

        coef = results.params['treatment']
        se = results.bse['treatment']
        pval = results.pvalues['treatment']
        n = len(data)

        rows.append({
            'Variable': label,
            'Control Mean': ctrl_mean,
            'Control SD': ctrl_sd,
            'Treatment Effect': coef,
            'SE': se,
            'p-value': pval,
            'Stars': stars(pval),
            'N': n,
        })

    tab = pd.DataFrame(rows)
    print(tab.to_string(index=False, float_format='{:.3f}'.format))
    return tab


# ================================================================
# TABLE 2: Loan Takeup and Characteristics
# ================================================================
def table2(df):
    """Reproduce Table 2: Treatment effects on loan takeup and characteristics."""
    print("\n" + "=" * 70)
    print("TABLE 2: Loan Takeup and Loan Characteristics")
    print("=" * 70)

    dep_vars = [
        ('took_loan', 'Took loan', None),
        ('ad_exploan_size', 'Loan size', None),
        ('ad_exploan_termm', 'Loan term (months)', 'took_loan==1'),
        ('ad_exploan_instalsiz', 'Installment size', None),
        ('outstanding20', 'Outstanding at 20m', None),
        ('outstanding30', 'Outstanding at 30m', None),
        ('ad_post1_3months', 'Another loan 1-3m', None),
    ]

    d = df[df['unique'] == 1].copy()

    rows = []
    for yvar, label, restriction in dep_vars:
        if yvar not in d.columns:
            print(f"  WARNING: {yvar} not in data")
            continue

        sample = d.copy()
        if restriction == 'took_loan==1':
            sample = sample[sample['took_loan'] == 1]

        ctrl = sample.loc[sample['treatment'] == 0, yvar].dropna()
        ctrl_mean = ctrl.mean()
        ctrl_sd = ctrl.std()

        res = run_areg(sample, yvar, ['treatment'], 'loid_baseline')
        if res is None:
            continue
        results, data = res

        coef = results.params['treatment']
        se = results.bse['treatment']
        pval = results.pvalues['treatment']
        n = len(data)

        rows.append({
            'Variable': label,
            'Control Mean': ctrl_mean,
            'Control SD': ctrl_sd,
            'Treatment Effect': coef,
            'SE': se,
            'p-value': pval,
            'Stars': stars(pval),
            'N': n,
        })

    tab = pd.DataFrame(rows)
    print(tab.to_string(index=False, float_format='{:.3f}'.format))
    return tab


# ================================================================
# TABLE 3: Repayment Outcomes
# ================================================================
def table3(df, penalty):
    """Reproduce Table 3: Treatment effects on repayment."""
    print("\n" + "=" * 70)
    print("TABLE 3: Loan Repayment Outcomes")
    print("=" * 70)

    d = df[df['unique'] == 1].copy()

    # Merge penalty data
    pen_vars = ['client_key', 'totpenalty24', 'value_late24']
    pen_available = [c for c in pen_vars if c in penalty.columns]
    d = d.merge(penalty[pen_available], on='client_key', how='left', suffixes=('', '_pen'))

    # Handle duplicate columns from merge
    for c in ['totpenalty24', 'value_late24']:
        if f'{c}_pen' in d.columns:
            d[c] = d[c].fillna(d[f'{c}_pen'])

    dep_vars = [
        ('perfect_repayment', 'Perfect repayment'),
        ('ad_exploan_totallate', 'Total days late'),
        ('totpenalty24', 'Total penalty (24m)'),
        ('value_late24', 'Daily value late (24m)'),
    ]

    rows = []
    for yvar, label in dep_vars:
        if yvar not in d.columns:
            print(f"  WARNING: {yvar} not in data")
            continue

        ctrl = d.loc[d['treatment'] == 0, yvar].dropna()
        ctrl_mean = ctrl.mean()
        ctrl_sd = ctrl.std()

        res = run_areg(d, yvar, ['treatment'], 'loid_baseline')
        if res is None:
            continue
        results, data = res

        coef = results.params['treatment']
        se = results.bse['treatment']
        pval = results.pvalues['treatment']
        n = len(data)

        rows.append({
            'Variable': label,
            'Control Mean': ctrl_mean,
            'Control SD': ctrl_sd,
            'Treatment Effect': coef,
            'SE': se,
            'p-value': pval,
            'Stars': stars(pval),
            'N': n,
        })

    tab = pd.DataFrame(rows)
    print(tab.to_string(index=False, float_format='{:.3f}'.format))
    return tab


# ================================================================
# TABLE 4: Business Outcomes (ATE and HTE)
# ================================================================
def table4(df, penalty):
    """Reproduce Table 4: Treatment effects on business and household outcomes."""
    print("\n" + "=" * 70)
    print("TABLE 4: Business and Household Outcomes")
    print("=" * 70)

    # Prepare data
    d = df.copy()

    # Merge penalty data
    pen_vars = ['client_key', 'totpenalty24', 'value_late24']
    pen_available = [c for c in pen_vars if c in penalty.columns]
    d = d.merge(penalty[pen_available], on='client_key', how='left', suffixes=('', '_pen'))
    for c in ['totpenalty24', 'value_late24']:
        if f'{c}_pen' in d.columns:
            d[c] = d[c].fillna(d[f'{c}_pen'])

    # Merge ML group data
    if os.path.exists(ML_GROUP_PATH):
        ml = pd.read_stata(ML_GROUP_PATH, convert_categoricals=False)
        ml_vars = [c for c in ml.columns if c in ['client_key', 'G1', 'G2', 'G3', 'G4',
                                                     'RF_HTE_S', 'RF_HTE_B', 'iw']]
        d = d.merge(ml[ml_vars], on='client_key', how='left')

    # Sample: round != 0, e_birth_year not missing
    d = d[d['round'] != 0].copy()
    d = d[d['e_birth_year'].notna()].copy() if 'e_birth_year' in d.columns else d

    # Winsorize endline outcomes
    for v in ['e_month_profit_all', 'e_month_rev_all', 'e_month_exp_all',
              'e_wagebill_all', 'e_hh_exp', 'e_totvalassets_all']:
        if v in d.columns:
            d[f'{v}_w'] = winsor(d[v], p=0.01)

    # TFP computation (IHS transform of rev, assets, wagebill -> regress -> residual)
    if all(c in d.columns for c in ['e_month_rev_all_w', 'e_totvalassets_all_w', 'e_wagebill_all']):
        for v in ['e_month_rev_all_w', 'e_totvalassets_all_w', 'e_wagebill_all']:
            d[f'ihs_{v}'] = np.arcsinh(d[v])

        import statsmodels.api as sm
        tfp_vars = ['ihs_e_totvalassets_all_w', 'ihs_e_wagebill_all', 'round']
        tfp_data = d[['ihs_e_month_rev_all_w'] + tfp_vars].dropna()
        if len(tfp_data) > 10:
            tfp_model = sm.OLS(tfp_data['ihs_e_month_rev_all_w'],
                               sm.add_constant(tfp_data[tfp_vars])).fit()
            d.loc[tfp_data.index, 'tfp'] = tfp_model.resid
            d['tfp'] = (d['tfp'] - d['tfp'].mean()) / d['tfp'].std()
            d['tfp_w'] = winsor(d['tfp'], p=0.01)

    # Mental health index
    mh_vars_neg = ['e_depressed', 'e_lack_of_concentration', 'e_worried_yesno']
    mh_vars_pos = ['e_on_top_of_things', 'e_peaceful_lastmonth']
    mh_components = []
    for v in mh_vars_neg:
        if v in d.columns:
            std_v = (d[v] - d[v].mean()) / d[v].std()
            d[f'std_{v}'] = -std_v  # negate
            mh_components.append(f'std_{v}')
    for v in mh_vars_pos:
        if v in d.columns:
            std_v = (d[v] - d[v].mean()) / d[v].std()
            d[f'std_{v}'] = std_v
            mh_components.append(f'std_{v}')
    if mh_components:
        d['mh_index'] = d[mh_components].sum(axis=1)
        d['std_mh_index'] = (d['mh_index'] - d['mh_index'].mean()) / d['mh_index'].std()

    # Ensure loid_bas exists
    if 'loid_bas' not in d.columns and 'loid_baseline' in d.columns:
        d['loid_bas'] = d['loid_baseline']

    # ---- Panel A: Profits ----
    print("\n--- Panel A: Monthly Profits ---")
    outcomes_a = [
        ('e_month_profit_all_w', 'Monthly profits (w)',
         ['treatment', 'b_monthly_profits_w', 'mis_b_monthly_profits_w', 'round']),
    ]

    # ---- Panel B: Other Business/HH Outcomes ----
    print("\n--- Panel B: Other Business/HH ---")
    outcomes_b = [
        ('e_month_rev_all_w', 'Monthly revenue (w)',
         ['treatment', 'b_monthly_revenue_w', 'mis_b_monthly_revenue_w', 'round']),
        ('e_month_exp_all_w', 'Monthly expenses (w)',
         ['treatment', 'b_monthly_expenses_w', 'mis_b_monthly_expenses_w', 'round']),
        ('e_wagebill_all_w', 'Wage bill (w)',
         ['treatment', 'b_wagebill_all_w', 'mis_b_wagebill_all_w', 'round']),
        ('tfp_w', 'TFP (w)', ['treatment', 'round']),
        ('e_hh_exp_w', 'HH expenditure (w)',
         ['treatment', 'b_hh_exp_w', 'mis_b_hh_exp_w', 'round']),
    ]

    # ---- Panel C: Status / Assets / Health / Penalties ----
    print("\n--- Panel C: Status / Health / Penalties ---")
    outcomes_c = [
        ('e_hasbusiness', 'Has business', ['treatment', 'round']),
        ('e_nemployees_all', 'N employees',
         ['treatment', 'b_num_employ_full', 'mis_b_num_employ_full', 'round']),
        ('e_totvalassets_all_w', 'Total assets (w)', ['treatment', 'round']),
        ('std_mh_index', 'Mental health index', ['treatment', 'round']),
        ('e_health_description', 'Physical health', ['treatment', 'round']),
        ('totpenalty24', 'Total penalty (24m)', ['treatment', 'round']),
        ('long_term', 'Long-term ABA loan', ['treatment', 'round']),
    ]

    all_outcomes = outcomes_a + outcomes_b + outcomes_c

    rows = []
    for yvar, label, xvars in all_outcomes:
        if yvar not in d.columns:
            print(f"  WARNING: {yvar} not in data")
            continue

        # Ensure control variables exist
        xvars_available = [v for v in xvars if v in d.columns]

        # Control mean
        ctrl = d.loc[d['treatment'] == 0, yvar].dropna()
        ctrl_mean = ctrl.mean()

        # ATE regression: areg Y treatment controls round, a(loid_bas) cl(client_key)
        res = run_areg(d, yvar, xvars_available, 'loid_bas', cluster_var='client_key')
        if res is None:
            print(f"  {label}: regression failed")
            continue
        results, data = res

        coef = results.params.get('treatment', np.nan)
        se = results.bse.get('treatment', np.nan)
        pval = results.pvalues.get('treatment', np.nan)
        n_obs = len(data)

        row = {
            'Variable': label,
            'Control Mean': ctrl_mean,
            'ATE Coef': coef,
            'ATE SE': se,
            'ATE p': pval,
            'ATE Stars': stars(pval),
            'N': n_obs,
        }

        # HTE regressions (G1-G4) if available
        if all(c in d.columns for c in ['G1', 'G2', 'G3', 'G4', 'iw', 'RF_HTE_S', 'RF_HTE_B']):
            hte_xvars = [v for v in xvars_available if v != 'treatment'] + \
                        ['G1', 'G2', 'G3', 'G4', 'RF_HTE_S', 'RF_HTE_B']
            hte_xvars = [v for v in hte_xvars if v in d.columns]

            res_hte = run_areg(d, yvar, hte_xvars, 'loid_bas',
                               cluster_var='client_key', weights='iw')
            if res_hte is not None:
                hte_results, hte_data = res_hte
                for g in ['G1', 'G2', 'G3', 'G4']:
                    if g in hte_results.params:
                        row[f'{g} Coef'] = hte_results.params[g]
                        row[f'{g} SE'] = hte_results.bse[g]
                        row[f'{g} p'] = hte_results.pvalues[g]
                        row[f'{g} Stars'] = stars(hte_results.pvalues[g])

        rows.append(row)

    tab = pd.DataFrame(rows)
    # Print ATE results
    ate_cols = ['Variable', 'Control Mean', 'ATE Coef', 'ATE SE', 'ATE p', 'ATE Stars', 'N']
    print("\nATE Results:")
    print(tab[[c for c in ate_cols if c in tab.columns]].to_string(index=False, float_format='{:.3f}'.format))

    # Print HTE results if available
    hte_cols = ['Variable', 'G1 Coef', 'G1 SE', 'G2 Coef', 'G2 SE',
                'G3 Coef', 'G3 SE', 'G4 Coef', 'G4 SE']
    if 'G1 Coef' in tab.columns:
        print("\nHTE Results:")
        print(tab[[c for c in hte_cols if c in tab.columns]].to_string(index=False, float_format='{:.3f}'.format))

    return tab


# ================================================================
# COMPARISON WITH STATA
# ================================================================
def compare_with_stata():
    """Compare our results with Stata-produced output."""
    print("\n" + "=" * 70)
    print("COMPARISON: Python vs Stata Results")
    print("=" * 70)

    # Load Stata output files if available
    stata_output_dir = '192297-V1/Output'
    if not os.path.exists(stata_output_dir):
        print("  Stata output directory not found. Skipping comparison.")
        return

    # Check for any output files
    output_files = os.listdir(stata_output_dir) if os.path.exists(stata_output_dir) else []
    print(f"  Found {len(output_files)} Stata output files: {output_files[:10]}")


# ================================================================
# MAIN
# ================================================================
def main():
    print("=" * 70)
    print("02_tables.py: Table Reproduction for 192297-V1")
    print("=" * 70)

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    merged, penalty = load_data()
    print(f"Loaded data: merged={merged.shape}, penalty={penalty.shape}")

    # Run tables
    t1 = table1(merged)
    t2 = table2(merged)
    t3 = table3(merged, penalty)
    t4 = table4(merged, penalty)

    compare_with_stata()

    print("\n" + "=" * 70)
    print("All tables completed.")
    print("=" * 70)


if __name__ == '__main__':
    main()
