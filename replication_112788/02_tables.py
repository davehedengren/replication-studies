"""
02_tables.py – Replicate Tables 1-3 and in-text statistics
Banerjee, Duflo & Hornbeck (2014)
"Bundling Health Insurance and Microfinance in India"
"""

import sys, os, warnings
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
warnings.filterwarnings('ignore')
from utils import *


# ══════════════════════════════════════════════════════════════════════════════
# TABLE 1: Baseline Household Characteristics
# ══════════════════════════════════════════════════════════════════════════════

def table_1():
    print_section("TABLE 1: Baseline Household Characteristics")

    bl = load_baseline()
    treat = bl['treatment']

    baseline_vars = [
        ('serious_health_events', 'Serious health events, previous year'),
        ('total_health_expense', 'Total health expenditures, previous year'),
        ('hosp_expense', 'Hospitalization expenditures, previous year'),
        ('hh_expenditure_yearly', 'Consumption of durables and non-durables, previous year'),
        ('healthinsurance_available', 'Formal health insurance, available in village'),
        ('healthinsurance_have', 'Formal health insurance, owned by household'),
    ]

    # Published values from table1.csv
    published = {
        'serious_health_events': {'all': 2.521, 'treat': 2.596, 'control': 2.440, 'diff': 0.141, 'se': 0.098, 'N': 4939},
        'total_health_expense': {'all': 4671.194, 'treat': 4898.737, 'control': 4429.916, 'diff': 422.764, 'se': 445.329, 'N': 5017},
        'hosp_expense': {'all': 602.856, 'treat': 653.482, 'control': 549.175, 'diff': 93.747, 'se': 96.487, 'N': 5017},
        'hh_expenditure_yearly': {'all': 47938.300, 'treat': 47721.570, 'control': 48165.890, 'diff': -774.012, 'se': 2164.823, 'N': 5232},
        'healthinsurance_available': {'all': 0.141, 'treat': 0.135, 'control': 0.148, 'diff': -0.019, 'se': 0.020, 'N': 5236},
        'healthinsurance_have': {'all': 0.004, 'treat': 0.005, 'control': 0.003, 'diff': 0.002, 'se': 0.002, 'N': 5236},
    }

    print(f"{'Variable':<45} {'All':>10} {'Treat':>10} {'Control':>10} {'Diff':>10} {'SE':>10} {'N':>6}")
    print(f"{'Published':<45} {'':>10} {'':>10} {'':>10} {'':>10} {'':>10} {'':>6}")
    print("-" * 101)

    for var, label in baseline_vars:
        data = bl[[var, 'treatment', 'stratify', 'village_id']].dropna()

        # Means and SDs
        all_mean = data[var].mean()
        all_sd = data[var].std()
        treat_mean = data.loc[data['treatment'] == 1, var].mean()
        treat_sd = data.loc[data['treatment'] == 1, var].std()
        ctrl_mean = data.loc[data['treatment'] == 0, var].mean()
        ctrl_sd = data.loc[data['treatment'] == 0, var].std()

        # Regression: areg var treat, absorb(stratify) cluster(village_id)
        result = areg(data, var, ['treatment'], absorb='stratify', cluster='village_id')
        diff = result.params['treatment']
        se = result.bse['treatment']
        N = int(result.nobs)

        pub = published[var]
        print(f"{label:<45} {all_mean:>10.3f} {treat_mean:>10.3f} {ctrl_mean:>10.3f} {diff:>10.3f} {se:>10.3f} {N:>6}")
        print(f"{'  Published':<45} {pub['all']:>10.3f} {pub['treat']:>10.3f} {pub['control']:>10.3f} {pub['diff']:>10.3f} {pub['se']:>10.3f} {pub['N']:>6}")
        print()


# ══════════════════════════════════════════════════════════════════════════════
# TABLE 2: Estimated Impacts of Treatment on SKS Loan Renewal
# ══════════════════════════════════════════════════════════════════════════════

def table_2():
    print_section("TABLE 2: Treatment on Loan Renewal")

    loans = load_loans()
    treat_df = load_treatment()
    asamp = load_analysis_sample()
    endline = load_endline()

    # Merge loans with treatment
    loans = loans.merge(treat_df[['center_id', 'village_id', 'stratify', 'treatment']],
                        on='center_id', how='left')
    # Merge with analysis sample flag
    loans = loans.merge(asamp[['sks_id', 'end_visit_date']].rename(columns={'end_visit_date': 'asamp_end_date'}),
                        on='sks_id', how='left', indicator='merge_analysis')
    loans['in_analysis'] = loans['merge_analysis'] == 'both'
    # Merge with self-reported
    loans = loans.merge(endline[['sks_id', 'sks_presentclient']], on='sks_id', how='left')

    # Published values
    published = [
        {'label': 'All clients, first renewal', 'coef': -0.161, 'se': 0.024, 'ctrl': 0.708, 'N': 14670},
        {'label': 'Analysis sample, first renewal', 'coef': -0.221, 'se': 0.029, 'ctrl': 0.724, 'N': 5366},
        {'label': 'Renewal at endline', 'coef': -0.162, 'se': 0.029, 'ctrl': 0.541, 'N': 5366},
        {'label': 'Self-reported SKS status', 'coef': -0.076, 'se': 0.024, 'ctrl': 0.717, 'N': 5232},
    ]

    print(f"{'Column':<40} {'Coef':>8} {'SE':>8} {'Ctrl Mean':>10} {'N':>7}  |  {'Pub Coef':>8} {'Pub SE':>8} {'Pub N':>7}")
    print("-" * 110)

    # Col 1: All eligible clients, renewal_first
    d1 = loans[loans['eligible'] == 1].dropna(subset=['renewal_first', 'treatment', 'stratify', 'village_id'])
    r1 = areg(d1, 'renewal_first', ['treatment'])
    ctrl1 = d1.loc[d1['treatment'] == 0, 'renewal_first'].mean()
    p = published[0]
    print(f"{'(1) All, first renewal':<40} {r1.params['treatment']:>8.3f} {r1.bse['treatment']:>8.3f} {ctrl1:>10.3f} {int(r1.nobs):>7}  |  {p['coef']:>8.3f} {p['se']:>8.3f} {p['N']:>7}")

    # Col 2: Analysis sample, renewal_first
    d2 = loans[(loans['eligible'] == 1) & (loans['in_analysis'])].dropna(subset=['renewal_first', 'treatment', 'stratify', 'village_id'])
    r2 = areg(d2, 'renewal_first', ['treatment'])
    ctrl2 = d2.loc[d2['treatment'] == 0, 'renewal_first'].mean()
    p = published[1]
    print(f"{'(2) Analysis sample, first renewal':<40} {r2.params['treatment']:>8.3f} {r2.bse['treatment']:>8.3f} {ctrl2:>10.3f} {int(r2.nobs):>7}  |  {p['coef']:>8.3f} {p['se']:>8.3f} {p['N']:>7}")

    # Col 3: Renewal at endline (analysis sample, all eligible)
    # The do-file uses all eligible clients for renewal_endline
    d3 = loans[loans['eligible'] == 1].dropna(subset=['renewal_endline', 'treatment', 'stratify', 'village_id'])
    r3 = areg(d3, 'renewal_endline', ['treatment'])
    ctrl3 = d3.loc[d3['treatment'] == 0, 'renewal_endline'].mean()
    p = published[2]
    print(f"{'(3) Renewal at endline':<40} {r3.params['treatment']:>8.3f} {r3.bse['treatment']:>8.3f} {ctrl3:>10.3f} {int(r3.nobs):>7}  |  {p['coef']:>8.3f} {p['se']:>8.3f} {p['N']:>7}")

    # Col 4: Self-reported SKS status
    d4 = loans[loans['eligible'] == 1].dropna(subset=['sks_presentclient', 'treatment', 'stratify', 'village_id'])
    r4 = areg(d4, 'sks_presentclient', ['treatment'])
    ctrl4 = d4.loc[d4['treatment'] == 0, 'sks_presentclient'].mean()
    p = published[3]
    print(f"{'(4) Self-reported status':<40} {r4.params['treatment']:>8.3f} {r4.bse['treatment']:>8.3f} {ctrl4:>10.3f} {int(r4.nobs):>7}  |  {p['coef']:>8.3f} {p['se']:>8.3f} {p['N']:>7}")


# ══════════════════════════════════════════════════════════════════════════════
# TABLE 3: Differential Impacts by Standardized HH Characteristics
# ══════════════════════════════════════════════════════════════════════════════

def table_3():
    print_section("TABLE 3: Adverse Selection Tests")

    # Load and merge data (matching AEA_Tables.do logic)
    loans = load_loans()
    bl = load_baseline()
    el = load_endline()

    # Keep analysis sample (merge loans with baseline)
    df = loans.merge(bl, on='sks_id', how='inner', suffixes=('', '_bl'))
    # Merge with endline
    df = df.merge(el, on='sks_id', how='inner')

    # Verify
    assert len(df) == 5366, f"Expected 5366, got {len(df)}"

    # Panel variable lists
    panelA_vars = ['chronic', 'family_chronic', 'health_scale', 'poor_health',
                   'consult_symptoms', 'smoke_drink']
    panelB_vars = ['plan_baby', 'num_female_age']
    panelD_vars = ['el_overnight', 'el_health_exp', 'el_baby']
    panelE_vars = ['hh_expenditure_yearly', 'self_financial', 'own_business']

    # Create predicted endline outcomes (Panel C)
    # Predict el_overnight, el_health_exp using panelA vars on control group
    for depvar in ['el_overnight', 'el_health_exp']:
        ctrl = df[df['treatment'] == 0].dropna(subset=[depvar] + panelA_vars).copy()
        X = sm.add_constant(ctrl[panelA_vars].astype(float))
        y = ctrl[depvar].astype(float)
        model = sm.OLS(y, X).fit(cov_type='cluster', cov_kwds={'groups': ctrl['village_id']})
        # Predict for all observations
        X_all = sm.add_constant(df[panelA_vars].astype(float))
        df[f'predict_{depvar}'] = model.predict(X_all)

    # Predict el_baby using panelB vars on control group
    ctrl = df[df['treatment'] == 0].dropna(subset=['el_baby'] + panelB_vars).copy()
    X = sm.add_constant(ctrl[panelB_vars].astype(float))
    y = ctrl['el_baby'].astype(float)
    model = sm.OLS(y, X).fit(cov_type='cluster', cov_kwds={'groups': ctrl['village_id']})
    X_all = sm.add_constant(df[panelB_vars].astype(float))
    df['predict_el_baby'] = model.predict(X_all)

    panelC_vars = ['predict_el_overnight', 'predict_el_health_exp', 'predict_el_baby']

    # Normalize all variables and create interactions
    all_vars = panelA_vars + panelB_vars + panelC_vars + panelD_vars + panelE_vars
    for var in all_vars:
        mean = df[var].mean()
        sd = df[var].std()
        df[f'norm_{var}'] = (df[var] - mean) / sd
        df[f'int_norm_{var}'] = df['treatment'] * df[f'norm_{var}']

    # Ensure treatment variable name is 'treatment' (matching do-file's 'treat')
    # The do-file uses 'treat' which is the same as 'treatment' in our data

    # Published values from table3.csv
    published = {
        'chronic': (-0.016, 0.013, 5312),
        'family_chronic': (-0.01, 0.013, 5312),
        'health_scale': (-0.002, 0.015, 5310),
        'poor_health': (-0.027, 0.013, 5310),
        'consult_symptoms': (-0.007, 0.015, 5201),
        'smoke_drink': (-0.015, 0.015, 5063),
        'plan_baby': (-0.014, 0.017, 5169),
        'num_female_age': (-0.002, 0.014, 5366),
        'predict_el_overnight': (-0.005, 0.014, 4946),
        'predict_el_health_exp': (-0.019, 0.015, 4946),
        'predict_el_baby': (-0.009, 0.016, 5169),
        'el_overnight': (0.013, 0.015, 5355),
        'el_health_exp': (-0.01, 0.014, 5358),
        'el_baby': (0.017, 0.013, 5366),
        'hh_expenditure_yearly': (0.032, 0.017, 5232),
        'self_financial': (-0.012, 0.017, 4881),
        'own_business': (-0.026, 0.019, 5233),
    }

    # Note: The table3.csv shows some different values from published paper
    # Paper Table 3: poor_health = -0.028, family_chronic = -0.015, own_business = -0.029
    # table3.csv: poor_health = -0.027, family_chronic = -0.01, own_business = -0.026

    panel_labels = {
        'chronic': 'Chronic disease, any in household',
        'family_chronic': 'Chronic disease, any family history',
        'health_scale': 'Self-reported health, household average',
        'poor_health': 'Any household member in poor health',
        'consult_symptoms': 'Any consultation for symptoms, previous 30 days',
        'smoke_drink': 'Any household member who smokes or drinks',
        'plan_baby': 'Any household member plans to have baby',
        'num_female_age': 'Number of females aged 17-to-24 in household',
        'predict_el_overnight': 'Spend night in hospital, predicted',
        'predict_el_health_exp': 'Health expenditures, predicted',
        'predict_el_baby': 'Pregnancy, predicted',
        'el_overnight': 'Spent night in hospital',
        'el_health_exp': 'Health expenditures',
        'el_baby': 'New baby, between baseline and endline',
        'hh_expenditure_yearly': 'Household consumption, previous year',
        'self_financial': 'Self-reported financial status',
        'own_business': 'Household owns business',
    }

    panels = [
        ('Panel A: Baseline Health Indicators', panelA_vars),
        ('Panel B: Baseline Pregnancy Indicators', panelB_vars),
        ('Panel C: Predicted Endline Health and Pregnancy', panelC_vars),
        ('Panel D: Endline Health and Pregnancy', panelD_vars),
        ('Panel E: Baseline Economic Indicators', panelE_vars),
    ]

    print(f"\n{'Variable':<50} {'Coef':>8} {'SE':>8} {'N':>6}  |  {'Pub':>8} {'Pub SE':>8} {'Pub N':>6}")
    print("-" * 105)

    for panel_name, panel_vars in panels:
        print(f"\n  {panel_name}")
        for var in panel_vars:
            # areg renewal_first int_norm_var norm_var treat, absorb(stratify) cluster(village_id)
            xvars = [f'int_norm_{var}', f'norm_{var}', 'treatment']
            data = df[list(set(['renewal_first'] + xvars + ['stratify', 'village_id']))].dropna()

            result = areg(data, 'renewal_first', xvars)
            coef = result.params[f'int_norm_{var}']
            se = result.bse[f'int_norm_{var}']
            N = int(result.nobs)

            pub = published.get(var, (np.nan, np.nan, np.nan))
            label = panel_labels.get(var, var)
            match_str = "✓" if abs(coef - pub[0]) < 0.002 else "✗"
            print(f"    {label:<46} {coef:>8.3f} {se:>8.3f} {N:>6}  |  {pub[0]:>8.3f} {pub[1]:>8.3f} {pub[2]:>6} {match_str}")

    # Joint F-test for Panels A+B
    print_subsection("Joint F-test (Panels A+B)")
    joint_vars = [f'norm_{v}' for v in panelA_vars + panelB_vars]
    int_joint_vars = [f'int_norm_{v}' for v in panelA_vars + panelB_vars]

    data_joint = df[list(set(['renewal_first'] + joint_vars + int_joint_vars +
                             ['treatment', 'stratify', 'village_id']))].dropna()

    # Without interactions
    r_without = areg(data_joint, 'renewal_first', joint_vars + ['treatment'])
    r2_without = r_without.rsquared

    # With interactions
    r_with = areg(data_joint, 'renewal_first', int_joint_vars + joint_vars + ['treatment'])
    r2_with = r_with.rsquared

    print(f"R² without interactions: {r2_without:.6f}")
    print(f"R² with interactions: {r2_with:.6f}")
    print(f"Difference: {abs(r2_with - r2_without):.6f}")

    # F-test on interaction terms
    from scipy.stats import f as f_dist
    # testparm on interaction vars
    r_matrix = np.zeros((len(int_joint_vars), len(r_with.params)))
    for i, var in enumerate(int_joint_vars):
        idx = list(r_with.params.index).index(var)
        r_matrix[i, idx] = 1
    f_test = r_with.f_test(r_matrix)
    print(f"F-test p-value: {float(f_test.pvalue):.4f}")


# ══════════════════════════════════════════════════════════════════════════════
# IN-TEXT STATISTICS (AEA_Paper_Numbers)
# ══════════════════════════════════════════════════════════════════════════════

def paper_numbers():
    print_section("IN-TEXT STATISTICS")

    # Health expenses
    print_subsection("Health Expenses (Baseline)")
    bl = load_baseline()
    he = bl['total_health_expense'].dropna()
    print(f"Average health expense: {he.mean():.4f}  (published: 4671.1945)")
    print(f"95th percentile: {he.quantile(0.95):.0f}  (published: 15570)")

    hosp = bl['hosp_expense'].dropna()
    print(f"Average hospitalization expense: {hosp.mean():.5f}  (published: 602.85629)")
    print(f"95th percentile hospitalization: {hosp.quantile(0.95):.0f}  (published: 2810)")

    he_treat = bl.loc[bl['treatment'] == 1, 'total_health_expense'].dropna()
    print(f"Avg health expense (treatment): {he_treat.mean():.3f}  (published: 4898.737)")
    print(f"95th pctl health expense (treatment): {he_treat.quantile(0.95):.0f}  (published: 15430)")

    # Attrition
    print_subsection("Attrition")
    att = load_attrition()
    att['attrition'] = np.nan
    mask_att = (att['merge_base_end'] == 1) & (att['merge_surveys_admin'] == 3) & (att['merge_treatment'] == 3)
    mask_no = (att['merge_base_end'] == 3) & (att['merge_surveys_admin'] == 3) & (att['merge_treatment'] == 3)
    att.loc[mask_att, 'attrition'] = 1
    att.loc[mask_no, 'attrition'] = 0
    att_valid = att.dropna(subset=['attrition'])
    print(f"Attrition rate: {att_valid['attrition'].mean():.6f}  (published: 0.012877)")
    print(f"N for attrition regression: {len(att_valid)}  (published: 5436)")

    # Attrition regression
    result = areg(att_valid, 'attrition', ['treatment'])
    print(f"Treatment coef on attrition: {result.params['treatment']:.7f}  (published: 0.0071324)")
    print(f"SE: {result.bse['treatment']:.7f}  (published: 0.0049405)")
    print(f"N: {int(result.nobs)}  (published: 5436)")

    # Missing rollout dates
    print_subsection("Missing Rollout Dates")
    treat = load_treatment()
    missing_centers = treat.loc[treat['treatment'] == 1, 'rollout_date'].isna().sum()
    print(f"Centers with missing rollout: {missing_centers}  (published: 32)")

    treat['village_rollout'] = treat.groupby('village_id')['rollout_date'].transform('max')
    villages = treat.drop_duplicates(subset=['village_id', 'treatment'])
    missing_villages = villages.loc[(villages['treatment'] == 1) & (villages['village_rollout'].isna())].shape[0]
    print(f"Villages with missing rollout: {missing_villages}  (published: 20)")


def main():
    table_1()
    table_2()
    table_3()
    paper_numbers()
    print_section("ALL TABLES COMPLETE")


if __name__ == '__main__':
    main()
