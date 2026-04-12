"""02_tables.py — Replicate Tables 1, 2, 3 from Bassi et al. (2025).

Table 1: Baseline summary stats (with treatment-control balance).
Table 2: Preschool & ITT impacts on primary school enrollment.
Table 3: Caregiver time investments.
"""
import os
import numpy as np
import pandas as pd
import statsmodels.api as sm
from utils import load_endline, load_baseline, areg_cluster, OUTPUT_DIR


def stata_tabstat_sd(series):
    """Stata's SD (n-1 denom, default)."""
    return float(series.std(ddof=1))


def table1():
    bl = load_baseline()

    # Match Stata: rename old _gsem versions away, then rename _zi to main name.
    bl = bl.rename(columns={'wealth_index': 'wealth_index_gsem',
                            'stimul_index': 'stimul_index_gsem'})
    bl = bl.rename(columns={'wealth_index_zi': 'wealth_index',
                            'stimul_index_zi': 'stimul_index'})

    target_vars = ['female', 'ageinmonth', 'zhfa_35', 'wenttopreschl',
                   'total_score_zi_nostd', 'score_com_zi_nostd',
                   'score_grossmotor_zi_nostd', 'score_finemotor_zi_nostd',
                   'score_pbmsolving_zi_nostd', 'score_socioperso_zi_nostd']
    caregiver_vars = ['IDADE', 'female', 'speaks_port', 'illiterate',
                      'noschooling', 'educ_year']
    hh_vars = ['AF05', 'nbr_children', 'nbr_children35', 'C01',
               'wealth_index', 'stimul_index']

    tgt_df = bl[(bl['target_35'] == 1) & (bl['TARGET_GROUP'] != 1)].copy()
    cg_df = bl[(bl['CH01_35'] == bl['MemberId']) & (bl['TARGET_GROUP'] != 1)].copy()
    hh_df = bl[(bl['relationToHead'] == 1) & (bl['TARGET_GROUP'] != 1)].copy()

    rows = []

    def compute_block(df, varlist, block_name):
        for v in varlist:
            sub = df.dropna(subset=[v, 'DIST_ID', 'COM_ID', 'treatment_ecd']).copy()
            ctrl = sub[sub['treatment_ecd'] == 0][v]
            trt = sub[sub['treatment_ecd'] == 1][v]
            res = areg_cluster(sub, v, ['treatment_ecd'], 'DIST_ID', 'COM_ID')
            rows.append({
                'block': block_name, 'var': v,
                'ctrl_mean': ctrl.mean(), 'ctrl_sd': stata_tabstat_sd(ctrl),
                'trt_mean': trt.mean(), 'trt_sd': stata_tabstat_sd(trt),
                'pval': res['pval'], 'N': res['N']
            })

    compute_block(tgt_df, target_vars, 'target')
    compute_block(cg_df, caregiver_vars, 'caregiver')
    compute_block(hh_df, hh_vars, 'hh')

    tab = pd.DataFrame(rows)
    print("\n" + "=" * 60)
    print("TABLE 1: Baseline Summary Statistics")
    print("=" * 60)
    print(tab.to_string(index=False, float_format='%.3f'))

    # Joint F-test on HH-level using the specific variable list the Stata code
    # passes in (global $tgt + $caregiver + $hh).
    jf_df = hh_df.copy()
    tgt_vars = ['zhfa_35', 'wenttopreschl_hh', 'total_score_zi_nostd',
                'score_com_zi_nostd', 'score_grossmotor_zi_nostd',
                'score_finemotor_zi_nostd', 'score_pbmsolving_zi_nostd',
                'score_socioperso_zi_nostd']
    allvars_unique = tgt_vars + caregiver_vars + hh_vars
    sub = jf_df.dropna(subset=allvars_unique + ['treatment_ecd', 'DIST_ID', 'COM_ID']).copy()
    dum = pd.get_dummies(sub['DIST_ID'].astype(int), prefix='d', drop_first=True).astype(float)
    X = pd.concat([sub[allvars_unique].astype(float).reset_index(drop=True),
                   dum.reset_index(drop=True)], axis=1)
    X = sm.add_constant(X, has_constant='add')
    y = sub['treatment_ecd'].astype(float).values
    res = sm.OLS(y, X.values.astype(float)).fit(
        cov_type='cluster',
        cov_kwds={'groups': sub['COM_ID'].astype(int).values},
        use_t=True
    )
    # Joint test: all the baseline covariates = 0 (exclude constant and FE dummies)
    test_idx = [i for i, c in enumerate(X.columns) if c in allvars_unique]
    R = np.zeros((len(test_idx), X.shape[1]))
    for r, i in enumerate(test_idx):
        R[r, i] = 1
    ft = res.f_test(R)
    print(f"\nJoint F-test (all vars, HH sample): F={float(ft.fvalue):.3f}  "
          f"p={float(ft.pvalue):.3f}  N={int(res.nobs)}")

    tab.to_csv(os.path.join(OUTPUT_DIR, 'table1.csv'), index=False)
    return tab


def table2():
    el = load_endline()
    d = el[el['istarget35'] == 1].copy()

    outcomes = ['wenttopreschl', 'enroledatschool', 'schl_repetition_uncon',
                'app_grade1', 'enrolment_zi']

    rows = []
    for y in outcomes:
        res = areg_cluster(d, y, ['treatment_ecd'], 'DIST_ID', 'COM_ID',
                           weight_col='equal_weights')
        ctrl_mean = d[d['treatment_ecd'] == 0][y].mean()
        rows.append({
            'outcome': y,
            'beta': res['beta'],
            'se': res['se'],
            'pval': res['pval'],
            'N': res['N'],
            'ctrl_mean': ctrl_mean,
        })
    tab = pd.DataFrame(rows)
    print("\n" + "=" * 60)
    print("TABLE 2: Preschool & ITT Impacts on Primary School Enrollment")
    print("=" * 60)
    print(tab.to_string(index=False, float_format='%.4f'))
    tab.to_csv(os.path.join(OUTPUT_DIR, 'table2.csv'), index=False)
    return tab


def table3():
    el = load_endline()
    d = el[el['istarget35'] == 1].copy()

    outcomes = ['Parental_stimul_index_EL_zi', 'Home_play_index_EL_zi',
                'pedschldrtr', 'times_pedschldrtr',
                'teacherschlmeet', 'times_teacherschlmeet',
                'school_committee']

    rows = []
    for y in outcomes:
        res = areg_cluster(d, y, ['treatment_ecd'], 'DIST_ID', 'COM_ID',
                           weight_col='equal_weights')
        ctrl_mean = d[d['treatment_ecd'] == 0][y].mean()
        rows.append({
            'outcome': y,
            'beta': res['beta'],
            'se': res['se'],
            'pval': res['pval'],
            'N': res['N'],
            'ctrl_mean': ctrl_mean,
        })
    tab = pd.DataFrame(rows)
    print("\n" + "=" * 60)
    print("TABLE 3: Caregiver Time Investments")
    print("=" * 60)
    print(tab.to_string(index=False, float_format='%.4f'))
    tab.to_csv(os.path.join(OUTPUT_DIR, 'table3.csv'), index=False)
    return tab


def main():
    table1()
    table2()
    table3()


if __name__ == '__main__':
    main()
