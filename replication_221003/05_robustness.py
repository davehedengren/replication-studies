"""05_robustness.py — Robustness checks for main ITT results (Table 2, Fig 2).

All spec as primary model:
    areg y treatment_ecd [pw=equal_weights] if istarget35==1,
         absorb(DIST_ID) vce(cluster COM_ID)

Each check varies one dimension and re-estimates the primary outcomes.
"""
import os
import numpy as np
import pandas as pd
import statsmodels.api as sm
from utils import load_endline, areg_cluster, OUTPUT_DIR

PRIMARY_OUTCOMES = [
    ('wenttopreschl', 'Preschool enrolled'),
    ('enroledatschool', 'Primary enrolled'),
    ('app_grade1', 'Appropriate grade'),
    ('enrolment_zi', 'Primary enrol. index'),
    ('skills_zi', 'Skills index'),
]


def run_spec(df, outcomes, weight='equal_weights', cluster='COM_ID',
             fe='DIST_ID'):
    rows = []
    for v, label in outcomes:
        r = areg_cluster(df, v, ['treatment_ecd'], fe, cluster,
                         weight_col=weight)
        rows.append({'outcome': label, 'var': v,
                     'beta': r['beta'], 'se': r['se'],
                     'pval': r['pval'], 'N': r['N']})
    return pd.DataFrame(rows)


def print_block(title, df):
    print("\n" + "-" * 70)
    print(title)
    print("-" * 70)
    print(df.to_string(index=False, float_format='%.4f'))


def main():
    el = load_endline()
    d0 = el[el['istarget35'] == 1].copy()

    print("=" * 70)
    print("BASELINE (for reference) — primary ITT spec")
    print("=" * 70)
    base = run_spec(d0, PRIMARY_OUTCOMES)
    print(base.to_string(index=False, float_format='%.4f'))

    # 1. No weights (unweighted OLS)
    print_block("1. UNWEIGHTED (no pw=equal_weights)",
                run_spec(d0, PRIMARY_OUTCOMES, weight=None))

    # 2. Cluster at district level instead of community
    print_block("2. CLUSTER SE AT DISTRICT (not community)",
                run_spec(d0, PRIMARY_OUTCOMES, cluster='DIST_ID'))

    # 3. No fixed effects (drop district FE → use province only via area_fixed)
    # Closest: pool districts, just rely on randomization. Drop absorb by
    # replacing fe with a single-valued column.
    d_tmp = d0.copy()
    d_tmp['_one'] = 1
    print_block("3. NO DISTRICT FE (pooled, community-clustered SE)",
                run_spec(d_tmp, PRIMARY_OUTCOMES, fe='_one'))

    # 4. Drop one province at a time (leave-one-province-out)
    for prov in sorted(d0['PROV_ID'].dropna().unique()):
        sub = d0[d0['PROV_ID'] != prov].copy()
        print_block(f"4. DROP PROVINCE {int(prov)}  (N={len(sub)})",
                    run_spec(sub, PRIMARY_OUTCOMES))

    # 5. Trim age distribution (drop 5th/95th tails of age_trgt)
    if 'age_trgt' in d0.columns:
        lo, hi = d0['age_trgt'].quantile([0.05, 0.95])
        sub = d0[(d0['age_trgt'] >= lo) & (d0['age_trgt'] <= hi)].copy()
        print_block(f"5. TRIM endline age tails (5/95) N={len(sub)}",
                    run_spec(sub, PRIMARY_OUTCOMES))

    # 6. Drop communities with fewer than 10 target children
    com_n = d0.groupby('COM_ID').size()
    keep = com_n[com_n >= 10].index
    sub = d0[d0['COM_ID'].isin(keep)].copy()
    print_block(f"6. Drop small communities (<10 target children). N={len(sub)}",
                run_spec(sub, PRIMARY_OUTCOMES))

    # 7. Winsorize skills index at 1/99
    d_win = d0.copy()
    for col in ['skills_zi', 'enrolment_zi']:
        s = d_win[col]
        lo, hi = s.quantile(0.01), s.quantile(0.99)
        d_win[col] = s.clip(lo, hi)
    print_block("7. WINSORIZE skills_zi / enrolment_zi at 1/99",
                run_spec(d_win, PRIMARY_OUTCOMES))

    # 8. Subgroup: female children
    sub = d0[d0['female'] == 1].copy()
    print_block(f"8a. GIRLS only (N={len(sub)})",
                run_spec(sub, PRIMARY_OUTCOMES))
    sub = d0[d0['female'] == 0].copy()
    print_block(f"8b. BOYS only (N={len(sub)})",
                run_spec(sub, PRIMARY_OUTCOMES))

    # 9. Placebo: shuffle treatment within districts (single draw) and re-run.
    #    Then full randomization inference on wenttopreschl and enrolment_zi.
    rng = np.random.default_rng(20260411)
    d_plac = d0.copy()
    # Assign community-level placebo, not child-level
    com_treat = (d0[['COM_ID', 'DIST_ID', 'treatment_ecd']]
                 .drop_duplicates('COM_ID'))
    com_treat['placebo'] = (com_treat.groupby('DIST_ID')['treatment_ecd']
                            .transform(lambda s: rng.permutation(s.values)))
    d_plac = d_plac.merge(com_treat[['COM_ID', 'placebo']], on='COM_ID')
    d_plac['treatment_ecd_real'] = d_plac['treatment_ecd']
    d_plac['treatment_ecd'] = d_plac['placebo']
    print_block("9. PLACEBO: within-district shuffle of treatment (1 draw)",
                run_spec(d_plac, PRIMARY_OUTCOMES))

    # 10. Randomization inference on wenttopreschl and enrolment_zi (200 reps)
    print("\n" + "-" * 70)
    print("10. RANDOMIZATION INFERENCE p-values (200 within-district shuffles)")
    print("-" * 70)
    observed = {v: areg_cluster(d0, v, ['treatment_ecd'], 'DIST_ID', 'COM_ID',
                                weight_col='equal_weights')['beta']
                for v, _ in PRIMARY_OUTCOMES}
    n_reps = 200
    exceed = {v: 0 for v in observed}
    comm_dist = (d0[['COM_ID', 'DIST_ID', 'treatment_ecd']]
                 .drop_duplicates('COM_ID').reset_index(drop=True))
    for rep in range(n_reps):
        comm_dist['placebo'] = (comm_dist.groupby('DIST_ID')['treatment_ecd']
                                .transform(lambda s: rng.permutation(s.values)))
        d_sh = d0.drop(columns='treatment_ecd').merge(
            comm_dist[['COM_ID', 'placebo']].rename(
                columns={'placebo': 'treatment_ecd'}), on='COM_ID')
        for v, _ in PRIMARY_OUTCOMES:
            b = areg_cluster(d_sh, v, ['treatment_ecd'], 'DIST_ID', 'COM_ID',
                             weight_col='equal_weights')['beta']
            if abs(b) >= abs(observed[v]):
                exceed[v] += 1
    for v, lbl in PRIMARY_OUTCOMES:
        p = exceed[v] / n_reps
        print(f"  {lbl:30s} observed β = {observed[v]:+.4f}   RI p = {p:.3f}")

    # 11. Alternative SE: heteroskedastic-robust (HC1, no cluster)
    print("\n" + "-" * 70)
    print("11. HC1 SE (no cluster) — should INFLATE t (understate true SE)")
    print("-" * 70)
    rows = []
    for v, label in PRIMARY_OUTCOMES:
        sub = d0.dropna(subset=[v, 'treatment_ecd', 'DIST_ID',
                                'equal_weights']).copy()
        dist_dum = pd.get_dummies(sub['DIST_ID'].astype(int), prefix='d',
                                  drop_first=True).astype(float)
        X = pd.concat([sub[['treatment_ecd']].astype(float)
                       .reset_index(drop=True),
                       dist_dum.reset_index(drop=True)], axis=1)
        X = sm.add_constant(X, has_constant='add')
        w = sub['equal_weights'].astype(float).values
        r = sm.WLS(sub[v].astype(float).values, X.values.astype(float),
                   weights=w).fit(cov_type='HC1')
        rows.append({'outcome': label, 'beta': r.params[1], 'se': r.bse[1],
                     'pval': r.pvalues[1], 'N': int(r.nobs)})
    print(pd.DataFrame(rows).to_string(index=False, float_format='%.4f'))

    # 12. Control for baseline child characteristics as additional covariates
    controls = ['female', 'calc_age_bl', 'zhfa_35_BL', 'educ_resp_BL']
    d_ctrl = d0.dropna(subset=controls).copy()
    rows = []
    for v, label in PRIMARY_OUTCOMES:
        sub = d_ctrl.dropna(subset=[v, 'treatment_ecd', 'DIST_ID',
                                    'COM_ID', 'equal_weights'] + controls)
        dist_dum = pd.get_dummies(sub['DIST_ID'].astype(int), prefix='d',
                                  drop_first=True).astype(float)
        X = pd.concat([sub[['treatment_ecd'] + controls].astype(float)
                       .reset_index(drop=True),
                       dist_dum.reset_index(drop=True)], axis=1)
        X = sm.add_constant(X, has_constant='add')
        w = sub['equal_weights'].astype(float).values
        r = sm.WLS(sub[v].astype(float).values, X.values.astype(float),
                   weights=w).fit(cov_type='cluster',
                                  cov_kwds={'groups': sub['COM_ID'].values})
        rows.append({'outcome': label, 'beta': r.params[1], 'se': r.bse[1],
                     'pval': r.pvalues[1], 'N': int(r.nobs)})
    print("\n" + "-" * 70)
    print("12. WITH BASELINE CONTROLS (female, age_bl, HFA z-bl, educ_bl)")
    print("-" * 70)
    print(pd.DataFrame(rows).to_string(index=False, float_format='%.4f'))

    base.to_csv(os.path.join(OUTPUT_DIR, 'robustness_baseline.csv'), index=False)


if __name__ == '__main__':
    main()
