"""
02_tables.py — Replicate main tables from Muralidharan, Singh, Ganimian (2019).

Tables replicated:
  - Table 2: ITT effect on endline test scores (math, hindi)
  - Table 5: Heterogeneity by baseline tercile
  - Table 6: Heterogeneity by grade-level vs below-grade material
  - Table 8: IV / dose-response using attendance
"""

import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import pandas as pd
import statsmodels.api as sm
from utils import load_wide, load_attendance, OUTPUT_DIR, ols_with_strata, format_coef

def run_table2():
    print('\n========== TABLE 2: ITT, endline scores on treatment ==========')
    df = load_wide()
    results = {}
    for subj, yvar, ybase, paper_beta, paper_se, paper_n in [
        ('Math',  'm_theta_mle2', 'm_theta_mle1', 0.36, 0.063, 529),
        ('Hindi', 'h_theta_mle2', 'h_theta_mle1', 0.22, 0.076, 533),
    ]:
        # Spec with lagged outcome + strata FE + HC1 SEs
        m = ols_with_strata(df, yvar, [ybase, 'treat'], strata_col='strata',
                            cov_type='HC1')
        beta = m.params['treat']; se = m.bse['treat']; pval = m.pvalues['treat']
        baseline_coef = m.params[ybase]
        print(f'{subj:6s}  β_treat={beta:.4f} (SE {se:.4f}, p={pval:.4f}) '
              f'β_baseline={baseline_coef:.4f}  N={int(m.nobs)} '
              f'| paper: {paper_beta} ({paper_se}), N={paper_n}')
        results[subj] = {'beta': beta, 'se': se, 'pval': pval,
                         'baseline_coef': baseline_coef, 'N': int(m.nobs),
                         'paper_beta': paper_beta, 'paper_se': paper_se,
                         'paper_N': paper_n}
    return results

def run_table6_grade():
    """Table 6 (paper): Grade-specific achievement — here we use test sub-scores
    if available; fall back to overall IRT score split by grade dummies.
    The paper uses at-grade vs below-grade ITEMS on the test, which requires
    item-level data (ms_mathqs, ms_hindiqs) + irt parameters. Approximate with
    heterogeneity by baseline score tercile instead (Table 5)."""
    print('\n========== TABLE 5 (tercile heterogeneity): ITT by baseline tercile ==========')
    df = load_wide()
    results = {}
    for subj, yvar, ybase, terc in [
        ('Math',  'm_theta_mle2', 'm_theta_mle1', 'terc_math1'),
        ('Hindi', 'h_theta_mle2', 'h_theta_mle1', 'terc_hindi1'),
    ]:
        d = df.dropna(subset=[yvar, ybase, 'treat', 'strata', terc]).copy()
        d['T_x_bot'] = d['treat'] * (d[terc] == 1).astype(int)
        d['T_x_mid'] = d['treat'] * (d[terc] == 2).astype(int)
        d['T_x_top'] = d['treat'] * (d[terc] == 3).astype(int)
        d['bot'] = (d[terc] == 1).astype(int)
        d['mid'] = (d[terc] == 2).astype(int)
        # baseline tercile main effects omit top as reference
        m = ols_with_strata(d, yvar,
                            [ybase, 'T_x_bot', 'T_x_mid', 'T_x_top', 'bot', 'mid'],
                            strata_col='strata', cov_type='HC1')
        print(f'{subj}:')
        for tk in ['T_x_bot','T_x_mid','T_x_top']:
            print(f'   {tk}: β={m.params[tk]:.4f} (SE {m.bse[tk]:.4f})')
        results[subj] = {'bot': m.params['T_x_bot'], 'mid': m.params['T_x_mid'],
                         'top': m.params['T_x_top'], 'N': int(m.nobs)}
    return results

def run_table8_iv():
    """Table 8: Dose-response IV using attendance. Second stage:
         Y_iks2 = α + γ Y_iks1 + β·days_attended + strata FE + ε
       Instrument: treatment indicator. Only treatment group has attendance.
       Use 2SLS: endogenous = att_tot, instrument = treat, sample = everyone."""
    print('\n========== TABLE 8: IV / dose-response on attendance ==========')
    df = load_wide()
    ei = load_attendance()[['st_id','att_tot']]
    d = df.merge(ei, on='st_id', how='left')
    # Control students have no att_tot; set to 0 for 2SLS (standard ITT-as-IV)
    d['att_tot'] = d['att_tot'].fillna(0.0)

    results = {}
    for subj, yvar, ybase, paper_beta in [
        ('Math',  'm_theta_mle2', 'm_theta_mle1', 0.0065),
        ('Hindi', 'h_theta_mle2', 'h_theta_mle1', 0.0040),
    ]:
        dd = d.dropna(subset=[yvar, ybase, 'treat', 'strata', 'att_tot']).copy()
        dd['strata'] = dd['strata'].astype(int)
        fe = pd.get_dummies(dd['strata'], prefix='s', drop_first=True).astype(float)

        # First stage: att_tot on treat (+ baseline + FE)
        X1 = pd.concat([dd[[ybase]].reset_index(drop=True).astype(float),
                        dd[['treat']].reset_index(drop=True).astype(float),
                        fe.reset_index(drop=True)], axis=1)
        X1 = sm.add_constant(X1)
        fs = sm.OLS(dd['att_tot'].reset_index(drop=True).astype(float), X1).fit(cov_type='HC1')
        first_stage_F = (fs.params['treat'] / fs.bse['treat']) ** 2
        dd['att_hat'] = fs.fittedvalues.values

        # Second stage
        X2 = pd.concat([dd[[ybase]].reset_index(drop=True).astype(float),
                        dd[['att_hat']].reset_index(drop=True),
                        fe.reset_index(drop=True)], axis=1)
        X2 = sm.add_constant(X2)
        ss = sm.OLS(dd[yvar].reset_index(drop=True).astype(float), X2).fit(cov_type='HC1')
        beta = ss.params['att_hat']; se = ss.bse['att_hat']; pval = ss.pvalues['att_hat']
        print(f'{subj:6s}  β_per_day={beta:.5f} (SE {se:.5f}, p={pval:.4f})  '
              f'first-stage F≈{first_stage_F:.1f}  N={int(ss.nobs)} '
              f'| paper per-day: {paper_beta}, 90-day impact ≈ {paper_beta*90:.3f}')
        results[subj] = {'beta_per_day': beta, 'se': se, 'pval': pval,
                         'impact_90d': beta * 90, 'N': int(ss.nobs),
                         'first_stage_F': first_stage_F}
    return results

def save_summary(t2, t5, t8):
    rows = []
    for subj, r in t2.items():
        rows.append({'table': 'Table 2 (ITT)', 'subject': subj,
                     'coef': r['beta'], 'se': r['se'], 'N': r['N'],
                     'paper_coef': r['paper_beta'], 'paper_se': r['paper_se'],
                     'match': abs(r['beta'] - r['paper_beta']) < 0.05})
    for subj, r in t5.items():
        rows.append({'table': 'Table 5 (terc_bot)', 'subject': subj,
                     'coef': r['bot'], 'se': np.nan, 'N': r['N'],
                     'paper_coef': np.nan, 'paper_se': np.nan, 'match': np.nan})
    for subj, r in t8.items():
        rows.append({'table': 'Table 8 (IV per-day)', 'subject': subj,
                     'coef': r['beta_per_day'], 'se': r['se'], 'N': r['N'],
                     'paper_coef': 0.0065 if subj=='Math' else 0.0040,
                     'paper_se': np.nan,
                     'match': abs(r['beta_per_day'] - (0.0065 if subj=='Math' else 0.0040)) < 0.002})
    pd.DataFrame(rows).to_csv(os.path.join(OUTPUT_DIR, 'table_results.csv'), index=False)
    print(f'\n[saved] output/table_results.csv ({len(rows)} rows)')

def main():
    t2 = run_table2()
    t5 = run_table6_grade()
    t8 = run_table8_iv()
    save_summary(t2, t5, t8)
    print('[done] 02_tables.py')

if __name__ == '__main__':
    main()
