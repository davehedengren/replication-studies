"""
05_robustness.py — Robustness checks for 113192.

Checks:
  1. No strata FE (simple diff)
  2. With demographic controls (age, female, SES, grade dummies)
  3. Cluster SE at ms_center1 level
  4. Lee (2009) bounds for differential attrition
  5. Winsorize test scores at 1/99 percentiles
  6. Drop 0-attendance students (treat intent-to-treat vs those who showed up)
  7. Placebo: shuffle treatment within strata
  8. Leave-one-strata-out
  9. Drop grade-missing students
 10. Alternative outcome: EAP estimator instead of MLE
"""

import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats
from utils import load_wide, load_attendance, OUTPUT_DIR, ols_with_strata

RNG = np.random.default_rng(20260409)

def get_itt(df, yvar, ybase, extra=None, cov='HC1', cluster=None):
    x_cols = [ybase, 'treat'] + (extra or [])
    d = df.dropna(subset=[yvar, ybase, 'treat', 'strata'] + (extra or [])).copy()
    if cluster:
        d = d.dropna(subset=[cluster])
    d['strata'] = d['strata'].astype(int)
    fe = pd.get_dummies(d['strata'], prefix='s', drop_first=True).astype(float)
    X = pd.concat([d[x_cols].astype(float).reset_index(drop=True),
                   fe.reset_index(drop=True)], axis=1)
    X = sm.add_constant(X)
    Y = d[yvar].astype(float).reset_index(drop=True)
    if cluster:
        groups = d[cluster].reset_index(drop=True)
        m = sm.OLS(Y, X).fit(cov_type='cluster', cov_kwds={'groups': groups})
    else:
        m = sm.OLS(Y, X).fit(cov_type=cov)
    return m.params['treat'], m.bse['treat'], m.pvalues['treat'], int(m.nobs)

def get_no_fe(df, yvar, ybase):
    d = df.dropna(subset=[yvar, ybase, 'treat'])
    X = sm.add_constant(d[[ybase, 'treat']].astype(float))
    m = sm.OLS(d[yvar].astype(float), X).fit(cov_type='HC1')
    return m.params['treat'], m.bse['treat'], m.pvalues['treat'], int(m.nobs)

def lee_bounds(df, yvar, ybase):
    """Lee (2009) bounds: trim from the group with lower attrition to equalize."""
    d = df.dropna(subset=[ybase, 'treat']).copy()
    t = d[d['treat']==1]; c = d[d['treat']==0]
    resp_t = t[yvar].notna().mean()
    resp_c = c[yvar].notna().mean()
    if resp_c > resp_t:
        keep_frac = resp_t / resp_c
        trim_grp = c
        other = t
    else:
        keep_frac = resp_c / resp_t
        trim_grp = t
        other = c
    obs = trim_grp[yvar].dropna().sort_values()
    k_low = int(np.floor(len(obs) * (1 - keep_frac)))  # drop top for low bound
    lo_mean = obs.iloc[:len(obs)-k_low].mean()
    hi_mean = obs.iloc[k_low:].mean()
    base = other[yvar].dropna().mean()
    if resp_c > resp_t:
        low = base - hi_mean
        high = base - lo_mean
    else:
        low = lo_mean - other[yvar].dropna().mean()
        high = hi_mean - other[yvar].dropna().mean()
    return low, high

def winsor(s, low=0.01, high=0.99):
    lo, hi = s.quantile(low), s.quantile(high)
    return s.clip(lo, hi)

def main():
    df = load_wide()
    ei = load_attendance()[['st_id','att_tot']]
    df = df.merge(ei, on='st_id', how='left')

    rows = []
    def add(name, subj, beta, se, pval, N):
        sig = '***' if pval<0.01 else ('**' if pval<0.05 else ('*' if pval<0.1 else ''))
        rows.append({'check': name, 'subject': subj, 'beta': round(beta,4),
                     'se': round(se,4) if np.isfinite(se) else np.nan,
                     'pval': round(pval,4), 'N': N, 'sig': sig})
        print(f'[{name:30s}] {subj:6s} β={beta:+.4f}{sig} SE={se:.4f} p={pval:.4f} N={N}')

    print('\n=== Baseline (Table 2) ===')
    for subj, y, yb in [('Math','m_theta_mle2','m_theta_mle1'),
                         ('Hindi','h_theta_mle2','h_theta_mle1')]:
        b, s, p, n = get_itt(df, y, yb)
        add('1. Baseline (strata FE, HC1)', subj, b, s, p, n)

    print('\n=== 2. No strata FE ===')
    for subj, y, yb in [('Math','m_theta_mle2','m_theta_mle1'),
                         ('Hindi','h_theta_mle2','h_theta_mle1')]:
        b, s, p, n = get_no_fe(df, y, yb)
        add('2. No strata FE', subj, b, s, p, n)

    print('\n=== 3. + demographic controls ===')
    for subj, y, yb in [('Math','m_theta_mle2','m_theta_mle1'),
                         ('Hindi','h_theta_mle2','h_theta_mle1')]:
        b, s, p, n = get_itt(df, y, yb, extra=['st_age1','st_female1','ses_index'])
        add('3. + demog controls', subj, b, s, p, n)

    print('\n=== 4. Cluster SE at ms_center1 ===')
    for subj, y, yb in [('Math','m_theta_mle2','m_theta_mle1'),
                         ('Hindi','h_theta_mle2','h_theta_mle1')]:
        b, s, p, n = get_itt(df, y, yb, cluster='ms_center1')
        add('4. Cluster SE (ms_center1)', subj, b, s, p, n)

    print('\n=== 5. Winsorize endline at 1/99 ===')
    dw = df.copy()
    dw['m_theta_mle2'] = winsor(dw['m_theta_mle2'])
    dw['h_theta_mle2'] = winsor(dw['h_theta_mle2'])
    for subj, y, yb in [('Math','m_theta_mle2','m_theta_mle1'),
                         ('Hindi','h_theta_mle2','h_theta_mle1')]:
        b, s, p, n = get_itt(dw, y, yb)
        add('5. Winsorize 1/99', subj, b, s, p, n)

    print('\n=== 6. Drop 0-day-attendance treatment students (compliers proxy) ===')
    d6 = df.copy()
    d6 = d6[~((d6['treat']==1) & (d6['att_tot']==0))]
    for subj, y, yb in [('Math','m_theta_mle2','m_theta_mle1'),
                         ('Hindi','h_theta_mle2','h_theta_mle1')]:
        b, s, p, n = get_itt(d6, y, yb)
        add('6. Drop zero-attend T', subj, b, s, p, n)

    print('\n=== 7. Placebo: shuffle treatment within strata (1000 draws) ===')
    d_p = df.dropna(subset=['m_theta_mle2','m_theta_mle1','treat','strata']).copy()
    d_p['strata'] = d_p['strata'].astype(int)
    def placebo_beta(y, yb, n_iter=1000):
        betas = []
        for _ in range(n_iter):
            tmp = d_p.copy()
            tmp['treat'] = tmp.groupby('strata')['treat'].transform(
                lambda s: RNG.permutation(s.values))
            fe = pd.get_dummies(tmp['strata'], prefix='s', drop_first=True).astype(float)
            X = pd.concat([tmp[[yb,'treat']].astype(float).reset_index(drop=True),
                           fe.reset_index(drop=True)], axis=1)
            X = sm.add_constant(X)
            m = sm.OLS(tmp[y].astype(float).reset_index(drop=True), X).fit()
            betas.append(m.params['treat'])
        return np.array(betas)
    for subj, y, yb, real in [('Math','m_theta_mle2','m_theta_mle1', 0.3738),
                               ('Hindi','h_theta_mle2','h_theta_mle1', 0.2379)]:
        pb = placebo_beta(y, yb, n_iter=500)
        pval = (np.abs(pb) >= abs(real)).mean()
        add(f'7. Placebo (permute in strata)', subj, real, pb.std(), pval, len(d_p))

    print('\n=== 8. Leave-one-strata-out (show range) ===')
    for subj, y, yb in [('Math','m_theta_mle2','m_theta_mle1'),
                         ('Hindi','h_theta_mle2','h_theta_mle1')]:
        betas = []
        for s_drop in sorted(df['strata'].dropna().unique()):
            sub = df[df['strata']!=s_drop]
            try:
                b, *_ = get_itt(sub, y, yb)
                betas.append(b)
            except Exception:
                pass
        add(f'8. LOSO range [{min(betas):.3f}, {max(betas):.3f}]',
            subj, float(np.mean(betas)), float(np.std(betas)), 0.0, len(betas))

    print('\n=== 9. Drop grade-missing students ===')
    d9 = df[df[['d_sch_grade4','d_sch_grade5','d_sch_grade6','d_sch_grade7','d_sch_grade8','d_sch_grade9']].sum(axis=1)==1]
    for subj, y, yb in [('Math','m_theta_mle2','m_theta_mle1'),
                         ('Hindi','h_theta_mle2','h_theta_mle1')]:
        b, s, p, n = get_itt(d9, y, yb)
        add('9. Drop grade-missing', subj, b, s, p, n)

    print('\n=== 10. Alternative outcome: EAP (posterior-mean) IRT score ===')
    for subj, y, yb in [('Math','m_theta_eap2','m_theta_eap1'),
                         ('Hindi','h_theta_eap2','h_theta_eap1')]:
        b, s, p, n = get_itt(df, y, yb)
        add('10. EAP outcome', subj, b, s, p, n)

    print('\n=== 11. Lee (2009) bounds ===')
    for subj, y, yb in [('Math','m_theta_mle2','m_theta_mle1'),
                         ('Hindi','h_theta_mle2','h_theta_mle1')]:
        lo, hi = lee_bounds(df, y, yb)
        rows.append({'check': '11. Lee bounds [lo, hi]', 'subject': subj,
                     'beta': round((lo+hi)/2,4), 'se': round((hi-lo)/2,4),
                     'pval': np.nan, 'N': df[y].notna().sum(),
                     'sig': f'[{lo:.3f}, {hi:.3f}]'})
        print(f'[11. Lee bounds            ] {subj:6s} [{lo:+.3f}, {hi:+.3f}]')

    pd.DataFrame(rows).to_csv(os.path.join(OUTPUT_DIR, 'robustness.csv'), index=False)
    print(f'\n[saved] output/robustness.csv ({len(rows)} rows)')

if __name__ == '__main__':
    main()
