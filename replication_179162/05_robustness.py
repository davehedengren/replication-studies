"""
05_robustness.py — Robustness checks for the Dads & Daughters replication.

Tests the sensitivity of the main finding (fathers spend less on daughters
but mothers do not) to alternative specifications.
"""

import sys, os
import numpy as np
import pandas as pd
import statsmodels.api as sm

sys.path.insert(0, os.path.dirname(__file__))
from utils import ANALYSIS_DATA_PATH, OUTPUT_DIR, _demean_by_group


def run_regression(df, y_var, x_vars, absorb_var, cluster_var, condition=None):
    """Run demeaned OLS with absorbed FE and clustered SEs."""
    data = df.copy()
    if condition is not None:
        data = data[condition].copy()

    x_vars = [v for v in x_vars if v in data.columns]
    data = data.dropna(subset=[y_var] + x_vars + [absorb_var, cluster_var]).copy()
    data = data.reset_index(drop=True)

    if len(data) < 10:
        return None

    groups_fe = data[absorb_var].values
    n_groups = data[absorb_var].nunique()

    y_dm = _demean_by_group(data[y_var].values.astype(float), groups_fe)
    X_dm = pd.DataFrame(index=data.index)
    for v in x_vars:
        X_dm[v] = _demean_by_group(data[v].values.astype(float), groups_fe)

    model = sm.OLS(y_dm, X_dm)
    cluster_groups = data[cluster_var].values

    try:
        results = model.fit(cov_type='cluster',
                           cov_kwds={'groups': cluster_groups, 'df_correction': True})
    except Exception:
        results = model.fit(cov_type='HC1')

    results.df_resid = len(data) - len(x_vars) - n_groups

    return results, data


def print_result(label, results, data, key_vars=None):
    """Print key coefficients from regression."""
    if key_vars is None:
        key_vars = ['girl', 'momXgirl', 'mom']

    res, d = results
    print(f"\n  {label} (N={len(d)})")
    for v in key_vars:
        if v in res.params.index:
            coef = res.params[v]
            se = res.bse[v]
            pval = res.pvalues[v]
            stars = '***' if pval < 0.01 else '**' if pval < 0.05 else '*' if pval < 0.1 else ''
            print(f"    {v:15s}  coef={coef:8.3f}  se={se:.3f}  p={pval:.3f} {stars}")


def main():
    print("=" * 60)
    print("05_robustness.py: Robustness Checks")
    print("=" * 60)

    df = pd.read_parquet(ANALYSIS_DATA_PATH)
    good_fe_cols = [c for c in df.columns if c.startswith('_G')]

    base_cond = (df['childgood'] == 1) & (df['toys_bin'] == 0)
    base_x = ['girl', 'momXgirl', 'mom', 'adult_wtp_std', 'adult_wtp_stdXfu',
              'fu', 'young'] + good_fe_cols

    # --- Baseline: reproduce Table 1 Col 2 ---
    print("\n" + "=" * 60)
    print("BASELINE: Table 1 Column 2")
    ret = run_regression(df, 'wtp_std', base_x, 'strat', 'hhid', base_cond)
    if ret:
        print_result("Baseline", ret, ret[1])

    # Published: girl=-0.102, momXgirl=0.131, mom=-0.095

    results_summary = []

    def record(name, ret):
        if ret is None:
            results_summary.append({'check': name, 'girl': np.nan, 'momXgirl': np.nan,
                                    'mom': np.nan, 'N': 0})
            return
        res = ret[0]
        results_summary.append({
            'check': name,
            'girl': res.params.get('girl', np.nan),
            'girl_se': res.bse.get('girl', np.nan),
            'girl_p': res.pvalues.get('girl', np.nan),
            'momXgirl': res.params.get('momXgirl', np.nan),
            'momXgirl_se': res.bse.get('momXgirl', np.nan),
            'momXgirl_p': res.pvalues.get('momXgirl', np.nan),
            'mom': res.params.get('mom', np.nan),
            'N': len(ret[1]),
        })

    record('Baseline', ret)

    # ================================================================
    # CHECK 1: Winsorize WTP at 5th/95th percentiles
    # ================================================================
    print("\n" + "-" * 60)
    print("CHECK 1: Winsorize WTP at 5th/95th percentiles")

    df_w = df.copy()
    main_mask = base_cond.copy()
    for tg in df_w['tempgood'].unique():
        mask = df_w['tempgood'] == tg
        vals = df_w.loc[mask, 'wtp']
        p5, p95 = vals.quantile(0.05), vals.quantile(0.95)
        df_w.loc[mask, 'wtp_w'] = vals.clip(p5, p95)

    # Re-standardize
    df_w['wtp_std_w'] = np.nan
    for tg in df_w['tempgood'].unique():
        mask = df_w['tempgood'] == tg
        sd = df_w.loc[mask, 'wtp_w'].std()
        if sd > 0:
            df_w.loc[mask, 'wtp_std_w'] = df_w.loc[mask, 'wtp_w'] / sd

    ret1 = run_regression(df_w, 'wtp_std_w', base_x, 'strat', 'hhid', main_mask)
    if ret1:
        print_result("Winsorized (5/95)", ret1, ret1[1])
    record('Winsorize 5/95', ret1)

    # ================================================================
    # CHECK 2: Log WTP (IHS transform)
    # ================================================================
    print("\n" + "-" * 60)
    print("CHECK 2: IHS transform of WTP")

    df_ihs = df.copy()
    df_ihs['wtp_ihs'] = np.arcsinh(df_ihs['wtp'])

    ret2 = run_regression(df_ihs, 'wtp_ihs', base_x, 'strat', 'hhid', base_cond)
    if ret2:
        print_result("IHS(WTP)", ret2, ret2[1])
    record('IHS transform', ret2)

    # ================================================================
    # CHECK 3: Drop one wave at a time
    # ================================================================
    print("\n" + "-" * 60)
    print("CHECK 3: Drop one wave at a time")

    # BL only
    bl_cond = base_cond & (df['fu'] == 0)
    x_bl = [v for v in base_x if v not in ['fu', 'adult_wtp_stdXfu']]
    ret3a = run_regression(df, 'wtp_std', x_bl, 'strat', 'hhid', bl_cond)
    if ret3a:
        print_result("BL only", ret3a, ret3a[1])
    record('BL only', ret3a)

    # FU only
    fu_cond = base_cond & (df['fu'] == 1)
    x_fu = [v for v in base_x if v not in ['fu', 'adult_wtp_stdXfu']]
    ret3b = run_regression(df, 'wtp_std', x_fu, 'strat', 'hhid', fu_cond)
    if ret3b:
        print_result("FU only", ret3b, ret3b[1])
    record('FU only', ret3b)

    # ================================================================
    # CHECK 4: Drop one good at a time (leave-one-out)
    # ================================================================
    print("\n" + "-" * 60)
    print("CHECK 4: Leave-one-good-out")

    child_goods = df.loc[base_cond, 'good'].unique()
    for drop_good in sorted(child_goods):
        cond = base_cond & (df['good'] != drop_good)
        # Filter good FE cols for remaining goods
        ret4 = run_regression(df, 'wtp_std', base_x, 'strat', 'hhid', cond)
        if ret4:
            res = ret4[0]
            girl_c = res.params.get('girl', np.nan)
            mg_c = res.params.get('momXgirl', np.nan)
            girl_p = res.pvalues.get('girl', np.nan)
            mg_p = res.pvalues.get('momXgirl', np.nan)
            girl_s = '***' if girl_p < 0.01 else '**' if girl_p < 0.05 else '*' if girl_p < 0.1 else ''
            mg_s = '***' if mg_p < 0.01 else '**' if mg_p < 0.05 else '*' if mg_p < 0.1 else ''
            print(f"    Drop {drop_good:12s}  girl={girl_c:7.3f}{girl_s:3s}  "
                  f"momXgirl={mg_c:7.3f}{mg_s:3s}  N={len(ret4[1])}")

    # ================================================================
    # CHECK 5: Subgroup by wave × incentivization
    # ================================================================
    print("\n" + "-" * 60)
    print("CHECK 5: Incentivized vs Non-incentivized")

    ret5a = run_regression(df, 'wtp_std', base_x, 'strat', 'hhid',
                           base_cond & (df['hypo'] == 0))
    if ret5a:
        print_result("Incentivized only", ret5a, ret5a[1])
    record('Incentivized', ret5a)

    ret5b = run_regression(df, 'wtp_std', base_x, 'strat', 'hhid',
                           base_cond & (df['hypo'] == 1))
    if ret5b:
        print_result("Non-incentivized only", ret5b, ret5b[1])
    record('Non-incentivized', ret5b)

    # ================================================================
    # CHECK 6: Alternative SEs — HC1 robust (no clustering)
    # ================================================================
    print("\n" + "-" * 60)
    print("CHECK 6: HC1 robust SEs (no clustering)")

    data6 = df[base_cond].copy()
    x6 = [v for v in base_x if v in data6.columns]
    data6 = data6.dropna(subset=['wtp_std'] + x6 + ['strat', 'hhid']).reset_index(drop=True)
    groups_fe = data6['strat'].values

    y_dm = _demean_by_group(data6['wtp_std'].values.astype(float), groups_fe)
    X_dm = pd.DataFrame(index=data6.index)
    for v in x6:
        X_dm[v] = _demean_by_group(data6[v].values.astype(float), groups_fe)

    model6 = sm.OLS(y_dm, X_dm)
    res6 = model6.fit(cov_type='HC1')

    print(f"\n  HC1 Robust SEs (N={len(data6)})")
    for v in ['girl', 'momXgirl', 'mom']:
        if v in res6.params.index:
            coef = res6.params[v]
            se = res6.bse[v]
            pval = res6.pvalues[v]
            stars = '***' if pval < 0.01 else '**' if pval < 0.05 else '*' if pval < 0.1 else ''
            print(f"    {v:15s}  coef={coef:8.3f}  se={se:.3f}  p={pval:.3f} {stars}")

    # ================================================================
    # CHECK 7: Placebo test — shuffle girl assignment
    # ================================================================
    print("\n" + "-" * 60)
    print("CHECK 7: Placebo test — shuffle girl assignment (100 permutations)")

    np.random.seed(42)
    n_perms = 100
    placebo_girl = []
    placebo_momXgirl = []

    df_perm = df[base_cond].copy()
    df_perm = df_perm.dropna(subset=['wtp_std', 'girl', 'mom'] + [v for v in base_x if v in df_perm.columns] + ['strat', 'hhid'])

    for i in range(n_perms):
        df_p = df_perm.copy()
        # Shuffle girl within strata (preserving stratification structure)
        for s in df_p['strat'].unique():
            mask = df_p['strat'] == s
            girls = df_p.loc[mask, 'girl'].values.copy()
            np.random.shuffle(girls)
            df_p.loc[mask, 'girl'] = girls
        df_p['momXgirl'] = df_p['mom'] * df_p['girl']

        ret_p = run_regression(df_p, 'wtp_std', base_x, 'strat', 'hhid')
        if ret_p:
            placebo_girl.append(ret_p[0].params.get('girl', np.nan))
            placebo_momXgirl.append(ret_p[0].params.get('momXgirl', np.nan))

    if placebo_girl:
        actual_girl = ret[0].params['girl']
        actual_momXgirl = ret[0].params['momXgirl']
        p_girl = np.mean([abs(g) >= abs(actual_girl) for g in placebo_girl])
        p_momXgirl = np.mean([abs(g) >= abs(actual_momXgirl) for g in placebo_momXgirl])
        print(f"  Actual girl coef: {actual_girl:.3f}")
        print(f"  Placebo girl: mean={np.mean(placebo_girl):.4f}, sd={np.std(placebo_girl):.4f}")
        print(f"  Permutation p-value (girl): {p_girl:.3f}")
        print(f"  Actual momXgirl coef: {actual_momXgirl:.3f}")
        print(f"  Placebo momXgirl: mean={np.mean(placebo_momXgirl):.4f}, sd={np.std(placebo_momXgirl):.4f}")
        print(f"  Permutation p-value (momXgirl): {p_momXgirl:.3f}")

    # ================================================================
    # CHECK 8: Health goods only vs non-health goods
    # ================================================================
    print("\n" + "-" * 60)
    print("CHECK 8: Health goods vs non-health child goods")

    ret8a = run_regression(df, 'wtp_std', base_x, 'strat', 'hhid',
                           base_cond & (df['health'] == 1))
    if ret8a:
        print_result("Health goods only", ret8a, ret8a[1])
    record('Health goods', ret8a)

    ret8b = run_regression(df, 'wtp_std', base_x, 'strat', 'hhid',
                           base_cond & (df['health'] == 0))
    if ret8b:
        print_result("Non-health child goods", ret8b, ret8b[1])
    record('Non-health goods', ret8b)

    # ================================================================
    # CHECK 9: Drop adult WTP control
    # ================================================================
    print("\n" + "-" * 60)
    print("CHECK 9: Without adult WTP control")

    x_no_adult = [v for v in base_x if v not in ['adult_wtp_std', 'adult_wtp_stdXfu']]
    ret9 = run_regression(df, 'wtp_std', x_no_adult, 'strat', 'hhid', base_cond)
    if ret9:
        print_result("No adult WTP control", ret9, ret9[1])
    record('No adult WTP', ret9)

    # ================================================================
    # CHECK 10: HH fixed effects instead of stratum FE
    # ================================================================
    print("\n" + "-" * 60)
    print("CHECK 10: Household FE (instead of stratum FE)")

    ret10 = run_regression(df, 'wtp_std', base_x, 'hhid', 'hhid', base_cond)
    if ret10:
        print_result("HH FE", ret10, ret10[1])
    record('HH FE', ret10)

    # ================================================================
    # CHECK 11: Interaction with young child indicator
    # ================================================================
    print("\n" + "-" * 60)
    print("CHECK 11: Add mom × young interaction")

    df_11 = df.copy()
    df_11['momXyoung'] = df_11['mom'] * df_11['young']
    x_11 = base_x + ['momXyoung']
    ret11 = run_regression(df_11, 'wtp_std', x_11, 'strat', 'hhid', base_cond)
    if ret11:
        print_result("With momXyoung", ret11, ret11[1], ['girl', 'momXgirl', 'mom', 'momXyoung'])
    record('Add momXyoung', ret11)

    # ================================================================
    # CHECK 12: Quantile regression (median)
    # ================================================================
    print("\n" + "-" * 60)
    print("CHECK 12: Quantile regression (median)")

    data12 = df[base_cond].copy()
    x12 = [v for v in base_x if v in data12.columns]
    data12 = data12.dropna(subset=['wtp_std'] + x12 + ['strat']).reset_index(drop=True)
    groups_fe = data12['strat'].values

    y_dm = _demean_by_group(data12['wtp_std'].values.astype(float), groups_fe)
    X_dm = pd.DataFrame(index=data12.index)
    for v in x12:
        X_dm[v] = _demean_by_group(data12[v].values.astype(float), groups_fe)

    try:
        qr = sm.QuantReg(y_dm, X_dm).fit(q=0.5)
        print(f"\n  Median regression (N={len(data12)})")
        for v in ['girl', 'momXgirl', 'mom']:
            if v in qr.params.index:
                coef = qr.params[v]
                se = qr.bse[v]
                pval = qr.pvalues[v]
                stars = '***' if pval < 0.01 else '**' if pval < 0.05 else '*' if pval < 0.1 else ''
                print(f"    {v:15s}  coef={coef:8.3f}  se={se:.3f}  p={pval:.3f} {stars}")
    except Exception as e:
        print(f"  Quantile regression failed: {e}")

    # ================================================================
    # SUMMARY TABLE
    # ================================================================
    print("\n" + "=" * 60)
    print("ROBUSTNESS SUMMARY")
    print("=" * 60)

    print(f"\n{'Check':30s} {'girl':>10s} {'momXgirl':>10s} {'mom':>10s} {'N':>8s}")
    print("-" * 70)
    for r in results_summary:
        girl_str = f"{r['girl']:.3f}" if not np.isnan(r.get('girl', np.nan)) else ''
        mg_str = f"{r['momXgirl']:.3f}" if not np.isnan(r.get('momXgirl', np.nan)) else ''
        mom_str = f"{r['mom']:.3f}" if not np.isnan(r.get('mom', np.nan)) else ''

        # Add stars
        if 'girl_p' in r and not np.isnan(r.get('girl_p', np.nan)):
            s = '***' if r['girl_p'] < 0.01 else '**' if r['girl_p'] < 0.05 else '*' if r['girl_p'] < 0.1 else ''
            girl_str += s
        if 'momXgirl_p' in r and not np.isnan(r.get('momXgirl_p', np.nan)):
            s = '***' if r['momXgirl_p'] < 0.01 else '**' if r['momXgirl_p'] < 0.05 else '*' if r['momXgirl_p'] < 0.1 else ''
            mg_str += s

        print(f"{r['check']:30s} {girl_str:>10s} {mg_str:>10s} {mom_str:>10s} {r['N']:>8d}")

    print(f"\n  Published baseline:          girl=-0.102***  momXgirl=0.131***  mom=-0.095***")


if __name__ == '__main__':
    main()
