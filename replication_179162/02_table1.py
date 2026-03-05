"""
02_table1.py — Table 1: Fathers spend less on girls but mothers do not

Reproduces Table 1 from Dizon-Ross & Jayachandran (2023, AER:Insights).
6 columns varying normalization, fixed effects, and goods included.
"""

import sys, os
import numpy as np
import pandas as pd
import statsmodels.api as sm

sys.path.insert(0, os.path.dirname(__file__))
from utils import ANALYSIS_DATA_PATH, run_areg


def run_table1_regression(df, y_var, x_vars, absorb_var, cluster_var, condition=None):
    """Run one column of Table 1 using demeaning to absorb FE."""
    from utils import _demean_by_group

    data = df.copy()
    if condition is not None:
        data = data[condition].copy()

    data = data.dropna(subset=[y_var] + x_vars + [absorb_var, cluster_var]).copy()
    data = data.reset_index(drop=True)

    if len(data) == 0:
        return None

    groups_fe = data[absorb_var].values
    n_groups = data[absorb_var].nunique()

    # Demean Y and X by absorb group (FWL theorem)
    y_dm = _demean_by_group(data[y_var].values.astype(float), groups_fe)

    X_dm = pd.DataFrame(index=data.index)
    for v in x_vars:
        X_dm[v] = _demean_by_group(data[v].values.astype(float), groups_fe)

    model = sm.OLS(y_dm, X_dm)
    cluster_groups = data[cluster_var].values

    n = len(data)
    k_x = len(x_vars)

    try:
        results = model.fit(
            cov_type='cluster',
            cov_kwds={'groups': cluster_groups, 'df_correction': True},
        )
        results.df_resid = n - k_x - n_groups
    except Exception:
        results = model.fit(cov_type='HC1')
        results.df_resid = n - k_x - n_groups

    # Father-son mean
    fs_mask = (data['mom'] == 0) & (data['girl'] == 0)
    father_son_mean = data.loc[fs_mask, y_var].mean() if fs_mask.any() else np.nan

    return results, data, father_son_mean


def main():
    print("=" * 60)
    print("02_table1.py: Table 1 — Main Results")
    print("=" * 60)

    df = pd.read_parquet(ANALYSIS_DATA_PATH)

    # Get good FE columns
    good_fe_cols = [c for c in df.columns if c.startswith('_G')]

    # Ensure adult_wtp_pct is available for Col 3
    # adult_wtp_pct = cup_wtp/3600 for BL, (jerry_wtp/4000 + poster_wtp/2000)/2 for FU
    if 'adult_wtp_pct' not in df.columns:
        # Compute from raw WTP values
        cup_pct = df.loc[df['good'] == 'cup', ['hhid', 'wtp']].rename(
            columns={'wtp': 'cup_pct'}).drop_duplicates('hhid')
        jerry_pct = df.loc[df['good'] == 'Fjerry', ['hhid', 'wtp']].rename(
            columns={'wtp': 'jerry_pct'}).drop_duplicates('hhid')
        poster_pct = df.loc[df['good'] == 'Fposter', ['hhid', 'wtp']].rename(
            columns={'wtp': 'poster_pct'}).drop_duplicates('hhid')

        df = df.merge(cup_pct, on='hhid', how='left')
        df = df.merge(jerry_pct, on='hhid', how='left')
        df = df.merge(poster_pct, on='hhid', how='left')

        df['adult_wtp_pct'] = np.where(
            df['fu'] == 0, df['cup_pct'],
            (df['jerry_pct'].fillna(0) + df['poster_pct'].fillna(0)) / 2)
        df.loc[(df['fu'] == 1) & df['jerry_pct'].isna() & df['poster_pct'].isna(),
               'adult_wtp_pct'] = np.nan
        df['adult_wtp_pctXfu'] = df['adult_wtp_pct'] * df['fu']
        df.drop(columns=['cup_pct', 'jerry_pct', 'poster_pct'], errors='ignore', inplace=True)

    # Base sample
    base_cond = (df['childgood'] == 1) & (df['toys_bin'] == 0)

    # Column specifications
    columns = {
        'Col 1: SD, Stratum FE, All': {
            'y': 'wtp_std',
            'x': ['girl', 'mom', 'adult_wtp_std', 'adult_wtp_stdXfu', 'fu', 'young'] + good_fe_cols,
            'absorb': 'strat',
            'cond': base_cond,
            'has_momXgirl': False,
        },
        'Col 2: SD, Stratum FE, All (interaction)': {
            'y': 'wtp_std',
            'x': ['girl', 'momXgirl', 'mom', 'adult_wtp_std', 'adult_wtp_stdXfu', 'fu', 'young'] + good_fe_cols,
            'absorb': 'strat',
            'cond': base_cond,
            'has_momXgirl': True,
        },
        'Col 3: Mkt price, Stratum FE, All': {
            'y': 'wtp',
            'x': ['girl', 'momXgirl', 'mom', 'adult_wtp_pct', 'adult_wtp_pctXfu', 'fu', 'young'] + good_fe_cols,
            'absorb': 'strat',
            'cond': base_cond,
            'has_momXgirl': True,
        },
        'Col 4: SD, HH FE, All': {
            'y': 'wtp_std',
            'x': ['girl', 'momXgirl', 'mom', 'adult_wtp_std', 'adult_wtp_stdXfu', 'fu', 'young'] + good_fe_cols,
            'absorb': 'hhid',
            'cond': base_cond,
            'has_momXgirl': True,
        },
        'Col 5: SD, Stratum FE, Incentivized': {
            'y': 'wtp_std',
            'x': ['girl', 'momXgirl', 'mom', 'adult_wtp_std', 'adult_wtp_stdXfu', 'fu', 'young'] + good_fe_cols,
            'absorb': 'strat',
            'cond': base_cond & (df['hypo'] == 0),
            'has_momXgirl': True,
        },
        'Col 6: SD, Stratum FE, Non-incentivized': {
            'y': 'wtp_std',
            'x': ['girl', 'momXgirl', 'mom', 'adult_wtp_std', 'adult_wtp_stdXfu', 'fu', 'young'] + good_fe_cols,
            'absorb': 'strat',
            'cond': base_cond & (df['hypo'] == 1),
            'has_momXgirl': True,
        },
    }

    # Published values for comparison
    published = {
        'Col 2: SD, Stratum FE, All (interaction)': {
            'girl': -0.102, 'momXgirl': 0.131, 'mom': -0.095, 'N': 6673, 'fs_mean': 1.943
        },
    }

    print("\n" + "=" * 100)
    print(f"{'':30s} {'Col 1':>10s} {'Col 2':>10s} {'Col 3':>10s} {'Col 4':>10s} {'Col 5':>10s} {'Col 6':>10s}")
    print("=" * 100)

    results_all = {}
    for col_name, spec in columns.items():
        x_vars_use = [v for v in spec['x'] if v in df.columns]
        ret = run_table1_regression(
            df, spec['y'], x_vars_use, spec['absorb'], 'hhid', spec['cond'])

        if ret is None:
            results_all[col_name] = None
            continue

        res, data, fs_mean = ret
        results_all[col_name] = {
            'results': res, 'data': data, 'fs_mean': fs_mean,
            'N': len(data), 'has_momXgirl': spec['has_momXgirl'],
        }

    # Print key coefficients
    key_vars = ['girl', 'momXgirl', 'mom']
    for var in key_vars:
        row_coef = f"  {var:28s}"
        row_se = f"  {'':28s}"
        for col_name in columns:
            r = results_all.get(col_name)
            if r is None or var not in r['results'].params.index:
                row_coef += f"{'':>10s}"
                row_se += f"{'':>10s}"
            else:
                coef = r['results'].params[var]
                se = r['results'].bse[var]
                pval = r['results'].pvalues[var]
                stars = '***' if pval < 0.01 else '**' if pval < 0.05 else '*' if pval < 0.1 else ''
                row_coef += f"{coef:>9.3f}{stars}"
                row_se += f"  ({se:.3f}) "
        print(row_coef)
        print(row_se)

    print("-" * 100)

    # Print stats
    for stat_name, stat_func in [
        ('N', lambda r: r['N']),
        ('Father-son mean', lambda r: r['fs_mean']),
    ]:
        row = f"  {stat_name:28s}"
        for col_name in columns:
            r = results_all.get(col_name)
            if r is None:
                row += f"{'':>10s}"
            else:
                val = stat_func(r)
                if stat_name == 'N':
                    row += f"{val:>10d}"
                else:
                    row += f"{val:>10.3f}"
        print(row)

    # P-value tests for cols with momXgirl
    row_p1 = f"  {'p: girl+momXgirl=0':28s}"
    row_p2 = f"  {'p: mom+momXgirl=0':28s}"
    for col_name in columns:
        r = results_all.get(col_name)
        if r is None or not r.get('has_momXgirl'):
            row_p1 += f"{'':>10s}"
            row_p2 += f"{'':>10s}"
        else:
            res = r['results']
            if 'girl' in res.params.index and 'momXgirl' in res.params.index:
                # Test girl + momXgirl = 0
                try:
                    t1 = res.t_test('girl + momXgirl = 0')
                    p1 = t1.pvalue.item() if hasattr(t1.pvalue, 'item') else float(t1.pvalue)
                    row_p1 += f"{p1:>10.3f}"
                except Exception:
                    row_p1 += f"{'err':>10s}"
            else:
                row_p1 += f"{'':>10s}"

            if 'mom' in res.params.index and 'momXgirl' in res.params.index:
                try:
                    t2 = res.t_test('mom + momXgirl = 0')
                    p2 = t2.pvalue.item() if hasattr(t2.pvalue, 'item') else float(t2.pvalue)
                    row_p2 += f"{p2:>10.3f}"
                except Exception:
                    row_p2 += f"{'err':>10s}"
            else:
                row_p2 += f"{'':>10s}"
    print(row_p1)
    print(row_p2)

    print("=" * 100)

    # Comparison with published values
    print("\n--- Comparison with Published Table 1 (Column 2) ---")
    pub = published.get('Col 2: SD, Stratum FE, All (interaction)')
    r2 = results_all.get('Col 2: SD, Stratum FE, All (interaction)')
    if pub and r2:
        res = r2['results']
        for var in ['girl', 'momXgirl', 'mom']:
            if var in res.params.index:
                ours = res.params[var]
                theirs = pub[var]
                diff = ours - theirs
                print(f"  {var:15s}  Ours: {ours:8.3f}  Published: {theirs:8.3f}  Diff: {diff:+8.3f}")
        print(f"  {'N':15s}  Ours: {r2['N']:8d}  Published: {pub['N']:8d}")
        print(f"  {'FS mean':15s}  Ours: {r2['fs_mean']:8.3f}  Published: {pub['fs_mean']:8.3f}")


if __name__ == '__main__':
    main()
