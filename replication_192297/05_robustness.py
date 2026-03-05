"""
05_robustness.py: Robustness checks for replication of
Bryan, Karlan & Osman (AER, 2024) "Big Loans to Small Businesses"
"""

import os, sys
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
from utils import (MERGED_PATH, PENALTY_PATH, ML_GROUP_PATH, CLEAN_DATA_DIR,
                   OUTPUT_DIR, run_areg, winsor, stars)


def load_data():
    merged = pd.read_parquet(MERGED_PATH)
    penalty = pd.read_parquet(PENALTY_PATH)
    return merged, penalty


def prepare_table4_sample(merged, penalty):
    """Prepare the Table 4 analysis sample."""
    d = merged.copy()

    # Merge penalty
    pen_vars = ['client_key', 'totpenalty24', 'value_late24']
    pen_available = [c for c in pen_vars if c in penalty.columns]
    d = d.merge(penalty[pen_available], on='client_key', how='left', suffixes=('', '_pen'))
    for c in ['totpenalty24', 'value_late24']:
        if f'{c}_pen' in d.columns:
            d[c] = d[c].fillna(d[f'{c}_pen'])

    # ML groups
    if os.path.exists(ML_GROUP_PATH):
        ml = pd.read_stata(ML_GROUP_PATH, convert_categoricals=False)
        ml_vars = [c for c in ml.columns if c in ['client_key', 'G1', 'G2', 'G3', 'G4',
                                                     'RF_HTE_S', 'RF_HTE_B', 'iw']]
        d = d.merge(ml[ml_vars], on='client_key', how='left')

    # Sample restrictions
    d = d[d['round'] != 0].copy()
    if 'e_birth_year' in d.columns:
        d = d[d['e_birth_year'].notna()].copy()

    # Winsorize
    for v in ['e_month_profit_all', 'e_month_rev_all', 'e_month_exp_all',
              'e_wagebill_all', 'e_hh_exp', 'e_totvalassets_all']:
        if v in d.columns:
            d[f'{v}_w'] = winsor(d[v], p=0.01)

    if 'loid_bas' not in d.columns and 'loid_baseline' in d.columns:
        d['loid_bas'] = d['loid_baseline']

    return d


def run_robustness(d, label, y_var, x_vars, absorb_var='loid_bas', cluster_var='client_key'):
    """Run a single robustness check and return results."""
    xvars_avail = [v for v in x_vars if v in d.columns]
    res = run_areg(d, y_var, xvars_avail, absorb_var, cluster_var=cluster_var)
    if res is None:
        return None
    results, data = res
    coef = results.params.get('treatment', np.nan)
    se = results.bse.get('treatment', np.nan)
    pval = results.pvalues.get('treatment', np.nan)
    return {'Label': label, 'Coef': coef, 'SE': se, 'p': pval, 'Stars': stars(pval), 'N': len(data)}


def main():
    print("=" * 60)
    print("05_robustness.py: Robustness Checks for 192297-V1")
    print("=" * 60)

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    merged, penalty = load_data()
    d = prepare_table4_sample(merged, penalty)

    base_xvars = ['treatment', 'b_monthly_profits_w', 'mis_b_monthly_profits_w', 'round']

    results = []

    # (1) Baseline: published specification
    print("\n--- (1) Baseline (Published Spec) ---")
    r = run_robustness(d, 'Baseline (published)', 'e_month_profit_all_w', base_xvars)
    if r: results.append(r)
    print(f"  Profits ATE: {r['Coef']:.1f} ({r['SE']:.1f}){r['Stars']}, N={r['N']}")

    # (2) Without winsorization
    print("\n--- (2) No Winsorization ---")
    r = run_robustness(d, 'No winsorization', 'e_month_profit_all', base_xvars)
    if r: results.append(r)
    print(f"  Profits ATE: {r['Coef']:.1f} ({r['SE']:.1f}){r['Stars']}, N={r['N']}")

    # (3) Winsorize at 5%
    print("\n--- (3) Winsorize at 5% ---")
    d['e_month_profit_all_w5'] = winsor(d['e_month_profit_all'], p=0.05)
    r = run_robustness(d, 'Winsorize 5%', 'e_month_profit_all_w5', base_xvars)
    if r: results.append(r)
    print(f"  Profits ATE: {r['Coef']:.1f} ({r['SE']:.1f}){r['Stars']}, N={r['N']}")

    # (4) Trim top/bottom 1%
    print("\n--- (4) Trim 1% ---")
    p01 = d['e_month_profit_all'].quantile(0.01)
    p99 = d['e_month_profit_all'].quantile(0.99)
    d_trim = d[(d['e_month_profit_all'] >= p01) & (d['e_month_profit_all'] <= p99)]
    r = run_robustness(d_trim, 'Trim 1%', 'e_month_profit_all', base_xvars)
    if r: results.append(r)
    print(f"  Profits ATE: {r['Coef']:.1f} ({r['SE']:.1f}){r['Stars']}, N={r['N']}")

    # (5) IHS transformation
    print("\n--- (5) IHS Transform ---")
    d['ihs_profit'] = np.arcsinh(d['e_month_profit_all'])
    r = run_robustness(d, 'IHS(profits)', 'ihs_profit', base_xvars)
    if r: results.append(r)
    print(f"  Profits ATE (IHS): {r['Coef']:.3f} ({r['SE']:.3f}){r['Stars']}, N={r['N']}")

    # (6) Log transform (positive profits only)
    print("\n--- (6) Log Transform (positive only) ---")
    d_pos = d[d['e_month_profit_all'] > 0].copy()
    d_pos['log_profit'] = np.log(d_pos['e_month_profit_all'])
    r = run_robustness(d_pos, 'Log(profits)', 'log_profit', base_xvars)
    if r: results.append(r)
    print(f"  Profits ATE (log): {r['Coef']:.3f} ({r['SE']:.3f}){r['Stars']}, N={r['N']}")

    # (7) Without baseline controls
    print("\n--- (7) No Baseline Controls ---")
    r = run_robustness(d, 'No BL controls', 'e_month_profit_all_w',
                       ['treatment', 'round'])
    if r: results.append(r)
    print(f"  Profits ATE: {r['Coef']:.1f} ({r['SE']:.1f}){r['Stars']}, N={r['N']}")

    # (8) Round 2 only
    print("\n--- (8) Round 2 Only (Long Run) ---")
    d_r2 = d[d['round'] == 2]
    r = run_robustness(d_r2, 'Round 2 only', 'e_month_profit_all_w',
                       ['treatment', 'b_monthly_profits_w', 'mis_b_monthly_profits_w'])
    if r: results.append(r)
    print(f"  Profits ATE: {r['Coef']:.1f} ({r['SE']:.1f}){r['Stars']}, N={r['N']}")

    # (9) Round 1 only
    print("\n--- (9) Round 1 Only (Short Run) ---")
    d_r1 = d[d['round'] == 1]
    r = run_robustness(d_r1, 'Round 1 only', 'e_month_profit_all_w',
                       ['treatment', 'b_monthly_profits_w', 'mis_b_monthly_profits_w'])
    if r: results.append(r)
    print(f"  Profits ATE: {r['Coef']:.1f} ({r['SE']:.1f}){r['Stars']}, N={r['N']}")

    # (10) Robust SE instead of clustered
    print("\n--- (10) HC1 SE (not clustered) ---")
    r = run_robustness(d, 'HC1 SE', 'e_month_profit_all_w', base_xvars,
                       cluster_var=None)
    if r: results.append(r)
    print(f"  Profits ATE: {r['Coef']:.1f} ({r['SE']:.1f}){r['Stars']}, N={r['N']}")

    # (11) Fix the profit double-counting bug
    print("\n--- (11) Fixed Profit (No Double-Count) ---")
    prof_cols_fixed = [f'e_month_profit_{x}' for x in range(1, 6) if f'e_month_profit_{x}' in d.columns]
    d['e_month_profit_all_fixed'] = d[prof_cols_fixed].fillna(0).sum(axis=1)
    d.loc[d.get('e_month_profit_1', pd.Series()).isna(), 'e_month_profit_all_fixed'] = np.nan
    d['e_month_profit_all_fixed_w'] = winsor(d['e_month_profit_all_fixed'], p=0.01)
    r = run_robustness(d, 'Fixed profit (no double-count)', 'e_month_profit_all_fixed_w', base_xvars)
    if r: results.append(r)
    print(f"  Profits ATE: {r['Coef']:.1f} ({r['SE']:.1f}){r['Stars']}, N={r['N']}")

    # (12) Female subsample
    print("\n--- (12) Female Subsample ---")
    if 'b_gender' in d.columns:
        d_f = d[d['b_gender'] == 1]
        r = run_robustness(d_f, 'Female only', 'e_month_profit_all_w', base_xvars)
        if r: results.append(r)
        print(f"  Profits ATE: {r['Coef']:.1f} ({r['SE']:.1f}){r['Stars']}, N={r['N']}")

    # (13) Male subsample
    print("\n--- (13) Male Subsample ---")
    if 'b_gender' in d.columns:
        d_m = d[d['b_gender'] == 0]
        r = run_robustness(d_m, 'Male only', 'e_month_profit_all_w', base_xvars)
        if r: results.append(r)
        print(f"  Profits ATE: {r['Coef']:.1f} ({r['SE']:.1f}){r['Stars']}, N={r['N']}")

    # (14) Quantile regression (median)
    print("\n--- (14) Quantile Regression (Median) ---")
    try:
        import statsmodels.api as sm
        from utils import _demean_by_group
        qd = d.dropna(subset=['e_month_profit_all_w'] + base_xvars + ['loid_bas']).copy()
        y_dm = _demean_by_group(qd['e_month_profit_all_w'].values.astype(float),
                                qd['loid_bas'].values)
        X_dm = pd.DataFrame(index=qd.index)
        for v in base_xvars:
            if v in qd.columns:
                X_dm[v] = _demean_by_group(qd[v].values.astype(float), qd['loid_bas'].values)
        qr = sm.QuantReg(y_dm, X_dm).fit(q=0.5)
        coef = qr.params.get('treatment', np.nan)
        se = qr.bse.get('treatment', np.nan)
        pval = qr.pvalues.get('treatment', np.nan)
        results.append({'Label': 'Median regression', 'Coef': coef, 'SE': se,
                        'p': pval, 'Stars': stars(pval), 'N': len(qd)})
        print(f"  Profits median: {coef:.1f} ({se:.1f}){stars(pval)}, N={len(qd)}")
    except Exception as e:
        print(f"  Quantile regression failed: {e}")

    # Summary table
    print("\n" + "=" * 60)
    print("ROBUSTNESS SUMMARY")
    print("=" * 60)
    tab = pd.DataFrame(results)
    print(tab.to_string(index=False, float_format='{:.3f}'.format))

    return tab


if __name__ == '__main__':
    main()
