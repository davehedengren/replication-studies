"""
05_robustness.py - Robustness checks for paper 206261
Immigration, Innovation, and Growth
"""
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

from utils import (OUTPUT, run_reghdfe, print_reg, demean_iterative,
                   PERIOD_AP1, PERIODS)

df = pd.read_pickle(f'{OUTPUT}/IIGFinalData.pkl')

def run_iv(data, yvar, xvar, zvar, fe_vars, cluster_var):
    """Run 2SLS IV regression."""
    all_vars = [yvar, xvar, zvar] + fe_vars + [cluster_var]
    all_vars = list(dict.fromkeys(all_vars))
    sub = data[all_vars].dropna().copy()
    if len(sub) < 50:
        return None

    # First stage
    sub_dm = demean_iterative(sub, xvar, [zvar], fe_vars)
    y1 = sub_dm[xvar].values
    X1 = sub_dm[[zvar]].values
    beta1 = np.linalg.solve(X1.T @ X1, X1.T @ y1)
    xhat = X1 @ beta1
    f_stat = (beta1[0] / 1)**2  # approximate

    # Second stage
    sub_dm2 = demean_iterative(sub, yvar, [xvar], fe_vars)
    y2 = sub_dm2[yvar].values
    X2 = xhat.reshape(-1, 1)
    beta2 = np.linalg.solve(X2.T @ X2, X2.T @ y2)
    resid2 = y2 - sub_dm[xvar].values.reshape(-1, 1) @ beta2.reshape(1)

    # Cluster SE
    clusters = sub[cluster_var].values
    unique_cl = np.unique(clusters)
    G = len(unique_cl)
    n = len(y2)
    S = np.zeros((1, 1))
    for c in unique_cl:
        mask = clusters == c
        u = (X2[mask].T @ resid2[mask].ravel()).reshape(-1, 1)
        S += u @ u.T
    dof_adj = (G / (G - 1)) * ((n - 1) / (n - 1))
    vcov = np.linalg.inv(X2.T @ X2) @ (S * dof_adj) @ np.linalg.inv(X2.T @ X2)
    se = np.sqrt(np.diag(vcov))
    t = beta2 / se

    # First stage F
    res_fs = run_reghdfe(sub, xvar, [zvar], fe_vars, [cluster_var])
    fs_f = res_fs['t'][0]**2 if len(res_fs['t']) > 0 else 0

    return {'b': beta2[0], 'se': se[0], 't': t[0], 'n': n, 'G': G, 'F': fs_f}

print("=" * 70)
print("ROBUSTNESS CHECKS")
print("=" * 70)

# ========================================================================
# 1. ALTERNATIVE OUTCOMES
# ========================================================================
print("\n--- ROBUSTNESS 1: ALTERNATIVE PATENT MEASURES ---")
patent_vars = [c for c in df.columns if c.startswith('diffPV_') and '_W' in c]
for pv in patent_vars:
    res = run_iv(df, pv, 'immigration_nEuro_d_', 'IhatM1_d',
                ['time', 'state_factor'], 'state_factor')
    if res:
        sig = '***' if abs(res['t']) > 2.576 else '**' if abs(res['t']) > 1.96 else '*' if abs(res['t']) > 1.645 else ''
        print(f"  {pv:<25} b={res['b']:>8.4f} se={res['se']:>8.4f} t={res['t']:>6.2f}{sig} N={res['n']} F={res['F']:.1f}")

# ========================================================================
# 2. ALTERNATIVE FE SPECIFICATIONS
# ========================================================================
print("\n--- ROBUSTNESS 2: ALTERNATIVE FE ---")
yvar = 'diffPV_a_W' if 'diffPV_a_W' in df.columns else [c for c in df.columns if 'diffPV' in c][0] if any('diffPV' in c for c in df.columns) else None
if yvar:
    for fe_label, fe_vars in [
        ("Time + State", ['time', 'state_factor']),
        ("Time + County", ['time', 'state_county_code_1990']),
    ]:
        res = run_iv(df, yvar, 'immigration_nEuro_d_', 'IhatM1_d', fe_vars, 'state_factor')
        if res:
            sig = '***' if abs(res['t']) > 2.576 else '**' if abs(res['t']) > 1.96 else '*' if abs(res['t']) > 1.645 else ''
            print(f"  FE: {fe_label:<20} b={res['b']:>8.4f} se={res['se']:>8.4f} t={res['t']:>6.2f}{sig} N={res['n']} F={res['F']:.1f}")

# ========================================================================
# 3. SAMPLE RESTRICTIONS
# ========================================================================
print("\n--- ROBUSTNESS 3: SAMPLE RESTRICTIONS ---")
if yvar:
    # Drop each period
    for drop_t in sorted(df['time'].unique()):
        sub = df[df['time'] != drop_t]
        res = run_iv(sub, yvar, 'immigration_nEuro_d_', 'IhatM1_d',
                    ['time', 'state_factor'], 'state_factor')
        if res:
            sig = '***' if abs(res['t']) > 2.576 else '**' if abs(res['t']) > 1.96 else '*' if abs(res['t']) > 1.645 else ''
            print(f"  Drop t={drop_t:<4} b={res['b']:>8.4f} se={res['se']:>8.4f} t={res['t']:>6.2f}{sig} N={res['n']}")

# ========================================================================
# 4. OLS vs IV COMPARISON
# ========================================================================
print("\n--- ROBUSTNESS 4: OLS vs IV COMPARISON ---")
if yvar:
    # OLS
    res_ols = run_reghdfe(df, yvar, ['immigration_nEuro_d_'],
                         ['time', 'state_factor'], ['state_factor'])
    if res_ols['n'] > 0:
        print(f"  OLS:  b={res_ols['b'][0]:>8.4f} se={res_ols['se'][0]:>8.4f} t={res_ols['t'][0]:>6.2f} N={res_ols['n']}")

    # IV
    res_iv = run_iv(df, yvar, 'immigration_nEuro_d_', 'IhatM1_d',
                   ['time', 'state_factor'], 'state_factor')
    if res_iv:
        print(f"  IV:   b={res_iv['b']:>8.4f} se={res_iv['se']:>8.4f} t={res_iv['t']:>6.2f} N={res_iv['n']} F={res_iv['F']:.1f}")

print("\n" + "=" * 70)
print("ROBUSTNESS CHECKS COMPLETE")
print("=" * 70)
