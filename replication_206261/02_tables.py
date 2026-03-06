"""
02_tables.py - Replicate Tables 1-4 for paper 206261
Burchardi, Chaney, Hassan, Tarquinio, Terry -
  Immigration, Innovation, and Growth
"""
import pandas as pd
import numpy as np
import statsmodels.api as sm
from scipy import stats as sp_stats
import warnings
warnings.filterwarnings('ignore')

from utils import (OUTPUT, TEMP, run_reghdfe, print_reg,
                   PERIOD_AP1, PERIODS, period_to_year)

# Load final dataset
df = pd.read_pickle(f'{OUTPUT}/IIGFinalData.pkl')
print(f"Dataset: {df.shape}")
print(f"Counties: {df['state_county_code_1990'].nunique()}")
print(f"Time periods: {sorted(df['time'].unique())}")
print(f"States: {df['state_code'].nunique()}")

# ========================================================================
# TABLE 1: SUMMARY STATISTICS
# ========================================================================
print("\n" + "=" * 70)
print("TABLE 1: SUMMARY STATISTICS")
print("=" * 70)

tab1_vars = {
    'immigration_nEuro_d_': 'Non-Euro Immigration (1000s)',
    'deltapop': 'Population Change (1000s)',
    'diffPV_a_W': 'Patent Flow Diff (per 100k)',
    'delta_avg_wage_adj': 'Wage Change (/100)',
    'immigration_25o_': 'Immigration 25+ (1000s)',
    'IhatM1_d': 'Instrument (predicted imm)',
}

print(f"\n{'Variable':<35} {'N':>6} {'Mean':>10} {'SD':>10} {'Min':>10} {'Max':>10}")
print("-" * 85)
for v, label in tab1_vars.items():
    if v in df.columns:
        s = df[v].dropna()
        print(f"{label:<35} {len(s):>6} {s.mean():>10.3f} {s.std():>10.3f} {s.min():>10.3f} {s.max():>10.3f}")

# ========================================================================
# TABLE 3: PATENTS ON IMMIGRATION (OLS and IV)
# ========================================================================
print("\n" + "=" * 70)
print("TABLE 3: PATENTS ON IMMIGRATION")
print("=" * 70)

yvar = 'diffPV_a_W'
endog = 'immigration_nEuro_d_'
instrument = 'IhatM1_d'

if yvar not in df.columns:
    print(f"  WARNING: {yvar} not in data, checking alternatives...")
    alt = [c for c in df.columns if 'diffPV' in c]
    print(f"  Available: {alt}")
    if alt:
        yvar = alt[0]

# --- OLS ---
print("\n--- Table 3: OLS ---")

# Column 1: Time + State FE
if yvar in df.columns and endog in df.columns:
    res_ols1 = run_reghdfe(df, yvar, [endog], ['time', 'state_factor'], ['state_factor'])
    print_reg(res_ols1, "OLS: Time + State FE")

    # Column 2: State-Time trends (add state_time interaction)
    df['state_time'] = df['state_factor'].astype(str) + '_' + df['time'].astype(str)
    res_ols2 = run_reghdfe(df, yvar, [endog], ['time', 'state_factor', 'state_time'], ['state_factor'])
    print_reg(res_ols2, "OLS: Time + State + State×Time FE")

    # Column 3: County FE
    res_ols3 = run_reghdfe(df, yvar, [endog], ['time', 'state_county_code_1990'], ['state_factor'])
    print_reg(res_ols3, "OLS: Time + County FE")

# --- First Stage ---
print("\n--- Table 3: First Stage ---")
if endog in df.columns and instrument in df.columns:
    # Column 1: Time + State FE
    res_fs1 = run_reghdfe(df, endog, [instrument], ['time', 'state_factor'], ['state_factor'])
    print_reg(res_fs1, "First Stage: Time + State FE")

    # F-stat
    if len(res_fs1['t']) > 0:
        f_stat = res_fs1['t'][0]**2
        print(f"  F-stat = {f_stat:.2f}")

# --- IV (2SLS) ---
print("\n--- Table 3: IV (2SLS) ---")
if yvar in df.columns and endog in df.columns and instrument in df.columns:
    # Manual 2SLS: first stage predicted values → second stage
    # First stage
    sub = df[[yvar, endog, instrument, 'time', 'state_factor', 'state_county_code_1990']].dropna().copy()

    # Demean for FE
    from utils import demean_iterative
    fe_vars = ['time', 'state_factor']
    sub_dm = demean_iterative(sub, endog, [instrument], fe_vars)

    # First stage: regress endog on instrument
    y1 = sub_dm[endog].values
    X1 = sub_dm[[instrument]].values
    XtX1 = X1.T @ X1
    beta1 = np.linalg.solve(XtX1, X1.T @ y1)
    endog_hat = X1 @ beta1

    # Second stage: regress outcome on predicted endog
    sub_dm2 = demean_iterative(sub, yvar, [endog], fe_vars)
    y2 = sub_dm2[yvar].values

    # Use predicted values in second stage
    X2 = endog_hat.reshape(-1, 1)
    XtX2 = X2.T @ X2
    beta2 = np.linalg.solve(XtX2, X2.T @ y2)
    resid2 = y2 - sub_dm[endog].values.reshape(-1, 1) @ beta2.reshape(1)  # Use actual endog for residuals

    # Cluster-robust SE
    clusters = sub['state_factor'].values
    unique_cl = np.unique(clusters)
    G = len(unique_cl)
    n = len(y2)
    k = 1
    XtX2_inv = np.linalg.inv(XtX2)

    S = np.zeros((k, k))
    for c in unique_cl:
        mask = clusters == c
        u = (X2[mask].T @ resid2[mask].ravel()).reshape(-1, 1)
        S += u @ u.T
    dof = (G / (G - 1)) * ((n - 1) / (n - k))
    vcov2 = XtX2_inv @ (S * dof) @ XtX2_inv
    se2 = np.sqrt(np.diag(vcov2))
    t2 = beta2 / se2

    print(f"\nIV (2SLS): Time + State FE")
    print("-" * 70)
    print(f"N = {n}")
    print(f"Clusters = {G}")
    sig = '***' if abs(t2[0]) > 2.576 else '**' if abs(t2[0]) > 1.96 else '*' if abs(t2[0]) > 1.645 else ''
    print(f"{'immigration_nEuro_d_':<35} {beta2[0]:>10.4f} {se2[0]:>10.4f} {t2[0]:>8.2f} {sig}")
    print(f"KP F-stat = {(beta1[0]/se2[0])**2:.2f}" if len(beta1) > 0 else "")

    # IV Column 2: County FE
    fe_vars2 = ['time', 'state_county_code_1990']
    sub_dm_c = demean_iterative(sub, endog, [instrument], fe_vars2)
    y1c = sub_dm_c[endog].values
    X1c = sub_dm_c[[instrument]].values
    beta1c = np.linalg.solve(X1c.T @ X1c, X1c.T @ y1c)
    endog_hat_c = X1c @ beta1c

    sub_dm2_c = demean_iterative(sub, yvar, [endog], fe_vars2)
    y2c = sub_dm2_c[yvar].values
    X2c = endog_hat_c.reshape(-1, 1)
    beta2c = np.linalg.solve(X2c.T @ X2c, X2c.T @ y2c)
    resid2c = y2c - sub_dm_c[endog].values.reshape(-1, 1) @ beta2c.reshape(1)

    S_c = np.zeros((1, 1))
    for c in unique_cl:
        mask = clusters == c
        u = (X2c[mask].T @ resid2c[mask].ravel()).reshape(-1, 1)
        S_c += u @ u.T
    vcov2c = np.linalg.inv(X2c.T @ X2c) @ (S_c * dof) @ np.linalg.inv(X2c.T @ X2c)
    se2c = np.sqrt(np.diag(vcov2c))
    t2c = beta2c / se2c

    print(f"\nIV (2SLS): Time + County FE")
    print("-" * 70)
    print(f"N = {n}")
    sig = '***' if abs(t2c[0]) > 2.576 else '**' if abs(t2c[0]) > 1.96 else '*' if abs(t2c[0]) > 1.645 else ''
    print(f"{'immigration_nEuro_d_':<35} {beta2c[0]:>10.4f} {se2c[0]:>10.4f} {t2c[0]:>8.2f} {sig}")

# ========================================================================
# TABLE 4: DYNAMISM, WAGES, AND INEQUALITY (IV)
# ========================================================================
print("\n" + "=" * 70)
print("TABLE 4: DYNAMISM, WAGES, AND INEQUALITY (IV)")
print("=" * 70)

tab4_outcomes = [
    ('delta_avg_wage_adj', 'immigration_25o_', 'IhatM1_d', 'Avg Wage Change'),
]

# Add education-specific wages if available
for edlabel in ['LessHS', 'HS', 'SomeCol', 'BA', 'Grad']:
    v = f'incwage_25oS_{edlabel}_delta'
    if v in df.columns:
        tab4_outcomes.append((v, 'immigration_25o_10y', 'IhatM1_d10y', f'Wage: {edlabel}'))

for yvar, xvar, zvar, label in tab4_outcomes:
    if yvar not in df.columns or xvar not in df.columns or zvar not in df.columns:
        print(f"\n  {label}: missing variables, skipping")
        continue

    sub = df[[yvar, xvar, zvar, 'time', 'state_factor']].dropna()
    if len(sub) < 50:
        print(f"\n  {label}: N={len(sub)}, too few obs")
        continue

    # First stage
    res_fs = run_reghdfe(sub, xvar, [zvar], ['time', 'state_factor'], ['state_factor'])

    # 2SLS
    fe_vars = ['time', 'state_factor']
    sub_dm = demean_iterative(sub, xvar, [zvar], fe_vars)
    y1 = sub_dm[xvar].values
    X1 = sub_dm[[zvar]].values
    beta1 = np.linalg.solve(X1.T @ X1, X1.T @ y1)
    xhat = X1 @ beta1

    sub_dm2 = demean_iterative(sub, yvar, [xvar], fe_vars)
    y2 = sub_dm2[yvar].values
    X2 = xhat.reshape(-1, 1)
    beta2 = np.linalg.solve(X2.T @ X2, X2.T @ y2)
    resid2 = y2 - sub_dm[xvar].values.reshape(-1, 1) @ beta2.reshape(1)

    clusters = sub['state_factor'].values
    unique_cl = np.unique(clusters)
    G = len(unique_cl)
    n = len(y2)
    S = np.zeros((1, 1))
    for c in unique_cl:
        mask = clusters == c
        u = (X2[mask].T @ resid2[mask].ravel()).reshape(-1, 1)
        S += u @ u.T
    dof_adj = (G / (G - 1)) * ((n - 1) / (n - 1))
    vcov2 = np.linalg.inv(X2.T @ X2) @ (S * dof_adj) @ np.linalg.inv(X2.T @ X2)
    se2 = np.sqrt(np.diag(vcov2))
    t2 = beta2 / se2

    sig = '***' if abs(t2[0]) > 2.576 else '**' if abs(t2[0]) > 1.96 else '*' if abs(t2[0]) > 1.645 else ''
    f_stat = res_fs['t'][0]**2 if len(res_fs['t']) > 0 else 0
    print(f"\n  {label:<25} beta={beta2[0]:>8.4f} se={se2[0]:>8.4f} t={t2[0]:>6.2f}{sig}  N={n}  F={f_stat:.1f}")

# ========================================================================
# TABLE 2: FIRST STAGE (if Stage 1 data available)
# ========================================================================
print("\n" + "=" * 70)
print("TABLE 2: FIRST STAGE REGRESSION")
print("=" * 70)

stage1_file = f'{OUTPUT}/IIGData_Stage1.pkl'
try:
    stage1 = pd.read_pickle(stage1_file)
    print(f"Stage 1 data: {stage1.shape}")

    # Find triple interaction columns
    triple_cols = [c for c in stage1.columns if c.startswith('Ahat_X_immihat')]
    print(f"Triple interaction cols: {len(triple_cols)}")

    if triple_cols and 'immigrants_' in stage1.columns:
        # Column 3: with FE (main spec)
        xvars = triple_cols + ['dist', 'distance_lat']
        xvars = [c for c in xvars if c in stage1.columns]
        res_t2 = run_reghdfe(stage1, 'immigrants_', xvars,
                            ['region_country_code', 'county_continent_code', 'time'],
                            'country_code_1990')
        print_reg(res_t2, "Table 2: First Stage (with FE)")

        # F-stat on excluded instruments
        triple_t = [res_t2['t'][res_t2['xvars'].index(c)]**2
                   for c in triple_cols if c in res_t2['xvars']]
        if triple_t:
            print(f"  Joint F on excluded instruments ≈ {np.mean(triple_t):.2f}")

except Exception as e:
    print(f"  Could not load Stage 1 data: {e}")

print("\n" + "=" * 70)
print("REPLICATION COMPLETE")
print("=" * 70)
