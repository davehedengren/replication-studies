"""
02_tables.py — Reproduce main tables for Andreolli & Surico (2025).

Paper: "Shock Sizes and the Marginal Propensity to Consume"
"""

import sys, os
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import norm
from utils import *

print('=' * 70)
print('02_tables.py — Table replication')
print('=' * 70)

df = load_analysis_data()

# ══════════════════════════════════════════════════════════════════════
# TABLE 1: Summary Statistics
# ══════════════════════════════════════════════════════════════════════
print('\n-- Table 1: Summary Statistics --')

sample10 = df[(df['marksample10'] == 1)].copy()

# For 2010: use marksample10 observations in the 2010 year row
# But marksample10 is set on anno==2010 rows, so these ARE the 2010 obs
stats_vars = {
    'r_cash': 'Cash-on-hand',
    'r_y': 'Income',
    'r_af': 'Financial assets',
    'male': 'Male',
    'married': 'Married',
    'educ': 'Education (years)',
    'ncomp': 'Family size',
    'south': 'South',
    'unempl': 'Unemployed',
    'mpc': 'MPC',
}

print(f'\n  {"Variable":<25} {"Mean":>8} {"p10":>8} {"p25":>8} {"p50":>8} {"p75":>8} {"p90":>8}')
print('  ' + '-' * 73)

for var, label in stats_vars.items():
    if var in sample10.columns:
        s = sample10[var].dropna()
        if len(s) > 0:
            print(f'  {label:<25} {s.mean():>8.2f} {s.quantile(0.1):>8.2f} '
                  f'{s.quantile(0.25):>8.2f} {s.quantile(0.5):>8.2f} '
                  f'{s.quantile(0.75):>8.2f} {s.quantile(0.9):>8.2f}')

# Additional 2012-specific variables
print(f'\n  Additional (from 2012 MPC):')
for var, label in [('dmpc_1012_in2010', 'MPC difference (S-L)'),
                    ('eatoutshare_2012_in2010', 'Eating out share')]:
    if var in sample10.columns:
        s = sample10[var].dropna()
        if len(s) > 0:
            print(f'  {label:<25} {s.mean():>8.3f} {s.quantile(0.1):>8.3f} '
                  f'{s.quantile(0.25):>8.3f} {s.quantile(0.5):>8.3f} '
                  f'{s.quantile(0.75):>8.3f} {s.quantile(0.9):>8.3f}')

print(f'  N = {len(sample10):,}')


# ══════════════════════════════════════════════════════════════════════
# TABLE 2: Baseline Tobit Regressions (MPC on cash-on-hand deciles)
# ══════════════════════════════════════════════════════════════════════
print('\n' + '=' * 70)
print('-- Table 2: Baseline Tobit Regressions --')
print('=' * 70)


def tobit_loglik(params, y, X, lower=0, upper=1):
    """Log-likelihood for doubly censored Tobit model."""
    beta = params[:-1]
    sigma = np.exp(params[-1])  # ensure positive
    xb = X @ beta
    ll = 0
    # Censored at lower
    mask_lo = y <= lower
    if mask_lo.sum() > 0:
        ll += np.sum(np.log(np.clip(norm.cdf((lower - xb[mask_lo]) / sigma), 1e-300, 1)))
    # Censored at upper
    mask_hi = y >= upper
    if mask_hi.sum() > 0:
        ll += np.sum(np.log(np.clip(1 - norm.cdf((upper - xb[mask_hi]) / sigma), 1e-300, 1)))
    # Interior
    mask_int = (~mask_lo) & (~mask_hi)
    if mask_int.sum() > 0:
        ll += np.sum(-0.5 * np.log(2 * np.pi) - np.log(sigma) -
                     0.5 * ((y[mask_int] - xb[mask_int]) / sigma) ** 2)
    return -ll


def run_tobit(y, X, lower=0, upper=1):
    """Run Tobit regression. Returns (coefs, std_errors, sigma, N)."""
    n, k = X.shape
    # OLS starting values
    beta0 = np.linalg.lstsq(X, y, rcond=None)[0]
    resid = y - X @ beta0
    sigma0 = np.log(np.std(resid))
    params0 = np.append(beta0, sigma0)

    result = minimize(tobit_loglik, params0, args=(y, X, lower, upper),
                      method='L-BFGS-B', options={'maxiter': 5000})

    beta = result.x[:-1]
    sigma = np.exp(result.x[-1])

    # Numerical Hessian for standard errors
    from scipy.optimize import approx_fprime
    eps = 1e-5
    hess = np.zeros((len(result.x), len(result.x)))
    for i in range(len(result.x)):
        def grad_i(p):
            return approx_fprime(p, lambda p: tobit_loglik(p, y, X, lower, upper), eps)[i]
        hess[i] = approx_fprime(result.x, grad_i, eps)

    try:
        cov = np.linalg.inv(hess)
        se = np.sqrt(np.clip(np.diag(cov), 0, None))[:k]
    except np.linalg.LinAlgError:
        se = np.full(k, np.nan)

    return beta, se, sigma, n


# Prepare data
sample = df[(df['marksample10'] == 1)].copy()
decile_cols = [f'q10cash_d{i}' for i in range(1, 11)]
demo_cols = ['age30', 'age45', 'age60', 'male', 'married', 'educ', 'ncomp', 'south', 'unempl']
demo_demean_cols = [f'{c}_demean' for c in demo_cols]
city_cols = ['acomd1', 'acomd2', 'acomd3']
city_demean_cols = [f'{c}_demean' for c in city_cols]

# Demean demographic controls
dm = demean(sample, demo_cols + city_cols, condition=pd.Series(True, index=sample.index))
for col in dm.columns:
    sample[col] = dm[col].values

# Run Tobit regressions: columns (1)-(3) without controls, (4)-(6) with controls
specs = [
    ('Small (no ctrl)', 'mpc_2010', decile_cols, 0, 1),
    ('Large (no ctrl)', 'mpc_2012_in2010', decile_cols, 0, 1),
    ('Diff (no ctrl)', 'dmpc_1012_in2010', decile_cols, -1, 1),
    ('Small (ctrl)', 'mpc_2010', decile_cols + demo_demean_cols + city_demean_cols, 0, 1),
    ('Large (ctrl)', 'mpc_2012_in2010', decile_cols + demo_demean_cols + city_demean_cols, 0, 1),
    ('Diff (ctrl)', 'dmpc_1012_in2010', decile_cols + demo_demean_cols + city_demean_cols + ['Dl_cash_1210_demean'] if 'Dl_cash_1210' in sample.columns else decile_cols + demo_demean_cols + city_demean_cols, -1, 1),
]

# Add Dl_cash_1210 demeaned
if 'Dl_cash_1210' in sample.columns:
    dm2 = demean(sample, ['Dl_cash_1210'], condition=pd.Series(True, index=sample.index))
    sample['Dl_cash_1210_demean'] = dm2['Dl_cash_1210_demean'].values

tobit_results = {}
for label, yvar, xvars, lo, hi in specs:
    valid_xvars = [c for c in xvars if c in sample.columns]
    mask = sample[yvar].notna()
    for c in valid_xvars:
        mask = mask & sample[c].notna()
    s = sample[mask]
    if len(s) < 20:
        print(f'  {label}: insufficient data ({len(s)} obs)')
        continue

    y = s[yvar].values
    X = s[valid_xvars].values

    beta, se, sigma, n = run_tobit(y, X, lower=lo, upper=hi)
    tobit_results[label] = (beta, se, sigma, n, valid_xvars)

# Display results (decile coefficients only)
print(f'\n  {"Decile":<15}', end='')
for label in tobit_results:
    print(f' {label:>15}', end='')
print()
print('  ' + '-' * (15 + 16 * len(tobit_results)))

for i in range(10):
    col_name = f'q10cash_d{i+1}'
    print(f'  {"D" + str(i+1):<15}', end='')
    for label, (beta, se, sigma, n, xvars) in tobit_results.items():
        idx = xvars.index(col_name) if col_name in xvars else -1
        if idx >= 0:
            star = '*' if abs(beta[idx] / se[idx]) > 1.96 else ''
            print(f' {beta[idx]:>14.3f}{star}', end='')
        else:
            print(f' {"":>15}', end='')
    print()
    # Standard errors
    print(f'  {"":>15}', end='')
    for label, (beta, se, sigma, n, xvars) in tobit_results.items():
        idx = xvars.index(col_name) if col_name in xvars else -1
        if idx >= 0:
            print(f' ({se[idx]:>12.3f})', end='')
        else:
            print(f' {"":>15}', end='')
    print()

# N and sigma
print(f'\n  {"N":<15}', end='')
for label, (beta, se, sigma, n, xvars) in tobit_results.items():
    print(f' {n:>15,}', end='')
print()
print(f'  {"Sigma":<15}', end='')
for label, (beta, se, sigma, n, xvars) in tobit_results.items():
    print(f' {sigma:>15.3f}', end='')
print()


# ══════════════════════════════════════════════════════════════════════
# TABLE 2 (OLS version - Table D.6): Baseline OLS regressions
# ══════════════════════════════════════════════════════════════════════
print('\n' + '=' * 70)
print('-- Table D.6: Baseline OLS Regressions --')
print('=' * 70)

import statsmodels.api as sm

ols_results = {}
for label, yvar, xvars, lo, hi in specs[:3]:  # No-controls only for OLS
    valid_xvars = [c for c in xvars if c in sample.columns]
    mask = sample[yvar].notna()
    for c in valid_xvars:
        mask = mask & sample[c].notna()
    s = sample[mask]

    y = s[yvar].values
    X = s[valid_xvars].values

    model = sm.OLS(y, X).fit(cov_type='HC1')
    ols_results[label] = model

print(f'\n  {"Decile":<15}', end='')
for label in ols_results:
    print(f' {label:>15}', end='')
print()
print('  ' + '-' * (15 + 16 * len(ols_results)))

for i in range(10):
    col_name = f'q10cash_d{i+1}'
    print(f'  {"D" + str(i+1):<15}', end='')
    for label, model in ols_results.items():
        idx = i  # decile cols are first
        b = model.params[idx]
        s = model.bse[idx]
        star = '*' if model.pvalues[idx] < 0.05 else ''
        print(f' {b:>14.3f}{star}', end='')
    print()
    print(f'  {"":>15}', end='')
    for label, model in ols_results.items():
        s = model.bse[i]
        print(f' ({s:>12.3f})', end='')
    print()

print(f'\n  {"N":<15}', end='')
for label, model in ols_results.items():
    print(f' {int(model.nobs):>15,}', end='')
print()
print(f'  {"R²":<15}', end='')
for label, model in ols_results.items():
    print(f' {model.rsquared:>15.3f}', end='')
print()


# ══════════════════════════════════════════════════════════════════════
# TABLE 4: Fiscal Experiments
# ══════════════════════════════════════════════════════════════════════
print('\n' + '=' * 70)
print('-- Table 4: Fiscal Experiments --')
print('=' * 70)

sample = df[(df['marksample10'] == 1)].copy()

# Need peso (survey weights) and consumption
if 'peso' in sample.columns and 'c' in sample.columns:
    c_to_GDP = 0.61011094
    totconsbase = (sample['c'] * sample['peso']).sum()

    # MPC-weighted spending
    sample['spendoutmpc_2010'] = sample['mpc_2010'] * sample['y'] / 12
    if 'mpc_2012_in2010' in sample.columns:
        sample['spendoutmpc_2012'] = sample['mpc_2012_in2010'] * sample['y']

    # Sort by cash for cumulative targeting
    sample_sorted = sample.sort_values('cash').copy()
    sample_sorted['y_peso'] = sample_sorted['y'] * sample_sorted['peso']
    sample_sorted['cost_cumul_up'] = sample_sorted['y_peso'].cumsum()

    sample_sorted_desc = sample.sort_values('cash', ascending=False).copy()
    sample_sorted_desc['y_peso'] = sample_sorted_desc['y'] * sample_sorted_desc['peso']
    sample_sorted_desc['cost_cumul_from'] = sample_sorted_desc['y_peso'].cumsum()

    print(f'\n  {"Panel":<10} {"Policy":<40} {"Avg Transfer":>13} {"MPC":>8} {"Agg C effect":>14}')
    print('  ' + '-' * 87)

    for pct_gdp in [0.5, 1, 2]:
        cost_policy = pct_gdp / c_to_GDP * totconsbase / 100

        # One month income to bottom X%
        eligible = sample_sorted[sample_sorted['cost_cumul_up'] / 12 <= cost_policy]
        if len(eligible) > 0:
            avg_transfer = (eligible['y'] * eligible['peso']).sum() / eligible['peso'].sum() / 12
            avg_spend = (eligible['spendoutmpc_2010'] * eligible['peso']).sum() / eligible['peso'].sum()
            mpc_eff = avg_spend / avg_transfer if avg_transfer > 0 else 0
            agg_effect = (eligible['spendoutmpc_2010'] * eligible['peso']).sum() / totconsbase * 100
            pct_covered = eligible['q10cash'].max() / 10 * 100 if 'q10cash' in eligible.columns else 0
            print(f'  {pct_gdp}% GDP   1m income to bottom {pct_covered:.0f}%          {avg_transfer*1000:>10.0f}€ {mpc_eff:>8.2f} {agg_effect:>13.2f}%')

        # One year income to bottom X%
        eligible_large = sample_sorted[sample_sorted['cost_cumul_up'] <= cost_policy]
        if len(eligible_large) > 0 and 'spendoutmpc_2012' in sample_sorted.columns:
            avg_transfer_l = (eligible_large['y'] * eligible_large['peso']).sum() / eligible_large['peso'].sum()
            avg_spend_l = (eligible_large['spendoutmpc_2012'] * eligible_large['peso']).sum() / eligible_large['peso'].sum()
            mpc_eff_l = avg_spend_l / avg_transfer_l if avg_transfer_l > 0 else 0
            agg_effect_l = (eligible_large['spendoutmpc_2012'] * eligible_large['peso']).sum() / totconsbase * 100
            print(f'           1y income to bottom (smaller group)   {avg_transfer_l*1000:>10.0f}€ {mpc_eff_l:>8.2f} {agg_effect_l:>13.2f}%')
else:
    print('  (peso or c not available for fiscal experiment)')


print('\n' + '=' * 70)
print('02_tables.py — DONE')
print('=' * 70)
