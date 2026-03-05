"""
Replication utilities for paper 112786
Rosenzweig & Udry (2014) - Rainfall Forecasts, Weather and Wages
"""
import os
import numpy as np
import pandas as pd
import statsmodels.api as sm

# Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, '..', '112786-V1', 'rosenzweig_data_and_programs')

# Data files
HARVEST_FILE = os.path.join(DATA_DIR, 'harvestwage.dta')
PLANTING_FILE = os.path.join(DATA_DIR, 'plantingwage.dta')
MIGRATION_FILE = os.path.join(DATA_DIR, 'earlymig.dta')


def load_data():
    """Load all three datasets."""
    harvest = pd.read_stata(HARVEST_FILE, convert_categoricals=False)
    planting = pd.read_stata(PLANTING_FILE, convert_categoricals=False)
    migration = pd.read_stata(MIGRATION_FILE, convert_categoricals=False)
    return harvest, planting, migration


def run_areg(df, yvar, xvars, absorb):
    """
    Replicate Stata's areg: OLS with absorbed fixed effects.
    Uses FWL (demeaning by group) with proper DOF correction.
    Stata areg without ,r uses conventional SEs with DOF = N - K - n_groups.
    Returns dict with coefficients, standard errors, t-stats, N, n_groups.
    """
    data = df[list(dict.fromkeys([yvar] + xvars + [absorb]))].dropna()
    y = data[yvar].astype(float)
    X = data[xvars].astype(float)
    groups = data[absorb]

    n = len(data)
    k = len(xvars)
    n_groups = groups.nunique()

    # Demean by group (FWL)
    group_means_y = y.groupby(groups).transform('mean')
    y_dm = y - group_means_y

    X_dm = X.copy()
    for col in xvars:
        group_means_x = X[col].groupby(groups).transform('mean')
        X_dm[col] = X[col] - group_means_x

    # OLS on demeaned data (no constant)
    XtX = X_dm.values.T @ X_dm.values
    Xty = X_dm.values.T @ y_dm.values
    beta = np.linalg.solve(XtX, Xty)

    # Residuals and DOF-corrected MSE
    resid = y_dm.values - X_dm.values @ beta
    dof = n - k - n_groups  # Stata areg DOF
    mse = (resid @ resid) / dof

    # Standard errors
    var_beta = mse * np.linalg.inv(XtX)
    se = np.sqrt(np.diag(var_beta))
    t_stats = beta / se

    results = {
        'coef': dict(zip(xvars, beta)),
        'se': dict(zip(xvars, se)),
        'tstat': dict(zip(xvars, t_stats)),
        'n': n,
        'n_groups': n_groups,
        'dof': dof,
        'r2_within': 1 - (resid @ resid) / (y_dm.values @ y_dm.values),
    }
    return results


def print_results(results, title=""):
    """Print regression results in a readable format."""
    if title:
        print(f"\n{'='*60}")
        print(title)
        print(f"{'='*60}")
    print(f"{'Variable':<30} {'Coef':>12} {'SE':>12} {'|t|':>8}")
    print("-" * 62)
    for var in results['coef']:
        coef = results['coef'][var]
        se = results['se'][var]
        t = abs(results['tstat'][var])
        print(f"{var:<30} {coef:>12.6f} {se:>12.6f} {t:>8.2f}")
    print(f"\nN = {results['n']}, Groups = {results['n_groups']}, DOF = {results['dof']}")
    print(f"Within R² = {results['r2_within']:.4f}")
