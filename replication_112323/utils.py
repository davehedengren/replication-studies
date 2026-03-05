"""
Shared utilities for replication of Samaniego (2008)
"Entry, Exit and Investment-Specific Technical Change"
Paper ID: 112323
"""
import os
import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats

# ── Paths ──────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, '..', '112323-V1')
OUTPUT_DIR = BASE_DIR

INDUSTRY_FILE = os.path.join(DATA_DIR, 'AER20071415r3industryvariables.dta')
DIFFINDIFF_FILE = os.path.join(DATA_DIR, 'AER20071414r3diffindiff.dta')

# ── Variable lists ─────────────────────────────────────
OUTCOMES = ['turnover', 'entry', 'exit']

# Interaction variables in diff-in-diff dataset
# DLLS entry cost interactions
DLLS_VARS = {
    '87st': 'etcdlls87st',   # ISTC 87-97 incl structures × DLLS
    '87eq': 'etcdlls',       # ISTC 87-97 excl structures × DLLS
    '00':   'etcdlls00',     # ISTC 47-00 excl structures × DLLS
}
# World Bank entry cost interactions
WB_VARS = {
    '87st': 'etcwb87st',     # ISTC 87-97 incl structures × WB
    '87eq': 'etcwb',         # ISTC 87-97 excl structures × WB
    '00':   'etcwb00',       # ISTC 47-00 excl structures × WB
}
# Lumpiness interactions
LUMPY_VARS = {
    'dlls': 'lumpydlls',
    'wb':   'lumpywb',
}

# ── Data loading ───────────────────────────────────────
def load_industry_data():
    """Load industry-level variables (41 industries)."""
    return pd.read_stata(INDUSTRY_FILE, convert_categoricals=False)


def load_did_data():
    """Load diff-in-diff panel data (18 countries × 41 industries)."""
    return pd.read_stata(DIFFINDIFF_FILE, convert_categoricals=False)


# ── Regression helpers ─────────────────────────────────
def run_areg_robust(df, yvar, xvar, absorb_var='c', fe_var='i'):
    """
    Replicate Stata: reg y x i.c i.i, robust

    With both country and industry FE, HC1 robust SEs.
    Returns dict with coefficient, p-value, N, R².
    """
    temp = df[[yvar, xvar, absorb_var, fe_var]].dropna().copy()

    # Create dummies for both FEs (drop_first=True for identification)
    c_dummies = pd.get_dummies(temp[absorb_var], prefix='c', drop_first=True).astype(float)
    i_dummies = pd.get_dummies(temp[fe_var], prefix='i', drop_first=True).astype(float)

    X = pd.concat([temp[[xvar]].reset_index(drop=True),
                   c_dummies.reset_index(drop=True),
                   i_dummies.reset_index(drop=True)], axis=1)
    X = sm.add_constant(X)
    y = temp[yvar].reset_index(drop=True)

    model = sm.OLS(y, X).fit(cov_type='HC1')

    coef = model.params[xvar]
    pval = model.pvalues[xvar]

    return {
        'coef': coef,
        'pval': pval,
        'N': int(model.nobs),
        'R2': model.rsquared,
        'se': model.bse[xvar],
        'model': model,
    }


def run_areg_cluster(df, yvar, xvar, absorb_var='c', cluster_var='i'):
    """
    Replicate Stata: areg y x, absorb(c) cluster(i)

    Country FE absorbed, clustered SEs by industry.
    NOTE: This does NOT include industry FEs (per README command).
    """
    temp = df[[yvar, xvar, absorb_var, cluster_var]].dropna().copy()

    c_dummies = pd.get_dummies(temp[absorb_var], prefix='c', drop_first=True).astype(float)

    X = pd.concat([temp[[xvar]].reset_index(drop=True),
                   c_dummies.reset_index(drop=True)], axis=1)
    X = sm.add_constant(X)
    y = temp[yvar].reset_index(drop=True)
    groups = temp[cluster_var].reset_index(drop=True)

    model = sm.OLS(y, X).fit(cov_type='cluster', cov_kwds={'groups': groups})

    coef = model.params[xvar]
    pval = model.pvalues[xvar]

    return {
        'coef': coef,
        'pval': pval,
        'N': int(model.nobs),
        'R2': model.rsquared,
        'se': model.bse[xvar],
        'model': model,
    }
