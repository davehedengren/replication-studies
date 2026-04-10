"""
utils.py — Paths, data loaders, and helpers for 113192-V1.

Paper: "Disrupting Education? Experimental Evidence on Technology-Aided
       Instruction in India"
Authors: Muralidharan, Singh, Ganimian (2019), AER
"""

import os
import numpy as np
import pandas as pd
import statsmodels.api as sm

# ── Paths ──────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(os.path.dirname(BASE_DIR), '113192-V1', 'data')
OUTPUT_DIR = os.path.join(BASE_DIR, 'output')
os.makedirs(OUTPUT_DIR, exist_ok=True)

STRATA_COL = 'strata'
TREAT_COL = 'treat'

# ── Data loaders ───────────────────────────────────────────────────────
def load_wide():
    """Main analysis file: 619 baseline students, wide format."""
    return pd.read_stata(os.path.join(DATA_DIR, 'ms_blel_jpal_wide.dta'),
                         convert_categoricals=False)

def load_long():
    return pd.read_stata(os.path.join(DATA_DIR, 'ms_blel_jpal_long.dta'),
                         convert_categoricals=False)

def load_attendance():
    """One row per treatment-group student (N=313) with att_tot = days attended."""
    return pd.read_stata(os.path.join(DATA_DIR, 'ms_ei.dta'),
                         convert_categoricals=False)

def load_school_results():
    return pd.read_stata(os.path.join(DATA_DIR, 'sc_results.dta'),
                         convert_categoricals=False)

def load_hh_survey():
    return pd.read_stata(os.path.join(DATA_DIR, 'hh_survey.dta'),
                         convert_categoricals=False)

# ── Normalization ──────────────────────────────────────────────────────
def normalize_to_baseline(df, col_baseline, col_endline=None):
    """Z-score using baseline mean/SD (control-group baseline is cleanest;
    we use full baseline since paper reports ITT-standardized effects)."""
    mu = df[col_baseline].mean()
    sd = df[col_baseline].std()
    df[col_baseline + '_z'] = (df[col_baseline] - mu) / sd
    if col_endline:
        df[col_endline + '_z'] = (df[col_endline] - mu) / sd
    return df, mu, sd

# ── Regression helper ─────────────────────────────────────────────────
def ols_with_strata(df, y, x_cols, strata_col='strata', cov_type='HC1'):
    """OLS with strata fixed effects and HC1 (Stata-robust equivalent) SEs."""
    d = df.dropna(subset=[y] + x_cols + [strata_col]).copy()
    d[strata_col] = d[strata_col].astype(int)
    fe = pd.get_dummies(d[strata_col], prefix='s', drop_first=True).astype(float)
    X = pd.concat([d[x_cols].astype(float).reset_index(drop=True),
                   fe.reset_index(drop=True)], axis=1)
    X = sm.add_constant(X)
    Y = d[y].astype(float).reset_index(drop=True)
    model = sm.OLS(Y, X).fit(cov_type=cov_type)
    return model

def format_coef(beta, se, pval):
    stars = '***' if pval < 0.01 else ('**' if pval < 0.05 else ('*' if pval < 0.1 else ''))
    return f'{beta:.4f}{stars} ({se:.4f})'
