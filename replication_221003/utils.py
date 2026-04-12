"""
utils.py — Paths, data loaders, and helpers for 221003-V1.

Paper: "From Access to Achievement: The Primary-School-Age Impacts of an
        At-Scale Preschool Construction Program in Highly Deprived Communities"
Authors: Bassi, Besbas, Dinarte-Diaz, Ravindran, Reynoso (NBER WP 33543, 2025)
"""

import os
import numpy as np
import pandas as pd

BASE_DIR = os.path.dirname(__file__)
DATA_DIR = os.path.join(os.path.dirname(BASE_DIR), '221003-V1', 'data')
OUTPUT_DIR = os.path.join(BASE_DIR, 'output')
os.makedirs(OUTPUT_DIR, exist_ok=True)


def load_endline():
    return pd.read_stata(os.path.join(DATA_DIR, 'data_EL.dta'),
                         convert_categoricals=False)


def load_baseline():
    return pd.read_stata(os.path.join(DATA_DIR, 'data_BL.dta'),
                         convert_categoricals=False)


def load_distance():
    return pd.read_stata(os.path.join(DATA_DIR, 'HH_Preschool_DICIPE_Distance.dta'),
                         convert_categoricals=False)


def load_distance_bl():
    return pd.read_stata(os.path.join(DATA_DIR, 'HH_Preschool_DICIPE_Distance_BL.dta'),
                         convert_categoricals=False)


def areg_cluster(df, yvar, xvars, fe_col, cluster_col, weight_col=None):
    """Replicate Stata: areg y x, absorb(fe) vce(cluster id) [pw=w].

    Uses statsmodels OLS with fixed-effect dummies and clustered SEs.
    Returns a dict with params, se, pvalues, N, control_mean (from y at treat==0).
    """
    import statsmodels.api as sm

    use_cols = [yvar] + list(xvars) + [fe_col, cluster_col]
    if weight_col:
        use_cols.append(weight_col)
    d = df.dropna(subset=use_cols).copy()

    y = d[yvar].astype(float).values
    fe_dum = pd.get_dummies(d[fe_col].astype(int), prefix='fe', drop_first=True).astype(float)
    X = pd.concat([d[list(xvars)].astype(float).reset_index(drop=True),
                   fe_dum.reset_index(drop=True)], axis=1)
    X = sm.add_constant(X, has_constant='add')

    if weight_col:
        w = d[weight_col].astype(float).values
        model = sm.WLS(y, X.values.astype(float), weights=w)
    else:
        model = sm.OLS(y, X.values.astype(float))

    res = model.fit(cov_type='cluster',
                    cov_kwds={'groups': d[cluster_col].astype(int).values},
                    use_t=True)

    # Stata "robust" correction: statsmodels already applies cluster dof adj
    idx = list(X.columns).index(xvars[0])
    return {
        'beta': res.params[idx],
        'se': res.bse[idx],
        'tval': res.tvalues[idx],
        'pval': res.pvalues[idx],
        'N': int(res.nobs),
        'result': res,
        'X_cols': list(X.columns),
    }


def star(p):
    if p < 0.01:
        return '***'
    if p < 0.05:
        return '**'
    if p < 0.1:
        return '*'
    return ''


def fmt(beta, se, pval):
    return f"{beta:.4f}{star(pval)} ({se:.4f})"
