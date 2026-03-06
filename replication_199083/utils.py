"""
utils.py - Utility functions for paper 199083 replication
Kantor & Whalley - Moonshot: Public R&D and Growth
"""
import pandas as pd
import numpy as np
import statsmodels.api as sm
from scipy import stats as sp_stats
import warnings
warnings.filterwarnings('ignore')

PROJ = '/Users/davehedengren/code/replication_studies/199083-V1'
RAW = f'{PROJ}/data/raw'
PROC = f'{PROJ}/data/process'

def load_dta(path, **kwargs):
    """Load a Stata .dta file, handling common issues."""
    return pd.read_stata(path, convert_categoricals=False, **kwargs)


def demean(df, var, groups):
    """FWL demeaning: subtract group means for each group variable."""
    x = df[var].astype(float).copy()
    for g in groups:
        means = df.groupby(g)[var].transform('mean')
        x = x - means
        # Re-center to avoid accumulating the grand mean subtraction
        if len(groups) > 1:
            x = x + x.mean()  # not exactly right for multi-way; iterate
    return x


def run_reghdfe(df, yvar, xvars, fe_vars, cluster_vars, weight_var=None):
    """
    Replicate Stata reghdfe with multi-way FE and multi-way clustering.
    Uses iterative demeaning (FWL) for fixed effects.

    Parameters
    ----------
    df : DataFrame
    yvar : str - dependent variable
    xvars : list of str - regressors (including treatment + controls)
    fe_vars : list of str - FE group variables (absorbed)
    cluster_vars : list or str - clustering variable(s)
    weight_var : str or None - analytical weight variable

    Returns
    -------
    dict with keys: b, se, t, p, n, r2, df_r, vcov, xvars
    """
    if isinstance(cluster_vars, str):
        cluster_vars = [cluster_vars]

    # Build complete variable list and drop missing
    all_vars = [yvar] + xvars + fe_vars + cluster_vars
    if weight_var:
        all_vars.append(weight_var)
    all_vars = list(dict.fromkeys(all_vars))  # deduplicate
    sub = df[all_vars].dropna().copy()

    if len(sub) == 0:
        return {'b': [], 'se': [], 'n': 0}

    # Iterative demeaning for multi-way FE
    max_iter = 50
    for iteration in range(max_iter):
        max_change = 0
        for fe in fe_vars:
            for v in [yvar] + xvars:
                old = sub[v].copy()
                group_means = sub.groupby(fe)[v].transform('mean')
                sub[v] = sub[v] - group_means
                change = (sub[v] - old).abs().max()
                if change > max_change:
                    max_change = change
        if max_change < 1e-10:
            break

    y = sub[yvar].values
    X = sub[xvars].values
    n = len(y)
    k = X.shape[1]

    # Count absorbed FE parameters
    n_fe = sum(sub[fe].nunique() for fe in fe_vars)

    if weight_var:
        w = sub[weight_var].values
        w = w / w.mean()  # normalize
        sqw = np.sqrt(w)
        Xw = X * sqw[:, None]
        yw = y * sqw
    else:
        Xw, yw = X, y

    # OLS
    XtX = Xw.T @ Xw
    try:
        XtX_inv = np.linalg.inv(XtX)
    except np.linalg.LinAlgError:
        XtX_inv = np.linalg.pinv(XtX)

    beta = XtX_inv @ (Xw.T @ yw)
    resid = yw - Xw @ beta

    # Multi-way cluster-robust variance (Cameron, Gelbach, Miller 2011)
    def meat_cluster(cl_var):
        clusters = sub[cl_var].values
        unique_cl = np.unique(clusters)
        G = len(unique_cl)
        S = np.zeros((k, k))
        for c in unique_cl:
            mask = clusters == c
            if weight_var:
                u = (Xw[mask].T @ resid[mask]).reshape(-1, 1)
            else:
                u = (X[mask].T @ resid[mask]).reshape(-1, 1)
            S += u @ u.T
        # DOF adjustment: G/(G-1) * (N-1)/(N-K)
        dof = (G / (G - 1)) * ((n - 1) / (n - k))
        return S * dof, G

    if len(cluster_vars) == 1:
        S, G = meat_cluster(cluster_vars[0])
        vcov = XtX_inv @ S @ XtX_inv
    elif len(cluster_vars) == 2:
        # Two-way clustering: V1 + V2 - V12
        S1, G1 = meat_cluster(cluster_vars[0])
        S2, G2 = meat_cluster(cluster_vars[1])
        # Intersection cluster
        sub['_cl12'] = sub[cluster_vars[0]].astype(str) + '_' + sub[cluster_vars[1]].astype(str)
        S12, G12 = meat_cluster('_cl12')
        S = S1 + S2 - S12
        vcov = XtX_inv @ S @ XtX_inv
        G = min(G1, G2)

    se = np.sqrt(np.diag(vcov))
    t_stats = beta / se

    # R-squared (using demeaned data)
    ss_res = np.sum(resid**2)
    ss_tot = np.sum((yw - yw.mean())**2)
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else np.nan

    return {
        'b': beta, 'se': se, 't': t_stats,
        'p': 2 * sp_stats.norm.sf(np.abs(t_stats)),
        'n': n, 'r2': r2, 'vcov': vcov, 'xvars': xvars,
        'n_cl': G if len(cluster_vars) <= 2 else None,
    }


def print_reg(res, title=""):
    """Print regression results in a formatted table."""
    if title:
        print(f"\n{title}")
        print("-" * 70)
    print(f"N = {res['n']}")
    if 'r2' in res:
        print(f"R2 = {res['r2']:.4f}")
    print(f"{'Variable':<35} {'Coef':>10} {'SE':>10} {'t':>8}")
    print("-" * 70)
    for i, v in enumerate(res['xvars']):
        sig = ''
        t = abs(res['t'][i])
        if t > 2.576: sig = '***'
        elif t > 1.96: sig = '**'
        elif t > 1.645: sig = '*'
        print(f"{v:<35} {res['b'][i]:>10.4f} {res['se'][i]:>10.4f} {res['t'][i]:>8.2f} {sig}")
