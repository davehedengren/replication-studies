"""
utils.py - Utility functions for paper 206261 replication
Burchardi, Chaney, Hassan, Tarquinio, Terry -
  Immigration, Innovation, and Growth
"""
import pandas as pd
import numpy as np
import statsmodels.api as sm
from scipy import stats as sp_stats
import warnings
warnings.filterwarnings('ignore')

PROJ = '/Users/davehedengren/code/replication_studies/206261-V1'
INPUT = f'{PROJ}/Analysis/Input'
OUTPUT = f'{PROJ}/Analysis/Output'
TEMP = f'{PROJ}/Analysis/Temp'
CODE = f'{PROJ}/Analysis/Code'

# Globals matching Stata setup
VAR_YR = 5
PERIODS = 26
PERIOD_A = 19
PERIOD_AP1 = 20

# Education cutoffs
ED_CUTOFF1 = 2
ED_CUTOFF2 = 4
ED_CUTOFF3 = 7
ED_CUTOFF4 = 10


def period_to_year(t):
    """Convert period index to calendar year."""
    return 1880 + 5 * t


def year_to_period(y):
    """Convert calendar year to period index."""
    return (y - 1880) // 5


def load_dta(path, **kwargs):
    """Load a Stata .dta file."""
    return pd.read_stata(path, convert_categoricals=False, **kwargs)


def read_ipums_fwf(filepath, colspecs, colnames, dtypes=None, chunksize=None):
    """Read IPUMS fixed-width file."""
    # Convert 1-indexed Stata specs to 0-indexed Python specs
    specs = [(s-1, e) for s, e in colspecs]
    if chunksize:
        return pd.read_fwf(filepath, colspecs=specs, names=colnames,
                          dtype=dtypes, chunksize=chunksize)
    return pd.read_fwf(filepath, colspecs=specs, names=colnames, dtype=dtypes)


def demean_iterative(df, yvar, xvars, fe_vars, max_iter=50, tol=1e-10):
    """Iterative demeaning (FWL) for multi-way FE."""
    sub = df.copy()
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
        if max_change < tol:
            break
    return sub


def run_reghdfe(df, yvar, xvars, fe_vars, cluster_vars, weight_var=None):
    """
    Replicate Stata reghdfe with multi-way FE and clustering.
    """
    if isinstance(cluster_vars, str):
        cluster_vars = [cluster_vars]

    all_vars = [yvar] + xvars + fe_vars + cluster_vars
    if weight_var:
        all_vars.append(weight_var)
    all_vars = list(dict.fromkeys(all_vars))
    sub = df[all_vars].dropna().copy()

    if len(sub) == 0:
        return {'b': np.array([]), 'se': np.array([]), 'n': 0, 'xvars': xvars}

    # Iterative demeaning
    sub = demean_iterative(sub, yvar, xvars, fe_vars)

    y = sub[yvar].values
    X = sub[xvars].values
    n = len(y)
    k = X.shape[1]

    if weight_var:
        w = sub[weight_var].values
        w = w / w.mean()
        sqw = np.sqrt(w)
        Xw = X * sqw[:, None]
        yw = y * sqw
    else:
        Xw, yw = X, y

    XtX = Xw.T @ Xw
    try:
        XtX_inv = np.linalg.inv(XtX)
    except np.linalg.LinAlgError:
        XtX_inv = np.linalg.pinv(XtX)

    beta = XtX_inv @ (Xw.T @ yw)
    resid = yw - Xw @ beta

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
        dof = (G / (G - 1)) * ((n - 1) / (n - k))
        return S * dof, G

    if len(cluster_vars) == 1:
        S, G = meat_cluster(cluster_vars[0])
        vcov = XtX_inv @ S @ XtX_inv
    elif len(cluster_vars) == 2:
        S1, G1 = meat_cluster(cluster_vars[0])
        S2, G2 = meat_cluster(cluster_vars[1])
        sub['_cl12'] = sub[cluster_vars[0]].astype(str) + '_' + sub[cluster_vars[1]].astype(str)
        S12, G12 = meat_cluster('_cl12')
        S = S1 + S2 - S12
        vcov = XtX_inv @ S @ XtX_inv
        G = min(G1, G2)
    else:
        # Single cluster fallback
        S, G = meat_cluster(cluster_vars[0])
        vcov = XtX_inv @ S @ XtX_inv

    se = np.sqrt(np.maximum(np.diag(vcov), 0))
    t_stats = np.where(se > 0, beta / se, 0)

    ss_res = np.sum(resid**2)
    ss_tot = np.sum((yw - yw.mean())**2)
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else np.nan

    return {
        'b': beta, 'se': se, 't': t_stats,
        'p': 2 * sp_stats.norm.sf(np.abs(t_stats)),
        'n': n, 'r2': r2, 'vcov': vcov, 'xvars': xvars,
        'n_cl': G, 'resid': resid,
        'fitted': Xw @ beta,
    }


def run_reghdfe_resid(df, yvar, xvars, fe_vars, cluster_var):
    """Run reghdfe and return residuals aligned to input df index."""
    all_vars = [yvar] + xvars + fe_vars + [cluster_var]
    all_vars = list(dict.fromkeys(all_vars))
    sub = df[all_vars].dropna().copy()
    idx = sub.index

    sub = demean_iterative(sub, yvar, xvars, fe_vars)

    y = sub[yvar].values
    X = sub[xvars].values

    XtX = X.T @ X
    try:
        XtX_inv = np.linalg.inv(XtX)
    except np.linalg.LinAlgError:
        XtX_inv = np.linalg.pinv(XtX)

    beta = XtX_inv @ (X.T @ y)
    resid = y - X @ beta

    out = pd.Series(np.nan, index=df.index)
    out.loc[idx] = resid
    return out


def print_reg(res, title=""):
    """Print regression results."""
    if title:
        print(f"\n{title}")
        print("-" * 70)
    print(f"N = {res['n']}")
    if 'r2' in res:
        print(f"R2 = {res['r2']:.4f}")
    if 'n_cl' in res and res['n_cl']:
        print(f"Clusters = {res['n_cl']}")
    print(f"{'Variable':<35} {'Coef':>10} {'SE':>10} {'t':>8}")
    print("-" * 70)
    for i, v in enumerate(res['xvars']):
        if i >= len(res['b']):
            break
        sig = ''
        t = abs(res['t'][i])
        if t > 2.576: sig = '***'
        elif t > 1.96: sig = '**'
        elif t > 1.645: sig = '*'
        print(f"{v:<35} {res['b'][i]:>10.4f} {res['se'][i]:>10.4f} {res['t'][i]:>8.2f} {sig}")


def winsorize(s, lower=0.01, upper=0.99):
    """Winsorize a series at given percentiles."""
    lo = s.quantile(lower)
    hi = s.quantile(upper)
    return s.clip(lo, hi)
