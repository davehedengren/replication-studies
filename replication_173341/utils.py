"""
Replication of 173341-V1: "Vulnerability and Clientelism"
Bobonis, Gertler, Gonzalez-Navarro, Nichter (2022), AER 112(11): 3627-3659

Paths, constants, and helper functions.
"""
import os
import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats

BASE = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE, '..', '173341-V1', 'data', 'final_data')
OUTPUT_DIR = os.path.join(BASE, 'output')
os.makedirs(OUTPUT_DIR, exist_ok=True)

INDIVIDUAL_DATA = os.path.join(DATA_DIR, 'clientelism_individual_data.dta')
HOUSEHOLD_DATA = os.path.join(DATA_DIR, 'clientelism_household_data.dta')
STACKED_DATA = os.path.join(DATA_DIR, 'clientelism_individual_data_stacked.dta')
VOTING_DATA = os.path.join(DATA_DIR, 'voting_data.dta')


def load_individual():
    return pd.read_stata(INDIVIDUAL_DATA, convert_categoricals=False)


def load_household():
    return pd.read_stata(HOUSEHOLD_DATA, convert_categoricals=False)


def load_stacked():
    return pd.read_stata(STACKED_DATA, convert_categoricals=False)


def load_voting():
    return pd.read_stata(VOTING_DATA, convert_categoricals=False)


def ols_cluster(y, X, cluster, add_const=True):
    """OLS with cluster-robust SEs (Stata's ,cluster())."""
    mask = y.notna() & X.notna().all(axis=1) & cluster.notna()
    y_, X_, cl_ = y[mask].astype(float), X[mask].astype(float), cluster[mask]
    if add_const:
        X_ = sm.add_constant(X_)
    model = sm.OLS(y_, X_).fit(
        cov_type='cluster', cov_kwds={'groups': cl_}
    )
    return model


def ols_with_fe(y, X, fe_var, cluster, data):
    """OLS with fixed effects (explicit dummies) and cluster-robust SEs.
    Matches Stata's: reg y x1 x2 i.fe_var, cluster(cluster)

    y: str - dependent variable name
    X: list of str - regressor names
    fe_var: str - fixed effect variable name
    cluster: str - cluster variable name
    data: DataFrame
    """
    cols = list(dict.fromkeys([y] + X + [fe_var, cluster]))
    df = data[cols].dropna().copy()

    # Create FE dummies (drop first to avoid perfect multicollinearity with constant)
    fe_dummies = pd.get_dummies(df[fe_var], prefix='_fe', drop_first=True, dtype=float)

    y_vec = df[y].astype(float)
    X_mat = pd.concat([df[X].astype(float), fe_dummies], axis=1)
    X_mat = sm.add_constant(X_mat)
    cl = df[cluster]

    model = sm.OLS(y_vec, X_mat).fit(
        cov_type='cluster', cov_kwds={'groups': cl}
    )

    n = len(y_vec)
    model._n_full = n
    model._n_fe = df[fe_var].nunique()

    return model, n


def get_adj_se(model):
    """Get SEs from model (no adjustment needed with explicit dummies)."""
    return model.bse


def reghdfe_absorb(y, X, absorb_var, cluster_var, data):
    """Replicate Stata's reghdfe with absorb() and cluster().

    Uses demeaning (FWL) with reghdfe DOF correction for SEs.
    """
    cols = list(dict.fromkeys([y] + X + [absorb_var, cluster_var]))
    df = data[cols].dropna().copy()

    # Demean by absorbed FE
    vars_to_demean = [y] + X
    group_means = df.groupby(absorb_var)[vars_to_demean].transform('mean')
    df_dm = df.copy()
    for v in vars_to_demean:
        df_dm[v] = (df[v] - group_means[v]).astype(float)

    y_dm = df_dm[y]
    X_dm = df_dm[X]
    cl = df_dm[cluster_var]

    model = sm.OLS(y_dm, X_dm).fit(
        cov_type='cluster', cov_kwds={'groups': cl}
    )

    n = len(y_dm)
    k = len(X)
    n_abs = df[absorb_var].nunique()
    n_clusters = cl.nunique()

    # reghdfe DOF: when absorb == cluster (nested), dof_absorbed = A - G = 0
    if absorb_var == cluster_var:
        dof_absorbed = 0
    else:
        dof_absorbed = n_abs

    # statsmodels scale (no constant): N/(N-k) * G/(G-1)
    # reghdfe scale: (N-1)/(N-k-dof_absorbed) * G/(G-1)
    sm_scale = n / (n - k) * n_clusters / (n_clusters - 1)
    reghdfe_scale = (n - 1) / (n - k - dof_absorbed) * n_clusters / (n_clusters - 1)
    adj = np.sqrt(reghdfe_scale / sm_scale)

    model._se_adj = adj
    model._n_full = n
    model._n_abs = n_abs
    model._n_clusters = n_clusters
    model._k = k

    return model, n


def lincom(model, indices, adj_se=None):
    """Compute linear combination of coefficients (sum) with SE.
    Equivalent to Stata's lincom b1 + b2.
    """
    params = model.params
    if adj_se is not None:
        # Use adjusted covariance
        cov = model.cov_params() * (adj_se[indices[0]]**2 / model.bse[indices[0]]**2)
    else:
        cov = model.cov_params()

    beta = sum(params.iloc[i] for i in indices)
    var = sum(cov.iloc[i, j] for i in indices for j in indices)
    se = np.sqrt(var)
    return beta, se


def fmt(x, decimals=3):
    """Format number to specified decimal places."""
    return f"{x:.{decimals}f}"
