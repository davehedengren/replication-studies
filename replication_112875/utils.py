"""
Shared utilities for replication of Callen & Long (2015)
"Institutional Corruption and Election Fraud: Evidence from a Field Experiment in Afghanistan"
AER 105(1): 354-381
"""

import os
import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats

# ── Paths ──────────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, '..', '112875-V1', '20120427_stata_files')
OUTPUT_DIR = os.path.join(BASE_DIR, 'output')
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ── Data loading ───────────────────────────────────────────────────────────
def load_vote_diff():
    """Load master_vote_difference_native.dta"""
    return pd.read_stata(os.path.join(DATA_DIR, 'master_vote_difference_native.dta'),
                         convert_categoricals=False)

def load_pc():
    """Load master_pc_native.dta"""
    return pd.read_stata(os.path.join(DATA_DIR, 'master_pc_native.dta'),
                         convert_categoricals=False)

def load_candidate_info():
    """Load di_candidate_information.dta"""
    return pd.read_stata(os.path.join(DATA_DIR, 'di_candidate_information.dta'),
                         convert_categoricals=False)

# ── Variable lists ─────────────────────────────────────────────────────────
CONNECTION_VARS = ['peo_fullsample', 'deo_fullsample', 'karzai_fullsample',
                   'gov_fullsample', 'incumbent']
CONTROLS1 = ['pashtun', 'tajik', 'treat_di_actual']
BALANCE_VARS_SURVEY = [
    'expectedturnout', 'secretvote', 'candknowsvote', 'parliamentaryknowledge',
    'onecandidatevote', 'transportproblems', 'policesafe', 'threatvote',
    'violenceown', 'ownqawmcandidate', 'traditionalauthority',
    'pashtun', 'tajik', 'employment', 'electrified', 'dgkeepsfair'
]
BALANCE_VARS_ADMIN = ['treat_di_actual', 'fefa_visit', 'inkproblem']

# ── Multiway clustering (Cameron, Gelbach, Miller 2011) ────────────────────
def cgmreg(y, X, cluster_vars, df):
    """
    OLS with multiway clustered standard errors (Cameron, Gelbach, Miller 2011).

    For two cluster dimensions A and B:
    V = V_A + V_B - V_AB
    where V_AB is the intersection cluster.
    """
    mask = y.notna() & X.notna().all(axis=1)
    for cv in cluster_vars:
        mask &= df[cv].notna()
    y_c = y[mask].values.astype(float)
    X_c = X[mask].values.astype(float)

    n = len(y_c)
    k = X_c.shape[1]

    # OLS
    beta = np.linalg.lstsq(X_c, y_c, rcond=None)[0]
    resid = y_c - X_c @ beta

    # Meat matrices for each cluster dimension and intersections
    from itertools import combinations

    cluster_cols = [df.loc[mask, cv].values for cv in cluster_vars]
    n_dims = len(cluster_vars)

    XtXinv = np.linalg.inv(X_c.T @ X_c)

    V = np.zeros((k, k))

    # For each subset of cluster dimensions
    for r in range(1, n_dims + 1):
        for combo in combinations(range(n_dims), r):
            # Create intersection cluster
            if len(combo) == 1:
                cl = cluster_cols[combo[0]]
            else:
                cl = np.array(['_'.join(str(c) for c in row)
                              for row in zip(*[cluster_cols[c] for c in combo])])

            unique_clusters = np.unique(cl)
            G = len(unique_clusters)

            # Small-sample correction: G/(G-1) * (n-1)/(n-k)
            correction = (G / (G - 1)) * ((n - 1) / (n - k))

            meat = np.zeros((k, k))
            for g in unique_clusters:
                idx = cl == g
                Xg = X_c[idx]
                eg = resid[idx]
                score_g = Xg.T @ eg
                meat += np.outer(score_g, score_g)

            meat *= correction
            V_cl = XtXinv @ meat @ XtXinv

            sign = (-1) ** (len(combo) + 1)
            V += sign * V_cl

    se = np.sqrt(np.diag(V))
    tstat = beta / se
    pval = 2 * (1 - stats.t.cdf(np.abs(tstat), n - k))

    # R-squared
    y_mean = np.mean(y_c)
    ss_tot = np.sum((y_c - y_mean) ** 2)
    ss_res = np.sum(resid ** 2)
    r2 = 1 - ss_res / ss_tot

    # Cluster counts
    n_clusters = [len(np.unique(cl)) for cl in cluster_cols]

    return {
        'beta': beta,
        'se': se,
        'tstat': tstat,
        'pval': pval,
        'r2': r2,
        'n': n,
        'n_clusters': n_clusters,
        'resid': resid,
        'V': V,
        'XtXinv': XtXinv,
    }


def ols_cluster(y, X, cluster, df=None):
    """
    OLS with cluster-robust SEs.
    y, X: Series/DataFrame aligned with df.
    cluster: array-like of cluster IDs.
    """
    if df is not None:
        mask = y.notna() & X.notna().all(axis=1) & pd.Series(cluster).notna()
        y_c = y[mask].values.astype(float)
        X_c = X[mask].values.astype(float)
        cl = np.asarray(cluster)[mask.values]
    else:
        y_c = np.asarray(y, dtype=float)
        X_c = np.asarray(X, dtype=float)
        cl = np.asarray(cluster)

    n = len(y_c)
    k = X_c.shape[1]

    beta = np.linalg.lstsq(X_c, y_c, rcond=None)[0]
    resid = y_c - X_c @ beta

    XtXinv = np.linalg.inv(X_c.T @ X_c)

    unique_clusters = np.unique(cl)
    G = len(unique_clusters)
    correction = (G / (G - 1)) * ((n - 1) / (n - k))

    meat = np.zeros((k, k))
    for g in unique_clusters:
        idx = cl == g
        Xg = X_c[idx]
        eg = resid[idx]
        score_g = Xg.T @ eg
        meat += np.outer(score_g, score_g)

    meat *= correction
    V = XtXinv @ meat @ XtXinv

    se = np.sqrt(np.diag(V))
    tstat = beta / se
    pval = 2 * (1 - stats.t.cdf(np.abs(tstat), G - 1))

    y_mean = np.mean(y_c)
    ss_tot = np.sum((y_c - y_mean) ** 2)
    ss_res = np.sum(resid ** 2)
    r2 = 1 - ss_res / ss_tot

    return {
        'beta': beta,
        'se': se,
        'tstat': tstat,
        'pval': pval,
        'r2': r2,
        'n': n,
        'n_clusters': G,
        'resid': resid,
        'V': V,
    }


def areg_cluster(y, X, absorb, cluster, df):
    """
    OLS with absorbed fixed effects (Stata areg equivalent) and cluster-robust SEs.
    Demeans y and X by the absorb variable, then runs OLS with cluster SEs.
    Reports R-squared from the full model (including FEs).
    """
    mask = y.notna() & X.notna().all(axis=1) & df[absorb].notna() & df[cluster].notna()
    y_c = y[mask].astype(float).copy()
    X_c = X[mask].astype(float).copy()
    abs_c = df.loc[mask, absorb].values
    cl_c = df.loc[mask, cluster].values

    # Demean by absorb groups
    y_dm = y_c.copy()
    X_dm = X_c.copy()

    groups = pd.Series(abs_c)
    for g in groups.unique():
        idx = groups.values == g
        y_dm.iloc[idx] = y_c.iloc[idx] - y_c.iloc[idx].mean()
        for col in X_dm.columns:
            X_dm.iloc[idx, X_dm.columns.get_loc(col)] = X_c.iloc[idx, X_c.columns.get_loc(col)] - X_c.iloc[idx, X_c.columns.get_loc(col)].mean()

    n = len(y_dm)
    n_groups = len(groups.unique())
    k_total = X_dm.shape[1] + n_groups  # total parameters for DOF

    y_arr = y_dm.values
    X_arr = X_dm.values
    k = X_arr.shape[1]

    beta = np.linalg.lstsq(X_arr, y_arr, rcond=None)[0]
    resid = y_arr - X_arr @ beta

    XtXinv = np.linalg.inv(X_arr.T @ X_arr)

    unique_clusters = np.unique(cl_c)
    G = len(unique_clusters)
    # Stata areg DOF correction: G/(G-1) * (n-1)/(n-k_total)
    correction = (G / (G - 1)) * ((n - 1) / (n - k_total))

    meat = np.zeros((k, k))
    for g in unique_clusters:
        idx = cl_c == g
        Xg = X_arr[idx]
        eg = resid[idx]
        score_g = Xg.T @ eg
        meat += np.outer(score_g, score_g)

    meat *= correction
    V = XtXinv @ meat @ XtXinv

    se = np.sqrt(np.diag(V))
    tstat = beta / se
    pval = 2 * (1 - stats.t.cdf(np.abs(tstat), G - 1))

    # R-squared from full model (with absorbed FEs)
    y_orig = y_c.values
    y_mean = np.mean(y_orig)
    ss_tot = np.sum((y_orig - y_mean) ** 2)
    ss_res = np.sum(resid ** 2)
    r2 = 1 - ss_res / ss_tot

    return {
        'beta': beta,
        'se': se,
        'tstat': tstat,
        'pval': pval,
        'r2': r2,
        'n': n,
        'n_clusters': G,
        'resid': resid,
        'V': V,
        'mask': mask,
    }


def lee_bounds(outcome, treatment, sample_mask, cluster_var=None, n_boot=250, seed=8221986):
    """
    Lee (2009) trimming bounds procedure.
    Returns dict with control stats, treatment stats, trimming ratio,
    trimmed means, and bounds.
    """
    rng = np.random.RandomState(seed)

    d = pd.DataFrame({
        'outcome': outcome,
        'treat': treatment,
        'sample': sample_mask,
    })
    if cluster_var is not None:
        d['cluster'] = cluster_var

    d = d[d['sample'] == 1].copy()

    # Control
    total_control = (d['treat'] == 0).sum()
    nonmissing_control = d.loc[d['treat'] == 0, 'outcome'].notna().sum()
    present_control = nonmissing_control / total_control
    control_mean = d.loc[(d['treat'] == 0) & d['outcome'].notna(), 'outcome'].mean()

    # Treatment
    total_treat = (d['treat'] == 1).sum()
    nonmissing_treat = d.loc[d['treat'] == 1, 'outcome'].notna().sum()
    present_treat = nonmissing_treat / total_treat
    treat_mean = d.loc[(d['treat'] == 1) & d['outcome'].notna(), 'outcome'].mean()

    # Trimming proportion
    p = (present_treat - present_control) / present_treat

    # Compute trimmed means
    treat_observed = d.loc[(d['treat'] == 1) & d['outcome'].notna(), 'outcome'].values
    treat_sorted = np.sort(treat_observed)
    n_treat_obs = len(treat_sorted)

    # Upper bound: trim bottom p percentiles
    n_trim = int(np.floor(p * n_treat_obs))
    upper_trimmed = treat_sorted[n_trim:]
    upper_mean = upper_trimmed.mean() if len(upper_trimmed) > 0 else np.nan
    upper_quantile = treat_sorted[n_trim] if n_trim < n_treat_obs else np.nan

    # Lower bound: trim top p percentiles
    n_keep = int(np.ceil((1 - p) * n_treat_obs))
    lower_trimmed = treat_sorted[:n_keep]
    lower_mean = lower_trimmed.mean() if len(lower_trimmed) > 0 else np.nan
    lower_quantile = treat_sorted[n_keep - 1] if n_keep > 0 else np.nan

    upper_bound = upper_mean - control_mean
    lower_bound = lower_mean - control_mean

    # Bootstrap SEs
    if n_boot > 0 and cluster_var is not None:
        boot_results = []
        clusters = d['cluster'].unique()
        for b in range(n_boot):
            boot_clusters = rng.choice(clusters, size=len(clusters), replace=True)
            boot_data = pd.concat([d[d['cluster'] == c].copy() for c in boot_clusters],
                                  ignore_index=True)
            try:
                br = _lee_core(boot_data)
                boot_results.append(br)
            except:
                continue

        if boot_results:
            boot_arr = np.array(boot_results)
            se_control = np.nanstd(boot_arr[:, 0], ddof=0)
            se_treat_upper = np.nanstd(boot_arr[:, 1], ddof=0)
            se_treat_lower = np.nanstd(boot_arr[:, 2], ddof=0)
            se_upper_bound = np.nanstd(boot_arr[:, 3], ddof=0)
            se_lower_bound = np.nanstd(boot_arr[:, 4], ddof=0)
        else:
            se_control = se_treat_upper = se_treat_lower = se_upper_bound = se_lower_bound = np.nan
    else:
        se_control = se_treat_upper = se_treat_lower = se_upper_bound = se_lower_bound = np.nan

    return {
        'total_control': total_control,
        'present_control': present_control,
        'control_mean': control_mean,
        'se_control': se_control,
        'total_treat': total_treat,
        'present_treat': present_treat,
        'treat_mean': treat_mean,
        'p': p,
        'upper_quantile': upper_quantile,
        'upper_mean': upper_mean,
        'lower_quantile': lower_quantile,
        'lower_mean': lower_mean,
        'upper_bound': upper_bound,
        'lower_bound': lower_bound,
        'se_upper_bound': se_upper_bound,
        'se_lower_bound': se_lower_bound,
        'se_treat_upper': se_treat_upper,
        'se_treat_lower': se_treat_lower,
    }


def _lee_core(d):
    """Core Lee bounds calculation for bootstrap."""
    total_control = (d['treat'] == 0).sum()
    nonmissing_control = d.loc[d['treat'] == 0, 'outcome'].notna().sum()
    present_control = nonmissing_control / total_control if total_control > 0 else 0
    control_mean = d.loc[(d['treat'] == 0) & d['outcome'].notna(), 'outcome'].mean()

    total_treat = (d['treat'] == 1).sum()
    nonmissing_treat = d.loc[d['treat'] == 1, 'outcome'].notna().sum()
    present_treat = nonmissing_treat / total_treat if total_treat > 0 else 0

    if present_treat == 0:
        return [control_mean, np.nan, np.nan, np.nan, np.nan]

    p = (present_treat - present_control) / present_treat

    treat_observed = d.loc[(d['treat'] == 1) & d['outcome'].notna(), 'outcome'].values
    treat_sorted = np.sort(treat_observed)
    n_treat_obs = len(treat_sorted)

    n_trim = int(np.floor(p * n_treat_obs))
    upper_trimmed = treat_sorted[n_trim:]
    upper_mean = upper_trimmed.mean() if len(upper_trimmed) > 0 else np.nan

    n_keep = int(np.ceil((1 - p) * n_treat_obs))
    lower_trimmed = treat_sorted[:n_keep]
    lower_mean = lower_trimmed.mean() if len(lower_trimmed) > 0 else np.nan

    return [control_mean, upper_mean, lower_mean,
            upper_mean - control_mean, lower_mean - control_mean]


def wald_test(V, R, beta, r=0):
    """
    Wald test: H0: R @ beta = r
    Returns F-stat and p-value.
    """
    diff = R @ beta - r
    Vr = R @ V @ R.T
    if np.isscalar(Vr):
        F = diff ** 2 / Vr
    else:
        F = diff @ np.linalg.inv(Vr) @ diff
    q = 1 if np.isscalar(diff) else len(diff)
    p = 1 - stats.chi2.cdf(F, q)
    return F, p


def print_comparison(label, python_val, published_val, tol=0.05):
    """Print comparison of Python vs published values."""
    if python_val is None or published_val is None:
        print(f"  {label}: Python={python_val}, Published={published_val}")
        return
    diff = abs(python_val - published_val)
    match = "MATCH" if diff < tol else f"DIFF ({diff:.4f})"
    print(f"  {label}: Python={python_val:.4f}, Published={published_val:.4f} [{match}]")
