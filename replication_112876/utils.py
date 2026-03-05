"""
Replication utilities for paper 112876
Rajan & Ramcharan (2014) - The Anatomy of a Credit Crisis
"""
import os
import numpy as np
import pandas as pd
import statsmodels.api as sm

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, '..', '112876-V1', 'AER_final_code')

# Standard control variable sets (matching Stata globals)
WIN_GEO_LOG = ['win_totrain_anave', 'win_totrain_anstd', 'win_county_area_log',
               'win_dis_mis_km_avg_log', 'win_dis_atl_km_avg_log',
               'win_dis_grl_km_avg_log', 'win_dis_pac_km_avg_log']
WIN_DEM_LOG = ['win_ngp_log', 'win_ubp_log', 'win_illit_log', 'win_shypy_log', 'win_ttlpy_log']
WIN_OTH2 = ['win_man_crop', 'win_vfmcropy_acres']
WIN_OTH = ['win_man_crop']


def load_dta(filename):
    """Load a Stata .dta file from the data directory."""
    return pd.read_stata(os.path.join(DATA_DIR, filename), convert_categoricals=False)


def run_reg_cluster(df, yvar, xvars, fe_var, cluster_var, condition=None):
    """
    Replicate Stata: xi: reg Y X i.fe_var, cluster(cluster_var)
    OLS with fixed effect dummies and cluster-robust SEs.
    """
    data = df.copy()
    if condition is not None:
        data = data[condition].copy()

    # Build variable list
    all_vars = list(dict.fromkeys([yvar] + xvars + [fe_var, cluster_var]))
    data = data.dropna(subset=all_vars)

    y = data[yvar].astype(float)

    # Create FE dummies (drop_first=True matches Stata i.var)
    fe_dummies = pd.get_dummies(data[fe_var], prefix=fe_var, drop_first=True, dtype=float)

    X = pd.concat([data[xvars].astype(float), fe_dummies], axis=1)
    X = sm.add_constant(X)

    # OLS
    model = sm.OLS(y, X)
    # Stata cluster SEs
    results = model.fit(cov_type='cluster', cov_kwds={'groups': data[cluster_var].values})

    # Extract coefficients for xvars only
    coef = {v: results.params[v] for v in xvars if v in results.params}
    se = {v: results.bse[v] for v in xvars if v in results.params}
    tstat = {v: results.tvalues[v] for v in xvars if v in results.params}
    pval = {v: results.pvalues[v] for v in xvars if v in results.params}

    return {
        'coef': coef, 'se': se, 'tstat': tstat, 'pval': pval,
        'n': int(results.nobs), 'r2': results.rsquared,
        'n_clusters': data[cluster_var].nunique(),
        'results': results,
        'data': data,
    }


def run_areg_cluster(df, yvar, xvars, absorb_var, cluster_var, condition=None):
    """
    Replicate Stata: areg Y X, absorb(absorb_var) cluster(cluster_var)
    FE via demeaning + cluster-robust SEs.
    """
    data = df.copy()
    if condition is not None:
        data = data[condition].copy()

    all_vars = list(dict.fromkeys([yvar] + xvars + [absorb_var, cluster_var]))
    data = data.dropna(subset=all_vars)

    y = data[yvar].astype(float)
    X = data[xvars].astype(float)
    groups = data[absorb_var]
    clusters = data[cluster_var]

    # Demean by absorb group
    y_dm = y - y.groupby(groups).transform('mean')
    X_dm = X.copy()
    for col in xvars:
        X_dm[col] = X[col] - X[col].groupby(groups).transform('mean')

    n = len(data)
    k = len(xvars)
    n_groups = groups.nunique()
    n_clusters = clusters.nunique()

    # OLS on demeaned data (no constant)
    XtX = X_dm.values.T @ X_dm.values
    Xty = X_dm.values.T @ y_dm.values
    beta = np.linalg.solve(XtX, Xty)
    resid = y_dm.values - X_dm.values @ beta

    # Cluster-robust variance (Stata areg cluster)
    # DOF correction: (G/(G-1)) * ((N-1)/(N-K-n_absorbed))
    # where n_absorbed = n_groups (number of absorbed FE groups)
    dof_resid = n - k - n_groups
    correction = (n_clusters / (n_clusters - 1)) * ((n - 1) / dof_resid)

    bread = np.linalg.inv(XtX)
    meat = np.zeros((k, k))

    # Reset index for alignment
    X_dm_arr = X_dm.values
    cluster_arr = clusters.values

    for g in np.unique(cluster_arr):
        mask = cluster_arr == g
        Xg = X_dm_arr[mask]
        eg = resid[mask]
        score_g = Xg.T @ eg
        meat += np.outer(score_g, score_g)

    var_cluster = correction * bread @ meat @ bread
    se = np.sqrt(np.diag(var_cluster))
    t_stats = beta / se

    return {
        'coef': dict(zip(xvars, beta)),
        'se': dict(zip(xvars, se)),
        'tstat': dict(zip(xvars, t_stats)),
        'n': n, 'n_groups': n_groups, 'n_clusters': n_clusters,
    }


def run_ivreg2_cluster(df, yvar, endog_vars, instruments, exog_vars, fe_var, cluster_var, condition=None):
    """
    Replicate Stata: ivreg2 Y exog (endog=instruments) i.fe_var, cluster(cluster_var)
    2SLS with FE dummies and cluster-robust SEs.
    """
    data = df.copy()
    if condition is not None:
        data = data[condition].copy()

    all_vars = list(dict.fromkeys([yvar] + endog_vars + instruments + exog_vars + [fe_var, cluster_var]))
    data = data.dropna(subset=all_vars)

    y = data[yvar].astype(float)

    # FE dummies
    fe_dummies = pd.get_dummies(data[fe_var], prefix=fe_var, drop_first=True, dtype=float)

    # Exogenous regressors (including FE dummies and constant)
    X_exog = pd.concat([data[exog_vars].astype(float), fe_dummies], axis=1)
    X_exog = sm.add_constant(X_exog)

    # Endogenous
    X_endog = data[endog_vars].astype(float)

    # Instruments (all exog + excluded instruments)
    Z = pd.concat([data[instruments].astype(float), X_exog], axis=1)

    # 2SLS
    from statsmodels.sandbox.regression.gmm import IV2SLS
    X_all = pd.concat([X_endog, X_exog], axis=1)

    model = IV2SLS(y.values, X_all.values, Z.values)
    results = model.fit()

    # Manual cluster-robust SEs for IV
    n = len(data)
    k = X_all.shape[1]
    n_clusters = data[cluster_var].nunique()

    # Predicted X from first stage
    Pz = Z.values @ np.linalg.lstsq(Z.values, X_all.values, rcond=None)[0]
    PzTPz_inv = np.linalg.inv(Pz.T @ Pz)
    resid = y.values - X_all.values @ results.params

    correction = (n_clusters / (n_clusters - 1)) * ((n - 1) / (n - k))
    meat = np.zeros((k, k))
    cluster_arr = data[cluster_var].values
    for g in np.unique(cluster_arr):
        mask = cluster_arr == g
        Pg = Pz[mask]
        eg = resid[mask]
        score_g = Pg.T @ eg
        meat += np.outer(score_g, score_g)

    var_cluster = correction * PzTPz_inv @ meat @ PzTPz_inv
    se = np.sqrt(np.diag(var_cluster))
    t_stats = results.params / se

    # Map back to variable names
    col_names = list(X_endog.columns) + list(X_exog.columns)
    report_vars = endog_vars + exog_vars
    coef = {}; se_d = {}; tstat_d = {}
    for v in report_vars:
        if v in col_names:
            idx = col_names.index(v)
            coef[v] = results.params[idx]
            se_d[v] = se[idx]
            tstat_d[v] = t_stats[idx]

    return {
        'coef': coef, 'se': se_d, 'tstat': tstat_d,
        'n': n, 'n_clusters': n_clusters,
    }


def print_results(results, title="", vars_to_show=None):
    """Print regression results."""
    if title:
        print(f"\n{'='*65}")
        print(title)
        print(f"{'='*65}")
    coefs = results['coef']
    if vars_to_show is None:
        vars_to_show = list(coefs.keys())
    print(f"{'Variable':<35} {'Coef':>10} {'SE':>10} {'t':>8}")
    print("-" * 65)
    for var in vars_to_show:
        if var in coefs:
            c = coefs[var]
            s = results['se'][var]
            t = results['tstat'][var]
            print(f"{var:<35} {c:>10.4f} {s:>10.4f} {t:>8.2f}")
    print(f"N = {results['n']}", end="")
    if 'n_clusters' in results:
        print(f", Clusters = {results['n_clusters']}", end="")
    if 'r2' in results:
        print(f", R² = {results['r2']:.4f}", end="")
    print()
