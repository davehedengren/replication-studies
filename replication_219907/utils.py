"""
utils.py — Paths, data loaders, and regression helpers for 219907-V1.

Paper: "Labor Market Power, Self-Employment, and Development"
Authors: Amodio, Medina & Morlacco (2025), AEJ: Macroeconomics
"""

import os
import numpy as np
import pandas as pd
import statsmodels.api as sm

# ── Paths ──────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(__file__)
DATA_DIR = os.path.join(os.path.dirname(BASE_DIR), '219907-V1', 'facts-stata', 'data')
OUTPUT_DIR = os.path.join(BASE_DIR, 'output')
os.makedirs(OUTPUT_DIR, exist_ok=True)


# ── Data loaders ───────────────────────────────────────────────────────
def load_firm_llmy():
    """Market-level (LLM × industry × year) firm concentration data."""
    return pd.read_stata(os.path.join(DATA_DIR, 'baseline_firm_llmy.dta'))


def load_firm():
    """Firm-level panel data (EEA survey)."""
    return pd.read_stata(os.path.join(DATA_DIR, 'baseline_firm.dta'))


def load_workers(columns=None):
    """Worker cross-section (ENAHO). Optionally select columns."""
    return pd.read_stata(os.path.join(DATA_DIR, 'baseline_workers.dta'),
                         columns=columns, convert_categoricals=False)


def load_workerpanel(columns=None):
    """Worker panel (ENAHO panel subsample)."""
    return pd.read_stata(os.path.join(DATA_DIR, 'baseline_workerpanel.dta'),
                         columns=columns, convert_categoricals=False)


def load_merged(columns=None):
    """Worker data merged with firm-level HHI (main estimation dataset)."""
    return pd.read_stata(os.path.join(DATA_DIR, 'merged_dataset_emp.dta'),
                         columns=columns, convert_categoricals=False)


def load_census():
    """2007 Economic Census."""
    return pd.read_stata(os.path.join(DATA_DIR, 'censo_dataset.dta'))


def load_electrification():
    """Electrification project data at LLM × year level."""
    return pd.read_stata(os.path.join(DATA_DIR, 'llmyear_iv_elect_exp.dta'))


def load_llm_hhi_se():
    """LLM-level HHI and self-employment shares for IV merging."""
    return pd.read_stata(os.path.join(DATA_DIR, 'llm_hhi_shempindep.dta'))


# ── Regression helpers ─────────────────────────────────────────────────
def reghdfe(df, yvar, xvars, fe_vars, cluster_var, subset=None):
    """OLS with absorbed high-dimensional FEs and clustered SEs.

    Uses linearmodels.AbsorbingLS (equivalent to Stata's reghdfe).
    """
    from linearmodels.iv import AbsorbingLS

    d = df.copy() if subset is None else df[subset].copy()
    all_vars = [yvar] + xvars + fe_vars + [cluster_var]
    d = d.dropna(subset=[v for v in all_vars if v in d.columns])

    y = d[yvar]
    X = d[xvars].astype(float)
    absorb = pd.DataFrame({v: pd.Categorical(d[v].astype(str)) for v in fe_vars})
    clusters = d[cluster_var]

    model = AbsorbingLS(y, X, absorb=absorb)
    result = model.fit(cov_type='clustered', clusters=clusters)

    return {
        'params': result.params,
        'se': result.std_errors,
        'pvalues': result.pvalues,
        'nobs': int(result.nobs),
        'n_clust': int(clusters.nunique()),
        'r2': result.rsquared,
        'result': result,
    }


def demean_by_fe(df, varnames, fe_vars):
    """Demean variables by high-dimensional fixed effects using iterative demeaning.

    Returns DataFrame with residualized variables.
    """
    from linearmodels.iv import AbsorbingLS

    d = df.copy()
    absorb = pd.DataFrame({v: pd.Categorical(d[v].astype(str)) for v in fe_vars})

    residuals = {}
    for var in varnames:
        y = d[var].astype(float)
        ones = pd.DataFrame({'_const': np.ones(len(d))}, index=d.index)
        model = AbsorbingLS(y, ones, absorb=absorb)
        res = model.fit()
        residuals[var] = res.resids

    return pd.DataFrame(residuals, index=d.index)


def ivreghdfe(df, yvar, endog_var, instrument_var, fe_vars, cluster_var):
    """IV regression with absorbed FEs using Frisch-Waugh-Lovell + manual 2SLS.

    Implements: ivreghdfe yvar (endog=instrument), absorb(fe_vars) cluster(cluster_var)
    """
    d = df.copy()
    all_vars = [yvar, endog_var, instrument_var, cluster_var] + fe_vars
    d = d.dropna(subset=[v for v in all_vars if v in d.columns])

    clusters = d[cluster_var].values
    n = len(d)
    G = len(np.unique(clusters))

    # Step 1: Demean y, X_endog, Z by FEs (FWL theorem)
    resids = demean_by_fe(d, [yvar, endog_var, instrument_var], fe_vars)

    y_tilde = resids[yvar].values
    x_tilde = resids[endog_var].values
    z_tilde = resids[instrument_var].values

    # Step 2: First stage — x_tilde = gamma * z_tilde + v
    first_ols = sm.OLS(x_tilde, z_tilde).fit()
    x_hat = first_ols.fittedvalues
    first_f = first_ols.fvalue

    # Step 3: Second stage — y_tilde = beta * x_hat + u
    # Coefficient from IV: beta = (Z'X)^-1 Z'y = (z'x)^-1 z'y (scalar case)
    beta_iv = np.dot(z_tilde, y_tilde) / np.dot(z_tilde, x_tilde)

    # Residuals using ORIGINAL x (not x_hat)
    u = y_tilde - beta_iv * x_tilde

    # Step 4: Cluster-robust SE
    # V = (X_hat'X_hat)^-1 * sum_g(X_hat_g' u_g)(u_g' X_hat_g) * (X_hat'X_hat)^-1
    x_hat_x_hat = np.dot(x_hat, x_hat)

    meat = 0.0
    for g in np.unique(clusters):
        mask = clusters == g
        x_g = x_hat[mask]
        u_g = u[mask]
        score_g = np.dot(x_g, u_g)
        meat += score_g ** 2

    # Small-sample correction: G/(G-1)
    correction = G / (G - 1)
    var_iv = correction / (x_hat_x_hat ** 2) * meat
    se_iv = np.sqrt(var_iv)
    t_stat = beta_iv / se_iv
    from scipy import stats
    p_val = 2 * (1 - stats.t.cdf(abs(t_stat), G - 1))

    return {
        'beta': beta_iv,
        'se': se_iv,
        'pval': p_val,
        'first_f': first_f,
        'nobs': n,
        'n_clust': G,
    }


def ivreghdfe_multi(df, yvar, endog_vars, instrument_vars, fe_vars, cluster_var):
    """IV regression with multiple endogenous variables and instruments.

    For specifications like: ivreghdfe y (x1 x2 x3 = z1 z2 z3), absorb(fe) cluster(cl)
    """
    d = df.copy()
    all_vars = [yvar] + endog_vars + instrument_vars + [cluster_var] + fe_vars
    d = d.dropna(subset=[v for v in all_vars if v in d.columns])

    clusters = d[cluster_var].values
    n = len(d)
    G = len(np.unique(clusters))
    k = len(endog_vars)

    # Step 1: Demean all variables by FEs
    all_demean = [yvar] + endog_vars + instrument_vars
    resids = demean_by_fe(d, all_demean, fe_vars)

    y = resids[yvar].values
    X = resids[endog_vars].values  # n × k
    Z = resids[instrument_vars].values  # n × k (just-identified)

    # Step 2: 2SLS
    # beta = (Z'X)^-1 Z'y
    ZX = Z.T @ X  # k × k
    Zy = Z.T @ y  # k × 1
    ZX_inv = np.linalg.inv(ZX)
    beta = ZX_inv @ Zy

    # Predicted X from first stage
    X_hat = Z @ (np.linalg.inv(Z.T @ Z) @ Z.T @ X)  # n × k

    # Residuals using original X
    u = y - X @ beta

    # Step 3: Cluster-robust variance
    # V = (X_hat'X_hat)^-1 * Σ_g (X_hat_g' u_g u_g' X_hat_g) * (X_hat'X_hat)^-1
    bread = np.linalg.inv(X_hat.T @ X_hat)
    meat = np.zeros((k, k))
    for g in np.unique(clusters):
        mask = clusters == g
        X_hat_g = X_hat[mask]
        u_g = u[mask]
        score_g = X_hat_g.T @ u_g  # k × 1
        meat += np.outer(score_g, score_g)

    correction = G / (G - 1)
    V = correction * bread @ meat @ bread
    se = np.sqrt(np.diag(V))

    from scipy import stats
    pvals = 2 * (1 - stats.t.cdf(np.abs(beta / se), G - 1))

    # First-stage F for each endogenous variable
    first_fs = []
    for j in range(k):
        fs = sm.OLS(X[:, j], Z).fit()
        first_fs.append(fs.fvalue)

    return {
        'beta': dict(zip(endog_vars, beta)),
        'se': dict(zip(endog_vars, se)),
        'pvals': dict(zip(endog_vars, pvals)),
        'first_f': first_fs,
        'nobs': n,
        'n_clust': G,
    }


def weighted_stats(series, weights):
    """Weighted mean and SD."""
    w = weights / weights.sum()
    mean = np.average(series, weights=weights)
    var = np.average((series - mean) ** 2, weights=weights)
    return mean, np.sqrt(var)


def binscatter(df, yvar, xvar, nbins=10, controls=None):
    """Simple binscatter: bin x into quantiles, compute mean y in each bin."""
    d = df[[yvar, xvar]].dropna().copy()
    if controls is not None:
        ctrl_cols = [c for c in controls if c in df.columns]
        d = df[[yvar, xvar] + ctrl_cols].dropna().copy()
        # Residualize y and x on controls
        for v in [yvar, xvar]:
            res = sm.OLS(d[v], sm.add_constant(d[ctrl_cols])).fit()
            d[v] = res.resid + d[v].mean()

    d['xbin'] = pd.qcut(d[xvar], nbins, labels=False, duplicates='drop')
    grouped = d.groupby('xbin').agg({xvar: 'mean', yvar: 'mean'}).reset_index()
    return grouped[xvar].values, grouped[yvar].values
