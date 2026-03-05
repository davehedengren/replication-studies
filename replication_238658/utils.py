"""
utils.py — Shared paths, loaders, and NC test functions for Danieli et al. (2025).

Paper: "Negative Control Falsification Tests for IV Designs"
Replication package: openICPSR 238658-V1
"""

import os
import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, '..', '238658-V1', 'Data_analysis')
LIT_DIR = os.path.join(BASE_DIR, '..', '238658-V1', 'Literature_survey')
OUTPUT_DIR = os.path.join(BASE_DIR, 'output')
os.makedirs(OUTPUT_DIR, exist_ok=True)


# ══════════════════════════════════════════════════════════════════════
# DATA LOADERS
# ══════════════════════════════════════════════════════════════════════

def load_adh():
    """Load ADH (Autor, Dorn, Hanson 2013) China trade data."""
    path = os.path.join(DATA_DIR, 'ADH', 'workfile_china.dta')
    return pd.read_stata(path, convert_categoricals=False)


def load_adh_preperiod():
    """Load ADH pre-period data."""
    path = os.path.join(DATA_DIR, 'ADH', 'workfile_china_preperiod.dta')
    return pd.read_stata(path, convert_categoricals=False)


def load_deming():
    """Load Deming (2014) school choice data."""
    path = os.path.join(DATA_DIR, 'Deming', 'cms_added_columns.dta')
    return pd.read_stata(path, convert_categoricals=False)


def load_ashraf_galor():
    """Load Ashraf & Galor (2013) genetic diversity data."""
    path = os.path.join(DATA_DIR, 'AshrafGalor', '20100971_Data', 'data', 'country.dta')
    return pd.read_stata(path, convert_categoricals=False)


def load_nunn_qian():
    """Load Nunn & Qian (2014) food aid data."""
    path = os.path.join(DATA_DIR, 'NunnQian', 'in_sample.dta')
    return pd.read_stata(path, convert_categoricals=False)


def load_literature_survey():
    """Load literature survey of 140 IV papers."""
    path = os.path.join(LIT_DIR, 'NC_literature_survey.csv')
    return pd.read_csv(path)


# ══════════════════════════════════════════════════════════════════════
# NC FALSIFICATION TEST FUNCTIONS
# ══════════════════════════════════════════════════════════════════════

def nc_f_test(Z, NC, cntrls=None, weights=None):
    """
    F-test: Z ~ NC + controls vs Z ~ controls.
    Tests H0: all NC coefficients = 0.
    Returns (p_value, F_stat).
    """
    n = len(Z)
    p = NC.shape[1]

    if cntrls is not None:
        X_full = np.column_stack([np.ones(n), NC, cntrls])
        X_null = np.column_stack([np.ones(n), cntrls])
    else:
        X_full = np.column_stack([np.ones(n), NC])
        X_null = np.ones((n, 1))

    if weights is not None:
        w = np.sqrt(weights)
        Z_w = Z * w
        X_full_w = X_full * w[:, np.newaxis]
        X_null_w = X_null * w[:, np.newaxis]
    else:
        Z_w, X_full_w, X_null_w = Z, X_full, X_null

    beta_full = np.linalg.lstsq(X_full_w, Z_w, rcond=None)[0]
    resid_full = Z_w - X_full_w @ beta_full
    sse_full = np.sum(resid_full ** 2)
    df_full = n - X_full.shape[1]

    beta_null = np.linalg.lstsq(X_null_w, Z_w, rcond=None)[0]
    resid_null = Z_w - X_null_w @ beta_null
    sse_null = np.sum(resid_null ** 2)

    F_stat = ((sse_null - sse_full) / p) / (sse_full / df_full)
    p_value = 1 - stats.f.cdf(F_stat, p, df_full)

    return p_value, F_stat


def nc_bonferroni(Z, NC, cntrls=None, weights=None, cluster=None):
    """
    Bonferroni correction (ADH-style): NC_i ~ Z + controls.

    For each NC_i, regress NC_i on Z (+ controls). Extract p-value for Z.
    Return min(p) * number_of_NCs.

    This matches R's estimate_lm_bonf: lm(x ~ Z, data=...) for each NC column x.
    When cluster is provided, uses vcovCL-style SEs with R's df = n-k.
    """
    p = NC.shape[1]
    n = len(Z)
    pvals = []

    for i in range(p):
        nc_i = NC[:, i]
        if cntrls is not None:
            X = np.column_stack([Z.reshape(-1, 1), cntrls])
        else:
            X = Z.reshape(-1, 1)
        X = sm.add_constant(X)
        k = X.shape[1]

        if weights is not None:
            model = sm.WLS(nc_i, X, weights=weights).fit()
        else:
            model = sm.OLS(nc_i, X).fit()

        if cluster is not None:
            # Match R's vcovCL: cluster-robust SE with df = n - k
            model_cl = model.get_robustcov_results(
                cov_type='cluster', groups=cluster, use_t=True)
            coef_z = model_cl.params[1]
            se_z = model_cl.bse[1]
            t_stat = coef_z / se_z
            # R uses df = n - k (residual df), not G - 1
            df = n - k
            pval = 2 * (1 - stats.t.cdf(abs(t_stat), df=df))
        else:
            pval = model.pvalues[1]

        pvals.append(pval)

    min_p = min(pvals)
    bonf_p = min(min_p * p, 1.0)
    return bonf_p, pvals


def nc_wald_test(Z, NC, cntrls=None, weights=None, cluster=None):
    """
    Wald test: Z ~ NC + controls, test H0: beta_NC = 0.

    Uses HC3 or cluster-robust SEs. Matches R's wald.robust.function
    which uses MASS::ginv for the covariance inverse.
    Returns (p_value, wald_stat).
    """
    n = len(Z)
    p = NC.shape[1]

    if cntrls is not None:
        X = np.column_stack([NC, cntrls])
    else:
        X = NC.copy()
    X = sm.add_constant(X)

    if weights is not None:
        model = sm.WLS(Z, X, weights=weights).fit()
    else:
        model = sm.OLS(Z, X).fit()

    if cluster is not None:
        model_r = model.get_robustcov_results(cov_type='cluster', groups=cluster)
    else:
        model_r = model.get_robustcov_results(cov_type='HC3')

    nc_indices = list(range(1, p + 1))
    beta_nc = model_r.params[nc_indices]
    vcov_nc = model_r.cov_params()[np.ix_(nc_indices, nc_indices)]

    # Use pseudo-inverse (matching R's MASS::ginv fallback)
    wald_stat = beta_nc @ np.linalg.pinv(vcov_nc) @ beta_nc

    p_value = 1 - stats.chi2.cdf(wald_stat, p)
    return p_value, wald_stat


def reduced_form_bonferroni(Y, NCs, cntrls=None, weights=None, cluster=None):
    """
    Bonferroni for reduced-form tests (Nunn-Qian style):
    Y ~ NC_i + controls for each NC_i.

    Returns min(p_NC) * number_of_NCs.
    """
    p = NCs.shape[1]
    pvals = []

    for i in range(p):
        nc_i = NCs[:, i]
        if cntrls is not None:
            X = np.column_stack([nc_i.reshape(-1, 1), cntrls])
        else:
            X = nc_i.reshape(-1, 1)
        X = sm.add_constant(X)

        if weights is not None:
            model = sm.WLS(Y, X, weights=weights).fit()
        else:
            model = sm.OLS(Y, X).fit()

        if cluster is not None:
            model_cl = model.get_robustcov_results(
                cov_type='cluster', groups=cluster, use_t=True)
            pval = model_cl.pvalues[1]
        else:
            pval = model.pvalues[1]

        pvals.append(pval)

    min_p = min(pvals)
    bonf_p = min(min_p * p, 1.0)
    return bonf_p, pvals


def reduced_form_wald(Y, NCs, cntrls=None, weights=None, cluster=None):
    """
    Wald test for reduced-form: Y ~ NCs + controls, test H0: beta_NC = 0.
    Uses HC or cluster-robust SEs with pseudo-inverse.
    """
    n = len(Y)
    p = NCs.shape[1]

    if cntrls is not None:
        X = np.column_stack([NCs, cntrls])
    else:
        X = NCs.copy()
    X = sm.add_constant(X)

    if weights is not None:
        model = sm.WLS(Y, X, weights=weights).fit()
    else:
        model = sm.OLS(Y, X).fit()

    if cluster is not None:
        model_r = model.get_robustcov_results(cov_type='cluster', groups=cluster)
    else:
        model_r = model.get_robustcov_results(cov_type='HC3')

    nc_indices = list(range(1, p + 1))
    beta_nc = model_r.params[nc_indices]
    vcov_nc = model_r.cov_params()[np.ix_(nc_indices, nc_indices)]

    wald_stat = beta_nc @ np.linalg.pinv(vcov_nc) @ beta_nc
    p_value = 1 - stats.chi2.cdf(wald_stat, p)
    return p_value, wald_stat
