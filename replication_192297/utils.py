"""
Shared utilities for the Big Loans replication study (192297-V1).

Paper: "Big Loans to Small Businesses: Predicting Winners and Losers
        in an Entrepreneurial Lending Experiment"
Authors: Bryan, Karlan & Osman (AER, 2024)
"""

import os
import numpy as np
import pandas as pd

RAW_DATA_DIR = '192297-V1/Raw Data'
CLEAN_DATA_DIR = '192297-V1/Clean Data'
OUTPUT_DIR = 'replication_192297/output'

# Paths to cleaned output
MERGED_PATH = 'replication_192297/output/16-ABA-All-Merged.parquet'
PENALTY_PATH = 'replication_192297/output/17-ABA-penalty-clean.parquet'
ML_GROUP_PATH = '192297-V1/Clean Data/20-ABA-ML-group.dta'  # pre-computed ML output


def _demean_by_group(arr, groups):
    """Subtract group means from array (FWL for absorbed FE)."""
    df_tmp = pd.DataFrame({'val': arr, 'g': groups})
    gm = df_tmp.groupby('g')['val'].transform('mean')
    return arr - gm.values


def run_areg(df, y_var, x_vars, absorb_var, cluster_var=None, weights=None):
    """
    Replicate Stata's areg: OLS with absorbed FE, optionally clustered SEs.
    Uses demeaning (FWL) to avoid large dummy matrices.
    """
    import statsmodels.api as sm

    all_vars = [y_var] + x_vars + [absorb_var]
    if cluster_var:
        all_vars.append(cluster_var)

    data = df.dropna(subset=all_vars).copy()
    data = data.reset_index(drop=True)

    if len(data) == 0:
        return None

    groups_fe = data[absorb_var].values
    n_groups = data[absorb_var].nunique()

    y_dm = _demean_by_group(data[y_var].values.astype(float), groups_fe)

    X_dm = pd.DataFrame(index=data.index)
    for v in x_vars:
        X_dm[v] = _demean_by_group(data[v].values.astype(float), groups_fe)

    w = None
    if weights is not None:
        w = data[weights].values.astype(float) if isinstance(weights, str) else weights

    model = sm.WLS(y_dm, X_dm, weights=w) if w is not None else sm.OLS(y_dm, X_dm)

    n = len(data)
    k_x = len(x_vars)

    if cluster_var:
        try:
            results = model.fit(
                cov_type='cluster',
                cov_kwds={'groups': data[cluster_var].values, 'df_correction': True})
        except Exception:
            results = model.fit(cov_type='HC1')
    else:
        results = model.fit(cov_type='HC1')

    results.df_resid = n - k_x - n_groups

    return results, data


def winsor(s, p=0.01, high_only=True):
    """Winsorize a Series at the p percentile (high only by default, matching Stata's winsor).
    Uses Stata's percentile method: value at position ceil((1-p)*n) in sorted order."""
    vals = s.dropna()
    sorted_vals = vals.sort_values().values
    n = len(sorted_vals)
    if n == 0:
        return s.copy()
    upper_pos = min(int(np.ceil((1 - p) * n)) - 1, n - 1)
    upper = sorted_vals[upper_pos]
    out = s.copy()
    out = out.clip(upper=upper)
    if not high_only:
        lower_pos = max(int(np.ceil(p * n)) - 1, 0)
        lower = sorted_vals[lower_pos]
        out = out.clip(lower=lower)
    return out


def stars(pval):
    if pval < 0.01: return '***'
    if pval < 0.05: return '**'
    if pval < 0.1: return '*'
    return ''
