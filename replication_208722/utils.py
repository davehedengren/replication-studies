"""
Shared utilities for replication of 208722-V1.

Paper: Sabet, Liebald & Friebel (2025)
"Terrorism and Voting: The Rise of Right-Wing Populism in Germany"
"""

import os
import numpy as np
import pandas as pd

# ── Paths ──────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(__file__)
DATA_DIR = os.path.join(BASE_DIR, '..', '208722-V1', 'SLF_replication_upload')
DERIVED_DIR = os.path.join(DATA_DIR, 'derived')
RAW_DIR = os.path.join(DATA_DIR, 'raw')
OUTPUT_DIR = os.path.join(BASE_DIR, 'output')
os.makedirs(OUTPUT_DIR, exist_ok=True)


def load_master():
    """Load the main analysis panel."""
    return pd.read_stata(os.path.join(DERIVED_DIR, 'master.dta'),
                         convert_categoricals=False)


def load_terror():
    """Load the cross-sectional terror attack dataset."""
    return pd.read_stata(os.path.join(DERIVED_DIR, 'terror.dta'),
                         convert_categoricals=False)


def load_master_propensity():
    """Load the propensity-score matched panel."""
    return pd.read_stata(os.path.join(DERIVED_DIR, 'master_propensity.dta'),
                         convert_categoricals=False)


def reghdfe(df, yvar, xvars, fe_vars, cluster_var, subset=None):
    """
    Replicate Stata's reghdfe: OLS with absorbed high-dimensional FEs.

    Uses linearmodels.AbsorbingLS for efficiency.

    Parameters
    ----------
    df : DataFrame
    yvar : str — dependent variable
    xvars : list[str] — regressors to report
    fe_vars : list[str] — variables to absorb as fixed effects
    cluster_var : str — cluster variable for SEs
    subset : Series[bool] or None — row filter

    Returns dict with keys: params, se, pvalues, nobs, n_clust, r2
    """
    from linearmodels.iv import AbsorbingLS
    import linearmodels

    d = df.copy() if subset is None else df[subset].copy()

    # Drop rows with missing in any relevant variable
    all_vars = [yvar] + xvars + fe_vars + [cluster_var]
    d = d.dropna(subset=[v for v in all_vars if v in d.columns])

    y = d[yvar]
    X = d[xvars]

    # Build absorbed effects
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


def balance_test(df, treatment_var, covariates, cluster_var=None, subset=None):
    """
    Simple balance table: mean of each covariate by treatment, with t-test.

    Returns DataFrame with columns: var, mean_0, mean_1, diff, se, pval, n
    """
    import statsmodels.api as sm

    d = df.copy() if subset is None else df[subset].copy()

    rows = []
    for var in covariates:
        dd = d[[var, treatment_var]].dropna()
        if cluster_var and cluster_var in d.columns:
            dd[cluster_var] = d.loc[dd.index, cluster_var]

        g0 = dd.loc[dd[treatment_var] == 0, var]
        g1 = dd.loc[dd[treatment_var] == 1, var]

        if len(g0) == 0 or len(g1) == 0:
            continue

        mean_0 = g0.mean()
        mean_1 = g1.mean()
        diff = mean_1 - mean_0

        # OLS regression for SE and p-value
        y = dd[var].values
        X = sm.add_constant(dd[treatment_var].values)

        if cluster_var and cluster_var in dd.columns:
            model = sm.OLS(y, X).fit(
                cov_type='cluster',
                cov_kwds={'groups': dd[cluster_var].values}
            )
        else:
            model = sm.OLS(y, X).fit(cov_type='HC1')

        se = model.bse[1]
        pval = model.pvalues[1]

        rows.append({
            'var': var, 'mean_0': mean_0, 'mean_1': mean_1,
            'diff': diff, 'se': se, 'pval': pval, 'n': len(dd)
        })

    return pd.DataFrame(rows)
