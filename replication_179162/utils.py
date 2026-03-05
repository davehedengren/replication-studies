"""
Shared utilities for the Dads & Daughters replication study (179162-V1).

Paper: "Detecting Mother-Father Differences in Spending on Children:
        A New Approach Using Willingness-to-Pay Elicitation"
Authors: Rebecca Dizon-Ross and Seema Jayachandran (AER: Insights, 2023)
"""

import os
import numpy as np
import pandas as pd

RAW_DATA_DIR = '179162-V1/raw_data'
OUTPUT_DIR = 'replication_179162/output'
ANALYSIS_DATA_PATH = 'replication_179162/output/analysis_data.parquet'

# Market prices (UGX) for each good
MARKET_PRICES = {
    'cup': 3600,
    'test': 6000,
    'sieve': 2000,   # approximate
    'toy': 10000,     # approximate
    'deworm': 4000,
    'shoes_a': 2500,  # older child (price list A)
    'shoes_b': 2000,  # younger child (price list B)
    'Fposter': 2000,
    'Fshoes': 2500,   # varies by child age
    'Fplainball': 1500,
    'Ffancyball': 1000,
    'Fcandy': 3000,
    'Fjerry': 4000,
    'Fworkbook': 4500,
    'Ftest': 6000,
    'Fdeworm': 4000,
    'Ftoy': 10000,     # approximate
}

# Good categories
HUMCAP_GOODS = ['test', 'deworm', 'shoes', 'Fshoes', 'Fworkbook', 'Ftest', 'Fdeworm']
HEALTH_GOODS = ['deworm', 'shoes', 'Fdeworm', 'Fshoes']
ENJOYMENT_GOODS = ['Fplainball', 'Ffancyball', 'Fcandy']
CHILD_GOODS = HUMCAP_GOODS + ENJOYMENT_GOODS + ['toy', 'Ftoy']
ADULT_GOODS = ['cup', 'sieve', 'Fposter', 'Fjerry']
TOYS_BIN_GOODS = ['toy', 'Ftoy', 'Ffancyball']  # excluded from main analysis
HYPO_GOODS_BL = ['sieve', 'toy', 'shoes', 'deworm']  # non-incentivized at baseline
HYPO_GOODS_FU = ['Ftest', 'Fdeworm', 'Ftoy']  # non-incentivized at follow-up


def _demean_by_group(arr, groups):
    """Subtract group means from array (Frisch-Waugh-Lovell for absorbed FE)."""
    df_tmp = pd.DataFrame({'val': arr, 'g': groups})
    gm = df_tmp.groupby('g')['val'].transform('mean')
    return arr - gm.values


def run_areg(df, y_var, x_vars, absorb_var, cluster_var):
    """
    Replicate Stata's areg: OLS with absorbed fixed effects and clustered SEs.
    Uses demeaning (FWL theorem) to absorb FE, avoiding large dummy matrices.

    Returns (results, data) where results has correct coefs/SEs for x_vars,
    and the DOF adjustment matches Stata's areg (N - K - n_groups).
    """
    import statsmodels.api as sm

    data = df.dropna(subset=[y_var] + x_vars + [absorb_var, cluster_var]).copy()
    data = data.reset_index(drop=True)

    if len(data) == 0:
        return None

    groups_fe = data[absorb_var].values
    n_groups = data[absorb_var].nunique()

    # Demean Y and each X by the absorb group
    y_dm = _demean_by_group(data[y_var].values.astype(float), groups_fe)

    X_dm = pd.DataFrame(index=data.index)
    for v in x_vars:
        X_dm[v] = _demean_by_group(data[v].values.astype(float), groups_fe)

    # OLS on demeaned data (no constant — absorbed by demeaning)
    model = sm.OLS(y_dm, X_dm)
    cluster_groups = data[cluster_var].values

    # Adjust DOF to match Stata's areg: df_resid = N - K_x - n_groups
    n = len(data)
    k_x = len(x_vars)

    try:
        results = model.fit(
            cov_type='cluster',
            cov_kwds={'groups': cluster_groups, 'df_correction': True},
        )
        # Manually fix residual DOF to match Stata areg
        results.df_resid = n - k_x - n_groups
    except Exception:
        results = model.fit(cov_type='HC1')
        results.df_resid = n - k_x - n_groups

    return results, data
