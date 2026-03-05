"""
Shared utilities for the "Underestimating Learning by Doing" replication study (212821-V1).

Paper: Horn, S. and Loewenstein, G. (2025). "Underestimating Learning by Doing."
"""

import os
import numpy as np
import pandas as pd

DATA_DIR = '212821-V1/ReplicationPackage/data/clean'
OUTPUT_DIR = 'replication_212821/output'

os.makedirs(OUTPUT_DIR, exist_ok=True)

# Study configurations
STUDIES = {
    1: {'name': 'Low-Difficulty Mirror Tracing', 'max_rounds': 10, 'ahead': [1, 4],
        'file': '1_Study1.dta'},
    2: {'name': 'Latin Translation', 'max_rounds': 7, 'ahead': [1, 3],
        'file': '2_Study2.dta'},
    3: {'name': 'High-Difficulty Mirror Tracing', 'max_rounds': 6, 'ahead': [1, 4],
        'file': '3_Study3.dta'},
    4: {'name': 'Study 4 (Performers + Predictors)', 'max_rounds': 10, 'ahead': [1, 4],
        'file': '4_Study4.dta'},
    0: {'name': 'Original Low-Pay Study 1', 'max_rounds': 10, 'ahead': [1, 4],
        'file': '0_Study1_LowPay.dta'},
}


def load_study(study_num):
    """Load a clean study DTA file."""
    cfg = STUDIES[study_num]
    return pd.read_stata(os.path.join(DATA_DIR, cfg['file']), convert_categoricals=False)


def clustered_mean_se(values, cluster_ids):
    """Compute mean and clustered SE (matching Stata's `reg y ones, cl(id)`)."""
    y = values.dropna()
    ids = cluster_ids.loc[y.index]
    n = len(y)
    mean = y.mean()

    # Clustered SE: regress on constant with clustering
    residuals = y - mean
    clusters = ids.unique()
    g = len(clusters)

    # Meat of sandwich estimator
    meat = 0.0
    for c in clusters:
        mask = ids == c
        u_c = residuals[mask].sum()
        meat += u_c ** 2

    # Small-sample adjustment: (g/(g-1)) * (n-1)/(n-k), k=1
    adj = (g / (g - 1)) * ((n - 1) / (n - 1))  # k=1 for constant
    var_b = adj * meat / (n ** 2)
    se = np.sqrt(var_b)

    return mean, se, n


def simple_mean_se(values):
    """Compute mean and simple SE (matching Stata's `ci means`)."""
    y = values.dropna()
    return y.mean(), y.std(ddof=1) / np.sqrt(len(y)), len(y)


def format_val(val, fmt="%9.2f"):
    """Format a value matching Stata's string() format."""
    return f"{val:.2f}"


def format_stars(pval):
    """Return significance stars."""
    if pval < 0.01:
        return '***'
    elif pval < 0.05:
        return '**'
    elif pval < 0.10:
        return '*'
    return ''
