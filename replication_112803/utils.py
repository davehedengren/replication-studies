"""
Shared utilities for replication of Acemoglu, Autor, Dorn, Hanson & Price (2014)
"Return of the Solow Paradox? IT, Productivity, and Employment in U.S. Manufacturing"
AER Papers & Proceedings, 104(5): 394-399.
"""
import os
import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.regression.linear_model import WLS

# ── Paths ──
BASE = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE, '..', '112803-V1', 'Autor-AER-PP-2014-Replication-Package')
DTA_DIR = os.path.join(DATA_DIR, 'dta')
CLEAN_DTA = os.path.join(DTA_DIR, 'nber-ces', 'clean', 'nber-ces-clean.dta')
OUT_DIR = os.path.join(BASE, 'output')
os.makedirs(OUT_DIR, exist_ok=True)

# ── Key parameters ──
YEAR_MIN = 1980
YEAR_MAX = 2009
BASE_YEAR = 1980  # coefficient normalized to zero in this year

# Computer sector definition (broad, following Houseman et al.)
COMP_BROAD_SICS = [
    # SIC 357x except 3579
    3571, 3572, 3575, 3577, 3578,
    # Audio/video
    3651, 3652,
    # SIC 366x
    3661, 3663,
    # SIC 367x
    3671, 3672, 3674, 3675, 3676, 3677, 3678, 3679,
    # Miscellaneous
    3695,
    # Instruments
    3812, 3822, 3823, 3824, 3825, 3826, 3829, 3844, 3845, 3873,
]

# CI weighting scheme: 5/32 on 77/82/87/92, 3/16 on 02/07
CI_WEIGHTS = {1977: 5/32, 1982: 5/32, 1987: 5/32, 1992: 5/32, 2002: 3/16, 2007: 3/16}


def load_clean_data():
    """Load the pre-built clean dataset."""
    df = pd.read_stata(CLEAN_DTA, convert_categoricals=False)
    return df


def compute_emp_weights(df, year_min=1980, year_max=2009):
    """
    Compute employment weights: each industry's average share of total
    manufacturing employment over the sample period.
    """
    sub = df.loc[(df['year'] >= year_min) & (df['year'] <= year_max),
                 ['sic87dd', 'year', 'emp']].copy()
    totemp = sub.groupby('year')['emp'].transform('sum')
    sub['empsh'] = sub['emp'] / totemp
    wt = sub.groupby('sic87dd')['empsh'].mean().reset_index()
    wt.columns = ['sic87dd', 'wt']
    return wt


def run_it_year_regression(df, depvar, itvar, weight_col='wt',
                           cluster_col='sic87dd', years=None):
    """
    Run the core regression from equation (1):
      log_Y = ind_FE + time_FE + sum_{t=81}^{09} beta_t * IT_j * 1{year==t} + e

    Weighted by employment shares, clustered SEs by industry.
    Returns DataFrame with columns: year, beta, se.
    """
    if years is None:
        years = list(range(YEAR_MIN, YEAR_MAX + 1))

    data = df[df['year'].isin(years)].copy()

    # Industry FE dummies
    ind_dummies = pd.get_dummies(data['sic87dd'], prefix='ind', drop_first=False).astype(float)

    # Year dummies (1981-2009, omitting base year 1980)
    year_dummies = pd.get_dummies(data['year'], prefix='time', drop_first=False).astype(float)
    year_cols_to_use = [c for c in year_dummies.columns if c != f'time_{YEAR_MIN}']
    year_dummies = year_dummies[year_cols_to_use]

    # IT x year interactions (1981-2009)
    it_cols = []
    for y in range(YEAR_MIN + 1, YEAR_MAX + 1):
        col = f'IT_{y}'
        data[col] = data[itvar].values * (data['year'].values == y).astype(float)
        it_cols.append(col)

    # Build X matrix (no constant — absorbed by industry FE)
    X = pd.concat([
        ind_dummies.reset_index(drop=True),
        year_dummies.reset_index(drop=True),
        data[it_cols].reset_index(drop=True)
    ], axis=1)

    y_var = data[depvar].reset_index(drop=True)
    w = data[weight_col].reset_index(drop=True)
    clusters = data[cluster_col].reset_index(drop=True)

    # Drop any rows with NaN
    mask = y_var.notna() & w.notna() & X.notna().all(axis=1)
    X = X.loc[mask]
    y_var = y_var.loc[mask]
    w = w.loc[mask]
    clusters = clusters.loc[mask]

    # WLS with cluster-robust SEs
    model = WLS(y_var, X, weights=w)
    result = model.fit(cov_type='cluster', cov_kwds={'groups': clusters}, use_t=True)

    # Extract IT x year coefficients
    rows = [{'year': YEAR_MIN, 'beta': 0.0, 'se': 0.0}]
    for y in range(YEAR_MIN + 1, YEAR_MAX + 1):
        col = f'IT_{y}'
        rows.append({
            'year': y,
            'beta': result.params[col],
            'se': result.bse[col]
        })

    out = pd.DataFrame(rows)
    out['beta_low'] = out['beta'] - 1.96 * out['se']
    out['beta_high'] = out['beta'] + 1.96 * out['se']
    return out, result
