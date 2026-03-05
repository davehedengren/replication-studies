"""
Shared utilities for replication of Altonji, Kahn, Speer (2014)
"Trends in Earnings Differentials Across College Majors and the Changing Task Composition of Jobs"
AEA Papers and Proceedings.
"""
import os
import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.regression.linear_model import WLS

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(os.path.dirname(BASE_DIR), '112802-V1', 'P2014_1129_data')
WORKING_DIR = os.path.join(DATA_DIR, 'Working-Data')
WORKING_DATA = os.path.join(WORKING_DIR, 'working_data2_3_14.dta')
WEIGHT = 'newweight'


def load_data():
    """Load working data and prepare variables for analysis."""
    df = pd.read_stata(WORKING_DATA, convert_categoricals=False)

    # Cast key variables to float64 for precision
    float_cols = ['majorbeta_none', 'majorbeta_hgc_wtime', 'lnearnings',
                  'abstract', 'routine', 'manual',
                  'male', 'black', 'hispanic', 'black_x_male', 'hispanic_x_male',
                  'survey2003', 'survey2009', 'surveydum1', 'surveydum2', 'surveydum3',
                  'potexp', 'hgc', WEIGHT, 'clust_var']
    for col in float_cols:
        df[col] = df[col].astype(np.float64)

    # Demean potexp (as done in paper_results.do)
    mean_pe = np.average(df['potexp'], weights=df[WEIGHT])
    df['potexp_dm'] = df['potexp'] - mean_pe
    df['potexp2_dm'] = df['potexp_dm'] ** 2
    df['potexp3_dm'] = df['potexp_dm'] ** 3

    # Rename task variables to match Stata code
    df['A'] = df['abstract'].astype(np.float64)
    df['R'] = df['routine'].astype(np.float64)
    df['M'] = df['manual'].astype(np.float64)

    # Year dummies (base=1993; 2003 and 2011 dropped for collinearity)
    df['year_2009'] = (df['year'] == 2009).astype(np.float64)
    df['year_2010'] = (df['year'] == 2010).astype(np.float64)

    # HGC dummies (base=16)
    df['hgc_18'] = (df['hgc'] == 18).astype(np.float64)
    df['hgc_19'] = (df['hgc'] == 19).astype(np.float64)
    df['hgc_20'] = (df['hgc'] == 20).astype(np.float64)

    # HGC x survey interactions (exclude hgc20 x survey for collinearity)
    for h in [16, 18, 19]:
        hd = (df['hgc'] == h).astype(np.float64)
        df[f'hgc{h}_s03'] = hd * df['survey2003']
        df[f'hgc{h}_s09'] = hd * df['survey2009']

    return df


def first_stage(df, betam_var):
    """Regress beta^m on A, R, M (WLS with newweight) to get residuals v^m."""
    y = df[betam_var].values
    X = sm.add_constant(df[['A', 'R', 'M']].values)
    w = df[WEIGHT].values

    mod = WLS(y, X, weights=w)
    res = mod.fit()

    resid = res.resid
    # Stata pweights normalize weights to sum=N for RMSE computation.
    # statsmodels doesn't normalize, so we correct: var_v = scale * N / sum(w)
    n = len(y)
    var_v = res.scale * n / np.sum(w)

    coefs = {'A': res.params[1], 'R': res.params[2], 'M': res.params[3]}
    r2 = res.rsquared

    return resid, var_v, coefs, r2


def wcov(x, y, w):
    """Weighted covariance matching Stata's [aw=w] covariance."""
    mx = np.average(x, weights=w)
    my = np.average(y, weights=w)
    sw = np.sum(w)
    return np.sum(w * (x - mx) * (y - my)) / (sw - 1)


def build_design_matrix(df, betam_var=None, grad_controls=True,
                         use_tasks=False, resid_var=None):
    """Build design matrix for earnings regression."""
    cols = {}

    if not use_tasks:
        # Columns 1-2: beta^m specification
        bm = df[betam_var].values
        cols[betam_var] = bm
        cols[f'{betam_var}_x_s03'] = bm * df['survey2003'].values
        cols[f'{betam_var}_x_s09'] = bm * df['survey2009'].values
        cols[f'{betam_var}_x_pe'] = bm * df['potexp_dm'].values
        cols[f'{betam_var}_x_pe2'] = bm * df['potexp2_dm'].values
    else:
        # Columns 3-4: task decomposition
        for var in ['A', 'R', 'M', resid_var]:
            v = df[var].values
            cols[var] = v
            cols[f'{var}_x_s03'] = v * df['survey2003'].values
            cols[f'{var}_x_s09'] = v * df['survey2009'].values
            cols[f'{var}_x_pe'] = v * df['potexp_dm'].values
            cols[f'{var}_x_pe2'] = v * df['potexp2_dm'].values

    if grad_controls:
        cols['hgc_18'] = df['hgc_18'].values
        cols['hgc_19'] = df['hgc_19'].values
        cols['hgc_20'] = df['hgc_20'].values
        for h in [16, 18, 19]:
            cols[f'hgc{h}_s03'] = df[f'hgc{h}_s03'].values
            cols[f'hgc{h}_s09'] = df[f'hgc{h}_s09'].values

    # Demographics
    cols['male'] = df['male'].values
    cols['black'] = df['black'].values
    cols['hispanic'] = df['hispanic'].values
    cols['black_x_male'] = df['black_x_male'].values
    cols['hispanic_x_male'] = df['hispanic_x_male'].values

    # Demographic x experience interactions (demeaned potexp)
    cols['male_pe'] = df['male'].values * df['potexp_dm'].values
    cols['male_pe2'] = df['male'].values * df['potexp2_dm'].values
    cols['hisp_pe'] = df['hispanic'].values * df['potexp_dm'].values
    cols['hisp_pe2'] = df['hispanic'].values * df['potexp2_dm'].values
    cols['black_pe'] = df['black'].values * df['potexp_dm'].values
    cols['black_pe2'] = df['black'].values * df['potexp2_dm'].values

    # Experience
    cols['potexp'] = df['potexp_dm'].values
    cols['potexp2'] = df['potexp2_dm'].values
    cols['potexp3'] = df['potexp3_dm'].values

    # Survey dummies (surveydum3 = base)
    cols['surveydum1'] = df['surveydum1'].values
    cols['surveydum2'] = df['surveydum2'].values

    # Year dummies
    cols['year_2009'] = df['year_2009'].values
    cols['year_2010'] = df['year_2010'].values

    # Constant
    cols['const'] = np.ones(len(df))

    return pd.DataFrame(cols)


def run_wls_cluster(y, X, weights, groups):
    """Run WLS with cluster-robust SEs."""
    mod = WLS(y, X, weights=weights)
    res = mod.fit(cov_type='cluster', cov_kwds={'groups': groups})
    return res


def print_comparison(label, python_val, stata_val, se_python=None, se_stata=None):
    """Print comparison of Python vs published values."""
    diff = python_val - stata_val
    if se_python is not None and se_stata is not None:
        print(f"  {label:30s}: Python={python_val:8.4f} (SE={se_python:.4f})  "
              f"Published={stata_val:8.4f} (SE={se_stata:.4f})  Diff={diff:+.4f}")
    else:
        print(f"  {label:30s}: Python={python_val:8.4f}  Published={stata_val:8.4f}  Diff={diff:+.4f}")
