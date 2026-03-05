"""
Utility functions for replication of:
Chan, Cropper, and Malik (2014) - "Why Are Power Plants in India Less Efficient
than Power Plants in the United States?"
AER Papers & Proceedings, 104(5): 586-590
"""
import os
import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.spatial.distance import cdist

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, '..', '112820-V1', 'P2014_1156_data')
OUTPUT_DIR = BASE_DIR


def load_us_data():
    return pd.read_stata(os.path.join(DATA_DIR, 'USdata_export.dta'),
                         convert_categoricals=False)


def load_india_data():
    return pd.read_stata(os.path.join(DATA_DIR, 'Indiadata_export.dta'),
                         convert_categoricals=False)


def load_cems_data():
    return pd.read_stata(os.path.join(DATA_DIR, 'USdata_cemsmerge.dta'),
                         convert_categoricals=False)


def _clean_us_base(us):
    """Common US cleaning steps shared by regression and matching samples."""
    us['india'] = 0
    us['nameplatesq'] = us['nameplate'] ** 2
    us['vintagesq'] = us['vintage'] ** 2
    us['w_vintagesq'] = us['w_vintage'] ** 2

    # Drop heatrate outliers (Stata treats missing as +inf, so also drop missing)
    us = us[us['heatrate'].notna() &
            (us['heatrate'] <= 20000) & (us['heatrate'] >= 6000)].copy()
    # Drop vintage outliers (same missing logic)
    us = us[us['w_vintage'].notna() & us['vintage'].notna() &
            (us['w_vintage'] >= 1910) & (us['vintage'] >= 1910)].copy()

    # Convert capfactor to percent
    us['capfactor'] = us['capfactor'] * 100
    us.loc[us['hours'] == 0, 'hours'] = np.nan
    us['availability'] = us['hours'] / (24 * 365) * 100

    # Drop CHP and non-utility sectors
    us = us[~((us['combinedheatpower'] == 1) | (us['eiasectornumber'] > 3))].copy()
    us['capfactorsq'] = us['capfactor'] ** 2

    # Correct US age/vintage to capacity-weighted
    us['age'] = us['w_age'] + 1
    us['agesq'] = us['w_agesq']
    us['agecube'] = us['w_agecube']
    us['vintage'] = us['w_vintage']
    us['vintagesq'] = us['w_vintagesq']
    return us


def build_regression_sample():
    """Build pooled regression sample following regressioncode.do."""
    us = _clean_us_base(load_us_data())
    india = load_india_data()
    df = pd.concat([us, india], ignore_index=True)
    df = df.rename(columns={'capfactor': 'capf'})

    df['log_heatrate'] = np.log(df['heatrate'])
    df['log_btu'] = np.log(df['btucontent'])
    df['missingbtu'] = df['log_btu'].isna().astype(float)
    df.loc[df['btucontent'].isna(), 'btucontent'] = 0
    df.loc[df['log_btu'].isna(), 'log_btu'] = 0

    df = df.dropna(subset=['capf', 'vintage', 'private']).copy()
    df = df[~((df['private'] == 1) & (df['india'] == 1))].copy()

    df['elec'] = df['elec'].fillna(0)
    df['elec_private'] = df['elec'] * df['private']
    df['avgnp'] = df.groupby('plantcode')['nameplate'].transform('mean')

    for y in range(1988, 2011):
        df[f'year{y}'] = (df['year'] == y).astype(float)

    df = df[df['nameplate'] >= 25].copy()

    for y in range(1988, 2011):
        df[f'indiayear{y}'] = df['india'] * df[f'year{y}']

    # Scale for numerical stability (matches Stata code)
    for vv in ['nameplate', 'nameplatesq', 'vintagesq', 'agesq', 'agecube',
               'capfactorsq']:
        df[vv] = df[vv] / 1000
    df['nameplatesq'] = df['nameplatesq'] / 1000  # extra /1000 => total /1e6

    df = df[df['year'] <= 2009].copy()
    return df


def build_matching_sample_heatrate():
    """Build matching sample for heat rate (nnmatch_heatrate.do)."""
    us = _clean_us_base(load_us_data())
    india = load_india_data()
    df = pd.concat([us, india], ignore_index=True)
    df = df.rename(columns={'capfactor': 'capf', 'nameplate': 'np'})
    df['log_heatrate'] = np.log(df['heatrate'])

    df = df.dropna(subset=['heatrate', 'capf', 'vintage', 'private']).copy()
    df = df[~((df['private'] == 1) & (df['india'] == 1))].copy()
    df['avgnp'] = df.groupby('plantcode')['np'].transform('mean')
    df = df[df['np'] >= 25].copy()
    return df


def build_matching_sample_aux():
    """Build matching sample for auxiliary gen (nnmatch_aux.do)."""
    cems = load_cems_data()
    cems['india'] = 0
    cems.loc[cems['grossloadmwh'] == 0, 'grossloadmwh'] = np.nan
    cems.loc[cems['total_generation'] == 0, 'total_generation'] = np.nan
    cems['aux'] = ((cems['grossloadmwh'] - cems['total_generation'])
                   / cems['grossloadmwh'] * 100)
    cems.loc[(cems['aux'] < 0) | (cems['aux'] > 30), 'aux'] = np.nan
    cems = cems.dropna(subset=['aux']).copy()
    cems = cems[~(cems['w_scrub'] > 0)].copy()

    india = load_india_data()
    df = pd.concat([cems, india], ignore_index=True)
    df = df.rename(columns={'nameplate': 'np', 'capfactor': 'capf',
                            'btucontent': 'gcvcoal'})

    df = df.dropna(subset=['aux', 'vintage', 'private', 'np']).copy()
    df = df[~((df['private'] == 1) & (df['india'] == 1))].copy()
    df['avgnp'] = df.groupby('plantcode')['np'].transform('mean')
    df = df[df['np'] >= 25].copy()
    return df


def nn_match(outcome_var, df_year, match_vars, M=5):
    """
    Nearest neighbor matching with Mahalanobis distance.
    Returns (SATT, SE).
    """
    treated = df_year[df_year['india'] == 1].copy()
    control = df_year[df_year['india'] == 0].copy()
    N_T, N_C = len(treated), len(control)

    if N_T == 0 or N_C == 0:
        return np.nan, np.nan

    X_T = treated[match_vars].values.astype(float)
    X_C = control[match_vars].values.astype(float)
    Y_T = treated[outcome_var].values.astype(float)
    Y_C = control[outcome_var].values.astype(float)

    # Mahalanobis distance using pooled covariance
    X_all = np.vstack([X_T, X_C])
    cov = np.cov(X_all.T)
    if cov.ndim == 0:
        cov = np.array([[cov]])
    cov_inv = np.linalg.inv(cov + np.eye(cov.shape[0]) * 1e-10)

    dists = cdist(X_T, X_C, metric='mahalanobis', VI=cov_inv)
    M_use = min(M, N_C)

    tau_i = np.zeros(N_T)
    for i in range(N_T):
        nn_idx = np.argsort(dists[i])[:M_use]
        Y_matched = Y_C[nn_idx]
        tau_i[i] = Y_T[i] - Y_matched.mean()

    satt = tau_i.mean()
    se = tau_i.std(ddof=1) / np.sqrt(N_T) if N_T > 1 else np.nan
    return satt, se


def run_ols_clustered(df, y_var, x_vars, cluster_var):
    """Run OLS with clustered standard errors."""
    reg_df = df.dropna(subset=[y_var] + x_vars + [cluster_var]).copy()
    # Drop columns with zero variance
    x_use = [c for c in x_vars if reg_df[c].std() > 0]
    X = sm.add_constant(reg_df[x_use])
    y = reg_df[y_var]
    model = sm.OLS(y, X).fit(
        cov_type='cluster', cov_kwds={'groups': reg_df[cluster_var]})
    return model, reg_df
