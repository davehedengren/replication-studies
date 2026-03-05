"""
Replication of: "Using School Choice Lotteries to Test Measures of School Effectiveness"
Author: David J. Deming (NBER WP 19803, 2014)
AER Papers & Proceedings

Shared paths, variable lists, and helper functions.
"""
import os
import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats

# === PATHS ===
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, '..', '112805-V1', 'Deming_AERPandP_datafolder')
DTA_PATH = os.path.join(DATA_DIR, 'cms_VAManalysis.dta')
OUTPUT_DIR = BASE_DIR

# === GRADES & YEARS ===
GRADES = list(range(4, 9))          # 4 through 8
LAG_GRADES = list(range(3, 8))      # 3 through 7
YEARS = list(range(1998, 2005))      # 1998 through 2004
VAM_YEARS_02 = [2002]               # 1-year prior
VAM_YEARS_2YR = [2001, 2002]        # 2-year prior
VAM_YEARS_ALL = list(range(1998, 2003))  # all years up to 2002

# Models available in public data (no demographics)
MODELS = [1, 2]
ESTIMATORS = ['ar', 'FE', 'mix']
SAMPLES = ['02', '2yr', 'all']
OUTCOMES = ['test']  # average of math + reading

# Schools to merge VA for
SCHOOL_TYPES = ['as', 'ch1', 'ch2', 'ch3', 'hm']


def load_data():
    """Load the main Stata dataset."""
    df = pd.read_stata(DTA_PATH, convert_categoricals=False)
    return df


def create_sample_indicators(df):
    """Create sample and onmargin_sample indicators matching Stata code."""
    df['sample'] = ((df['future_grd'] >= 4) & (df['future_grd'] <= 8) &
                    (df['miss_02'] == 0)).astype(int)
    df['onmargin_sample'] = ((df['onmargin'] == 1) & (df['sample'] == 1)).astype(int)
    return df


def create_year_vars(df):
    """Create year_g variables: year when student was in grade g with prior year in grade g-1."""
    for g in GRADES:
        df[f'year_{g}'] = np.nan
        f = g - 1
        for y in range(1998, 2005):
            x = y - 1
            mask = (df[f'grade{y}'] == g) & (df[f'grade{x}'] == f)
            df.loc[mask, f'year_{g}'] = y
    return df


def create_school_cohort_fe(df):
    """Create school-by-cohort FE variables."""
    for g in GRADES:
        # Group school_g and year_g
        temp = df[[f'school_{g}', f'year_{g}']].copy()
        mask = temp.notna().all(axis=1)
        df[f'schXcohortFE_{g}'] = np.nan
        if mask.any():
            codes, _ = pd.factorize(
                temp.loc[mask].apply(lambda r: f"{r.iloc[0]}_{r.iloc[1]}", axis=1)
            )
            df.loc[mask, f'schXcohortFE_{g}'] = codes
    # Also year-based FE
    for y in range(1999, 2005):
        temp = df[[f'school{y}', f'grade{y}']].copy()
        mask = temp.notna().all(axis=1)
        df[f'schXcohortFE_{y}'] = np.nan
        if mask.any():
            codes, _ = pd.factorize(
                temp.loc[mask].apply(lambda r: f"{r.iloc[0]}_{r.iloc[1]}", axis=1)
            )
            df.loc[mask, f'schXcohortFE_{y}'] = codes
    return df


def create_test_scores(df):
    """Create average test scores (test_g = rowmean of math_g, read_g)."""
    for g in range(3, 9):
        m = df[f'math_{g}']
        r = df[f'read_{g}']
        df[f'test_{g}'] = (m + r) / 2
        # Handle cases where one is missing
        both_miss = m.isna() & r.isna()
        one_miss = m.isna() ^ r.isna()
        df.loc[one_miss & m.isna(), f'test_{g}'] = r.loc[one_miss & m.isna()]
        df.loc[one_miss & r.isna(), f'test_{g}'] = m.loc[one_miss & r.isna()]
        df.loc[both_miss, f'test_{g}'] = np.nan
    return df


def create_imputed_scores(df):
    """Create imputed scores with missing indicators and polynomials for lag grades."""
    # Grade-based lags (grades 3-7)
    for y in range(3, 8):
        for x in ['math', 'read']:
            col = f'{x}_{y}'
            df[f'{x}_{y}_imp'] = df[col].copy()
            df[f'{x}_{y}_miss'] = df[col].isna().astype(float)
            df.loc[df[f'{x}_{y}_miss'] == 1, f'{x}_{y}_imp'] = 0.0
            df[f'{x}_{y}_imp_sq'] = df[f'{x}_{y}_imp'] ** 2
            df[f'{x}_{y}_imp_cub'] = df[f'{x}_{y}_imp'] ** 3

    # Year-based scores (1998-2004)
    for y in range(1998, 2005):
        for x in ['math', 'read']:
            col = f'{x}z{y}'
            df[f'{x}_{y}_imp'] = df[col].copy()
            df[f'{x}_{y}_miss'] = df[col].isna().astype(float)
            df.loc[df[f'{x}_{y}_miss'] == 1, f'{x}_{y}_imp'] = 0.0
            df[f'{x}_{y}_imp_sq'] = df[f'{x}_{y}_imp'] ** 2
            df[f'{x}_{y}_imp_cub'] = df[f'{x}_{y}_imp'] ** 3
    return df


def create_year_test_scores(df):
    """Create yearly average test scores: testz{year} = rowmean(mathz{year}, readz{year})."""
    for y in range(1998, 2005):
        m = df[f'mathz{y}']
        r = df[f'readz{y}']
        df[f'testz{y}'] = (m + r) / 2
        both_miss = m.isna() & r.isna()
        one_miss = m.isna() ^ r.isna()
        df.loc[one_miss & m.isna(), f'testz{y}'] = r.loc[one_miss & m.isna()]
        df.loc[one_miss & r.isna(), f'testz{y}'] = m.loc[one_miss & r.isna()]
        df.loc[both_miss, f'testz{y}'] = np.nan
    return df


def create_peer_scores(df):
    """Create peer prior test scores: avg_test{lag} = mean of test_{lag} for non-margin students in same school."""
    for g in GRADES:
        lag = g - 1
        school_col = f'school_{g}'
        test_col = f'test_{lag}'
        # Only non-margin students
        mask = df['onmargin_sample'] != 1
        temp = df.loc[mask, [school_col, test_col]].dropna()
        means = temp.groupby(school_col)[test_col].mean()
        df[f'avg_test{lag}'] = df[school_col].map(means)
    return df


def estimate_vam_ar(df, outcome, covariates, school_col, sample_mask):
    """
    Average residual approach: regress outcome on covariates, take school mean of residuals.
    Returns a Series indexed by school with VAM estimates.
    """
    sub = df.loc[sample_mask, [outcome, school_col] + covariates].dropna().copy()
    if len(sub) < 10:
        return pd.Series(dtype=float)

    y = sub[outcome].values
    if len(covariates) > 0:
        X = sm.add_constant(sub[covariates].values.astype(float))
    else:
        X = np.ones((len(y), 1))

    try:
        model = sm.OLS(y, X).fit()
        sub['resid'] = model.resid
    except Exception:
        return pd.Series(dtype=float)

    return sub.groupby(school_col)['resid'].mean()


def estimate_vam_fe(df, outcome, covariates, school_col, sample_mask):
    """
    Fixed effects approach: demean by school, regress, extract school effect (u).
    Returns a Series indexed by school with VAM estimates.
    """
    sub = df.loc[sample_mask, [outcome, school_col] + covariates].dropna().copy()
    if len(sub) < 10:
        return pd.Series(dtype=float)

    # Demean by school
    y = sub[outcome].astype(float)
    school_means_y = sub.groupby(school_col)[outcome].transform('mean')

    if len(covariates) > 0:
        X = sub[covariates].astype(float)
        school_means_X = sub.groupby(school_col)[covariates].transform('mean')
        y_dm = y - school_means_y
        X_dm = X - school_means_X

        try:
            model = sm.OLS(y_dm.values, X_dm.values).fit()
            sub['resid'] = y.values - X.values @ model.params
        except Exception:
            return pd.Series(dtype=float)
    else:
        sub['resid'] = y.values

    # School FE = school mean of residual (equivalent to xtreg predict, u)
    grand_mean = sub['resid'].mean()
    school_fe = sub.groupby(school_col)['resid'].mean() - grand_mean
    return school_fe


def estimate_vam_mix(df, outcome, covariates, school_col, sample_mask, group_col=None):
    """
    Mixed model (random effects) approach: random intercept for school.
    Returns a Series indexed by school with predicted random effects (BLUP).
    """
    sub = df.loc[sample_mask, [outcome, school_col] + covariates +
                 ([group_col] if group_col else [])].dropna().copy()
    if len(sub) < 10:
        return pd.Series(dtype=float)

    y = sub[outcome].astype(float)
    if len(covariates) > 0:
        X = sm.add_constant(sub[covariates].astype(float))
    else:
        X = np.ones((len(y), 1))

    groups = sub[school_col]

    try:
        model = sm.MixedLM(y, X, groups=groups)
        result = model.fit(reml=True, maxiter=200)
        re = result.random_effects
        vam = pd.Series({k: v.iloc[0] for k, v in re.items()})
        return vam
    except Exception:
        return pd.Series(dtype=float)


def run_2sls_fe(y, endog, exog, instrument, fe_groups, covariates_df=None):
    """
    Run 2SLS with FE (demeaning approach).

    Parameters:
    - y: outcome (Series/array)
    - endog: endogenous variable (Series/array)
    - instrument: instrument (Series/array)
    - fe_groups: group variable for FE (Series/array)
    - covariates_df: DataFrame of exogenous covariates

    Returns dict with coefficient, se, pvalue, n, f_test_eq_1
    """
    # Build dataframe
    data = pd.DataFrame({
        'y': y, 'endog': endog, 'instrument': instrument, 'fe': fe_groups
    }).copy()
    if covariates_df is not None:
        for c in covariates_df.columns:
            data[c] = covariates_df[c].values

    data = data.dropna().reset_index(drop=True)
    n = len(data)
    if n < 20:
        return None

    # Demean by FE
    fe_col = 'fe'
    cols_to_demean = ['y', 'endog', 'instrument']
    if covariates_df is not None:
        cols_to_demean += list(covariates_df.columns)

    for c in cols_to_demean:
        data[c] = data[c].astype(float)
        gm = data.groupby(fe_col)[c].transform('mean')
        data[f'{c}_dm'] = data[c] - gm

    # First stage: endog_dm ~ instrument_dm + covariates_dm
    exog_cols_dm = ['instrument_dm']
    if covariates_df is not None:
        exog_cols_dm += [f'{c}_dm' for c in covariates_df.columns]

    X_fs = data[exog_cols_dm].values
    y_fs = data['endog_dm'].values

    fs_model = sm.OLS(y_fs, X_fs).fit()
    endog_hat = fs_model.fittedvalues

    # Second stage: y_dm ~ endog_hat_dm + covariates_dm
    exog_cols_ss = ['endog_hat']
    data['endog_hat'] = endog_hat
    if covariates_df is not None:
        exog_cols_ss += [f'{c}_dm' for c in covariates_df.columns]

    X_ss = data[exog_cols_ss].values
    y_ss = data['y_dm'].values

    ss_model = sm.OLS(y_ss, X_ss).fit()

    # Get proper 2SLS standard errors
    resid = data['y_dm'].values - np.column_stack(
        [data['endog_dm'].values] +
        ([data[[f'{c}_dm' for c in covariates_df.columns]].values] if covariates_df is not None else [])
    ) @ ss_model.params if covariates_df is not None else (
        data['y_dm'].values - data['endog_dm'].values * ss_model.params[0]
    )

    coef = ss_model.params[0]

    return {
        'coef': coef,
        'n': n,
        'fs_model': fs_model,
        'ss_model': ss_model,
        'data': data
    }


def bootstrap_2sls_fe(y, endog, instrument, fe_groups, covariates_df=None,
                       n_reps=100, cluster_var=None):
    """
    Block bootstrap 2SLS with FE, clustering at cluster_var level.
    Returns dict with coef, se, pval, pval_eq1, n.
    """
    data = pd.DataFrame({
        'y': y, 'endog': endog, 'instrument': instrument, 'fe': fe_groups
    }).copy()
    if covariates_df is not None:
        for c in covariates_df.columns:
            data[c] = covariates_df[c].values
    if cluster_var is not None:
        data['cluster'] = cluster_var

    data = data.dropna().reset_index(drop=True)
    n = len(data)

    # Point estimate
    def estimate_2sls(d):
        fe_col = 'fe'
        cols_to_demean = ['y', 'endog', 'instrument']
        cov_cols = [c for c in d.columns if c not in ['y', 'endog', 'instrument', 'fe', 'cluster']]

        cols_to_demean += cov_cols
        for c in cols_to_demean:
            d[c] = d[c].astype(float)
            gm = d.groupby(fe_col)[c].transform('mean')
            d[f'{c}_dm'] = d[c] - gm

        # First stage
        exog_fs = ['instrument_dm'] + [f'{c}_dm' for c in cov_cols]
        X_fs = d[exog_fs].values
        y_fs = d['endog_dm'].values
        try:
            fs = sm.OLS(y_fs, X_fs).fit()
        except Exception:
            return np.nan
        endog_hat = fs.fittedvalues

        # Second stage
        d['endog_hat'] = endog_hat
        exog_ss = ['endog_hat'] + [f'{c}_dm' for c in cov_cols]
        X_ss = d[exog_ss].values
        y_ss = d['y_dm'].values
        try:
            ss = sm.OLS(y_ss, X_ss).fit()
        except Exception:
            return np.nan
        return ss.params[0]

    point_est = estimate_2sls(data.copy())

    # Bootstrap
    if cluster_var is not None:
        clusters = data['cluster'].unique()
    else:
        clusters = None

    boot_coefs = []
    np.random.seed(42)
    for _ in range(n_reps):
        if clusters is not None:
            # Resample clusters
            boot_clusters = np.random.choice(clusters, size=len(clusters), replace=True)
            boot_data = pd.concat(
                [data[data['cluster'] == c].copy() for c in boot_clusters],
                ignore_index=True
            )
        else:
            boot_data = data.sample(n=n, replace=True).reset_index(drop=True)

        coef = estimate_2sls(boot_data)
        if not np.isnan(coef):
            boot_coefs.append(coef)

    boot_coefs = np.array(boot_coefs)
    se = np.std(boot_coefs, ddof=1) if len(boot_coefs) > 1 else np.nan

    # p-value for coef = 0
    if se > 0:
        t_stat = point_est / se
        pval = 2 * (1 - stats.norm.cdf(abs(t_stat)))
        # F-test for VA = 1
        t_stat_1 = (point_est - 1) / se
        pval_eq1 = 2 * (1 - stats.norm.cdf(abs(t_stat_1)))
    else:
        pval = np.nan
        pval_eq1 = np.nan

    return {
        'coef': point_est,
        'se': se,
        'pval': pval,
        'pval_eq1': pval_eq1,
        'n': n,
        'n_boot': len(boot_coefs)
    }


def run_xtreg_fe(y, X, fe_groups, cluster_var=None):
    """
    Fixed effects regression with optional clustering.
    Like Stata's xtreg, fe vce(cluster ...).
    Returns statsmodels result.
    """
    data = pd.DataFrame({'y': y}).copy()
    data['fe'] = fe_groups.values if hasattr(fe_groups, 'values') else fe_groups
    for i, col in enumerate(X.columns if hasattr(X, 'columns') else range(X.shape[1])):
        data[f'x_{i}'] = X.iloc[:, i].values if hasattr(X, 'iloc') else X[:, i]
    if cluster_var is not None:
        data['cluster'] = cluster_var.values if hasattr(cluster_var, 'values') else cluster_var

    data = data.dropna().reset_index(drop=True)

    # Demean
    x_cols = [c for c in data.columns if c.startswith('x_')]
    for c in ['y'] + x_cols:
        data[c] = data[c].astype(float)
        gm = data.groupby('fe')[c].transform('mean')
        data[f'{c}_dm'] = data[c] - gm

    X_dm = data[[f'{c}_dm' for c in x_cols]].values
    y_dm = data['y_dm'].values

    if cluster_var is not None:
        model = sm.OLS(y_dm, X_dm).fit(
            cov_type='cluster',
            cov_kwds={'groups': data['cluster'].values}
        )
    else:
        model = sm.OLS(y_dm, X_dm).fit(cov_type='HC1')

    return model


def sd_of_school_effects(vam_series):
    """Compute SD of school-level VAM estimates."""
    return vam_series.std() if len(vam_series) > 1 else np.nan
