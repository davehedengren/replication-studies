"""
Shared utilities for the Ghana Gender Gaps replication study.

Includes:
- zindex: Z-score index construction (replicates Stata zindex.ado)
- Regression helper functions
- Constants and variable lists
"""

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats


# ── Variable lists used across scripts ──────────────────────────────────────

PRODUCT_DUMMIES = [
    'product_aginput', 'product_livestock', 'product_crop', 'product_cocoa',
    'product_forestry', 'product_fish', 'product_processed'
]

CONTROLS = ['age', 'years_operation', 'business_primary_inc']

TABLE1_VARS = [
    'profit', 'years_operation', 'workers_total', 'business_at_home',
    'hours_business', 'hours_salary', 'hours_childcare',
    'product_aginput', 'product_livestock', 'product_crop', 'product_cocoa',
    'product_forestry', 'product_fish', 'product_processed', 'product_other',
    'married', 'age', 'collegedegree', 'children',
    'business_primary_inc', 'work_for_pay'
]

TABLE1_LABELS = {
    'profit': 'Profit last month (GHS)',
    'years_operation': 'Years in operation',
    'workers_total': 'Total workers (excl. owner)',
    'business_at_home': 'Business located at home',
    'hours_business': 'Hours worked at business last week',
    'hours_salary': 'Hours worked at wage employment last week',
    'hours_childcare': 'Hours spent caregiving last week',
    'product_aginput': 'Agricultural inputs',
    'product_livestock': 'Livestock',
    'product_crop': 'Crops',
    'product_cocoa': 'Cocoa',
    'product_forestry': 'Forestry',
    'product_fish': 'Fishing',
    'product_processed': 'Agro-Processing',
    'product_other': 'Other',
    'married': 'Married or in partnership',
    'age': 'Age',
    'collegedegree': 'College degree',
    'children': 'Number of children',
    'business_primary_inc': 'Primary income from business',
    'work_for_pay': 'Work for pay',
}

NETWORK_SIZE_VARS = [
    'business_help_n', 'meet_other_business', 'meet_other_business_n',
    'business_association', 'total_collab', 'suppliers_n', 'clients_n'
]

NETWORK_SIZE_LABELS = {
    'business_help_n': '# Business Owners\nfor Advice',
    'meet_other_business': 'Meet Other\nBusiness Owners',
    'meet_other_business_n': '# Business Owners\nMeet Regularly',
    'business_association': 'Business\nAssociation',
    'total_collab': '# Business\nCollaborations',
    'suppliers_n': '# Suppliers',
    'clients_n': '# Clients',
}

# Figure 1a only plots 5 of the 7 (matching Stata code which plots indices 1,3,5,6,7)
FIGURE1A_VARS = [
    'business_help_n', 'meet_other_business_n',
    'total_collab', 'suppliers_n', 'clients_n'
]

FIGURE1A_LABELS = [
    '# Business Owners\nfor Advice',
    '# Business Owners\nMeet Regularly',
    '# Business\nCollaborations',
    '# Suppliers',
    '# Clients',
]

FEMALE_SHARE_VARS = [
    'share_business_help_women', 'share_meet_business_women',
    'share_suppliers_impt_women', 'share_clients_impt_women'
]

FRIENDREL_SHARE_VARS = [
    'share_business_help_friendrel', 'share_meet_business_friendrel',
    'share_suppliers_impt_friendrel', 'share_clients_impt_friendrel'
]

FIGURE1B_LABELS = [
    'Business Owners\nfor Advice',
    'Business Owners\nMeet Regularly',
    'Suppliers',
    'Business Clients',
]

DATA_PATH = '228101-V1/data/raw/gender_gaps_ghana.dta'
CLEAN_DATA_PATH = 'replication/output/gender_gaps_ghana_clean.parquet'


# ── zindex: Z-score index construction ──────────────────────────────────────

def zindex(df, varlist, gen_name=None):
    """
    Replicate Stata's zindex.ado:
    1. For each variable, compute z-score: (x - mean) / sd
    2. Take row mean of z-scores (ignoring NaN, like Stata's egen rowmean)
    3. Re-standardize the row mean: (rowmean - mean(rowmean)) / sd(rowmean)

    Parameters
    ----------
    df : DataFrame
    varlist : list of str
        Column names to include in the index.
    gen_name : str, optional
        Name for the resulting index column (only used in diagnostic output).

    Returns
    -------
    pd.Series
        The z-score index values.
    """
    zscores = pd.DataFrame(index=df.index)

    for var in varlist:
        col = df[var].astype(float)
        m = col.mean()   # nanmean by default in pandas
        s = col.std()    # ddof=1 by default (matches Stata)
        zscores[f'z_{var}'] = (col - m) / s

    # Row mean ignoring NaN (matches Stata's egen rowmean)
    z_index_mean = zscores.mean(axis=1, skipna=True)

    # Re-standardize
    m2 = z_index_mean.mean()
    s2 = z_index_mean.std()  # ddof=1
    result = (z_index_mean - m2) / s2

    # Diagnostic: count missing
    if gen_name:
        for var in varlist:
            n_miss = df[var].isna().sum()
            if n_miss > 0:
                print(f"  zindex({gen_name}): {n_miss}/{len(df)} missing for {var}")

    return result


# ── Regression helpers ──────────────────────────────────────────────────────

def run_ols(df, y_var, x_vars, robust=True):
    """
    Run OLS regression with robust (HC1) standard errors.

    Parameters
    ----------
    df : DataFrame
    y_var : str
        Dependent variable name.
    x_vars : list of str
        Independent variable names (constant is added automatically).
    robust : bool
        Whether to use HC1 robust standard errors.

    Returns
    -------
    statsmodels RegressionResultsWrapper
    """
    subset = df[[y_var] + x_vars].dropna()
    y = subset[y_var]
    X = sm.add_constant(subset[x_vars])

    model = sm.OLS(y, X)
    if robust:
        results = model.fit(cov_type='HC1')
    else:
        results = model.fit()

    return results


def add_missing_indicators(df, varnames, fill_value=-9):
    """
    Replicate Stata's missing indicator approach:
    For each variable, create m_<varname> = 1 if missing, 0 otherwise,
    then replace missing values with fill_value.

    Parameters
    ----------
    df : DataFrame
    varnames : list of str
    fill_value : numeric

    Returns
    -------
    DataFrame with added m_ columns and filled values.
    list of str: names of the missing indicator columns created.
    """
    df = df.copy()
    m_cols = []
    for var in varnames:
        m_col = f'm_{var}'
        df[m_col] = df[var].isna().astype(int)
        df[var] = df[var].fillna(fill_value)
        m_cols.append(m_col)
    return df, m_cols


def format_stars(pval):
    """Return significance stars based on p-value (Stata convention)."""
    if pval < 0.01:
        return '***'
    elif pval < 0.05:
        return '**'
    elif pval < 0.10:
        return '*'
    return ''


def format_regression_table(results_list, dep_var_label, keep_vars, var_labels=None,
                            show_industry=None, show_mean=None):
    """
    Format a list of regression results into a display table.

    Parameters
    ----------
    results_list : list of statsmodels results
    dep_var_label : str
    keep_vars : list of str
        Variables to display (in order).
    var_labels : dict, optional
        Mapping from variable name to display label.
    show_industry : list of str, optional
        "Yes"/"No" for each column.
    show_mean : list of float, optional
        Mean of dependent variable for each column.

    Returns
    -------
    str : Formatted table string.
    """
    if var_labels is None:
        var_labels = {}

    n_cols = len(results_list)
    col_headers = [f'({i+1})' for i in range(n_cols)]

    lines = []
    lines.append(f"Dependent variable: {dep_var_label}")
    lines.append("-" * (30 + 15 * n_cols))
    header = f"{'':30s}" + "".join(f"{h:>15s}" for h in col_headers)
    lines.append(header)
    lines.append("-" * (30 + 15 * n_cols))

    for var in keep_vars:
        label = var_labels.get(var, var)
        coef_row = f"{label:30s}"
        se_row = f"{'':30s}"

        for res in results_list:
            if var in res.params.index:
                coef = res.params[var]
                se = res.bse[var]
                pval = res.pvalues[var]
                stars = format_stars(pval)
                coef_row += f"{coef:>12.3f}{stars:3s}"
                se_row += f"{'(' + f'{se:.3f}' + ')':>15s}"
            else:
                coef_row += f"{'':>15s}"
                se_row += f"{'':>15s}"

        lines.append(coef_row)
        lines.append(se_row)

    lines.append("-" * (30 + 15 * n_cols))

    if show_industry:
        row = f"{'Product FE':30s}" + "".join(f"{v:>15s}" for v in show_industry)
        lines.append(row)

    if show_mean:
        row = f"{'Mean':30s}" + "".join(f"{v:>15.3f}" for v in show_mean)
        lines.append(row)

    r2_row = f"{'R-squared':30s}" + "".join(f"{res.rsquared:>15.3f}" for res in results_list)
    lines.append(r2_row)

    n_row = f"{'N':30s}" + "".join(f"{int(res.nobs):>15d}" for res in results_list)
    lines.append(n_row)

    lines.append("-" * (30 + 15 * n_cols))

    return "\n".join(lines)
