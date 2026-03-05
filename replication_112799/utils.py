"""
Replication utilities for Paper 112799:
Arceo-Gomez & Campos-Vazquez (2014)
"Race and Marriage in the Labor Market: A Discrimination Correspondence Study"

Correspondence study in Mexico: ~8,000 fictitious CVs sent to ~1,000 job ads.
Varies gender, phenotype (European/Mestizo/Indigenous photos), marital status.
Method: LPM with clustered SEs at firm (id_offer) level.
"""

import os
import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats

# ── Paths ──────────────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "..", "112799-V1", "Data", "data")
DTA_FILE = os.path.join(DATA_DIR, "Arceo_Campos_AEAP-P2014.dta")

# ── Variable lists ─────────────────────────────────────────────────────────────
# Outcome
DEPVAR = "callback"

# Key treatment variables
TREATMENTS = ["sex", "photo", "married"]

# Photo dummies (omitted = photo 3 = indigenous)
# photo1 = European, photo2 = Mestizo, photo4 = No photo
PHOTO_DUMMIES = ["photo1", "photo2", "photo4"]

# Control variables (matching Stata global X)
CONTROLS = [
    "ss_degree",          # social science degree (=business major)
    "scholarship",        # scholarship dummy
    "public_highschool",  # public high school dummy
    "other_language",     # foreign language dummy
    "some_availab",       # time availability dummy
    "leadership",         # leadership activity dummy
    "age",                # age (continuous)
    "english",            # English proficiency dummy
]

# Core regressors for Table 4 Column [1] (no photo dummies)
TABLE4_COL1 = ["sex", "public_college", "married"]

# Core regressors for Table 4 Column [2] (with individual photo dummies)
TABLE4_COL2 = ["sex", "public_college", "married", "photo1", "photo2", "photo4"]

# For women-only regressions (Table 5), drop sex
TABLE5_CORE = ["public_college", "married"]

# Cluster variable
CLUSTER_VAR = "id_offer"

# Sample filter for balanced sample (8 CVs per firm)
BALANCED_FILTER = "all8"


# ── Helper functions ───────────────────────────────────────────────────────────
def load_data():
    """Load the main dataset."""
    df = pd.read_stata(DTA_FILE, convert_categoricals=False)
    return df


def run_ols_cluster(df, depvar, regressors, cluster_var, add_constant=True):
    """
    Run OLS with clustered standard errors.
    Stata: reg y x, robust cluster(id_offer)
    Returns statsmodels RegressionResults.
    """
    y = df[depvar].astype(float)
    X = df[regressors].astype(float)
    if add_constant:
        X = sm.add_constant(X)
    model = sm.OLS(y, X, missing='drop')
    # Stata cluster SEs with HC1 adjustment
    result = model.fit(cov_type='cluster', cov_kwds={'groups': df[cluster_var]})
    return result


def run_fe_ols_cluster(df, depvar, regressors, fe_var, cluster_var):
    """
    Run OLS with entity fixed effects and clustered SEs.
    Stata: xtreg y x, fe i(id_offer) robust cluster(id_offer)
    Uses FWL (demeaning) approach.
    """
    # Demean variables by fe_var
    y = df[depvar].astype(float)
    X = df[regressors].astype(float)

    group_means_y = y.groupby(df[fe_var]).transform('mean')
    y_dm = y - group_means_y

    X_dm = X.copy()
    for col in X_dm.columns:
        group_means = X_dm[col].groupby(df[fe_var]).transform('mean')
        X_dm[col] = X_dm[col] - group_means

    # No constant after demeaning
    model = sm.OLS(y_dm, X_dm, missing='drop')

    n_groups = df[fe_var].nunique()
    n_obs = len(df)
    k = len(regressors)

    result = model.fit(
        cov_type='cluster',
        cov_kwds={'groups': df[cluster_var]},
    )

    # Adjust DOF for absorbed FE (Stata xtreg,fe style)
    # Stata DOF: (N - n_groups - k) / (N - 1) * (n_clusters - 1) / n_clusters
    # The cluster adjustment already handles some of this
    return result


def callback_rate(df, group_col=None, group_val=None):
    """Compute callback rate (percentage) for a subset."""
    if group_col is not None:
        sub = df[df[group_col] == group_val]
    else:
        sub = df
    return sub[DEPVAR].mean() * 100


def ttest_by_group(df, group_col, val1, val2, outcome=DEPVAR):
    """Two-sample t-test for callback rates between two groups."""
    g1 = df[df[group_col] == val1][outcome].dropna()
    g2 = df[df[group_col] == val2][outcome].dropna()
    t_stat, p_val = stats.ttest_ind(g1, g2)
    return t_stat, p_val


def chi2_independence(df, row_var, col_var):
    """Chi-squared test of independence (Pearson)."""
    ct = pd.crosstab(df[row_var], df[col_var])
    chi2, p, dof, expected = stats.chi2_contingency(ct)
    return chi2, p


def stars(p):
    """Return significance stars."""
    if p < 0.01:
        return "***"
    elif p < 0.05:
        return "**"
    elif p < 0.10:
        return "*"
    return ""


def format_coef(coef, se, p):
    """Format coefficient with SE in brackets and stars."""
    return f"{coef:.3f}{stars(p)}  [{se:.3f}]"


if __name__ == "__main__":
    df = load_data()
    print(f"Loaded data: {df.shape[0]} obs, {df.shape[1]} vars")
    print(f"Men: {(df.sex == 0).sum()}, Women: {(df.sex == 1).sum()}")
    print(f"Callback rate: {df.callback.mean()*100:.2f}%")
    print(f"all8 sample: {(df.all8 == 1).sum()}")
    print(f"Unique firms: {df.id_offer.nunique()}")
