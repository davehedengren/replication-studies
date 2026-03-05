"""
utils.py – Shared utilities for replication of
Banerjee, Duflo & Hornbeck (2014)
"Bundling Health Insurance and Microfinance in India"
AER 104(5): 291-297
"""

import os
import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, '..', '112788-V1', 'P2014_1150_data', 'Data')
OUTPUT_DIR = os.path.join(BASE_DIR, 'output')
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ── Data Loading ──────────────────────────────────────────────────────────────

def load_analysis_sample():
    """Load analysis_sample_PUBLISH.dta (5366 obs, unit = sks_id)."""
    return pd.read_stata(os.path.join(DATA_DIR, 'analysis_sample_PUBLISH.dta'),
                         convert_categoricals=False)

def load_loans():
    """Load clean_loans_annual_PUBLISH.dta (56060 obs)."""
    return pd.read_stata(os.path.join(DATA_DIR, 'clean_loans_annual_PUBLISH.dta'),
                         convert_categoricals=False)

def load_baseline():
    """Load clean_baseline_PUBLISH.dta (5366 obs)."""
    return pd.read_stata(os.path.join(DATA_DIR, 'clean_baseline_PUBLISH.dta'),
                         convert_categoricals=False)

def load_endline():
    """Load clean_endline_PUBLISH.dta (5366 obs)."""
    return pd.read_stata(os.path.join(DATA_DIR, 'clean_endline_PUBLISH.dta'),
                         convert_categoricals=False)

def load_treatment():
    """Load clean_treatment_PUBLISH.dta (447 centers)."""
    return pd.read_stata(os.path.join(DATA_DIR, 'clean_treatment_PUBLISH.dta'),
                         convert_categoricals=False)

def load_attrition():
    """Load attrition_sample_PUBLISH.dta."""
    return pd.read_stata(os.path.join(DATA_DIR, 'attrition_sample_PUBLISH.dta'),
                         convert_categoricals=False)

# ── Baseline raw data loaders ─────────────────────────────────────────────────

def load_baseline_crosswalk():
    return pd.read_stata(os.path.join(DATA_DIR, 'baseline', 'baseline_crosswalk_PUBLISH.dta'),
                         convert_categoricals=False)

def load_baseline_hh():
    return pd.read_stata(os.path.join(DATA_DIR, 'baseline', 'hh_master_PUBLISH.dta'),
                         convert_categoricals=False)

def load_baseline_hh_events():
    return pd.read_stata(os.path.join(DATA_DIR, 'baseline', 'hh_master_e_PUBLISH.dta'),
                         convert_categoricals=False)

def load_baseline_adult():
    return pd.read_stata(os.path.join(DATA_DIR, 'baseline', 'adult_master_PUBLISH.dta'),
                         convert_categoricals=False)

def load_endline_crosswalk():
    return pd.read_stata(os.path.join(DATA_DIR, 'endline', 'endline_crosswalk_PUBLISH.dta'),
                         convert_categoricals=False)

# ── Statistical Helpers ───────────────────────────────────────────────────────

def areg(df, yvar, xvars, absorb='stratify', cluster='village_id'):
    """
    Stata-style areg: OLS with absorbed FE + cluster-robust SEs.

    Parameters
    ----------
    df : DataFrame
    yvar : str – dependent variable
    xvars : list of str – regressors (treatment, interactions, etc.)
    absorb : str – variable whose FE are absorbed (demeaned out)
    cluster : str – cluster variable for robust SEs

    Returns
    -------
    statsmodels RegressionResultsWrapper
    """
    data = df[list(set([yvar] + xvars + [absorb, cluster]))].dropna().copy()

    # Demean by absorbed FE
    for v in [yvar] + xvars:
        data[v] = data[v].astype(float)
        group_means = data.groupby(absorb)[v].transform('mean')
        data[v] = data[v] - group_means

    y = data[yvar]
    X = sm.add_constant(data[xvars])

    model = sm.OLS(y, X).fit(
        cov_type='cluster',
        cov_kwds={'groups': data[cluster]},
        use_t=True
    )
    return model


def areg_with_stats(df, yvar, xvars, absorb='stratify', cluster='village_id'):
    """
    Like areg but also returns N and control group mean.
    """
    data = df[list(set([yvar] + xvars + [absorb, cluster, 'treatment']))].dropna().copy()
    control_mean = data.loc[data['treatment'] == 0, yvar].mean()
    n = len(data)

    result = areg(df, yvar, xvars, absorb, cluster)
    return result, control_mean, n


# ── Display Helpers ───────────────────────────────────────────────────────────

def print_section(title):
    print(f"\n{'═' * 70}")
    print(f"  {title}")
    print(f"{'═' * 70}")

def print_subsection(title):
    print(f"\n── {title} {'─' * max(0, 60 - len(title))}")

def fmt_coef(b, se, stars=True, pval=None):
    """Format coefficient with stars."""
    star = ''
    if stars and pval is not None:
        if pval <= 0.01:
            star = '***'
        elif pval <= 0.05:
            star = '**'
        elif pval <= 0.1:
            star = '*'
    return f"{b:.3f}{star}", f"({se:.3f})"

def fmt_mean_sd(mean, sd):
    """Format mean [sd]."""
    return f"{mean:.3f}", f"[{sd:.3f}]"
