"""
Replication of: "The Declining Fortunes of the Young Since 2000"
Beaudry, Green, and Sand (2014), AER Papers & Proceedings 104(5): 381-386

Shared paths, variable lists, and helper functions.
"""
import os
import numpy as np
import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

# ── Paths ──
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, '..', '112801-V1', 'DataAndPrograms_P2014_1128')
RAW_DTA = os.path.join(DATA_DIR, 'MorgWorking.dta')
OUTPUT_DIR = os.path.join(BASE_DIR, 'output')
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ── Education codes ──
EDUC_BA = 4       # exactly college degree
EDUC_POST = 5     # post-college degree

# ── Occupation codes (occ4) ──
OCC_COGNITIVE = 1
OCC_CLERICAL = 2
OCC_SERVICE = 3   # called "Manual" in code label, "Service" in paper
OCC_PRODUCTION = 4
OCC_NILF = 5      # not employed

OCC_LABELS = {1: 'Cognitive', 2: 'Clerical', 3: 'Service', 4: 'Production', 5: 'NILF'}

# ── Cohort/time parameters ──
JEB_MIN = 1990
JEB_MAX = 2010
MAX_EXPERIENCE_YEARS = 4  # keep if myrs <= 4 (first 5 years: 0,1,2,3,4)


def load_working_data():
    """Load MorgWorking.dta with numeric codes."""
    return pd.read_stata(RAW_DTA, convert_categoricals=False)


def prepare_sample(df, educ_code, gender='all', drop_nilf=True):
    """
    Prepare the analysis sample matching Figures-Final.do logic.

    Parameters:
        df: raw MorgWorking dataframe
        educ_code: 4 (BA) or 5 (post-college)
        gender: 'all', 'male', or 'female'
        drop_nilf: if True, drop occ4==5 (employed only, for employment shares)

    Returns:
        DataFrame ready for analysis
    """
    # Filter by gender and education
    d = df.copy()
    if gender == 'male':
        d = d[d['female'] == 0]
    elif gender == 'female':
        d = d[d['female'] == 1]
    d = d[d['educ'] == educ_code]

    # Job entry year
    d['je'] = d['year'] - d['pexp']

    # Birth year
    d['dob'] = d['year'] - d['age']

    # 2-year bins for job entry
    d['jeb'] = d['je'].copy()
    d.loc[d['je'] % 2 == 1, 'jeb'] = d['je'] - 1

    # 2-year bins for birth year
    d['dobb'] = d['dob'].copy()
    d.loc[d['dob'] % 2 == 1, 'dobb'] = d['dob'] - 1

    # Cohort based on dob + 25
    d['cohort'] = d['dob'] + 25
    d['bcoh'] = d['cohort'].copy()
    d.loc[d['cohort'] % 2 == 1, 'bcoh'] = d['cohort'] - 1

    # Keep job entry bins 1990-2010
    d = d[(d['jeb'] >= JEB_MIN) & (d['jeb'] <= JEB_MAX)]

    # Replace allocated wages with missing
    d.loc[d['alloc'] == 1, 'lnrhrw_cpi'] = np.nan

    # Drop NILF if requested (for employment share figures)
    if drop_nilf:
        d = d[d['occ4'] != OCC_NILF]

    # Occupation dummies
    for i in range(1, 6):
        d[f'Op{i}'] = (d['occ4'] == i).astype(float)

    # Years on market
    d['myrs'] = d['year'] - d['jeb']

    # Keep first 5 years of experience
    d = d[d['myrs'] <= MAX_EXPERIENCE_YEARS]

    return d.reset_index(drop=True)


def collapse_by_cohort(d):
    """
    Collapse data by year/educ/jeb, matching Stata collapse command.
    Computes mean occupation shares, median log wage, obs count, sum of weights.
    """
    def agg_func(group):
        w = group['wgt'].values
        total_w = w.sum()
        result = {}
        for i in range(1, 6):
            col = f'Op{i}'
            if col in group.columns:
                result[col] = np.average(group[col].values, weights=w)

        # Median wage (unweighted p50 in Stata collapse)
        valid_wage = group['lnrhrw_cpi'].dropna()
        result['med'] = valid_wage.median() if len(valid_wage) > 0 else np.nan

        result['obs'] = len(group)
        result['wgt'] = total_w
        return pd.Series(result)

    collapsed = d.groupby(['year', 'educ', 'jeb']).apply(agg_func).reset_index()
    collapsed['myrs'] = collapsed['year'] - collapsed['jeb']
    return collapsed


def smooth_profiles(collapsed, educ_code, variables=None):
    """
    Regression smoothing matching Stata code:
    reg var i.jeb#(c.myrs) c.myrs i.jeb [aw=wgt]

    This fits: var = constant + sum_j(beta_j * jeb_j * myrs) + gamma*myrs + sum_j(delta_j * jeb_j)
    i.e., cohort-specific slopes on experience + cohort fixed effects + common experience effect
    """
    if variables is None:
        variables = ['med'] + [f'Op{i}' for i in range(1, 6) if f'Op{i}' in collapsed.columns]

    d = collapsed[collapsed['educ'] == educ_code].copy()

    # Build design matrix: jeb dummies, myrs, jeb*myrs interactions
    jeb_vals = sorted(d['jeb'].unique())
    base_jeb = jeb_vals[0]  # Stata uses first as base for i.jeb

    results = {}
    for var in variables:
        if var not in d.columns:
            continue
        valid = d.dropna(subset=[var, 'wgt'])
        if len(valid) == 0:
            continue

        y = valid[var].values
        myrs = valid['myrs'].values
        w = valid['wgt'].values
        jeb = valid['jeb'].values

        # Build X: constant, myrs, jeb dummies (excl base), jeb*myrs interactions (excl base)
        X_parts = [np.ones(len(y)), myrs]
        for j in jeb_vals[1:]:
            X_parts.append((jeb == j).astype(float))
        for j in jeb_vals[1:]:
            X_parts.append(((jeb == j) * myrs).astype(float))

        X = np.column_stack(X_parts)

        # WLS
        try:
            model = sm.WLS(y, X, weights=w)
            fit = model.fit()
            pred = fit.predict(X)
            d.loc[valid.index, f'sm_{var}'] = pred
        except Exception as e:
            print(f"  Warning: smoothing failed for {var}: {e}")

    return d


def apred_weighted_means(d, variables, weight_col='wgt'):
    """
    Replicate the apred ado: compute weighted means by year within each jeb group.
    Returns a dataframe with year, jeb, and the weighted means.
    """
    results = []
    for jeb_val in sorted(d['jeb'].unique()):
        sub = d[d['jeb'] == jeb_val]
        for yr in sorted(sub['year'].unique()):
            row = {'year': yr, 'jeb': jeb_val}
            yr_data = sub[sub['year'] == yr]
            w = yr_data[weight_col].values
            if w.sum() == 0:
                continue
            for var in variables:
                if var in yr_data.columns:
                    vals = yr_data[var].values
                    mask = ~np.isnan(vals) & ~np.isnan(w)
                    if mask.sum() > 0:
                        row[var] = np.average(vals[mask], weights=w[mask])
                    else:
                        row[var] = np.nan
            results.append(row)
    return pd.DataFrame(results)


def index_var_demean(df, variables, by_col='myrs', filter_val=None):
    """
    Replicate IndexVar with index(dm): subtract the mean by group.
    var_i = var - mean(var) within by group
    """
    d = df.copy()
    if filter_val is not None:
        d = d[d[by_col] == filter_val]

    for var in variables:
        if var in d.columns:
            group_mean = d.groupby(by_col)[var].transform('mean')
            d[f'{var}_i'] = d[var] - group_mean
    return d


def make_cohort_profile_plot(data, var, ylabel, title, filename,
                             smoothed=True, year_col='year', jeb_col='jeb'):
    """Create a cohort profile plot with one line per job-entry cohort."""
    fig, ax = plt.subplots(figsize=(7, 5))

    markers = ['o', 's', '^', 'v', 'D', '<', '>', 'p', 'h', '*', '+', 'x']
    linestyles = ['-', '--', '-.', ':']

    jeb_vals = sorted(data[jeb_col].unique())
    for idx, jeb in enumerate(jeb_vals):
        sub = data[data[jeb_col] == jeb].sort_values(year_col)
        marker = markers[idx % len(markers)]
        ls = linestyles[idx % len(linestyles)]
        ax.plot(sub[year_col], sub[var], marker=marker, linestyle=ls,
                markersize=4, linewidth=1, label=str(int(jeb)))

    ax.set_xlabel('Year')
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, filename), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved {filename}")


def print_section(title):
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")
