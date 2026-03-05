"""
Utility functions for replication of Nielsen & Rehbeck (2022)
"When Choices Are Mistakes"
Paper ID: 164661
"""

import os
import numpy as np
import pandas as pd
from scipy import stats

# ── Paths ──────────────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, '..', '164661-V1')
OUTPUT_DIR = os.path.join(BASE_DIR, 'output')
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ── Constants ──────────────────────────────────────────────────────────────────
AXIOM_LABELS = {1: 'IIA', 2: 'FOSD', 3: 'TRANS', 4: 'IND', 5: 'BRANCH', 6: 'CONS'}
AXIOM_ORDER = [1, 2, 3, 4, 5, 6]

# Number of lottery questions per axiom
QUESTIONS_PER_AXIOM = {1: 4, 2: 4, 3: 3, 4: 3, 5: 1, 6: 2}  # total = 17

RECONCILE_LABELS = {0: 'Keep Inconsistent', 1: 'Unselect Rule',
                    2: 'Change Lotteries', 3: 'Change Both'}

RULEREVISE_LABELS = {1: 'Keep Both', 2: 'Unselect c-Axiom',
                     3: 'Unselect Axiom', 4: 'Unselect Both'}


# ── Data Loading ───────────────────────────────────────────────────────────────
def load_data():
    """Load main.dta and return DataFrame."""
    fp = os.path.join(DATA_DIR, 'main.dta')
    df = pd.read_stata(fp, convert_categoricals=False)
    return df


def filter_lab_nocost(df):
    return df[(df['online'] == 0) & (df['cost'] == 0)].copy()


def filter_lab_cost(df):
    return df[(df['online'] == 0) & (df['cost'] == 1)].copy()


def filter_online(df):
    return df[df['online'] == 1].copy()


def filter_lab_all(df):
    return df[df['online'] == 0].copy()


def filter_nocost_all(df):
    return df[df['cost'] == 0].copy()


# ── Statistical Helpers ────────────────────────────────────────────────────────
def crosstab_row_pct(df, row_var, col_var, margins=True):
    """Cross-tabulation with row percentages (like Stata's tab ..., column)."""
    ct = pd.crosstab(df[row_var], df[col_var], margins=margins)
    ct_pct = ct.div(ct.iloc[:, -1] if margins else ct.sum(axis=1), axis=0) * 100
    return ct, ct_pct


def fisher_exact_2x2(table):
    """Fisher exact test on 2x2 table. Returns (odds_ratio, p_value_two_sided)."""
    return stats.fisher_exact(table)


def wilcoxon_ranksum(x, y):
    """Wilcoxon rank-sum test (Mann-Whitney U). Returns (statistic, p_value)."""
    x_clean = np.array(x).astype(float)
    y_clean = np.array(y).astype(float)
    x_clean = x_clean[~np.isnan(x_clean)]
    y_clean = y_clean[~np.isnan(y_clean)]
    return stats.ranksums(x_clean, y_clean)


def wilcoxon_signrank(x, y=None):
    """Wilcoxon signed-rank test. If y is None, test x against 0."""
    x_clean = np.array(x).astype(float)
    x_clean = x_clean[~np.isnan(x_clean)]
    if y is not None:
        y_clean = np.array(y).astype(float)
        y_clean = y_clean[~np.isnan(y_clean)]
        # Need paired data
        return stats.wilcoxon(x_clean, y_clean)
    else:
        return stats.wilcoxon(x_clean)


def signrank_vs_constant(x, c=0.5):
    """Stata's signrank x = c: signed-rank test of x against constant c."""
    x_clean = np.array(x).astype(float)
    x_clean = x_clean[~np.isnan(x_clean)]
    d = x_clean - c
    d = d[d != 0]
    if len(d) == 0:
        return None, 1.0
    return stats.wilcoxon(d)


def spearman_corr(x, y):
    """Spearman rank correlation. Returns (rho, p_value)."""
    mask = ~(np.isnan(x) | np.isnan(y))
    return stats.spearmanr(np.array(x)[mask], np.array(y)[mask])


def ttest_ind(x, y):
    """Independent samples t-test. Returns (statistic, p_value)."""
    x_clean = np.array(x).astype(float)
    y_clean = np.array(y).astype(float)
    x_clean = x_clean[~np.isnan(x_clean)]
    y_clean = y_clean[~np.isnan(y_clean)]
    return stats.ttest_ind(x_clean, y_clean)


# ── Display Helpers ────────────────────────────────────────────────────────────
def axiom_label(num):
    return AXIOM_LABELS.get(int(num), str(num))


def print_section(title):
    print(f"\n{'='*70}")
    print(f"  {title}")
    print(f"{'='*70}")


def print_subsection(title):
    print(f"\n--- {title} ---")


def n_subjects(df):
    """Count unique subjects in a DataFrame."""
    return df['subject'].nunique()
