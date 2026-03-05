"""
05_table2.py — Table 2: Network Index Regressions

6 OLS regressions of business_network_index on female + mechanism variables.
Uses robust (HC1) standard errors and missing indicator approach.

Columns:
  (1) female + controls
  (2) + collegedegree
  (3) + business_at_home
  (4) + hours_childcare
  (5) all mechanism variables
  (6) all mechanism variables + product FE

NOTE: output.do restores business_network_index (replaces -9 back to NaN)
before running Table 2. This is because for Table 3, business_network_index
was used as a regressor (filled with -9), but for Table 2, it's the
dependent variable, so missing values should be dropped naturally.
"""

import sys
import os
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
from utils import (run_ols, add_missing_indicators, format_regression_table,
                   format_stars, CONTROLS, PRODUCT_DUMMIES, CLEAN_DATA_PATH)


def run_table2(df):
    """Run all 6 regressions for Table 2."""
    dep_var = 'business_network_index'

    # Create missing indicators for mechanism variables (output.do lines 213-217)
    mechanism_vars = ['collegedegree', 'business_at_home', 'hours_childcare']
    df, m_cols = add_missing_indicators(df, mechanism_vars)

    results = []
    industry_flags = []

    # (1) female + controls only
    x1 = ['female'] + CONTROLS
    results.append(run_ols(df, dep_var, x1))
    industry_flags.append('No')

    # (2) + collegedegree
    x2 = ['female', 'collegedegree', 'm_collegedegree'] + CONTROLS
    results.append(run_ols(df, dep_var, x2))
    industry_flags.append('No')

    # (3) + business_at_home
    x3 = ['female', 'business_at_home', 'm_business_at_home'] + CONTROLS
    results.append(run_ols(df, dep_var, x3))
    industry_flags.append('No')

    # (4) + hours_childcare
    x4 = ['female', 'hours_childcare', 'm_hours_childcare'] + CONTROLS
    results.append(run_ols(df, dep_var, x4))
    industry_flags.append('No')

    # (5) all mechanism variables
    x5 = (['female', 'collegedegree', 'business_at_home', 'hours_childcare',
            'm_collegedegree', 'm_business_at_home', 'm_hours_childcare'] + CONTROLS)
    results.append(run_ols(df, dep_var, x5))
    industry_flags.append('No')

    # (6) + product FE
    x6 = (['female', 'collegedegree', 'business_at_home', 'hours_childcare',
            'm_collegedegree', 'm_business_at_home', 'm_hours_childcare'] +
           CONTROLS + PRODUCT_DUMMIES)
    results.append(run_ols(df, dep_var, x6))
    industry_flags.append('Yes')

    return results, industry_flags


def display_table2(results, industry_flags, df):
    """Display Table 2 results."""
    dep_var = 'business_network_index'
    keep_vars = ['female', 'collegedegree', 'business_at_home', 'hours_childcare']
    var_labels = {
        'female': 'Female',
        'collegedegree': 'College Degree',
        'business_at_home': 'Business at Home',
        'hours_childcare': 'Hours Spent Caregiving Last Week',
    }

    dep_mean = df[dep_var].mean()
    show_mean = [dep_mean] * len(results)

    output = format_regression_table(
        results, "Networking Index",
        keep_vars, var_labels,
        show_industry=industry_flags,
        show_mean=show_mean,
    )

    print("\n" + "=" * 120)
    print("TABLE 2: Determinants of Business Network Index")
    print("=" * 120)
    print(output)

    return output


def main():
    print("=" * 60)
    print("05_table2.py: Network Index Regressions (Table 2)")
    print("=" * 60)

    df = pd.read_parquet(CLEAN_DATA_PATH)
    print(f"Data loaded: {len(df)} observations")

    results, industry_flags = run_table2(df)
    output = display_table2(results, industry_flags, df)

    with open('replication/output/table2.txt', 'w') as f:
        f.write("TABLE 2: Determinants of Business Network Index\n")
        f.write(output)
    print("\nSaved: replication/output/table2.txt")

    return results


if __name__ == '__main__':
    main()
