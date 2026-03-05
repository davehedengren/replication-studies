"""
04_table3.py — Table 3: Profit Regressions with Network Indices

6 OLS regressions of profit on female + network indices + controls.
Uses robust (HC1) standard errors and the missing indicator approach
(replace missing index values with -9, add missing indicator dummy).

Columns:
  (1) female + controls
  (2) + business_network_index
  (3) + business_network_female_index
  (4) + business_network_friendrel_index
  (5) all three indices
  (6) all three indices + product FE
"""

import sys
import os
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
from utils import (run_ols, add_missing_indicators, format_regression_table,
                   format_stars, CONTROLS, PRODUCT_DUMMIES, CLEAN_DATA_PATH)


def prepare_data(df):
    """Prepare data with missing indicators for network indices."""
    # Create missing indicators and fill with -9 (matching output.do lines 153-160)
    index_vars = [
        'business_network_index',
        'business_network_female_index',
        'business_network_friendrel_index',
    ]
    df, m_cols = add_missing_indicators(df, index_vars)
    return df, m_cols


def run_table3(df):
    """Run all 6 regressions for Table 3."""
    df, m_cols = prepare_data(df)

    # Dependent variable
    dep_var = 'profit'

    # Index variable names and their missing indicators
    bni = 'business_network_index'
    bfi = 'business_network_female_index'
    bfri = 'business_network_friendrel_index'
    m_bni = 'm_business_network_index'
    m_bfi = 'm_business_network_female_index'
    m_bfri = 'm_business_network_friendrel_index'

    results = []
    industry_flags = []

    # (1) female + controls
    x1 = ['female'] + CONTROLS
    results.append(run_ols(df, dep_var, x1))
    industry_flags.append('No')

    # (2) + business_network_index
    x2 = ['female', bni, m_bni] + CONTROLS
    results.append(run_ols(df, dep_var, x2))
    industry_flags.append('No')

    # (3) + business_network_female_index
    x3 = ['female', bfi, m_bfi] + CONTROLS
    results.append(run_ols(df, dep_var, x3))
    industry_flags.append('No')

    # (4) + business_network_friendrel_index
    x4 = ['female', bfri, m_bfri] + CONTROLS
    results.append(run_ols(df, dep_var, x4))
    industry_flags.append('No')

    # (5) all three indices
    x5 = ['female', bni, bfi, bfri, m_bni, m_bfi, m_bfri] + CONTROLS
    results.append(run_ols(df, dep_var, x5))
    industry_flags.append('No')

    # (6) all three + product FE
    x6 = ['female', bni, bfi, bfri, m_bni, m_bfi, m_bfri] + CONTROLS + PRODUCT_DUMMIES
    results.append(run_ols(df, dep_var, x6))
    industry_flags.append('Yes')

    return results, industry_flags


def display_table3(results, industry_flags, df):
    """Display Table 3 results."""
    dep_var = 'profit'
    keep_vars = [
        'female',
        'business_network_index',
        'business_network_female_index',
        'business_network_friendrel_index',
    ]
    var_labels = {
        'female': 'Female',
        'business_network_index': 'Networking Index',
        'business_network_female_index': 'Female Networking Index',
        'business_network_friendrel_index': 'Friends/Relatives Networking Index',
    }

    # Compute mean of dep var (unconditional, matching Stata's "sum profit")
    dep_mean = df[dep_var].mean()
    show_mean = [dep_mean] * len(results)

    output = format_regression_table(
        results, f"Profit last month (GHS)",
        keep_vars, var_labels,
        show_industry=industry_flags,
        show_mean=show_mean,
    )

    print("\n" + "=" * 120)
    print("TABLE 3: Effect of Networking on Profits")
    print("=" * 120)
    print(output)

    return output


def main():
    print("=" * 60)
    print("04_table3.py: Profit Regressions (Table 3)")
    print("=" * 60)

    df = pd.read_parquet(CLEAN_DATA_PATH)
    print(f"Data loaded: {len(df)} observations")

    results, industry_flags = run_table3(df)
    output = display_table3(results, industry_flags, df)

    with open('replication/output/table3.txt', 'w') as f:
        f.write("TABLE 3: Effect of Networking on Profits\n")
        f.write(output)
    print("\nSaved: replication/output/table3.txt")

    return results


if __name__ == '__main__':
    main()
