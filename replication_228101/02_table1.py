"""
02_table1.py — Table 1: Summary Statistics

Replicates Table 1 from the paper:
- Column A: All sample means
- Column B: Male means
- Column C: Female means
- Column D: Difference (t-test) with p-values

Uses two-sample t-test with unequal variance (Welch's t-test),
matching Stata's estpost ttest.
"""

import sys
import os
import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, os.path.dirname(__file__))
from utils import TABLE1_VARS, TABLE1_LABELS, CLEAN_DATA_PATH


def compute_table1(df):
    """Compute summary statistics and t-tests for Table 1."""
    male = df[df['female'] == 0]
    female = df[df['female'] == 1]

    rows = []
    for var in TABLE1_VARS:
        all_mean = df[var].mean()
        all_n = df[var].notna().sum()
        male_mean = male[var].mean()
        male_n = male[var].notna().sum()
        female_mean = female[var].mean()
        female_n = female[var].notna().sum()

        # Two-sample t-test (Welch's, unequal variance)
        male_vals = male[var].dropna()
        female_vals = female[var].dropna()
        if len(male_vals) > 0 and len(female_vals) > 0:
            tstat, pval = stats.ttest_ind(male_vals, female_vals, equal_var=False)
            diff = male_mean - female_mean
        else:
            tstat, pval, diff = np.nan, np.nan, np.nan

        rows.append({
            'variable': var,
            'label': TABLE1_LABELS.get(var, var),
            'all_mean': all_mean,
            'all_n': all_n,
            'male_mean': male_mean,
            'male_n': male_n,
            'female_mean': female_mean,
            'female_n': female_n,
            'diff': diff,
            'tstat': tstat,
            'pval': pval,
        })

    return pd.DataFrame(rows)


def format_table1(table_df):
    """Format Table 1 for display."""
    lines = []
    lines.append("=" * 100)
    lines.append("TABLE 1: Summary Statistics")
    lines.append("=" * 100)
    header = (f"{'Variable':40s} {'All':>10s} {'Male':>10s} {'Female':>10s} "
              f"{'Diff':>10s} {'p-value':>10s}")
    lines.append(header)
    lines.append("-" * 100)

    prev_was_product = False
    for _, row in table_df.iterrows():
        var = row['variable']
        label = row['label']

        # Add "Products" header before first product dummy
        if var == 'product_aginput' and not prev_was_product:
            lines.append(f"{'Products':40s}")
            prev_was_product = True

        if var.startswith('product_'):
            label = f"  {label}"

        stars = ''
        if row['pval'] < 0.01:
            stars = '***'
        elif row['pval'] < 0.05:
            stars = '**'
        elif row['pval'] < 0.10:
            stars = '*'

        line = (f"{label:40s} {row['all_mean']:10.2f} {row['male_mean']:10.2f} "
                f"{row['female_mean']:10.2f} {row['diff']:10.2f}{stars:3s} "
                f"({row['pval']:.2f})")
        lines.append(line)

    lines.append("-" * 100)
    lines.append(f"{'N':40s} {table_df['all_n'].max():10.0f} "
                 f"{table_df['male_n'].max():10.0f} "
                 f"{table_df['female_n'].max():10.0f}")
    lines.append("=" * 100)
    lines.append("Stars: * p<0.10, ** p<0.05, *** p<0.01")
    lines.append("Difference = Male - Female (Welch's t-test, unequal variance)")

    return "\n".join(lines)


def main():
    print("=" * 60)
    print("02_table1.py: Summary Statistics (Table 1)")
    print("=" * 60)

    df = pd.read_parquet(CLEAN_DATA_PATH)
    print(f"Data loaded: {len(df)} observations")
    print(f"  Male: {(df['female']==0).sum()}, Female: {(df['female']==1).sum()}, "
          f"Missing gender: {df['female'].isna().sum()}")

    table_df = compute_table1(df)
    output = format_table1(table_df)
    print("\n" + output)

    # Save
    table_df.to_csv('replication/output/table1.csv', index=False)
    with open('replication/output/table1.txt', 'w') as f:
        f.write(output)
    print("\nSaved: replication/output/table1.csv, table1.txt")

    return table_df


if __name__ == '__main__':
    main()
