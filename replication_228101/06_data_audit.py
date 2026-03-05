"""
06_data_audit.py — Data Quality Checks

Performs comprehensive data quality audits:
1. Outlier detection (IQR method) for continuous variables
2. Logical consistency checks
3. Missing data patterns by gender
4. Duplicate detection
5. Distribution anomalies (heaping, suspicious patterns)
6. Coding error checks
"""

import sys
import os
import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, os.path.dirname(__file__))
from utils import CLEAN_DATA_PATH


def section(title):
    print(f"\n{'='*80}")
    print(f"  {title}")
    print(f"{'='*80}\n")


# ── 1. Outlier Detection ────────────────────────────────────────────────────

def check_outliers(df):
    section("1. OUTLIER DETECTION")

    continuous_vars = [
        ('profit', 'Profit last month (GHS)'),
        ('sales_lastmonth', 'Sales last month (GHS)'),
        ('hours_business', 'Hours worked at business'),
        ('hours_salary', 'Hours worked for salary'),
        ('hours_childcare', 'Hours caregiving'),
        ('workers_total', 'Total workers'),
        ('workers_female', 'Female workers'),
        ('years_operation', 'Years in operation'),
        ('age', 'Age'),
        ('children', 'Number of children'),
        ('business_help_n', '# people helped'),
        ('meet_other_business_n', '# business owners meet'),
        ('total_collab', '# collaborations'),
        ('suppliers_n', '# suppliers'),
        ('clients_n', '# clients'),
    ]

    print(f"{'Variable':35s} {'N':>6s} {'Mean':>10s} {'SD':>10s} {'Min':>10s} "
          f"{'P25':>10s} {'Median':>10s} {'P75':>10s} {'Max':>10s} {'IQR Out':>8s}")
    print("-" * 145)

    outlier_summary = []
    for var, label in continuous_vars:
        if var not in df.columns:
            continue
        col = df[var].dropna()
        if len(col) == 0:
            continue
        q1 = col.quantile(0.25)
        q3 = col.quantile(0.75)
        iqr = q3 - q1
        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr
        n_outliers = ((col < lower) | (col > upper)).sum()

        print(f"{label:35s} {len(col):6d} {col.mean():10.1f} {col.std():10.1f} "
              f"{col.min():10.1f} {q1:10.1f} {col.median():10.1f} {q3:10.1f} "
              f"{col.max():10.1f} {n_outliers:8d}")

        if n_outliers > 0:
            outlier_summary.append((var, label, n_outliers, col.max(), upper))

    print("\nVariables with IQR outliers:")
    for var, label, n, max_val, threshold in outlier_summary:
        print(f"  {label}: {n} outliers (max={max_val:.1f}, upper fence={threshold:.1f})")

    # Top extreme profit values
    print("\nTop 10 profit values:")
    top_profits = df.nlargest(10, 'profit')[['caseid', 'profit', 'female', 'age',
                                              'years_operation', 'workers_total']]
    print(top_profits.to_string(index=False))

    return outlier_summary


# ── 2. Logical Consistency ──────────────────────────────────────────────────

def check_logical_consistency(df):
    section("2. LOGICAL CONSISTENCY CHECKS")
    issues = []

    # Check: workers_female <= workers_total
    mask = (df['workers_female'] > df['workers_total']) & df['workers_female'].notna()
    n = mask.sum()
    print(f"workers_female > workers_total: {n} cases")
    if n > 0:
        issues.append(('workers_female > workers_total', n))
        print(df.loc[mask, ['caseid', 'workers_female', 'workers_total']].head())

    # Check: total hours plausibility (< 168 hours/week)
    df['total_hours'] = df[['hours_business', 'hours_salary', 'hours_childcare']].sum(
        axis=1, skipna=True
    )
    mask = (df['total_hours'] > 168) & (df['total_hours'] > 0)
    n = mask.sum()
    print(f"\nTotal hours (business + salary + childcare) > 168/week: {n} cases")
    if n > 0:
        issues.append(('total_hours > 168', n))
        print(df.loc[mask, ['caseid', 'hours_business', 'hours_salary',
                            'hours_childcare', 'total_hours']].head(10))

    # Very high hours
    mask_high = (df['total_hours'] > 100)
    n_high = mask_high.sum()
    print(f"Total hours > 100/week: {n_high} cases")

    # Check: shares in [0,1]
    share_vars = [
        'share_business_help_women', 'share_business_help_friendrel',
        'share_meet_business_women', 'share_meet_business_friendrel',
        'share_suppliers_impt_women', 'share_suppliers_impt_friendrel',
        'share_clients_impt_women', 'share_clients_impt_friendrel',
        'workers_female_share',
    ]
    print("\nShare variables outside [0,1]:")
    for var in share_vars:
        if var not in df.columns:
            continue
        col = df[var].dropna()
        n_below = (col < 0).sum()
        n_above = (col > 1).sum()
        if n_below > 0 or n_above > 0:
            print(f"  {var}: {n_below} below 0, {n_above} above 1")
            issues.append((f'{var} outside [0,1]', n_below + n_above))
        else:
            print(f"  {var}: OK")

    # Check: years_operation vs age consistency
    df['age_at_start'] = df['age'] - df['years_operation']
    mask = (df['age_at_start'] < 10) & df['age_at_start'].notna()
    n = mask.sum()
    print(f"\nStarted business before age 10 (age - years_operation < 10): {n} cases")
    if n > 0:
        issues.append(('business started before age 10', n))
        print(df.loc[mask, ['caseid', 'age', 'years_operation', 'age_at_start']].head(10))

    mask2 = (df['age_at_start'] < 15) & df['age_at_start'].notna()
    print(f"Started business before age 15: {mask2.sum()} cases")

    print(f"\nTotal logical issues found: {sum(i[1] for i in issues)}")
    return issues


# ── 3. Missing Data Patterns ───────────────────────────────────────────────

def check_missing_patterns(df):
    section("3. MISSING DATA PATTERNS BY GENDER")

    key_vars = [
        'profit', 'years_operation', 'workers_total', 'business_at_home',
        'hours_business', 'hours_salary', 'hours_childcare',
        'business_help_n', 'meet_other_business', 'meet_other_business_n',
        'business_association', 'total_collab', 'suppliers_n', 'clients_n',
        'share_business_help_women', 'share_meet_business_women',
        'share_suppliers_impt_women', 'share_clients_impt_women',
        'business_network_index', 'business_network_female_index',
        'business_network_friendrel_index',
        'collegedegree', 'married', 'children',
    ]

    male = df[df['female'] == 0]
    female = df[df['female'] == 1]

    print(f"{'Variable':45s} {'All %miss':>10s} {'Male %miss':>12s} "
          f"{'Female %miss':>13s} {'Diff':>8s}")
    print("-" * 95)

    for var in key_vars:
        if var not in df.columns:
            continue
        all_pct = df[var].isna().mean() * 100
        male_pct = male[var].isna().mean() * 100
        female_pct = female[var].isna().mean() * 100
        diff = male_pct - female_pct

        flag = ''
        if abs(diff) > 5:
            flag = ' ***'
        elif abs(diff) > 2:
            flag = ' *'

        print(f"{var:45s} {all_pct:10.1f} {male_pct:12.1f} "
              f"{female_pct:13.1f} {diff:8.1f}{flag}")

    # Overall missingness
    print(f"\n{'Total observations':45s} {'':>10s} {len(male):12d} {len(female):13d}")

    # Test whether missingness is systematically related to gender
    print("\nChi-squared tests for systematic missingness by gender:")
    for var in ['profit', 'business_network_index', 'business_network_female_index',
                'hours_childcare', 'collegedegree']:
        if var not in df.columns:
            continue
        subset = df[df['female'].notna()]
        contingency = pd.crosstab(subset['female'], subset[var].isna())
        if contingency.shape == (2, 2):
            chi2, p, dof, expected = stats.chi2_contingency(contingency)
            print(f"  {var:40s} chi2={chi2:.3f}, p={p:.3f}")


# ── 4. Duplicate Detection ─────────────────────────────────────────────────

def check_duplicates(df):
    section("4. DUPLICATE DETECTION")

    # Check caseid uniqueness
    n_unique = df['caseid'].nunique()
    n_total = len(df)
    n_dup = n_total - n_unique
    print(f"Total rows: {n_total}")
    print(f"Unique caseids: {n_unique}")
    print(f"Duplicate caseids: {n_dup}")

    if n_dup > 0:
        dups = df[df['caseid'].duplicated(keep=False)]
        print(f"\nDuplicate caseid values:")
        print(dups[['caseid', 'gender', 'age', 'profit']].to_string())

    # Check for suspiciously similar rows (same age, gender, profit, years_operation)
    check_cols = ['gender', 'age', 'years_operation', 'g6']
    subset = df[check_cols].dropna()
    n_dup_rows = subset.duplicated().sum()
    print(f"\nRows with identical (gender, age, years_operation, profit): {n_dup_rows}")
    if n_dup_rows > 0 and n_dup_rows < 20:
        print(df.loc[subset[subset.duplicated(keep=False)].index,
                     ['caseid'] + check_cols].sort_values(check_cols).head(20))


# ── 5. Distribution Anomalies ──────────────────────────────────────────────

def check_distribution_anomalies(df):
    section("5. DISTRIBUTION ANOMALIES")

    # Heaping in profits and sales
    print("Profit heaping (round numbers):")
    profit_valid = df['profit'].dropna()
    for divisor, label in [(1000, 'divisible by 1000'), (500, 'divisible by 500'),
                           (100, 'divisible by 100')]:
        n_round = (profit_valid % divisor == 0).sum()
        pct = n_round / len(profit_valid) * 100
        print(f"  {label}: {n_round}/{len(profit_valid)} ({pct:.1f}%)")

    # Zero profits
    n_zero = (profit_valid == 0).sum()
    print(f"  Zero profit: {n_zero}/{len(profit_valid)} ({n_zero/len(profit_valid)*100:.1f}%)")

    # Negative profits
    n_neg = (profit_valid < 0).sum()
    print(f"  Negative profit: {n_neg}/{len(profit_valid)} ({n_neg/len(profit_valid)*100:.1f}%)")

    # Sales heaping
    print("\nSales heaping:")
    sales_valid = df['sales_lastmonth'].dropna()
    for divisor, label in [(1000, 'divisible by 1000'), (500, 'divisible by 500')]:
        n_round = (sales_valid % divisor == 0).sum()
        pct = n_round / len(sales_valid) * 100
        print(f"  {label}: {n_round}/{len(sales_valid)} ({pct:.1f}%)")

    # Hours distributions - check for digit preference
    print("\nHours distributions:")
    for var, label in [('hours_business', 'Business hours'),
                       ('hours_salary', 'Salary hours'),
                       ('hours_childcare', 'Childcare hours')]:
        col = df[var].dropna()
        if len(col) == 0:
            continue
        print(f"\n  {label} (N={len(col)}):")
        vc = col.value_counts().head(10)
        for val, cnt in vc.items():
            print(f"    {val:6.0f}: {cnt:5d} ({cnt/len(col)*100:.1f}%)")

    # Skewness/kurtosis of key variables
    print("\nSkewness and Kurtosis:")
    for var in ['profit', 'sales_lastmonth', 'hours_business', 'workers_total']:
        col = df[var].dropna()
        if len(col) > 3:
            sk = col.skew()
            ku = col.kurtosis()
            print(f"  {var:30s}  skewness={sk:8.2f}  kurtosis={ku:8.2f}")


# ── 6. Coding Error Checks ─────────────────────────────────────────────────

def check_coding_errors(df):
    section("6. CODING ERROR CHECKS")

    # Check binary variables are actually 0/1
    binary_vars = [
        'female', 'collegedegree', 'married', 'business_primary_inc',
        'work_for_pay', 'business_at_home', 'export',
        'any_business_help', 'business_association', 'meet_other_business',
        'product_aginput', 'product_livestock', 'product_crop',
        'product_cocoa', 'product_forestry', 'product_fish',
        'product_processed', 'product_other',
    ]

    print("Binary variable checks (should be 0/1/NaN only):")
    for var in binary_vars:
        if var not in df.columns:
            continue
        vals = df[var].dropna().unique()
        valid = set(vals).issubset({0, 0.0, 1, 1.0})
        if not valid:
            print(f"  WARNING: {var} has non-binary values: {sorted(vals)}")
        else:
            n_one = (df[var] == 1).sum()
            n_zero = (df[var] == 0).sum()
            n_na = df[var].isna().sum()
            print(f"  {var:35s} 0={n_zero:5d}  1={n_one:5d}  NA={n_na:5d}  OK")

    # Check count variables are non-negative
    print("\nCount variable checks (should be >= 0):")
    count_vars = [
        'business_help_n', 'meet_other_business_n', 'total_collab',
        'suppliers_n', 'clients_n', 'workers_total', 'workers_female',
        'children', 'years_operation',
    ]
    for var in count_vars:
        if var not in df.columns:
            continue
        col = df[var].dropna()
        n_neg = (col < 0).sum()
        if n_neg > 0:
            print(f"  WARNING: {var} has {n_neg} negative values")
        else:
            print(f"  {var:35s} min={col.min():8.1f}  max={col.max():8.1f}  OK")

    # Check the profit variable discrepancy
    print("\n\nCODE BUG NOTE:")
    print("  output.do references 'profit' but clean.do creates 'profits_lastmonth'.")
    print("  Variable 'profit' does not exist in raw data nor is created in clean.do.")
    print("  Our Python code creates 'profit' as an alias of 'profits_lastmonth' (g6).")
    print(f"  profit N={df['profit'].notna().sum()}, mean={df['profit'].mean():.2f}")
    print(f"  profits_lastmonth N={df['profits_lastmonth'].notna().sum()}, "
          f"mean={df['profits_lastmonth'].mean():.2f}")
    print(f"  Values match: {(df['profit'] == df['profits_lastmonth']).all() or (df['profit'].isna() == df['profits_lastmonth'].isna()).all()}")


def main():
    print("=" * 80)
    print("06_data_audit.py: Data Quality Checks")
    print("=" * 80)

    df = pd.read_parquet(CLEAN_DATA_PATH)
    # Also load raw for reference
    raw = pd.read_stata('228101-V1/data/raw/gender_gaps_ghana.dta', convert_categoricals=False)
    df['g6'] = raw['g6'].values  # Keep raw profit for comparison

    print(f"Data loaded: {len(df)} observations")

    outliers = check_outliers(df)
    issues = check_logical_consistency(df)
    check_missing_patterns(df)
    check_duplicates(df)
    check_distribution_anomalies(df)
    check_coding_errors(df)

    section("AUDIT SUMMARY")
    print(f"Total observations: {len(df)}")
    print(f"Valid gender: {df['female'].notna().sum()} (Male={int((df['female']==0).sum())}, "
          f"Female={int((df['female']==1).sum())})")
    print(f"Variables with IQR outliers: {len(outliers)}")
    print(f"Logical consistency issues: {len(issues)}")

    # Save audit report
    import io
    old_stdout = sys.stdout
    sys.stdout = buffer = io.StringIO()

    print(f"Data loaded: {len(df)} observations")
    check_outliers(df)
    check_logical_consistency(df)
    check_missing_patterns(df)
    check_duplicates(df)
    check_distribution_anomalies(df)
    check_coding_errors(df)

    sys.stdout = old_stdout
    with open('replication/output/data_audit.txt', 'w') as f:
        f.write(buffer.getvalue())
    print("\nSaved: replication/output/data_audit.txt")


if __name__ == '__main__':
    main()
