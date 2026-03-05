"""
04_data_audit.py: Data quality audit for replication of
Bryan, Karlan & Osman (AER, 2024) "Big Loans to Small Businesses"
"""

import os, sys
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
from utils import MERGED_PATH, PENALTY_PATH, OUTPUT_DIR


def main():
    print("=" * 60)
    print("04_data_audit.py: Data Quality Audit for 192297-V1")
    print("=" * 60)

    merged = pd.read_parquet(MERGED_PATH)
    penalty = pd.read_parquet(PENALTY_PATH)

    issues = []

    # (1) Sample coverage
    print("\n--- (1) Sample Coverage ---")
    n_clients = merged['client_key'].nunique()
    n_unique = merged[merged['unique'] == 1]['client_key'].nunique()
    n_panel = len(merged)
    print(f"  Total clients: {n_clients}")
    print(f"  Unique==1 observations: {n_unique}")
    print(f"  Panel rows: {n_panel} ({n_panel // n_clients} rounds × {n_clients} clients)")

    # Treatment balance
    d = merged[merged['unique'] == 1]
    treat_n = d['treatment'].value_counts()
    print(f"  Treatment: {treat_n.get(1, 0)}, Control: {treat_n.get(0, 0)}")
    if abs(treat_n.get(1, 0) - treat_n.get(0, 0)) > 50:
        issues.append("Large treatment imbalance")

    # (2) Missing data patterns
    print("\n--- (2) Missing Data Patterns ---")
    key_vars = ['b_age', 'b_gender', 'b_monthly_profits', 'b_monthly_revenue',
                'b_monthly_expenses', 'treatment', 'loid_baseline']
    for v in key_vars:
        if v in d.columns:
            n_miss = d[v].isna().sum()
            pct = 100 * n_miss / len(d)
            flag = ' [HIGH]' if pct > 5 else ''
            print(f"  {v}: {n_miss} missing ({pct:.1f}%){flag}")
            if pct > 10:
                issues.append(f"High missingness in {v}: {pct:.1f}%")

    # Differential attrition by treatment
    panel = merged[merged['round'] != 0]
    for r in [1, 2]:
        rd = panel[panel['round'] == r]
        survey_col = 'e_tooksurvey' if 'e_tooksurvey' in rd.columns else None
        if survey_col:
            t_attrition = 1 - rd.loc[rd['treatment'] == 1, survey_col].mean()
            c_attrition = 1 - rd.loc[rd['treatment'] == 0, survey_col].mean()
            diff = t_attrition - c_attrition
            print(f"  Round {r} attrition: T={t_attrition:.3f}, C={c_attrition:.3f}, diff={diff:+.3f}")
            if abs(diff) > 0.05:
                issues.append(f"Differential attrition in round {r}: {diff:+.3f}")

    # (3) Outlier analysis
    print("\n--- (3) Outlier Analysis ---")
    financial_vars = ['b_monthly_profits', 'b_monthly_revenue', 'b_monthly_expenses',
                      'e_month_profit_all', 'e_month_rev_all', 'e_month_exp_all']
    for v in financial_vars:
        if v in merged.columns:
            vals = merged[v].dropna()
            q1, q3 = vals.quantile(0.25), vals.quantile(0.75)
            iqr = q3 - q1
            upper = q3 + 3 * iqr
            lower = q1 - 3 * iqr
            n_outliers = ((vals > upper) | (vals < lower)).sum()
            pct = 100 * n_outliers / len(vals)
            print(f"  {v}: {n_outliers} outliers ({pct:.1f}%), range=[{vals.min():.0f}, {vals.max():.0f}]")
            if pct > 5:
                issues.append(f"Many outliers in {v}: {pct:.1f}%")

    # (4) Logical consistency
    print("\n--- (4) Logical Consistency ---")

    # Age vs experience
    if 'b_age' in d.columns and 'b_experience' in d.columns:
        impossible = d[(d['b_experience'] > d['b_age'] - 10) & d['b_experience'].notna() & d['b_age'].notna()]
        print(f"  Experience > age-10: {len(impossible)} cases")
        if len(impossible) > 10:
            issues.append(f"Experience > age-10: {len(impossible)} cases")

    # Profits = revenue - expenses
    if all(c in d.columns for c in ['b_monthly_profits', 'b_monthly_revenue', 'b_monthly_expenses']):
        calc_profit = d['b_monthly_revenue'] - d['b_monthly_expenses']
        diff = (d['b_monthly_profits'] - calc_profit).dropna()
        exact_match = (diff.abs() < 1).sum()
        print(f"  Profits ≈ Revenue - Expenses: {exact_match}/{len(diff)} ({100*exact_match/len(diff):.1f}%)")

    # Loan size consistency
    if 'ad_exploan_size' in d.columns and 'took_loan' in d.columns:
        no_loan_nonzero = d[(d['took_loan'] == 0) & (d['ad_exploan_size'] > 0)].shape[0]
        print(f"  took_loan=0 but exploan_size>0: {no_loan_nonzero}")
        if no_loan_nonzero > 0:
            issues.append(f"Inconsistent: {no_loan_nonzero} with took_loan=0 but loan size>0")

    # (5) Distribution checks
    print("\n--- (5) Distribution Checks ---")

    # Heaping in financial variables
    for v in ['b_monthly_profits', 'b_monthly_revenue']:
        if v in d.columns:
            vals = d[v].dropna()
            # Check for round number heaping
            round_100 = (vals % 100 == 0).mean()
            round_1000 = (vals % 1000 == 0).mean()
            print(f"  {v}: {100*round_100:.1f}% round to 100, {100*round_1000:.1f}% round to 1000")

    # (6) Duplicate detection
    print("\n--- (6) Duplicate Detection ---")
    dup_clients = d['client_key'].duplicated().sum()
    print(f"  Duplicate client_key in unique==1: {dup_clients}")
    if dup_clients > 0:
        issues.append(f"Duplicate client_key: {dup_clients}")

    # (7) Loan officer fixed effects
    print("\n--- (7) Loan Officer Fixed Effects ---")
    if 'loid_baseline' in d.columns:
        lo_counts = d['loid_baseline'].value_counts()
        print(f"  N loan officers: {len(lo_counts)}")
        print(f"  Min clients per LO: {lo_counts.min()}, Max: {lo_counts.max()}")
        singletons = (lo_counts == 1).sum()
        if singletons > 0:
            print(f"  Singleton LO groups: {singletons}")
            issues.append(f"Singleton loan officer groups: {singletons}")

    # (8) Treatment assignment
    print("\n--- (8) Treatment Assignment ---")
    if 'loid_baseline' in d.columns:
        lo_treat = d.groupby('loid_baseline')['treatment'].agg(['mean', 'count'])
        pure_treat = (lo_treat['mean'] == 1).sum()
        pure_ctrl = (lo_treat['mean'] == 0).sum()
        print(f"  LOs with only treatment: {pure_treat}")
        print(f"  LOs with only control: {pure_ctrl}")
        if pure_treat + pure_ctrl > 5:
            issues.append(f"LOs with no within-group variation: {pure_treat + pure_ctrl}")

    # (9) Coding bug: profit double-counting
    print("\n--- (9) Known Coding Issues ---")
    print("  e_month_profit_all double-counts e_month_profit_1 via e_month_profit_main")
    print("  (Stata: rowtotal(e_month_profit_*) includes _main = copy of _1)")
    issues.append("CODING BUG: e_month_profit_all double-counts primary business profits")

    # Summary
    print("\n" + "=" * 60)
    print("AUDIT SUMMARY")
    print("=" * 60)
    if issues:
        for i, issue in enumerate(issues, 1):
            print(f"  {i}. {issue}")
    else:
        print("  No issues found.")

    return issues


if __name__ == '__main__':
    main()
