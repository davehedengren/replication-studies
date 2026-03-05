"""
04_data_audit.py — Data quality checks for Horn & Loewenstein (2025).
"""

import numpy as np
import pandas as pd
from utils import OUTPUT_DIR, STUDIES, load_study
import os

print("=" * 60)
print("04_data_audit.py — Data quality audit")
print("=" * 60)


# ══════════════════════════════════════════════════════════════════════
# 1. COVERAGE
# ══════════════════════════════════════════════════════════════════════

print("\n── 1. Coverage ──\n")

for study_num in [1, 2, 3, 4, 0]:
    cfg = STUDIES[study_num]
    df = load_study(study_num)
    max_r = cfg['max_rounds']
    print(f"  Study {study_num} ({cfg['name']}): {len(df)} obs, {df.shape[1]} vars")

    # Missing performance data
    perf_cols = [f'r{r}_perf' for r in range(1, max_r + 1)]
    n_complete = df[perf_cols].notna().all(axis=1).sum()
    print(f"    Complete performance data: {n_complete}/{len(df)} "
          f"({100*n_complete/len(df):.1f}%)")


# ══════════════════════════════════════════════════════════════════════
# 2. DISTRIBUTIONS
# ══════════════════════════════════════════════════════════════════════

print("\n── 2. Distributions ──\n")

for study_num in [1, 2, 3]:
    cfg = STUDIES[study_num]
    df = load_study(study_num)
    max_r = cfg['max_rounds']

    perf_cols = [f'r{r}_perf' for r in range(1, max_r + 1)]
    perf_data = df[perf_cols]

    print(f"\n  Study {study_num} ({cfg['name']}):")
    print(f"    Performance range: [{perf_data.min().min():.0f}, {perf_data.max().max():.0f}]")
    print(f"    Mean (across rounds): {perf_data.mean().mean():.2f}")
    print(f"    Median duration: {df['duration'].median():.0f} min")

    # Check for zero-performance participants
    zero_all = (perf_data == 0).all(axis=1).sum()
    print(f"    Zero performance in all rounds: {zero_all}")

    # Improvement check
    improved = (df[f'r{max_r}_perf'] > df['r1_perf']).sum()
    pct_improved = 100 * improved / len(df)
    print(f"    Improved (R{max_r} > R1): {improved}/{len(df)} ({pct_improved:.0f}%)")


# ══════════════════════════════════════════════════════════════════════
# 3. LOGICAL CONSISTENCY
# ══════════════════════════════════════════════════════════════════════

print("\n── 3. Logical consistency ──\n")

for study_num in [1, 2, 3]:
    cfg = STUDIES[study_num]
    df = load_study(study_num)
    max_r = cfg['max_rounds']
    ahead_list = cfg['ahead']

    print(f"\n  Study {study_num}:")

    # Check: perfdiff = predicted - actual
    issues = 0
    for ahead in ahead_list:
        max_t = max_r - ahead
        for t in range(0, max_t + 1):
            r = t + ahead
            diff_col = f't{t}r{ahead}_perfdiff'
            pred_col = f't{t}_pred_r{r}'
            perf_col = f'r{r}_perf'

            if all(c in df.columns for c in [diff_col, pred_col, perf_col]):
                computed = df[pred_col] - df[perf_col]
                diff = (df[diff_col] - computed).abs()
                bad = (diff > 0.001).sum()
                if bad > 0:
                    issues += bad
                    print(f"    {diff_col}: {bad} rows where diff != pred - actual")

    if issues == 0:
        print(f"    All perfdiff variables consistent with pred - actual ✓")

    # Check learning diff consistency
    learn_issues = 0
    for col in [c for c in df.columns if c.endswith('_learndiff')]:
        t = int(col.split('_')[0][1:])
        # learndiff_t = predicted_learning - actual_learning
        ahead = ahead_list[1]  # learning uses the larger ahead value
        r1 = t + 1
        r2 = t + ahead
        pred_learn_col = f't{t}_pred_r{r1}r{r2}'
        actual_learn_col = f'learning_r{r1}r{r2}'

        if all(c in df.columns for c in [pred_learn_col, actual_learn_col]):
            computed = df[pred_learn_col] - df[actual_learn_col]
            diff = (df[col] - computed).abs()
            bad = (diff > 0.001).sum()
            if bad > 0:
                learn_issues += bad

    if learn_issues == 0:
        print(f"    All learndiff variables consistent ✓")


# Study 4: Check performer assignment
print("\n  Study 4:")
df4 = load_study(4)
print(f"    Total: {len(df4)} (Performers: {(df4['performer']==1).sum()}, "
      f"Predictors: {(df4['performer']==0).sum()})")
# Check if performer IDs match Study 1
df1 = load_study(1)
perf_ids = set(df4[df4['performer'] == 1]['id'].unique())
s1_ids = set(df1['id'].unique())
overlap = len(perf_ids & s1_ids)
print(f"    Performer IDs overlapping with Study 1: {overlap}/{len(perf_ids)}")


# ══════════════════════════════════════════════════════════════════════
# 4. MISSING DATA PATTERNS
# ══════════════════════════════════════════════════════════════════════

print("\n── 4. Missing data patterns ──\n")

for study_num in [1, 2, 3]:
    cfg = STUDIES[study_num]
    df = load_study(study_num)
    max_r = cfg['max_rounds']

    # Check for any missing performance or prediction data
    perf_miss = df[[f'r{r}_perf' for r in range(1, max_r + 1)]].isna().sum().sum()
    pred_cols = [c for c in df.columns if '_pred_' in c]
    pred_miss = df[pred_cols].isna().sum().sum()
    diff_cols = [c for c in df.columns if 'perfdiff' in c or 'learndiff' in c]
    diff_miss = df[diff_cols].isna().sum().sum()

    print(f"  Study {study_num}: perf missing={perf_miss}, "
          f"pred missing={pred_miss}, diff missing={diff_miss}")


# ══════════════════════════════════════════════════════════════════════
# 5. DUPLICATES
# ══════════════════════════════════════════════════════════════════════

print("\n── 5. Duplicates ──\n")

for study_num in [1, 2, 3, 4, 0]:
    df = load_study(study_num)
    n_dupes = df.duplicated(subset=['id']).sum()
    print(f"  Study {study_num}: {n_dupes} duplicate IDs")


# ══════════════════════════════════════════════════════════════════════
# 6. DEMOGRAPHICS
# ══════════════════════════════════════════════════════════════════════

print("\n── 6. Demographics ──\n")

for study_num in [1, 2, 3, 4, 0]:
    cfg = STUDIES[study_num]
    df = load_study(study_num)
    print(f"\n  Study {study_num} ({cfg['name']}):")

    if 'gender' in df.columns:
        gender_cts = df['gender'].value_counts()
        print(f"    Gender: {dict(gender_cts)}")

    if 'age' in df.columns:
        print(f"    Age: mean={df['age'].mean():.1f}, median={df['age'].median():.0f}, "
              f"range=[{df['age'].min():.0f}, {df['age'].max():.0f}]")

    if 'duration' in df.columns:
        print(f"    Duration: mean={df['duration'].mean():.0f} min, "
              f"median={df['duration'].median():.0f} min")


print("\n" + "=" * 60)
print("04_data_audit.py — DONE")
print("=" * 60)
