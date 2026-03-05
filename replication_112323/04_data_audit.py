"""
04_data_audit.py – Data quality audit for Samaniego (2008) replication.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import pandas as pd
from scipy import stats
from utils import load_industry_data, load_did_data

df_ind = load_industry_data()
df_did = load_did_data()

print("=" * 70)
print("DATA AUDIT: Samaniego (2008)")
print("=" * 70)

# ── 1. Coverage ──────────────────────────────────────────
print("\n1. COVERAGE")
print("-" * 50)
print(f"Industry variables: {df_ind.shape[0]} industries, {df_ind.shape[1]} variables")
print(f"DiD panel: {df_did.shape[0]} obs = {df_did['c'].nunique()} countries × "
      f"{df_did['i'].nunique()} industries")
print(f"Expected: 18 × 41 = 738 → Actual: {df_did.shape[0]} ({'MATCH' if df_did.shape[0] == 738 else 'MISMATCH'})")

# Variable completeness
print("\n  DiD variable completeness:")
for col in df_did.columns:
    n = df_did[col].notna().sum()
    pct = n / len(df_did) * 100
    if pct < 100:
        print(f"    {col:<15s}: {n:>4d}/{len(df_did)} ({pct:.1f}%)")

print("\n  Industry variable completeness:")
for col in df_ind.columns:
    n = df_ind[col].notna().sum()
    if n < len(df_ind):
        print(f"    {col:<15s}: {n:>3d}/{len(df_ind)}")

# ── 2. Summary statistics ────────────────────────────────
print("\n2. SUMMARY STATISTICS")
print("-" * 50)

print("\n  DiD panel outcomes:")
for col in ['entry', 'exit', 'turnover']:
    s = df_did[col].describe()
    print(f"    {col:<10s}: mean={s['mean']:7.2f} sd={s['std']:7.2f} "
          f"min={s['min']:7.2f} max={s['max']:7.2f} N={int(s['count'])}")

print("\n  DiD interaction variables (should be ~mean=0, sd=1 if normalized):")
for col in ['etcdlls87st', 'etcwb87st', 'etcdlls00', 'etcwb00', 'etcdlls', 'etcwb']:
    s = df_did[col].describe()
    print(f"    {col:<15s}: mean={s['mean']:8.4f} sd={s['std']:6.3f}")

print("\n  Industry ISTC:")
print(f"    istc:  mean={df_ind['istc'].mean():.2f} sd={df_ind['istc'].std():.2f} "
      f"range=[{df_ind['istc'].min():.2f}, {df_ind['istc'].max():.2f}]")
print(f"    istceq: mean={df_ind['istceq'].mean():.2f} sd={df_ind['istceq'].std():.2f} "
      f"range=[{df_ind['istceq'].min():.2f}, {df_ind['istceq'].max():.2f}]")

# ── 3. Plausibility and bounds ──────────────────────────
print("\n3. PLAUSIBILITY CHECKS")
print("-" * 50)

# Entry and exit rates should be non-negative
neg_entry = (df_did['entry'] < 0).sum()
neg_exit = (df_did['exit'] < 0).sum()
print(f"  Negative entry rates: {neg_entry}")
print(f"  Negative exit rates: {neg_exit}")

# Turnover should equal entry + exit
check = df_did.dropna(subset=['entry', 'exit', 'turnover']).copy()
check['diff'] = abs(check['turnover'] - check['entry'] - check['exit'])
print(f"  Turnover = entry + exit: max deviation = {check['diff'].max():.4f}")
violations = (check['diff'] > 0.01).sum()
print(f"  Violations (>0.01): {violations}/{len(check)}")

# Zero values
for col in ['entry', 'exit', 'turnover']:
    n_zero = (df_did[col] == 0).sum()
    if n_zero > 0:
        print(f"  Zero {col}: {n_zero} obs")

# ── 4. Outliers ──────────────────────────────────────────
print("\n4. OUTLIER ANALYSIS")
print("-" * 50)

for col in ['entry', 'exit', 'turnover']:
    q1, q3 = df_did[col].quantile([0.25, 0.75])
    iqr = q3 - q1
    lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
    outliers = df_did[(df_did[col] < lower) | (df_did[col] > upper)]
    print(f"  {col}: IQR=[{q1:.1f}, {q3:.1f}], bounds=[{lower:.1f}, {upper:.1f}], "
          f"outliers={len(outliers)}")
    if len(outliers) > 0:
        # Show top outliers
        top = outliers.nlargest(3, col)
        for _, row in top.iterrows():
            print(f"    c={int(row['c'])}, i={int(row['i'])}: {col}={row[col]:.1f}")

# ── 5. Missing data patterns ─────────────────────────────
print("\n5. MISSING DATA PATTERNS")
print("-" * 50)

# By country
print("  Missing turnover by country:")
for c in sorted(df_did['c'].unique()):
    sub = df_did[df_did['c'] == c]
    n_miss = sub['turnover'].isna().sum()
    if n_miss > 0:
        miss_i = sub[sub['turnover'].isna()]['i'].tolist()
        print(f"    Country {int(c)}: {n_miss} missing (industries: {miss_i})")

# By industry
print("\n  Missing turnover by industry:")
for i in sorted(df_did['i'].unique()):
    sub = df_did[df_did['i'] == i]
    n_miss = sub['turnover'].isna().sum()
    if n_miss > 0:
        miss_c = sub[sub['turnover'].isna()]['c'].tolist()
        print(f"    Industry {int(i)}: {n_miss} missing (countries: {miss_c})")

# Check if missingness is related to manuf status
manuf_miss = df_did[df_did['manuf'] == 1]['turnover'].isna().mean()
nonmanuf_miss = df_did[df_did['manuf'] == 0]['turnover'].isna().mean()
print(f"\n  Turnover missing rate: manuf={manuf_miss:.1%}, non-manuf={nonmanuf_miss:.1%}")

# ── 6. Duplicates and panel balance ──────────────────────
print("\n6. PANEL BALANCE AND DUPLICATES")
print("-" * 50)

# Check for duplicate c-i pairs
dups = df_did.groupby(['c', 'i']).size()
dup_count = (dups > 1).sum()
print(f"  Duplicate (c, i) pairs: {dup_count}")

# Check if all c-i combinations exist
expected = df_did['c'].nunique() * df_did['i'].nunique()
actual = len(df_did)
print(f"  Expected obs (complete panel): {expected}")
print(f"  Actual obs: {actual}")
print(f"  Panel is {'balanced' if expected == actual else 'UNBALANCED'}")

# ── 7. Identical observations ────────────────────────────
print("\n7. ANOMALIES")
print("-" * 50)

# Check for industries with identical values
df_ind['i'] = range(1, 42)
for col in ['entry', 'exit', 'turnover']:
    groups = df_ind.groupby(col)
    for val, group in groups:
        if len(group) > 1:
            names = group['Industry'].tolist()
            print(f"  IDENTICAL {col}={val:.4f}: {names}")

# Check if interaction variables are truly mean-zero
print("\n  Interaction variable means (should be ≈0):")
for col in ['etcdlls87st', 'etcwb87st', 'etcdlls', 'etcwb']:
    print(f"    {col}: mean={df_did[col].mean():.2e}")

# Check correlation between DLLS and WB interaction terms
r, p = stats.pearsonr(df_did['etcdlls87st'], df_did['etcwb87st'])
print(f"\n  Correlation between DLLS and WB interaction terms: {r:.3f} (p={p:.3f})")
print(f"  Paper states 68% correlation between entry cost measures")

# ── 8. Distribution of entry/exit across countries ───────
print("\n8. CROSS-COUNTRY VARIATION")
print("-" * 50)
for col in ['turnover', 'entry', 'exit']:
    means = df_did.groupby('c')[col].mean()
    print(f"  {col}: country means range [{means.min():.1f}, {means.max():.1f}], "
          f"cv={means.std()/means.mean():.2f}")

print("\n" + "=" * 70)
print("DATA AUDIT COMPLETE")
print("=" * 70)
