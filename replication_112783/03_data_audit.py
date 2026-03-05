"""
03_data_audit.py - Data quality audit for paper 112783.
Checks coverage, distributions, logical consistency, missing data, outliers.
"""

import sys
sys.path.insert(0, '.')
from utils import *

df = load_data()

print("=" * 70)
print("DATA AUDIT: Paper 112783")
print("=" * 70)

# ============================================================
# 1. Coverage
# ============================================================
print("\n1. COVERAGE")
print("-" * 50)
print(f"Total observations: {len(df):,}")
print(f"Killer apps: {df['killerappgros'].sum():,.0f} ({df['killerappgros'].mean()*100:.1f}%)")
print(f"Non-killer apps: {(df['killerappgros']==0).sum():,}")
print(f"Games (cat5=1): {df['cat5'].sum():,}")
print(f"Non-games (cat5=0): {(df['cat5']==0).sum():,}")

# Category distribution
print(f"\nCategory distribution:")
for i in range(1, 19):
    n = df[f'cat{i}'].sum()
    pct = n / len(df) * 100
    print(f"  cat{i}: {n:>6,} ({pct:5.1f}%)")

# Check categories are mutually exclusive
cat_sum = sum(df[f'cat{i}'] for i in range(1, 19))
print(f"\nCategory sum per obs: min={cat_sum.min()}, max={cat_sum.max()}, mean={cat_sum.mean():.4f}")
if cat_sum.min() == 1 and cat_sum.max() == 1:
    print("  Categories are mutually exclusive and exhaustive.")

# Cohort distribution
print(f"\nCohort distribution:")
print(f"  Number of cohorts: {df['cohort'].nunique()}")
cohort_counts = df.groupby('cohort').size()
print(f"  Min cohort size: {cohort_counts.min()} (cohort {cohort_counts.idxmin():.0f})")
print(f"  Max cohort size: {cohort_counts.max()} (cohort {cohort_counts.idxmax():.0f})")
print(f"  Mean cohort size: {cohort_counts.mean():.1f}")

# Check cohort dummies sum to 1
cohort_sum = sum(df[f'cohort{i}'] for i in range(1, 39))
print(f"  Cohort dummy sum: min={cohort_sum.min()}, max={cohort_sum.max()}")

# ============================================================
# 2. Variable Distributions
# ============================================================
print("\n2. VARIABLE DISTRIBUTIONS")
print("-" * 50)
key_vars = ['countapp', 'numverapp', 'noupdates', 'avdeltatime',
            'avprice', 'avsize', 'numcomapp', 'scoreapp', 'lnnumcomapp']

for v in key_vars:
    s = df[v]
    q1, q3 = s.quantile(0.25), s.quantile(0.75)
    iqr = q3 - q1
    n_outliers = ((s < q1 - 1.5 * iqr) | (s > q3 + 1.5 * iqr)).sum()
    print(f"\n  {VAR_LABELS.get(v, v)} ({v}):")
    print(f"    Mean={s.mean():.4f}, SD={s.std():.4f}, Median={s.median():.2f}")
    print(f"    Min={s.min():.2f}, Max={s.max():.2f}")
    print(f"    Q1={q1:.2f}, Q3={q3:.2f}, IQR={iqr:.2f}")
    print(f"    Outliers (IQR method): {n_outliers:,} ({n_outliers/len(df)*100:.1f}%)")
    print(f"    Top 5: {sorted(s.nlargest(5).values, reverse=True)}")

# ============================================================
# 3. Logical Consistency
# ============================================================
print("\n3. LOGICAL CONSISTENCY")
print("-" * 50)

# noupdates should be 1 when numverapp==1
ver1 = df[df['numverapp'] == 1]
noupd_when_v1 = ver1['noupdates'].mean()
print(f"noupdates when numverapp==1: mean={noupd_when_v1:.4f} (expect ~1.0)")
print(f"  numverapp==1 count: {len(ver1)}")
print(f"  of those, noupdates==1: {(ver1['noupdates']==1).sum()}")
print(f"  of those, noupdates==0: {(ver1['noupdates']==0).sum()}")

# When noupdates==1, avdeltatime should be 0
noupd = df[df['noupdates'] == 1]
print(f"\nWhen noupdates==1:")
print(f"  avdeltatime: mean={noupd['avdeltatime'].mean():.4f}, max={noupd['avdeltatime'].max():.2f}")
print(f"  numverapp: mean={noupd['numverapp'].mean():.4f}, min={noupd['numverapp'].min():.0f}, max={noupd['numverapp'].max():.0f}")

# When noupdates==0, numverapp should be >1
upd = df[df['noupdates'] == 0]
print(f"\nWhen noupdates==0:")
print(f"  numverapp: mean={upd['numverapp'].mean():.4f}, min={upd['numverapp'].min():.0f}, max={upd['numverapp'].max():.0f}")
print(f"  numverapp==1 count: {(upd['numverapp']==1).sum()}")

# scoreapp bounds [0, 5]
print(f"\nScore range: [{df['scoreapp'].min():.2f}, {df['scoreapp'].max():.2f}] (expect [0,5])")
print(f"  Score==0: {(df['scoreapp']==0).sum():,} ({(df['scoreapp']==0).mean()*100:.1f}%)")

# Price should be >= 0
print(f"Price range: [{df['avprice'].min():.2f}, {df['avprice'].max():.2f}] (expect >=0)")
print(f"  Free apps (price=0): {(df['avprice']==0).sum():,} ({(df['avprice']==0).mean()*100:.1f}%)")

# Size should be > 0
print(f"Size range: [{df['avsize'].min():.2f}, {df['avsize'].max():.2f}] (expect >0)")
print(f"  Size==0: {(df['avsize']==0).sum()}")

# lnnumcomapp consistency
print(f"\nlnnumcomapp vs log(numcomapp+1):")
manual_ln = np.log(df['numcomapp'] + 1)
diff = (df['lnnumcomapp'] - manual_ln).abs()
print(f"  Max absolute diff: {diff.max():.6f}")
print(f"  Mean absolute diff: {diff.mean():.6f}")
# Also try ln(numcomapp)
pos = df[df['numcomapp'] > 0]
manual_ln2 = np.log(pos['numcomapp'])
diff2 = (pos['lnnumcomapp'] - manual_ln2).abs()
print(f"  ln(numcomapp) for numcomapp>0: max diff={diff2.max():.6f}, mean diff={diff2.mean():.6f}")

# Check zero numcomapp vs lnnumcomapp
zero_com = df[df['numcomapp'] == 0]
print(f"  numcomapp==0: {len(zero_com):,}, their lnnumcomapp: mean={zero_com['lnnumcomapp'].mean():.4f}, min={zero_com['lnnumcomapp'].min():.4f}, max={zero_com['lnnumcomapp'].max():.4f}")

# countapp (App Order) should be >= 1
print(f"\ncountapp range: [{df['countapp'].min():.0f}, {df['countapp'].max():.0f}] (expect >=1)")

# ============================================================
# 4. Missing Data Patterns
# ============================================================
print("\n4. MISSING DATA PATTERNS")
print("-" * 50)
total_missing = df.isnull().sum().sum()
print(f"Total missing values: {total_missing}")
if total_missing > 0:
    for col in df.columns:
        if df[col].isnull().sum() > 0:
            print(f"  {col}: {df[col].isnull().sum()}")
else:
    print("  No missing values in any variable.")

# Check for zeros that might represent missing
print(f"\nPotential missing-as-zero patterns:")
for v in ['scoreapp', 'numcomapp', 'avdeltatime']:
    n_zero = (df[v] == 0).sum()
    print(f"  {v}==0: {n_zero:,} ({n_zero/len(df)*100:.1f}%)")
    # By killer status
    for k in [0, 1]:
        sub = df[df['killerappgros'] == k]
        nz = (sub[v] == 0).sum()
        print(f"    killerappgros={k}: {nz:,} ({nz/len(sub)*100:.1f}%)")

# ============================================================
# 5. Duplicates
# ============================================================
print("\n5. DUPLICATES")
print("-" * 50)
# Check for exact duplicate rows
n_dup = df.duplicated().sum()
print(f"Exact duplicate rows: {n_dup}")

# Check for potential duplicates on key variables
key = ['countapp', 'numverapp', 'avprice', 'avsize', 'numcomapp', 'scoreapp', 'cat']
n_dup_key = df.duplicated(subset=key).sum()
print(f"Duplicate on key vars ({key}): {n_dup_key}")

# ============================================================
# 6. Killer App Rate by Category
# ============================================================
print("\n6. KILLER APP RATE BY CATEGORY")
print("-" * 50)
for i in range(1, 19):
    sub = df[df[f'cat{i}'] == 1]
    if len(sub) > 0:
        rate = sub['killerappgros'].mean() * 100
        n_killer = sub['killerappgros'].sum()
        print(f"  cat{i}: {len(sub):>5,} apps, {n_killer:>4.0f} killer ({rate:5.1f}%)")

# ============================================================
# 7. Cohort Balance
# ============================================================
print("\n7. COHORT BALANCE")
print("-" * 50)
cohort_stats = df.groupby('cohort').agg(
    n=('killerappgros', 'size'),
    killer_rate=('killerappgros', 'mean'),
    mean_countapp=('countapp', 'mean'),
    mean_numver=('numverapp', 'mean'),
).reset_index()

print(f"{'Cohort':>8}{'N':>8}{'Killer%':>10}{'AvgAppOrd':>12}{'AvgNumVer':>12}")
for _, row in cohort_stats.iterrows():
    print(f"{row['cohort']:>8.0f}{row['n']:>8}{row['killer_rate']*100:>10.1f}{row['mean_countapp']:>12.2f}{row['mean_numver']:>12.2f}")

print(f"\nKiller rate range: {cohort_stats['killer_rate'].min()*100:.1f}% - {cohort_stats['killer_rate'].max()*100:.1f}%")

# ============================================================
# 8. Extreme Values / Outlier Analysis
# ============================================================
print("\n8. EXTREME VALUES")
print("-" * 50)
# Top 10 most extreme observations
for v in ['avprice', 'avsize', 'countapp', 'numcomapp']:
    top = df.nlargest(10, v)[[v, 'killerappgros', 'cat5']].reset_index(drop=True)
    print(f"\nTop 10 {v}:")
    print(top.to_string())

print("\n" + "=" * 70)
print("DATA AUDIT COMPLETE")
print("=" * 70)
