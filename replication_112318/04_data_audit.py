"""
04_data_audit.py - Data quality audit for Urquiola (2005) replication.
"""
import sys
import numpy as np
import pandas as pd
sys.path.insert(0, '.')
from utils import *

print("=" * 70)
print("DATA AUDIT: Urquiola (2005)")
print("=" * 70)

# ============================================================
# 1. COVERAGE
# ============================================================
print("\n--- 1. COVERAGE ---")

ma = load_ma()
dr = load_district_race()
de = load_district_education()
drs = load_district_race_stacked()
des = load_district_edu_stacked()
sc = load_schools()
scs = load_schools_stacked()
mp = load_ma_private()

print(f"\n  MA-level dataset:")
print(f"    Total MAs: {len(ma)} (unique: {ma['ma9dn'].nunique()})")
print(f"    After dropping 5000, 5990: {len(ma[~ma['ma9dn'].isin([5000, 5990])])} (published: 331)")
print(f"    Region distribution: N={ma['rN'].sum():.0f}, S={ma['rS'].sum():.0f}, "
      f"M={ma['rM'].sum():.0f}, W={ma['rW'].sum():.0f}")
print(f"    Sum regions: {(ma['rN']+ma['rS']+ma['rM']+ma['rW']).sum():.0f} (should = {len(ma)})")

print(f"\n  District race (unstacked): {len(dr)} obs, {dr['ma9dn'].nunique()} MAs")
print(f"  District education (unstacked): {len(de)} obs, {de['ma9dn'].nunique()} MAs")
print(f"  District race (stacked): {len(drs)} obs, {drs['ma9dn'].nunique()} MAs")
print(f"  District education (stacked): {len(des)} obs, {des['ma9dn'].nunique()} MAs")
print(f"  Schools (unstacked): {len(sc)} obs, {sc['ma9dn'].nunique()} MAs")
print(f"  Schools (stacked): {len(scs)} obs, {scs['ma9dn'].nunique()} MAs")
print(f"  MA private enrollment (stacked): {len(mp)} obs, {mp['ma9dn'].nunique()} MAs")

# Variable completeness
print(f"\n  Variable completeness (MA level):")
for col in ['n9d', 'rve9s', 'hmrc9f', 'hmede9s', 'hmrc1e9s', 'rblkn9g']:
    pct = (1 - ma[col].isna().mean()) * 100
    print(f"    {col}: {pct:.1f}% non-missing")

# ============================================================
# 2. DISTRIBUTIONS
# ============================================================
print("\n--- 2. DISTRIBUTIONS ---")

print("\n  Key outcome variables:")
for df, name, col in [(dr, 'District race sorting', 'yrc1e9s'),
                       (de, 'District educ sorting', 'yede9s'),
                       (sc, 'School race sorting', 'ysrc9f'),
                       (ma, 'Private enrollment', 'rve9s')]:
    v = df[col].dropna()
    print(f"    {name} ({col}):")
    print(f"      N={len(v)}, mean={v.mean():.2f}, sd={v.std():.2f}, "
          f"min={v.min():.2f}, p25={v.quantile(.25):.2f}, "
          f"p50={v.median():.2f}, p75={v.quantile(.75):.2f}, max={v.max():.2f}")

print("\n  Key independent variables (MA level):")
for col in ['n9d', 'ln9d', 'nm9f', 'riin9c', 'rcldn9c', 'mdy9c', 'n9c']:
    v = ma[col].dropna()
    print(f"    {col}: N={len(v)}, mean={v.mean():.3f}, sd={v.std():.3f}, "
          f"min={v.min():.3f}, max={v.max():.3f}")

# Outliers
print("\n  Outlier analysis (IQR method, MA level):")
for col in ['n9d', 'rve9s', 'hmrc9f', 'hmede9s']:
    v = ma[col].dropna()
    q1, q3 = v.quantile(0.25), v.quantile(0.75)
    iqr = q3 - q1
    low = q1 - 1.5 * iqr
    high = q3 + 1.5 * iqr
    n_out = ((v < low) | (v > high)).sum()
    print(f"    {col}: IQR=[{q1:.1f}, {q3:.1f}], outliers: {n_out} "
          f"({n_out/len(v)*100:.1f}%)")

# ============================================================
# 3. LOGICAL CONSISTENCY
# ============================================================
print("\n--- 3. LOGICAL CONSISTENCY ---")

# Districts: primary + secondary type counts
print(f"\n  District type consistency (MA level):")
print(f"    n9d = n19d + n29d + n39d: "
      f"all match = {(ma['n9d'] == ma['n19d'] + ma['n29d'] + ma['n39d']).all()}")
print(f"    prm9d = n19d + n39d: "
      f"all match = {(ma['prm9d'] == ma['n19d'] + ma['n39d']).all()}")
print(f"    sec9d = n29d + n39d: "
      f"all match = {(ma['sec9d'] == ma['n29d'] + ma['n39d']).all()}")
print(f"    prm9d >= sec9d: "
      f"{(ma['prm9d'] >= ma['sec9d']).mean()*100:.1f}% of MAs")

# Heterogeneity in [0,1] (before *100)
print(f"\n  Heterogeneity bounds:")
for col in ['hmrc9f', 'hmede9s', 'hmrc1e9s']:
    v = ma[col].dropna()
    print(f"    {col}: min={v.min():.1f}, max={v.max():.1f} "
          f"(should be in [0, 100] after scaling)")

# Relative sorting: should be <= 100 if districts more homogeneous than MAs
print(f"\n  Relative sorting > 100 (districts more heterogeneous than MA):")
v = dr['yrc1e9s'].dropna()
print(f"    Race (district): {(v > 100).sum()} obs ({(v > 100).mean()*100:.1f}%)")
v = de['yede9s'].dropna()
print(f"    Educ (district): {(v > 100).sum()} obs ({(v > 100).mean()*100:.1f}%)")

# Private enrollment bounds
print(f"\n  Private enrollment rate bounds:")
v = ma['rve9s'].dropna()
print(f"    min={v.min():.2f}, max={v.max():.2f} (should be in [0, 100])")

# ============================================================
# 4. MISSING DATA PATTERNS
# ============================================================
print("\n--- 4. MISSING DATA PATTERNS ---")

print(f"\n  Stacked district race data:")
for zsec_val in [0, 1]:
    sub = drs[drs['zsec'] == zsec_val]
    print(f"    zsec={zsec_val}: {len(sub)} obs, "
          f"yrc1e9s missing: {sub['yrc1e9s'].isna().sum()}")

print(f"\n  Stacked district education data:")
for zsec_val in [0, 1]:
    sub = des[des['zsec'] == zsec_val]
    print(f"    zsec={zsec_val}: {len(sub)} obs, "
          f"yede9s missing: {sub['yede9s'].isna().sum()}")

# Missing by region
print(f"\n  Missing race data by region (MA level):")
for reg, col in [('North', 'rN'), ('South', 'rS'), ('Midwest', 'rM'), ('West', 'rW')]:
    sub = ma[ma[col] == 1]
    pct_miss = sub['hmrc9f'].isna().mean() * 100
    print(f"    {reg}: {len(sub)} MAs, hmrc9f missing: {pct_miss:.1f}%")

# ============================================================
# 5. PANEL BALANCE / STACKING
# ============================================================
print("\n--- 5. STACKING BALANCE ---")

print(f"\n  District race stacked:")
p = drs[drs['zsec'] == 0]
s = drs[drs['zsec'] == 1]
print(f"    Primary obs: {len(p)}, Secondary obs: {len(s)}")
print(f"    Unique MAs (primary): {p['ma9dn'].nunique()}, (secondary): {s['ma9dn'].nunique()}")
both = set(p['ma9dn'].unique()) & set(s['ma9dn'].unique())
print(f"    MAs with both levels: {len(both)}")

print(f"\n  MA private enrollment stacked:")
p = mp[mp['zsec'] == 0]
s = mp[mp['zsec'] == 1]
print(f"    Primary obs: {len(p)}, Secondary obs: {len(s)}")
print(f"    Balanced: {len(p) == len(s)}")

# ============================================================
# 6. DUPLICATES AND ANOMALIES
# ============================================================
print("\n--- 6. DUPLICATES AND ANOMALIES ---")

# Check for duplicate MAs in MA-level data
dups = ma['ma9dn'].duplicated().sum()
print(f"\n  MA-level duplicates: {dups}")

# Check district data for duplicates
print(f"  District race obs per MA (top 10):")
vc = dr.groupby('ma9dn').size().sort_values(ascending=False)
print(f"    Max: {vc.max()} districts in MA {vc.idxmax()}")
print(f"    Mean: {vc.mean():.1f}, Median: {vc.median():.0f}")

# Negative or zero values where unexpected
print(f"\n  Zero/negative enrollment:")
print(f"    e9d <= 0 (districts): {(dr['e9d'] <= 0).sum()}")
print(f"    n9d <= 0 (MA): {(ma['n9d'] <= 0).sum()}")

# Check region assignment: should sum to 1 for each MA
print(f"\n  Region assignment check:")
region_sum = ma['rN'] + ma['rS'] + ma['rM'] + ma['rW']
print(f"    MAs with region sum != 1: {(region_sum != 1).sum()}")
if (region_sum != 1).any():
    bad = ma[region_sum != 1][['ma9dn', 'rN', 'rS', 'rM', 'rW']]
    print(f"    Affected MAs:")
    for _, row in bad.head(10).iterrows():
        print(f"      MA {int(row['ma9dn'])}: N={row['rN']:.0f} S={row['rS']:.0f} "
              f"M={row['rM']:.0f} W={row['rW']:.0f}")

print("\n[DONE] Data audit complete.")
