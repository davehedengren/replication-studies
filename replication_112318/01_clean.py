"""
01_clean.py - Load and validate all datasets for Urquiola (2005) replication.
Checks sample sizes and basic variable ranges against published values.
"""
import sys
import numpy as np
import pandas as pd
sys.path.insert(0, '.')
from utils import *

print("=" * 70)
print("PHASE 1: Loading and validating all datasets")
print("=" * 70)

# ============================================================
# 1. District-level race data (Table 3, Panel A, cols 1-2)
# ============================================================
print("\n--- District Race (arc1e9s.raw) ---")
dr = load_district_race()
print(f"  Obs: {len(dr)}")
print(f"  MAs: {dr['ma9dn'].nunique()}")
print(f"  yrc1e9s: mean={dr['yrc1e9s'].mean():.1f}, sd={dr['yrc1e9s'].std():.1f}")
print(f"  n9d range: {dr['n9d'].min()} - {dr['n9d'].max()}")
print(f"  Published Table 3 Panel A N=5,555 (cols 1-2)")

# ============================================================
# 2. District-level education data (Table 3, Panel B, cols 1-2)
# ============================================================
print("\n--- District Education (aede9s.raw) ---")
de = load_district_education()
print(f"  Obs: {len(de)}")
print(f"  MAs: {de['ma9dn'].nunique()}")
print(f"  yede9s: mean={de['yede9s'].mean():.1f}, sd={de['yede9s'].std():.1f}")
print(f"  Published Table 3 Panel B N=5,554 (cols 1-2)")

# ============================================================
# 3. Stacked district race data (Table 3, Panel A, cols 3-5)
# ============================================================
print("\n--- District Race Stacked (asrc1e9s.raw) ---")
drs = load_district_race_stacked()
print(f"  Obs: {len(drs)}")
print(f"  MAs: {drs['ma9dn'].nunique()}")
print(f"  Primary obs (zsec=0): {(drs['zsec']==0).sum()}")
print(f"  Secondary obs (zsec=1): {(drs['zsec']==1).sum()}")
print(f"  Published Table 3 Panel A N=9,452 (cols 3-5)")

# ============================================================
# 4. Stacked district education data (Table 3, Panel B, cols 3-5)
# ============================================================
print("\n--- District Education Stacked (asede9s.raw) ---")
des = load_district_edu_stacked()
print(f"  Obs: {len(des)}")
print(f"  MAs: {des['ma9dn'].nunique()}")
print(f"  Published Table 3 Panel B N=9,458 (cols 3-5)")

# ============================================================
# 5. School-level data (Table 4, cols 1-3)
# ============================================================
print("\n--- Schools (sch.raw) ---")
sc = load_schools()
print(f"  Obs: {len(sc)}")
print(f"  MAs: {sc['ma9dn'].nunique()}")
print(f"  ysrc9f: mean={sc['ysrc9f'].dropna().mean():.1f}, sd={sc['ysrc9f'].dropna().std():.1f}")
print(f"  Published Table 4 N=48,075 (cols 1-3)")

# ============================================================
# 6. Stacked school data (Table 4, cols 4-6)
# ============================================================
print("\n--- Schools Stacked (ssch.raw) ---")
scs = load_schools_stacked()
print(f"  Obs: {len(scs)}")
print(f"  MAs: {scs['ma9dn'].nunique()}")
print(f"  Published Table 4 N=50,224 (cols 4-6)")

# ============================================================
# 7. MA-level data (Tables 2, 5 cols 1-2)
# ============================================================
print("\n--- MA Level (ma9.raw + referee.raw) ---")
ma = load_ma()
print(f"  Obs: {len(ma)}")
print(f"  Published Table 2: N=331")
print(f"  n9d: mean={ma['n9d'].mean():.1f}, sd={ma['n9d'].std():.1f}")
print(f"  rve9s: mean={ma['rve9s'].mean():.1f}, sd={ma['rve9s'].std():.1f}")

# Check Rothstein data exclusions for Table 2
ma_t2 = ma[~ma['ma9dn'].isin([5000, 5990])].copy()
print(f"  After dropping Miami (5000) and ? (5990): {len(ma_t2)}")

# ============================================================
# 8. MA stacked private enrollment (Table 5, cols 3-6)
# ============================================================
print("\n--- MA Private Enrollment Stacked (ma9s.raw) ---")
mp = load_ma_private()
print(f"  Obs: {len(mp)}")
print(f"  MAs: {mp['ma9dn'].nunique()}")
print(f"  Published Table 5 N=582 (cols 3-6)")

# ============================================================
# 9. Table 1 descriptive statistics comparison
# ============================================================
print("\n" + "=" * 70)
print("TABLE 1 VALIDATION: Descriptive Statistics")
print("=" * 70)

print("\n--- MA Level Data (N=331 expected) ---")
print(f"  Racial heterogeneity (hmrc9f):  mean={ma['hmrc9f'].mean():.1f}, sd={ma['hmrc9f'].std():.1f}, "
      f"published: 35.1, 17.0")
print(f"  Educ heterogeneity (hmede9s):   mean={ma['hmede9s'].mean():.1f}, sd={ma['hmede9s'].std():.1f}, "
      f"published: 72.1, 2.4")
print(f"  Number of districts (n9d):      mean={ma['n9d'].mean():.1f}, sd={ma['n9d'].std():.1f}, "
      f"published: 18.8, 24.3")
print(f"  Log districts (ln9d):           mean={ma['ln9d'].mean():.1f}, sd={ma['ln9d'].std():.1f}, "
      f"published: 2.3, 1.2")

print("\n--- District Level Data ---")
print(f"  Racial het relative to MA (yrc1e9s):  mean={dr['yrc1e9s'].mean():.1f}, sd={dr['yrc1e9s'].std():.1f}, "
      f"published: 55.9, 19.2")
print(f"  Educ het relative to MA (yede9s):     mean={de['yede9s'].mean():.1f}, sd={de['yede9s'].std():.1f}, "
      f"published: 91.0, 13.1")

print("\n--- School Level Data ---")
print(f"  Racial het (hsrc9f):       mean={sc['hsrc9f'].dropna().mean():.1f}, sd={sc['hsrc9f'].dropna().std():.1f}, "
      f"published: 28.5, 21.3")
print(f"  Rel to MA (ysrc9f):        mean={sc['ysrc9f'].dropna().mean():.1f}, sd={sc['ysrc9f'].dropna().std():.1f}, "
      f"published: 69.4, 55.8")
print(f"  Rel to district (vsrc9f):  mean={sc['vsrc9f'].dropna().mean():.1f}, sd={sc['vsrc9f'].dropna().std():.1f}, "
      f"published: 90.9, 49.8")

print("\n[DONE] All datasets loaded successfully.")
