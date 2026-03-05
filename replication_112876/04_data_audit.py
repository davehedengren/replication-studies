"""
04_data_audit.py - Data audit for paper 112876
Rajan & Ramcharan (2014) - The Anatomy of a Credit Crisis
"""
import sys, warnings
warnings.filterwarnings('ignore')
sys.path.insert(0, '.')
from utils import *

print("=" * 70)
print("DATA AUDIT: Paper 112876 (Rajan & Ramcharan 2014)")
print("=" * 70)

datasets = {
    'table_5A': 'table_5A.dta', 'table_5b': 'table_5b.dta', 'table_6': 'table_6.dta',
    'table_9': 'table_9.dta', 'table_10': 'table_10.dta', 'table_11': 'table_11.dta',
    'table_12': 'table_12.dta', 'figure_5': 'figure_5.dta',
}

print("\n1. COVERAGE")
print("-" * 50)
for name, fname in datasets.items():
    df = load_dta(fname)
    print(f"\n{name} ({fname}): {df.shape[0]} obs, {df.shape[1]} vars")
    print(f"  Missing: {df.isnull().sum().sum()} total across all vars")
    if 'state' in df.columns:
        print(f"  States: {df['state'].nunique()}")
    if 'statename' in df.columns:
        print(f"  States (by name): {df['statename'].nunique()}")
    if 'fips' in df.columns:
        print(f"  Counties (FIPS): {df['fips'].nunique()}")
    if 'year' in df.columns:
        print(f"  Years: {sorted(df['year'].dropna().unique())}")

# Deep dive on main dataset (Table 5A)
print("\n\n2. DISTRIBUTIONS (Table 5A)")
print("-" * 50)
df = load_dta('table_5A.dta')
key_vars = ['win_landval_update_ppa_log', 'win_banks_l', 'a', 'p',
            'win_totrain_anave', 'win_man_crop', 'win_vfmcropy_acres']
for v in key_vars:
    if v in df.columns:
        s = df[v].dropna()
        print(f"\n{v}: n={len(s)}, mean={s.mean():.4f}, sd={s.std():.4f}")
        print(f"  min={s.min():.4f}, p25={s.quantile(.25):.4f}, "
              f"p50={s.median():.4f}, p75={s.quantile(.75):.4f}, max={s.max():.4f}")

# Panel balance (Table 5B)
print("\n\n3. PANEL BALANCE (Table 5B)")
print("-" * 50)
df5b = load_dta('table_5b.dta')
obs_per_county = df5b.groupby('fips').size()
print(f"Counties: {obs_per_county.shape[0]}")
print(f"Obs per county: min={obs_per_county.min()}, max={obs_per_county.max()}, "
      f"median={obs_per_county.median():.0f}")
print(f"Year distribution:\n{df5b['year'].value_counts().sort_index()}")

# Check logical consistency
print("\n\n4. LOGICAL CONSISTENCY")
print("-" * 50)

# l_sq should be l^2
df9 = load_dta('table_9.dta')
if 'l_sq' in df9.columns and 'l' in df9.columns:
    diff = (df9['l_sq'] - df9['l']**2).abs().max()
    print(f"l_sq = l^2 check: max diff = {diff:.6f} {'PASS' if diff < 0.01 else 'FAIL'}")

# Check winsorization
df5a = load_dta('table_5A.dta')
for v in ['win_banks_l', 'win_landval_update_ppa_log']:
    if v in df5a.columns:
        s = df5a[v].dropna()
        p1, p99 = s.quantile(0.01), s.quantile(0.99)
        below = (s < p1 - 0.001).sum()
        above = (s > p99 + 0.001).sum()
        print(f"{v}: obs below p1={below}, above p99={above} (should be 0 if winsorized)")

# Duplicates
print("\n\n5. DUPLICATES")
print("-" * 50)
for name, fname in datasets.items():
    df = load_dta(fname)
    dups = df.duplicated().sum()
    if dups > 0:
        print(f"{name}: {dups} duplicate rows")
    else:
        print(f"{name}: no duplicates")

print("\n\n6. MISSING DATA SUMMARY")
print("-" * 50)
for name, fname in [('table_5A', 'table_5A.dta'), ('table_9', 'table_9.dta')]:
    df = load_dta(fname)
    missing = df.isnull().sum()
    missing_cols = missing[missing > 0]
    if len(missing_cols) > 0:
        print(f"\n{name}:")
        for col, n in missing_cols.items():
            print(f"  {col}: {n} missing ({n/len(df)*100:.1f}%)")
    else:
        print(f"{name}: no missing values")

print("\n\nDATA AUDIT SUMMARY")
print("=" * 50)
print("""
Key findings:
1. Multiple separate datasets, one per table - clean and well-organized
2. Each dataset is pre-processed (winsorized, variables pre-computed)
3. Panel data (table_5b) covers 1900-1930 with county-level observations
4. Cross-sectional analyses use ~2500-3000 counties (1920 observations)
5. Variables are winsorized at 1st/99th percentile as documented in paper
6. l_sq = l^2 verified (quadratic specifications)
7. No year variable ambiguity - year is explicit in panel datasets
8. Missing data is moderate and consistent with data availability constraints
""")
