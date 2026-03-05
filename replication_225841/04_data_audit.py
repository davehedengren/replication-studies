"""
04_data_audit.py - Data audit for paper 225841
Greenwald & Guren (2025) - Do Credit Conditions Move House Prices?
"""
import sys, warnings
warnings.filterwarnings('ignore')
sys.path.insert(0, '.')
from utils import *

print("=" * 70)
print("DATA AUDIT: Paper 225841 (Greenwald & Guren 2025)")
print("=" * 70)

df = load_dta('ls_analysis_data_nc.dta')

print(f"\n1. COVERAGE")
print("-" * 50)
print(f"Total observations: {df.shape[0]}")
print(f"Total variables: {df.shape[1]}")
print(f"CBSAs: {df['cbsa'].nunique()}")
print(f"Years: {sorted(df['year'].unique())}")
print(f"Panel structure: {df.shape[0]} = {df['cbsa'].nunique()} CBSAs x {df['year'].nunique()} years")

print(f"\n2. SAMPLE INDICATORS")
print("-" * 50)
for s in ['HVS_all_sample', 'HVS_common_sample', 'HVS_common_sample_allyrs',
          'HVS_horall_sample', 'gg_common_sample', 'gg_all_sample']:
    if s in df.columns:
        n = (df[s] == 1).sum()
        ncbsa = df.loc[df[s] == 1, 'cbsa'].nunique() if n > 0 else 0
        print(f"  {s}: {n} obs, {ncbsa} CBSAs")

print(f"\n3. KEY VARIABLE DISTRIBUTIONS")
print("-" * 50)
key_vars = ['lprr', 'lhor_hvs', 'lhpi', 'z', 'Lfraction', 'pop2000']
for v in key_vars:
    if v in df.columns:
        s = df[v].dropna()
        print(f"\n  {v}: n={len(s)}, mean={s.mean():.4f}, sd={s.std():.4f}")
        print(f"    min={s.min():.4f}, p50={s.median():.4f}, max={s.max():.4f}")

print(f"\n4. MISSING DATA")
print("-" * 50)
miss = df[['lprr', 'lhor_hvs', 'lhpi', 'z', 'Lfraction', 'L2fraction', 'Lz',
           'Llprr', 'Llhor_hvs', 'pop2000']].isnull().sum()
for col, n in miss.items():
    if n > 0:
        print(f"  {col}: {n} missing ({n/len(df)*100:.1f}%)")

print(f"\n5. PANEL BALANCE")
print("-" * 50)
obs_per_cbsa = df.groupby('cbsa').size()
print(f"Obs per CBSA: min={obs_per_cbsa.min()}, max={obs_per_cbsa.max()}, median={obs_per_cbsa.median():.0f}")
print(f"Balanced panel: {'Yes' if obs_per_cbsa.nunique() == 1 else 'No'}")

print(f"\n6. INSTRUMENT VARIATION")
print("-" * 50)
z = df['z'].dropna()
print(f"LS instrument (z): n={len(z)}, mean={z.mean():.4f}, sd={z.std():.4f}")
z_within = df.groupby('cbsa')['z'].transform(lambda x: x - x.mean())
print(f"Within-CBSA sd: {z_within.dropna().std():.4f}")
z_between = df.groupby('cbsa')['z'].mean()
print(f"Between-CBSA sd: {z_between.dropna().std():.4f}")

print(f"\n7. PSEUDODATA NOTE")
print("-" * 50)
print("This is the non-confidential version with pseudodata for some variables.")
print("Coefficient values may differ from published results.")

print("\n\nDATA AUDIT COMPLETE")
