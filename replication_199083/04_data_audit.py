"""
04_data_audit.py - Data audit for paper 199083
Kantor & Whalley - Moonshot: Public R&D and Growth
"""
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

PROJ = '/Users/davehedengren/code/replication_studies/199083-V1'
PROC = f'{PROJ}/data/process'

df = pd.read_pickle(f'{PROC}/final/3.compile.pkl')

print("=" * 70)
print("DATA AUDIT: Paper 199083 - Moonshot")
print("=" * 70)

# 1. Panel structure
print(f"\n1. PANEL STRUCTURE")
print(f"   Observations: {len(df)}")
print(f"   Counties (fips): {df['fips'].nunique()}")
print(f"   Industries (SIC): {df['sic'].nunique()}")
print(f"   Years: {sorted(df['year'].unique())}")
print(f"   Obs per year: {df.groupby('year').size().to_dict()}")

# Panel balance
fips_sic = df.groupby(['fips', 'sic']).size().reset_index(name='n_years')
print(f"   Obs per fips-sic: min={fips_sic['n_years'].min()}, max={fips_sic['n_years'].max()}, mean={fips_sic['n_years'].mean():.1f}")
print(f"   Balanced (10 years): {(fips_sic['n_years'] == 10).sum()} / {len(fips_sic)} cells")

# 2. Key variable distributions
print(f"\n2. KEY VARIABLE DISTRIBUTIONS")
for v in ['lva', 'lemp', 'lcapital', 'ltfp', 'mval_add', 'memp', 'capital']:
    if v in df.columns:
        s = df[v].dropna()
        print(f"   {v:<20} N={len(s):<6} mean={s.mean():<12.2f} sd={s.std():<12.2f} min={s.min():<12.2f} max={s.max():<12.2f}")

# 3. Treatment variable distributions
print(f"\n3. TREATMENT VARIABLES")
for v in ['amspost_sv2_stemmed', 'amspost_sv2_stemmed_sr', 'amspost_sv2_stemmed_psr',
          'namspost_sv2_stemmed_sr', 'namspost_sv2_stemmed_psr', 'nasa_ind']:
    if v in df.columns:
        s = df[v].dropna()
        print(f"   {v:<35} mean={s.mean():.4f}  =1: {(s==1).sum():<6}  =0: {(s==0).sum():<6}")

# 4. Geographic coverage
print(f"\n4. GEOGRAPHIC COVERAGE")
df['state_fips'] = df['fips'] // 1000
state_counts = df.groupby('state_fips')['fips'].nunique().sort_values(ascending=False)
print(f"   States: {len(state_counts)}")
print(f"   Top 5 states by county count: {state_counts.head().to_dict()}")

# 5. MSA clustering
print(f"\n5. CLUSTERING")
if 'msa_code' in df.columns:
    print(f"   MSA clusters: {df['msa_code'].nunique()}")
print(f"   SIC clusters: {df['sic'].nunique()}")

# 6. Missing data
print(f"\n6. MISSING DATA")
key_vars = ['lva', 'lemp', 'lcapital', 'ltfp', 'mval_add', 'memp', 'capital', 'minvest',
            'amspost_sv2_stemmed', 'msa_code']
for v in key_vars:
    if v in df.columns:
        miss = df[v].isna().sum()
        print(f"   {v:<25} missing: {miss} ({100*miss/len(df):.1f}%)")

# 7. Patent variables
print(f"\n7. PATENT VARIABLES")
pat_vars = ['smsfpatent5', 'smsfnasa_patent5', 'smsfarmy_patent5', 'smsfnavy_patent5', 'smsfgov_patent5']
for v in pat_vars:
    if v in df.columns:
        s = df[v].dropna()
        print(f"   {v:<25} N={len(s):<6} mean={s.mean():.2f} >0: {(s>0).sum()}")

# 8. NASA spending
print(f"\n8. NASA CONTRACTOR SPENDING")
if 'r58_s2_nasa_contractor_spend' in df.columns:
    s = df['r58_s2_nasa_contractor_spend']
    print(f"   Non-missing: {s.notna().sum()}")
    print(f"   Positive: {(s > 0).sum()}")
    print(f"   Mean (unconditional): {s.mean():.4f}")
    print(f"   Mean (conditional on >0): {s[s>0].mean():.4f}" if (s>0).sum() > 0 else "   No positive values")
