"""
04_data_audit.py - Data audit for paper 206261
Immigration, Innovation, and Growth
"""
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

from utils import OUTPUT, TEMP, period_to_year

df = pd.read_pickle(f'{OUTPUT}/IIGFinalData.pkl')

print("=" * 70)
print("DATA AUDIT: Paper 206261 - Immigration, Innovation, and Growth")
print("=" * 70)

# 1. Panel structure
print(f"\n1. PANEL STRUCTURE")
print(f"   Observations: {len(df)}")
print(f"   Counties: {df['state_county_code_1990'].nunique()}")
print(f"   Time periods: {sorted(df['time'].unique())}")
print(f"   Years: {[period_to_year(t) for t in sorted(df['time'].unique())]}")
print(f"   States: {df['state_code'].nunique()}")
print(f"   Obs per period: {df.groupby('time').size().to_dict()}")

# Panel balance
ct = df.groupby('state_county_code_1990').size().reset_index(name='n_periods')
print(f"   Periods per county: min={ct['n_periods'].min()}, max={ct['n_periods'].max()}, mean={ct['n_periods'].mean():.1f}")
max_periods = len(df['time'].unique())
print(f"   Balanced ({max_periods} periods): {(ct['n_periods'] == max_periods).sum()} / {len(ct)} counties")

# 2. Key variable distributions
print(f"\n2. KEY VARIABLE DISTRIBUTIONS")
key_vars = ['immigration_nEuro_d_', 'IhatM1_d', 'diffPV_a_W', 'delta_avg_wage_adj',
            'deltapop', 'immigration_25o_', 'IhatM1_d10y', 'immigration_nEuro_d_10y']
for v in key_vars:
    if v in df.columns:
        s = df[v].dropna()
        print(f"   {v:<30} N={len(s):<6} mean={s.mean():<12.4f} sd={s.std():<12.4f} min={s.min():<12.4f} max={s.max():<12.4f}")

# 3. Geographic coverage
print(f"\n3. GEOGRAPHIC COVERAGE")
print(f"   States: {sorted(df['state_code'].unique())}")
if 'state_name' in df.columns:
    state_counts = df.groupby('state_name')['state_county_code_1990'].nunique().sort_values(ascending=False)
    print(f"   Top 5 states: {state_counts.head().to_dict()}")

# 4. Missing data
print(f"\n4. MISSING DATA")
for v in key_vars + ['time', 'state_county_code_1990', 'state_factor']:
    if v in df.columns:
        miss = df[v].isna().sum()
        print(f"   {v:<30} missing: {miss} ({100*miss/len(df):.1f}%)")

# 5. Instrument quality
print(f"\n5. INSTRUMENT DIAGNOSTICS")
if 'IhatM1_d' in df.columns and 'immigration_nEuro_d_' in df.columns:
    corr = df[['IhatM1_d', 'immigration_nEuro_d_']].dropna().corr().iloc[0, 1]
    print(f"   Correlation(instrument, endog): {corr:.4f}")
    print(f"   Instrument mean: {df['IhatM1_d'].mean():.4f}")
    print(f"   Instrument SD: {df['IhatM1_d'].std():.4f}")
    print(f"   Instrument range: [{df['IhatM1_d'].min():.4f}, {df['IhatM1_d'].max():.4f}]")

# 6. All columns
print(f"\n6. ALL COLUMNS ({len(df.columns)})")
for c in sorted(df.columns):
    dtype = df[c].dtype
    miss = df[c].isna().sum()
    print(f"   {c:<40} {str(dtype):<10} missing={miss}")
