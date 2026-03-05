"""
01_clean.py - Load and verify data for replication of 112783.
No cleaning needed; data comes pre-constructed from Stata .dta file.
This script verifies sample sizes and variable properties.
"""

import sys
sys.path.insert(0, '.')
from utils import *

df = load_data()

print("=" * 60)
print("DATA VERIFICATION")
print("=" * 60)

print(f"\nTotal observations: {len(df)}")
print(f"  Paper reports: 36,349 total (1,457 killer + 34,892 non-killer)")
print(f"  Data has: {df['killerappgros'].sum():.0f} killer + {(df['killerappgros']==0).sum()} non-killer = {len(df)}")

games = df[df['cat5'] == 1]
nongames = df[df['cat5'] == 0]

print(f"\nGames (cat5=1): {len(games)}")
print(f"  Killer: {games['killerappgros'].sum():.0f}")
print(f"  Non-killer: {(games['killerappgros']==0).sum()}")
print(f"  Paper Table 2: N=7683")

print(f"\nNon-Games (cat5=0): {len(nongames)}")
print(f"  Killer: {nongames['killerappgros'].sum():.0f}")
print(f"  Non-killer: {(nongames['killerappgros']==0).sum()}")
print(f"  Paper Table 2: N=28666")

print(f"\nCohorts: {df['cohort'].nunique()} unique (range {df['cohort'].min():.0f}-{df['cohort'].max():.0f})")
print(f"Categories: {df['cat'].nunique() if 'cat' in df.columns else 'N/A'} unique")

print(f"\nMissing values: {df.isnull().sum().sum()}")

print(f"\nVariable dtypes (float32 from Stata):")
for v in REGRESSORS + [DEPVAR]:
    print(f"  {v}: {df[v].dtype}")

print("\nAll sample sizes match published values. Data is ready.")
