"""
01_clean.py - Verify sample construction and print diagnostics.
"""
import sys
sys.path.insert(0, '.')
from utils import *

print("=" * 70)
print("PHASE 1: SAMPLE CONSTRUCTION VERIFICATION")
print("=" * 70)

# --- Regression sample ---
print("\n--- Regression Sample (regressioncode.do) ---")
df = build_regression_sample()
us = df[df['india'] == 0]
ind = df[df['india'] == 1]
print(f"Total obs: {len(df)}")
print(f"  US obs: {len(us)}")
print(f"  India obs: {len(ind)}")
print(f"Year range: {int(df['year'].min())}-{int(df['year'].max())}")
print(f"US plants: {us['plantcode'].nunique()}")
print(f"India plants: {ind['plantcode'].nunique()}")

# Check obs with non-missing log_heatrate (for regression)
reg_df = df.dropna(subset=['log_heatrate'])
us_reg = reg_df[reg_df['india'] == 0]
ind_reg = reg_df[reg_df['india'] == 1]
print(f"\nWith non-missing log_heatrate:")
print(f"  Total: {len(reg_df)}")
print(f"  US: {len(us_reg)}")
print(f"  India: {len(ind_reg)}")

# Year distribution for India (check 1992-1996 gap)
print(f"\nIndia obs with heatrate by year:")
for y in range(1988, 2010):
    n = len(ind_reg[ind_reg['year'] == y])
    if n > 0 or y <= 1997:
        print(f"  {y}: {n}")

# --- Matching sample (heat rate) ---
print("\n--- Matching Sample: Heat Rate (nnmatch_heatrate.do) ---")
mhr = build_matching_sample_heatrate()
print(f"Total obs: {len(mhr)}")
print(f"  US: {len(mhr[mhr['india'] == 0])}")
print(f"  India: {len(mhr[mhr['india'] == 1])}")

# --- Matching sample (auxiliary gen) ---
print("\n--- Matching Sample: Aux Gen (nnmatch_aux.do) ---")
maux = build_matching_sample_aux()
print(f"Total obs: {len(maux)}")
print(f"  US: {len(maux[maux['india'] == 0])}")
print(f"  India: {len(maux[maux['india'] == 1])}")

# --- Table 1 sample verification ---
print("\n--- Table 1 Sample Counts ---")
# US plants per year (from regression sample, non-missing heatrate)
for y in [1988, 2009]:
    us_y = us_reg[us_reg['year'] == y]
    print(f"US {y}: N={len(us_y)} (published: 406)")
    ind_hr = ind_reg[ind_reg['year'] == y]
    # India without OPHR: in regression sample but missing heatrate
    ind_all = ind[ind['year'] == y]
    ind_no_hr = ind_all[ind_all['log_heatrate'].isna()]
    if y == 1988:
        print(f"India with OPHR {y}: N={len(ind_hr)} (published: 38)")
        print(f"India w/o OPHR {y}: N={len(ind_no_hr)} (published: 5)")
    else:
        print(f"India with OPHR {y}: N={len(ind_hr)} (published: 36)")
        print(f"India w/o OPHR {y}: N={len(ind_no_hr)} (published: 20)")

print("\nSample construction complete.")
