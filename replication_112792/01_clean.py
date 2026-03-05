"""
01_clean.py — Load and inspect both datasets, verify basic properties.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from utils import *

print("=" * 70)
print("  DATA INSPECTION: MA_state_data.dta")
print("=" * 70)
sdf = pd.read_stata(STATE_DTA, convert_categoricals=False)
sdf["q_fld"] = q_fld_to_int(sdf["q_fld"])
print(f"Shape: {sdf.shape}")
print(f"States: {sorted(sdf['state'].unique())}")
print(f"q_fld range: {sdf['q_fld'].min()} - {sdf['q_fld'].max()}")
print(f"\nColumns: {list(sdf.columns)}")
print(f"\nSummary stats:")
print(sdf[APP_VARS_STATE + ["ue", "wapop"]].describe().round(4))

print(f"\n\n{'=' * 70}")
print("  DATA INSPECTION: MA_county_data.dta")
print("=" * 70)
cdf = pd.read_stata(COUNTY_DTA, convert_categoricals=False)
cdf["q_fld"] = q_fld_to_int(cdf["q_fld"])
print(f"Shape (before drops): {cdf.shape}")
print(f"States: {sorted(cdf['state'].unique())}")
print(f"q_fld range: {cdf['q_fld'].min()} - {cdf['q_fld'].max()}")
print(f"Unique counties: {cdf['county'].nunique()}")

# Check which counties have zeros (suppressed cells)
cdf["anymissing"] = ((cdf["DIonly"] == 0) | (cdf["SSIonly"] == 0) | (cdf["concurrent"] == 0)).astype(int)
problem_counties = cdf.loc[cdf["anymissing"] == 1, "county"].unique()
print(f"\nCounties with any zero in DIonly/SSIonly/concurrent: {len(problem_counties)}")
print(f"  {sorted(problem_counties)}")
print(f"  Expected from Stata code: {len(DROP_COUNTIES)} counties")

# After dropping
cdf_clean = cdf[~cdf["county"].isin(DROP_COUNTIES)].copy()
print(f"\nShape (after drops): {cdf_clean.shape}")
print(f"Unique counties after drops: {cdf_clean['county'].nunique()}")

# MA counties
ma_counties = cdf_clean.loc[cdf_clean["state"] == "MA", "county"].unique()
print(f"MA counties: {len(ma_counties)}")

print(f"\nColumns: {list(cdf.columns)}")
print(f"\nSummary stats (after drops):")
print(cdf_clean[APP_VARS_COUNTY + ["ue", "wapop", "nohi05"]].describe().round(4))

# Check lowHI threshold
cdf_clean["lowHI"] = (cdf_clean["nohi05"] >= 0.12).astype(int)
ma_q187 = cdf_clean[(cdf_clean["state"] == "MA") & (cdf_clean["q_fld"] == 187)]
print(f"\nMA counties at q_fld=187: {len(ma_q187)}")
print(f"  lowHI=1: {ma_q187['lowHI'].sum()}, lowHI=0: {(1-ma_q187['lowHI']).sum()}")
# Weighted share
w_low = ma_q187.loc[ma_q187["lowHI"] == 1, "wapop"].sum()
w_all = ma_q187["wapop"].sum()
print(f"  Weighted share lowHI: {w_low/w_all:.3f}")
