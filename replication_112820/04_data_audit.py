"""
04_data_audit.py - Data quality and integrity checks.
"""
import sys, warnings
warnings.filterwarnings('ignore')
sys.path.insert(0, '.')
from utils import *

print("=" * 80)
print("DATA AUDIT")
print("=" * 80)

us = load_us_data()
india = load_india_data()
cems = load_cems_data()

# ===================================================================
# 1. COVERAGE
# ===================================================================
print("\n1. COVERAGE")
print("-" * 60)

print(f"\nUS data: {us.shape[0]} obs, {us['plantcode'].nunique()} plants, "
      f"years {int(us['year'].min())}-{int(us['year'].max())}")
print(f"India data: {india.shape[0]} obs, {india['plantcode'].nunique()} plants, "
      f"years {int(india['year'].min())}-{int(india['year'].max())}")
print(f"CEMS data: {cems.shape[0]} obs, {cems['plantcode'].nunique()} plants, "
      f"years {int(cems['year'].min())}-{int(cems['year'].max())}")

# Plants per year
print(f"\nPlants per year:")
print(f"  {'Year':>6s}  {'US':>6s}  {'India':>6s}  {'India private':>14s}")
for y in range(1988, 2011):
    n_us = len(us[us['year'] == y])
    n_ind = len(india[india['year'] == y])
    n_priv = len(india[(india['year'] == y) & (india['private'] == 1)])
    if n_us > 0 or n_ind > 0:
        print(f"  {y:6d}  {n_us:6d}  {n_ind:6d}  {n_priv:14d}")


# ===================================================================
# 2. VARIABLE COMPLETENESS
# ===================================================================
print("\n2. VARIABLE COMPLETENESS")
print("-" * 60)

key_vars_us = ['heatrate', 'nameplate', 'age', 'vintage', 'capfactor',
               'btucontent', 'private', 'aux']
key_vars_ind = ['heatrate', 'nameplate', 'age', 'vintage', 'capfactor',
                'btucontent', 'private', 'aux']

print(f"\nUS data missing rates:")
for v in key_vars_us:
    if v in us.columns:
        miss = us[v].isna().sum()
        pct = miss / len(us) * 100
        print(f"  {v:20s}: {miss:6d} missing ({pct:.1f}%)")

print(f"\nIndia data missing rates:")
for v in key_vars_ind:
    if v in india.columns:
        miss = india[v].isna().sum()
        pct = miss / len(india) * 100
        print(f"  {v:20s}: {miss:6d} missing ({pct:.1f}%)")

# India heatrate missing by year
print(f"\nIndia heatrate availability by year:")
for y in range(1988, 2010):
    iy = india[(india['year'] == y) & (india['private'] == 0)]
    hr_avail = iy['heatrate'].notna().sum()
    total = len(iy)
    if total > 0:
        print(f"  {y}: {hr_avail}/{total} ({hr_avail/total*100:.0f}%)")


# ===================================================================
# 3. DISTRIBUTIONS AND PLAUSIBILITY
# ===================================================================
print("\n3. DISTRIBUTIONS AND PLAUSIBILITY")
print("-" * 60)

print(f"\nUS heatrate (MMBtu/kWh):")
hr_us = us['heatrate'].dropna()
print(f"  Range: [{hr_us.min():.0f}, {hr_us.max():.0f}]")
print(f"  Mean: {hr_us.mean():.0f}, Median: {hr_us.median():.0f}")
print(f"  IQR: [{hr_us.quantile(0.25):.0f}, {hr_us.quantile(0.75):.0f}]")
print(f"  Extreme values (>20000 or <6000): "
      f"{((hr_us > 20000) | (hr_us < 6000)).sum()}")

print(f"\nIndia heatrate (MMBtu/kWh):")
hr_ind = india['heatrate'].dropna()
print(f"  Range: [{hr_ind.min():.0f}, {hr_ind.max():.0f}]")
print(f"  Mean: {hr_ind.mean():.0f}, Median: {hr_ind.median():.0f}")
print(f"  IQR: [{hr_ind.quantile(0.25):.0f}, {hr_ind.quantile(0.75):.0f}]")
print(f"  Values >20000: {(hr_ind > 20000).sum()}")

print(f"\nUS capacity factor:")
cf_us = us['capfactor'].dropna()
print(f"  Range: [{cf_us.min():.3f}, {cf_us.max():.3f}]")
print(f"  Values > 1 (>100%): {(cf_us > 1).sum()}")
print(f"  Negative values: {(cf_us < 0).sum()}")

print(f"\nIndia capacity factor:")
cf_ind = india['capfactor'].dropna()
print(f"  Range: [{cf_ind.min():.1f}, {cf_ind.max():.1f}]")
print(f"  Values > 100: {(cf_ind > 100).sum()}")

print(f"\nUS nameplate capacity (MW):")
np_us = us['nameplate'].dropna()
print(f"  Range: [{np_us.min():.0f}, {np_us.max():.0f}]")
print(f"  < 25 MW: {(np_us < 25).sum()} ({(np_us < 25).sum()/len(np_us)*100:.1f}%)")

print(f"\nIndia nameplate capacity (MW):")
np_ind = india['nameplate'].dropna()
print(f"  Range: [{np_ind.min():.0f}, {np_ind.max():.0f}]")
print(f"  < 25 MW: {(np_ind < 25).sum()}")

print(f"\nUS BTU content of coal:")
btu_us = us['btucontent'].dropna()
print(f"  Range: [{btu_us.min():.0f}, {btu_us.max():.0f}]")
print(f"  Zero values: {(btu_us == 0).sum()}")

print(f"\nIndia BTU content of coal:")
btu_ind = india['btucontent'].dropna()
print(f"  Range: [{btu_ind.min():.0f}, {btu_ind.max():.0f}]")


# ===================================================================
# 4. OUTLIERS
# ===================================================================
print("\n4. OUTLIER ANALYSIS")
print("-" * 60)

# Top 10 US heatrates
print(f"\nTop 10 US heatrate values:")
top_us = us.nlargest(10, 'heatrate')[['plantcode', 'year', 'heatrate', 'nameplate']]
print(top_us.to_string(index=False))

# Top 10 India heatrates
print(f"\nTop 10 India heatrate values:")
top_ind = india.nlargest(10, 'heatrate')[['plantcode', 'year', 'heatrate', 'nameplate']]
print(top_ind.to_string(index=False))

# India aux gen extremes
print(f"\nIndia aux gen extremes:")
aux_ind = india['aux'].dropna()
print(f"  Range: [{aux_ind.min():.1f}, {aux_ind.max():.1f}]")
print(f"  Values > 20%: {(aux_ind > 20).sum()}")
print(f"  Values = 0: {(aux_ind == 0).sum()}")


# ===================================================================
# 5. PANEL BALANCE
# ===================================================================
print("\n5. PANEL BALANCE")
print("-" * 60)

print(f"\nUS panel structure:")
us_panel = us.groupby('plantcode')['year'].agg(['count', 'min', 'max'])
print(f"  Plants: {len(us_panel)}")
print(f"  Obs per plant: mean={us_panel['count'].mean():.1f}, "
      f"min={us_panel['count'].min()}, max={us_panel['count'].max()}")
print(f"  Balanced (23 years): {(us_panel['count'] == 23).sum()} plants")

print(f"\nIndia panel structure:")
ind_panel = india.groupby('plantcode')['year'].agg(['count', 'min', 'max'])
print(f"  Plants: {len(ind_panel)}")
print(f"  Obs per plant: mean={ind_panel['count'].mean():.1f}, "
      f"min={ind_panel['count'].min()}, max={ind_panel['count'].max()}")


# ===================================================================
# 6. LOGICAL CONSISTENCY
# ===================================================================
print("\n6. LOGICAL CONSISTENCY")
print("-" * 60)

# Age + vintage should equal year
print(f"\nAge + vintage consistency (should ≈ year):")
us_check = us.dropna(subset=['age', 'vintage', 'year']).copy()
us_check['age_check'] = us_check['year'] - us_check['vintage']
diff = (us_check['age'] - us_check['age_check']).abs()
print(f"  US: age vs (year-vintage) diff > 1: {(diff > 1).sum()} of {len(us_check)}")

# Capacity-weighted age: w_age
us_check2 = us.dropna(subset=['w_age', 'w_vintage', 'year']).copy()
us_check2['w_age_check'] = us_check2['year'] - us_check2['w_vintage']
diff2 = (us_check2['w_age'] - us_check2['w_age_check']).abs()
print(f"  US w_age vs (year-w_vintage) diff > 1: {(diff2 > 1).sum()} of {len(us_check2)}")

ind_check = india.dropna(subset=['age', 'vintage', 'year']).copy()
ind_check['age_check'] = ind_check['year'] - ind_check['vintage']
diff3 = (ind_check['age'] - ind_check['age_check']).abs()
print(f"  India: age vs (year-vintage) diff > 1: {(diff3 > 1).sum()} of {len(ind_check)}")

# Private variable consistency
print(f"\nOwnership distribution (US):")
for v in ['federal', 'stateown', 'coop', 'muni', 'private']:
    if v in us.columns:
        n = us[v].dropna()
        print(f"  {v}: mean={n.mean():.3f} ({(n==1).sum()} obs)")

# Heatrate squared consistency
us_sq = us.dropna(subset=['agesq', 'age']).copy()
sq_check = (us_sq['agesq'] - us_sq['age']**2).abs()
print(f"\nUS agesq vs age^2: max diff = {sq_check.max():.1f}")

# Check India age vs age2
ind_a = india.dropna(subset=['age', 'age2']).copy()
print(f"\nIndia age2 (not same as age^2):")
sq_diff = (ind_a['age2'] - ind_a['age']**2).abs()
print(f"  age2 vs age^2: max diff = {sq_diff.max():.1f} "
      f"(age2 is a different variable)")


# ===================================================================
# 7. MISSING DATA PATTERNS
# ===================================================================
print("\n7. MISSING DATA PATTERNS")
print("-" * 60)

# India: plants without OPHR by year
print(f"\nIndia state-owned plants without OPHR:")
state = india[india['private'] == 0]
for y in [1988, 1997, 2000, 2005, 2009]:
    sy = state[state['year'] == y]
    no_hr = sy['heatrate'].isna()
    print(f"  {y}: {no_hr.sum()}/{len(sy)} missing heatrate "
          f"({no_hr.sum()/len(sy)*100:.0f}%)")

# Are plants missing heatrate systematically smaller/older?
print(f"\nIndia 2009 comparison: plants with vs without OPHR")
sy = state[state['year'] == 2009].copy()
with_hr = sy[sy['heatrate'].notna()]
without_hr = sy[sy['heatrate'].isna()]
for v in ['nameplate', 'age', 'capfactor', 'aux']:
    if v in sy.columns:
        m1 = with_hr[v].mean()
        m2 = without_hr[v].mean()
        print(f"  {v:15s}: with OPHR={m1:.1f}, without OPHR={m2:.1f}")


# ===================================================================
# 8. DUPLICATES
# ===================================================================
print("\n8. DUPLICATES")
print("-" * 60)

us_dup = us.duplicated(subset=['plantcode', 'year']).sum()
ind_dup = india.duplicated(subset=['plantcode', 'year']).sum()
print(f"US duplicate (plantcode, year) pairs: {us_dup}")
print(f"India duplicate (plantcode, year) pairs: {ind_dup}")

print("\nData audit complete.")
