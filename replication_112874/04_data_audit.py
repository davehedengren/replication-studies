"""
04_data_audit.py - Data quality audit for Kelly, Lustig & Van Nieuwerburgh (2016)
"""

import numpy as np
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from utils import *

print("=" * 80)
print("DATA AUDIT: Kelly, Lustig & Van Nieuwerburgh (2016)")
print("=" * 80)

# =============================================================================
# 1. COVERAGE
# =============================================================================
print("\n" + "=" * 60)
print("1. COVERAGE")
print("=" * 60)

opts = load_strike_matched_options()
datelist = opts['datelist'].flatten()
T = len(datelist)
N = opts['N'][0, 0]

print(f"\nSample period: {datelist[0]} to {datelist[-1]}")
print(f"Trading days: {T}")
print(f"Max stocks: {N}")

# Date range check
start_date = 20030102
end_date = 20090630
print(f"\nExpected: Jan 2003 - Jun 2009")
print(f"Actual start: {datelist[0]} ({'MATCH' if datelist[0] == start_date else 'DIFF'})")
print(f"Actual end: {datelist[-1]} ({'MATCH' if datelist[-1] == end_date else 'DIFF'})")

# Sector coverage
SPgics = opts['SPgics']
ETFgiclist = opts['ETFgiclist']

print("\nSector composition (average number of stocks per day):")
for i, name in enumerate(SECTOR_NAMES):
    gic1, gic2 = ETFgiclist[i]
    mask = (SPgics == gic1)
    if not np.isnan(gic2):
        mask = mask | (SPgics == gic2)
    counts = mask.sum(axis=1)
    print(f"  {name:>12s}: mean={np.mean(counts):.1f}, min={np.min(counts)}, max={np.max(counts)}")

# Options data completeness
PUTst = opts['PUTst']
CALLst = opts['CALLst']
PUTix = opts['PUTix']
CALLix = opts['CALLix']

print(f"\nStock-level put option coverage: {(~np.isnan(PUTst)).sum()}/{PUTst.size} "
      f"({100*(~np.isnan(PUTst)).sum()/PUTst.size:.1f}%)")
print(f"Stock-level call option coverage: {(~np.isnan(CALLst)).sum()}/{CALLst.size} "
      f"({100*(~np.isnan(CALLst)).sum()/CALLst.size:.1f}%)")
print(f"Index put option coverage: {(~np.isnan(PUTix)).sum()}/{PUTix.size} "
      f"({100*(~np.isnan(PUTix)).sum()/PUTix.size:.1f}%)")
print(f"Index call option coverage: {(~np.isnan(CALLix)).sum()}/{CALLix.size} "
      f"({100*(~np.isnan(CALLix)).sum()/CALLix.size:.1f}%)")

# =============================================================================
# 2. DISTRIBUTIONS
# =============================================================================
print("\n" + "=" * 60)
print("2. DISTRIBUTIONS")
print("=" * 60)

# Option prices
BSIVPst = opts['BSIVPst']
BSIVCst = opts['BSIVCst']
BSIVPix = opts['BSIVPix']
BSIVCix = opts['BSIVCix']

for name, arr in [('Stock Put IV', BSIVPst), ('Stock Call IV', BSIVCst),
                   ('Index Put IV', BSIVPix), ('Index Call IV', BSIVCix)]:
    valid = arr[~np.isnan(arr)]
    print(f"\n{name}:")
    print(f"  N={len(valid)}, Mean={np.mean(valid):.4f}, Std={np.std(valid):.4f}")
    print(f"  Min={np.min(valid):.4f}, Q25={np.percentile(valid,25):.4f}, "
          f"Median={np.median(valid):.4f}, Q75={np.percentile(valid,75):.4f}, Max={np.max(valid):.4f}")

# Strike prices (as fraction of spot, since spot=1 in this data)
KPst = opts['KPst']
KCst = opts['KCst']
KPix = opts['KPix']
KCix = opts['KCix']

print("\nPut strike prices (as fraction of spot):")
for name, arr in [('Stock', KPst), ('Index', KPix)]:
    valid = arr[~np.isnan(arr)]
    print(f"  {name}: Mean={np.mean(valid):.4f}, Median={np.median(valid):.4f}, "
          f"Min={np.min(valid):.4f}, Max={np.max(valid):.4f}")

print("\nCall strike prices (as fraction of spot):")
for name, arr in [('Stock', KCst), ('Index', KCix)]:
    valid = arr[~np.isnan(arr)]
    print(f"  {name}: Mean={np.mean(valid):.4f}, Median={np.median(valid):.4f}, "
          f"Min={np.min(valid):.4f}, Max={np.max(valid):.4f}")

# Risk-free rate
rf = opts['rf'].flatten()
print(f"\nRisk-free rate: Mean={np.mean(rf):.4f}, Std={np.std(rf):.4f}, "
      f"Min={np.min(rf):.4f}, Max={np.max(rf):.4f}")

# =============================================================================
# 3. LOGICAL CONSISTENCY
# =============================================================================
print("\n" + "=" * 60)
print("3. LOGICAL CONSISTENCY")
print("=" * 60)

# Load spread data
sd = np.load(os.path.join(OUTPUT_DIR, 'spread_data.npz'))

# Check: basket-index put spread should be non-negative for strike-matched
put_spread_f = sd['put_spread_f']
put_spread_nf = sd['put_spread_nf']

neg_fin = np.sum(put_spread_f[~np.isnan(put_spread_f)] < -0.001)
neg_nf = np.sum(put_spread_nf[~np.isnan(put_spread_nf)] < -0.001)
print(f"\nNegative put spread (fin): {neg_fin}/{np.sum(~np.isnan(put_spread_f))} "
      f"({100*neg_fin/np.sum(~np.isnan(put_spread_f)):.1f}%)")
print(f"Negative put spread (nonfin): {neg_nf}/{np.sum(~np.isnan(put_spread_nf))} "
      f"({100*neg_nf/np.sum(~np.isnan(put_spread_nf)):.1f}%)")

if neg_fin > 0:
    min_neg = np.nanmin(put_spread_f)
    print(f"  Min negative fin spread: {100*min_neg:.4f} cents")
    print(f"  NOTE: Small negatives possible with delta-matched basket (no-arb only holds for strike-matched)")

# Check: implied vols should be positive
for name, arr in [('Stock Put IV', BSIVPst), ('Stock Call IV', BSIVCst),
                   ('Index Put IV', BSIVPix), ('Index Call IV', BSIVCix)]:
    neg = np.sum(arr[~np.isnan(arr)] <= 0)
    if neg > 0:
        print(f"  WARNING: {neg} non-positive values in {name}")
    else:
        print(f"  {name}: All positive (OK)")

# Check: put-call parity consistency
# For a given stock, call - put = S*exp(-q*T) - K*exp(-r*T)
# With S=1, this means call - put should equal exp(-q*T) - K*exp(-r*T)
print("\nPut-call parity check (Index, sector 4=Financials):")
valid_pcp = ~np.isnan(PUTix[:, FIN_IDX] + CALLix[:, FIN_IDX])
lhs = CALLix[valid_pcp, FIN_IDX] - PUTix[valid_pcp, FIN_IDX]
rhs = np.exp(-opts['Yix'][valid_pcp, FIN_IDX]) - KPix[valid_pcp, FIN_IDX] * np.exp(-rf[valid_pcp])
pcp_error = lhs - rhs
print(f"  Mean PCP error: {np.mean(pcp_error):.6f}")
print(f"  Max abs PCP error: {np.max(np.abs(pcp_error)):.6f}")
if np.max(np.abs(pcp_error)) < 0.01:
    print(f"  Put-call parity holds (OK)")
else:
    print(f"  NOTE: PCP errors present - expected for American-style options on ETFs")

# =============================================================================
# 4. MISSING DATA PATTERNS
# =============================================================================
print("\n" + "=" * 60)
print("4. MISSING DATA PATTERNS")
print("=" * 60)

# Missing data by time period
pre = datelist < PRE_CRISIS_END
cri = datelist > CRISIS_START

# Stock put option missing rates by period
put_miss_pre = np.isnan(PUTst[pre]).mean()
put_miss_cri = np.isnan(PUTst[cri]).mean()
print(f"\nStock put option missing rate:")
print(f"  Pre-crisis: {100*put_miss_pre:.1f}%")
print(f"  Crisis: {100*put_miss_cri:.1f}%")

# Missing by sector
print("\nMissing rate by sector (stock-level puts):")
for i, name in enumerate(SECTOR_NAMES):
    gic1, gic2 = ETFgiclist[i]
    mask = (SPgics == gic1)
    if not np.isnan(gic2):
        mask = mask | (SPgics == gic2)
    sector_puts = PUTst.copy()
    sector_puts[~mask] = np.nan
    valid_cells = mask.sum()
    missing = np.isnan(sector_puts[mask]).sum() if valid_cells > 0 else 0
    miss_rate = missing / valid_cells if valid_cells > 0 else 0
    print(f"  {name:>12s}: {100*miss_rate:.1f}% missing ({missing}/{valid_cells})")

# Index option completeness
print("\nIndex option completeness (by sector):")
for i, name in enumerate(SECTOR_NAMES):
    valid = (~np.isnan(PUTix[:, i])).sum()
    print(f"  {name:>12s}: {valid}/{T} days ({100*valid/T:.1f}%)")

# =============================================================================
# 5. PANEL BALANCE
# =============================================================================
print("\n" + "=" * 60)
print("5. PANEL BALANCE AND COMPOSITION CHANGES")
print("=" * 60)

# Track S&P 500 sector composition changes
SPpermno = opts.get('SPpermno', None)
if SPpermno is not None:
    # Not available in strike-matched file, check another source
    pass

# Number of stocks per sector over time
print("\nFinancial sector stock count over time:")
gic_fin = ETFgiclist[FIN_IDX]
fin_mask = (SPgics == gic_fin[0])
fin_counts = fin_mask.sum(axis=1)
yearly = {}
for t in range(T):
    year = datelist[t] // 10000
    if year not in yearly:
        yearly[year] = []
    yearly[year].append(fin_counts[t])
for y in sorted(yearly.keys()):
    vals = yearly[y]
    print(f"  {y}: mean={np.mean(vals):.0f}, min={np.min(vals)}, max={np.max(vals)}")

# =============================================================================
# 6. DUPLICATES AND ANOMALIES
# =============================================================================
print("\n" + "=" * 60)
print("6. DUPLICATES AND ANOMALIES")
print("=" * 60)

# Check for duplicate dates
unique_dates, counts = np.unique(datelist, return_counts=True)
dups = counts[counts > 1]
if len(dups) > 0:
    print(f"  WARNING: {len(dups)} duplicate dates found")
else:
    print(f"  No duplicate dates (OK)")

# Check for extreme values
P_data = opts['P']
S_data = opts['S']
ME_data = opts['ME']

print(f"\nShares outstanding (P):")
valid_P = P_data[~np.isnan(P_data)]
print(f"  Range: {np.min(valid_P):.0f} to {np.max(valid_P):.0f}")
print(f"  Median: {np.median(valid_P):.0f}")

print(f"\nMarket equity (ME):")
valid_ME = ME_data[~np.isnan(ME_data)]
print(f"  Range: ${np.min(valid_ME)/1e6:.1f}M to ${np.max(valid_ME)/1e6:.1f}M")
print(f"  Median: ${np.median(valid_ME)/1e6:.1f}M")

# Extreme implied vols
extreme_iv = BSIVPst[~np.isnan(BSIVPst)]
n_extreme_high = np.sum(extreme_iv > 2.0)  # >200% vol
n_extreme_low = np.sum(extreme_iv < 0.05)  # <5% vol
print(f"\nExtreme stock put implied vols:")
print(f"  >200%: {n_extreme_high} ({100*n_extreme_high/len(extreme_iv):.2f}%)")
print(f"  <5%: {n_extreme_low} ({100*n_extreme_low/len(extreme_iv):.2f}%)")

# Spread outliers
put_spr = 100 * put_spread_f[~np.isnan(put_spread_f)]
q1, q3 = np.percentile(put_spr, [25, 75])
iqr = q3 - q1
outliers_high = np.sum(put_spr > q3 + 3 * iqr)
outliers_low = np.sum(put_spr < q1 - 3 * iqr)
print(f"\nFinancial put spread outliers (3x IQR):")
print(f"  IQR: [{q1:.2f}, {q3:.2f}], range = {iqr:.2f}")
print(f"  High outliers: {outliers_high}")
print(f"  Low outliers: {outliers_low}")

# Top 10 financial put spread days
put_spr_full = 100 * put_spread_f
valid_mask = ~np.isnan(put_spr_full)
sorted_idx = np.argsort(-put_spr_full * valid_mask)[:10]
print(f"\nTop 10 financial put spread days:")
for rank, idx in enumerate(sorted_idx):
    print(f"  {rank+1}. Date={datelist[idx]}, Spread={put_spr_full[idx]:.2f} cents")

# =============================================================================
# 7. ESTIMATION RESULTS CONSISTENCY
# =============================================================================
print("\n" + "=" * 60)
print("7. ESTIMATION RESULTS CONSISTENCY")
print("=" * 60)

# Check that estimation data matches raw data
for sec in ['financials']:
    d_bail = load_estimation_results(sec, bailout=True)
    oth = d_bail['othinfo']
    est_data = oth['data_full'][0, 0]
    est_dl = oth['datelist_full'][0, 0].flatten()

    print(f"\n{sec.upper()} estimation vs raw data:")
    print(f"  Estimation datelist: {est_dl[0]} to {est_dl[-1]} ({len(est_dl)} obs)")
    print(f"  Raw datelist: {datelist[0]} to {datelist[-1]} ({T} obs)")

    # Compare put CPDI from estimation data column 0 (putcpdibsk) with our computed values
    sd = np.load(os.path.join(OUTPUT_DIR, 'spread_data.npz'))
    our_putcpdibskf = sd['putcpdibskf']

    # The estimation data is in units that need scaling
    est_putcpdibskf = est_data[:, 0]

    # Check correlation
    valid = ~np.isnan(our_putcpdibskf) & ~np.isnan(est_putcpdibskf)
    corr = np.corrcoef(100 * our_putcpdibskf[valid], est_putcpdibskf[valid])[0, 1]
    print(f"  Correlation between our putcpdibsk and estimation data: {corr:.6f}")

print("\n04_data_audit.py complete.")
