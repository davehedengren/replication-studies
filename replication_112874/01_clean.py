"""
01_clean.py - Data loading, cleaning, and basket-index spread computation
Replicates the data processing in T1_F1_F2.m and T2.m

Key outputs saved to output/:
- spread_data.npz: basket and index CPDI, spreads, correlations, volatilities
"""

import numpy as np
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from utils import *

print("=" * 70)
print("PHASE 1: Loading and cleaning data")
print("=" * 70)

# =============================================================================
# 1. Load main options data (Strike-matched-options-delta25-365day.mat)
# =============================================================================
print("\n--- Loading strike-matched options data ---")
opts = load_strike_matched_options()

useobs = slice(0, 1635)  # All observations
datelist = opts['datelist'][useobs].flatten()
T = len(datelist)
N = opts['N'][0, 0]
ttm = 1  # 365-day maturity

SPgics = opts['SPgics'][useobs]
ETFgiclist = opts['ETFgiclist']  # (9, 2) sector codes

# Stock-level data
BSIVCst = opts['BSIVCst'][useobs]
BSIVPst = opts['BSIVPst'][useobs]
CALLst = opts['CALLst'][useobs]
PUTst = opts['PUTst'][useobs]
KCst = opts['KCst'][useobs]
KPst = opts['KPst'][useobs]
Yst = opts['Yst'][useobs]

# Index-level data
BSIVCix = opts['BSIVCix'][useobs]
BSIVPix = opts['BSIVPix'][useobs]
CALLix = opts['CALLix'][useobs]
PUTix = opts['PUTix'][useobs]
KCix = opts['KCix'][useobs]
KPix = opts['KPix'][useobs]
Yix = opts['Yix'][useobs]

# Market data
ME = opts['ME'][useobs]
P = opts['P'][useobs]  # Shares outstanding
S = opts['S'][useobs]  # Stock prices (not used directly but needed for weighting)
rf = opts['rf'][useobs].flatten()
rfSP = np.tile(rf.reshape(-1, 1), (1, N))

print(f"  Dates: {datelist[0]} to {datelist[-1]} ({T} obs)")
print(f"  Stocks: {N}")

# =============================================================================
# 2. European/American adjustment for individual stocks (by sector)
# Replicate T1_F1_F2.m lines 54-100
# =============================================================================
print("\n--- Computing European-adjusted stock option prices ---")

putst = np.full((T, N), np.nan)
callst = np.full((T, N), np.nan)
putstrst = np.full((T, N), np.nan)
callstrst = np.full((T, N), np.nan)
yldst = np.full((T, N), np.nan)

for i in range(9):
    # Find stocks in this sector with valid data
    gic1 = ETFgiclist[i, 0]
    gic2 = ETFgiclist[i, 1]

    sector_mask = (SPgics == gic1)
    if not np.isnan(gic2):
        sector_mask = sector_mask | (SPgics == gic2)

    data_mask = ~np.isnan(PUTst + KPst + CALLst + KCst)
    loc = np.where(sector_mask & data_mask)

    if len(loc[0]) == 0:
        continue

    # Extract data for this sector
    callvol = BSIVCst[loc]
    putvol = BSIVPst[loc]
    put_K = KPst[loc]
    call_K = KCst[loc]
    rf_vals = rfSP[loc]
    yld_vals = Yst[loc]

    # Remove NaN in callvol
    valid = ~np.isnan(callvol)
    rows, cols = loc[0][valid], loc[1][valid]
    callvol_v = callvol[valid]
    putvol_v = putvol[valid]
    put_K_v = put_K[valid]
    call_K_v = call_K[valid]
    rf_v = rf_vals[valid]
    yld_v = yld_vals[valid]

    # European/American adjustment via Black-Scholes repricing
    # MATLAB: [call,~] = blsprice(1, calldata(:,2), calldata(:,4), calldata(:,3), callvol, yld)
    call_euro, _ = blsprice(1, call_K_v, rf_v, ttm, callvol_v, yld_v)
    _, put_euro = blsprice(1, put_K_v, rf_v, ttm, putvol_v, yld_v)

    putst[rows, cols] = put_euro
    callst[rows, cols] = call_euro
    putstrst[rows, cols] = put_K_v
    callstrst[rows, cols] = call_K_v
    yldst[rows, cols] = yld_v

    print(f"  Sector {i} ({SECTOR_NAMES[i]}): {len(rows)} obs")

# =============================================================================
# 3. European/American adjustment for index options
# Replicate T1_F1_F2.m lines 106-152
# =============================================================================
print("\n--- Computing European-adjusted index option prices ---")

putix = np.full((T, 9), np.nan)
callix = np.full((T, 9), np.nan)
putstrix = np.full((T, 9), np.nan)
callstrix = np.full((T, 9), np.nan)
yldix = np.full((T, 9), np.nan)

for i in range(9):
    callvol = BSIVCix[:, i]
    putvol = BSIVPix[:, i]
    call_K = KCix[:, i]
    put_K = KPix[:, i]
    yld_vals = Yix[:, i]

    # Remove NaN
    valid_check = np.column_stack([callvol, putvol, PUTix[:, i], KPix[:, i],
                                    CALLix[:, i], KCix[:, i]])
    valid = ~np.isnan(valid_check.sum(axis=1))
    loc = np.where(valid)[0]

    if len(loc) == 0:
        continue

    call_euro, _ = blsprice(1, call_K[loc], rf[loc], ttm, callvol[loc], yld_vals[loc])
    _, put_euro = blsprice(1, put_K[loc], rf[loc], ttm, putvol[loc], yld_vals[loc])

    putix[loc, i] = put_euro
    callix[loc, i] = call_euro
    putstrix[loc, i] = put_K[loc]
    callstrix[loc, i] = call_K[loc]
    yldix[loc, i] = yld_vals[loc]

    print(f"  Sector {i} ({SECTOR_NAMES[i]}): {len(loc)} obs")

# =============================================================================
# 4. Compute basket and index CPDI
# Replicate T1_F1_F2.m lines 156-198
# =============================================================================
print("\n--- Computing basket-index spreads ---")

putcpdiix = putix / putstrix
callcpdiix = callix / callstrix

totme = np.full((T, 9), np.nan)
putbsk = np.full((T, 9), np.nan)
putstrbsk = np.full((T, 9), np.nan)
callbsk = np.full((T, 9), np.nan)
callstrbsk = np.full((T, 9), np.nan)
yldbsk = np.full((T, 9), np.nan)

for i in range(9):
    gic1 = ETFgiclist[i, 0]
    gic2 = ETFgiclist[i, 1]
    for t in range(T):
        # Find stocks in this sector
        sector_mask = (SPgics[t, :] == gic1)
        if not np.isnan(gic2):
            sector_mask = sector_mask | (SPgics[t, :] == gic2)
        loc = np.where(sector_mask)[0]

        # Further filter for valid data
        valid = ~np.isnan(putst[t, loc] + P[t, loc] + S[t, loc])
        loc = loc[valid]

        if len(loc) == 0:
            continue

        # Market-cap weights
        me = P[t, loc] * S[t, loc]
        total_me = me.sum()
        totme[t, i] = total_me

        # Value-weighted basket
        putbsk[t, i] = np.sum(putst[t, loc] * me) / total_me
        putstrbsk[t, i] = np.sum(putstrst[t, loc] * me) / total_me
        callbsk[t, i] = np.sum(callst[t, loc] * me) / total_me
        callstrbsk[t, i] = np.sum(callstrst[t, loc] * me) / total_me
        yldbsk[t, i] = np.sum(yldst[t, loc] * me) / total_me

putcpdibsk = putbsk / putstrbsk
callcpdibsk = callbsk / callstrbsk

# Financial sector
putcpdiixf = putcpdiix[:, FIN_IDX]
putcpdibskf = putcpdibsk[:, FIN_IDX]
callcpdiixf = callcpdiix[:, FIN_IDX]
callcpdibskf = callcpdibsk[:, FIN_IDX]

# Non-financial (value-weighted average)
putcpdiixnf = compute_vw_nonfin(putcpdiix, totme)
putcpdibsknf = compute_vw_nonfin(putcpdibsk, totme)
callcpdiixnf = compute_vw_nonfin(callcpdiix, totme)
callcpdibsknf = compute_vw_nonfin(callcpdibsk, totme)

# Put spread
put_spread_f = putcpdibskf - putcpdiixf
put_spread_nf = putcpdibsknf - putcpdiixnf
call_spread_f = callcpdibskf - callcpdiixf
call_spread_nf = callcpdibsknf - callcpdiixnf

print(f"  Financial put spread (full sample mean): {100*np.nanmean(put_spread_f):.2f} cents")
print(f"  Non-fin put spread (full sample mean): {100*np.nanmean(put_spread_nf):.2f} cents")

# =============================================================================
# 5. Validate against pre-computed values in ImpliedVols.mat
# =============================================================================
print("\n--- Validating against ImpliedVols.mat pre-computed values ---")
iv = load_implied_vols()

# Compare put CPDI for financials
iv_putcpdibskf = iv['putcpdibskf'].flatten()
iv_putcpdiixf = iv['putcpdiixf'].flatten()
iv_putspreadf = iv_putcpdibskf - iv_putcpdiixf

my_putspreadf = put_spread_f

pre = datelist < PRE_CRISIS_END
cri = datelist > CRISIS_START

print("Put spread validation (cents):")
print_comparison("Pre-crisis mean (fin)",
                 100 * np.nanmean(my_putspreadf[pre]),
                 100 * np.nanmean(iv_putspreadf[pre]))
print_comparison("Crisis mean (fin)",
                 100 * np.nanmean(my_putspreadf[cri]),
                 100 * np.nanmean(iv_putspreadf[cri]))
print_comparison("Crisis max (fin)",
                 100 * np.nanmax(my_putspreadf[cri]),
                 100 * np.nanmax(iv_putspreadf[cri]))

# =============================================================================
# 6. Load correlation and volatility data for Table 2
# =============================================================================
print("\n--- Loading correlation and volatility data ---")
rc = load_realized_corr()
equicorrf = rc['equicorrf'].flatten()
equicorrnf = rc['equicorrnf'].flatten()

ic = load_implied_corrs()
impcorr = ic['impcorr']

# Also get volatility from ImpliedVols
callixivf = iv['callixivf'].flatten()
callixivnf = iv['callixivnf'].flatten()

print(f"  Realized corr (fin, full sample mean): {np.nanmean(equicorrf):.3f}")
print(f"  Realized corr (nonfin, full sample mean): {np.nanmean(equicorrnf):.3f}")

# =============================================================================
# 7. Save all computed data
# =============================================================================
print("\n--- Saving processed data ---")
np.savez(os.path.join(OUTPUT_DIR, 'spread_data.npz'),
         datelist=datelist,
         # CPDI by sector
         putcpdiix=putcpdiix, callcpdiix=callcpdiix,
         putcpdibsk=putcpdibsk, callcpdibsk=callcpdibsk,
         # Financial
         putcpdiixf=putcpdiixf, putcpdibskf=putcpdibskf,
         callcpdiixf=callcpdiixf, callcpdibskf=callcpdibskf,
         # Non-financial
         putcpdiixnf=putcpdiixnf, putcpdibsknf=putcpdibsknf,
         callcpdiixnf=callcpdiixnf, callcpdibsknf=callcpdibsknf,
         # Spreads
         put_spread_f=put_spread_f, put_spread_nf=put_spread_nf,
         call_spread_f=call_spread_f, call_spread_nf=call_spread_nf,
         # Market data
         totme=totme,
         # Correlation and volatility
         equicorrf=equicorrf, equicorrnf=equicorrnf,
         impcorr=impcorr,
         callixivf=callixivf, callixivnf=callixivnf,
         # Raw basket/index prices and strikes
         putbsk=putbsk, callbsk=callbsk,
         putstrbsk=putstrbsk, callstrbsk=callstrbsk,
         putix=putix, callix=callix,
         putstrix=putstrix, callstrix=callstrix,
         yldbsk=yldbsk, yldix=yldix, rf=rf)

print("  Saved to output/spread_data.npz")
print("\n01_clean.py complete.")
