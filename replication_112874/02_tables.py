"""
02_tables.py - Reproduce Tables 1-10 from Kelly, Lustig & Van Nieuwerburgh (2016)
"""

import numpy as np
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from utils import *

# Load processed data from 01_clean.py
data = np.load(os.path.join(OUTPUT_DIR, 'spread_data.npz'))
datelist = data['datelist']
T = len(datelist)

pre = datelist < PRE_CRISIS_END
cri = datelist > CRISIS_START

# =============================================================================
# TABLE 1: Basket-Index Spread Summary Statistics
# Panel I: Delta-matched, TTM=365, Delta=25 (strike-matched in original)
# =============================================================================
print("=" * 80)
print("TABLE 1: Basket-Index Spread (cents per dollar insured)")
print("=" * 80)

put_spread_f = data['put_spread_f']
put_spread_nf = data['put_spread_nf']
call_spread_f = data['call_spread_f']
call_spread_nf = data['call_spread_nf']

# Financial sector
fin_tab = np.zeros((6, 2))
fin_tab[0] = [100 * np.nanmean(put_spread_f[pre]), 100 * np.nanmean(call_spread_f[pre])]
fin_tab[1] = [100 * np.nanstd(put_spread_f[pre], ddof=1), 100 * np.nanstd(call_spread_f[pre], ddof=1)]
fin_tab[2] = [100 * np.nanmax(put_spread_f[pre]), 100 * np.nanmax(call_spread_f[pre])]
fin_tab[3] = [100 * np.nanmean(put_spread_f[cri]), 100 * np.nanmean(call_spread_f[cri])]
fin_tab[4] = [100 * np.nanstd(put_spread_f[cri], ddof=1), 100 * np.nanstd(call_spread_f[cri], ddof=1)]
fin_tab[5] = [100 * np.nanmax(put_spread_f[cri]), 100 * np.nanmax(call_spread_f[cri])]

# Non-financial sector
nfin_tab = np.zeros((6, 2))
nfin_tab[0] = [100 * np.nanmean(put_spread_nf[pre]), 100 * np.nanmean(call_spread_nf[pre])]
nfin_tab[1] = [100 * np.nanstd(put_spread_nf[pre], ddof=1), 100 * np.nanstd(call_spread_nf[pre], ddof=1)]
nfin_tab[2] = [100 * np.nanmax(put_spread_nf[pre]), 100 * np.nanmax(call_spread_nf[pre])]
nfin_tab[3] = [100 * np.nanmean(put_spread_nf[cri]), 100 * np.nanmean(call_spread_nf[cri])]
nfin_tab[4] = [100 * np.nanstd(put_spread_nf[cri], ddof=1), 100 * np.nanstd(call_spread_nf[cri], ddof=1)]
nfin_tab[5] = [100 * np.nanmax(put_spread_nf[cri]), 100 * np.nanmax(call_spread_nf[cri])]

# Difference (fin - nonfin)
diff_tab = fin_tab - nfin_tab

# DD: (fin put - nonfin put) - (fin call - nonfin call)
dd_put_f = put_spread_f - put_spread_nf
dd_call_f = call_spread_f - call_spread_nf
dd = dd_put_f - dd_call_f

dd_tab = np.zeros((6, 1))
dd_tab[0] = 100 * np.nanmean(dd[pre])
dd_tab[1] = 100 * np.nanstd(dd[pre], ddof=1)
dd_tab[2] = 100 * np.nanmax(dd[pre])
dd_tab[3] = 100 * np.nanmean(dd[cri])
dd_tab[4] = 100 * np.nanstd(dd[cri], ddof=1)
dd_tab[5] = 100 * np.nanmax(dd[cri])

row_labels = ['Pre: Mean', 'Pre: Std', 'Pre: Max', 'Cri: Mean', 'Cri: Std', 'Cri: Max']

print("\n{:>12s}  {:>8s} {:>8s}  {:>8s} {:>8s}  {:>8s} {:>8s}  {:>8s}".format(
    '', 'Fin Put', 'Fin Call', 'NF Put', 'NF Call', 'Dif Put', 'Dif Call', 'DDD'))
print("-" * 85)
for r in range(6):
    print(f"{row_labels[r]:>12s}  {fin_tab[r,0]:8.2f} {fin_tab[r,1]:8.2f}  "
          f"{nfin_tab[r,0]:8.2f} {nfin_tab[r,1]:8.2f}  "
          f"{diff_tab[r,0]:8.2f} {diff_tab[r,1]:8.2f}  {dd_tab[r,0]:8.2f}")

# Published values for comparison (Table I, Panel I, from paper p.12)
print("\n--- Comparison with published Table 1 ---")
print_comparison("Pre mean put fin", fin_tab[0, 0], 0.81, "cents")
print_comparison("Pre std put fin", fin_tab[1, 0], 0.20, "cents")
print_comparison("Cri mean put fin", fin_tab[3, 0], 3.79, "cents")
print_comparison("Cri std put fin", fin_tab[4, 0], 2.39, "cents")
print_comparison("Cri max put fin", fin_tab[5, 0], 12.46, "cents")
print_comparison("Cri mean put nonfin", nfin_tab[3, 0], 1.57, "cents")

# =============================================================================
# TABLE 2: Correlation and Volatility
# =============================================================================
print("\n" + "=" * 80)
print("TABLE 2: Correlation and Volatility")
print("=" * 80)

equicorrf = data['equicorrf']
equicorrnf = data['equicorrnf']
impcorr = data['impcorr']
totme = data['totme']
callixivf = data['callixivf']
callixivnf = data['callixivnf']

# Implied correlations
fin_ic = impcorr[:, FIN_IDX]
nonfin_ic = np.nansum(impcorr[:, NF_IDX] * totme[:, NF_IDX], axis=1) / np.nansum(totme[:, NF_IDX], axis=1)

# Load IV data for volatility
iv = load_implied_vols()
# Index volatility from ImpliedVols
ixvol_data = iv.get('callixivf', np.full((T, 1), np.nan)).flatten()
ixvol_nf_data = iv.get('callixivnf', np.full((T, 1), np.nan)).flatten()

mat = np.column_stack([equicorrf, equicorrnf, fin_ic, nonfin_ic,
                        callixivf**2, callixivnf**2])

# Convert volatility columns to levels (sqrt) after computing means
means_pre = np.nanmean(mat[pre], axis=0)
max_pre = np.nanmax(mat[pre], axis=0)
means_cri = np.nanmean(mat[cri], axis=0)
max_cri = np.nanmax(mat[cri], axis=0)

# For vol columns (4,5), take sqrt of means
tab2 = np.array([means_pre, max_pre, means_cri, max_cri])
tab2[:, 4:6] = np.sqrt(tab2[:, 4:6])

diff_row = tab2[2] - tab2[0]

print("\n{:>12s}  {:>8s} {:>8s}  {:>8s} {:>8s}  {:>8s} {:>8s}".format(
    '', 'RC Fin', 'RC NF', 'IC Fin', 'IC NF', 'IV Fin', 'IV NF'))
print("-" * 70)
labels2 = ['Pre Mean', 'Pre Max', 'Cri Mean', 'Cri Max', 'Change']
for r in range(4):
    print(f"{labels2[r]:>12s}  {tab2[r,0]:8.3f} {tab2[r,1]:8.3f}  "
          f"{tab2[r,2]:8.3f} {tab2[r,3]:8.3f}  {tab2[r,4]:8.3f} {tab2[r,5]:8.3f}")
print(f"{labels2[4]:>12s}  {diff_row[0]:8.3f} {diff_row[1]:8.3f}  "
      f"{diff_row[2]:8.3f} {diff_row[3]:8.3f}  {diff_row[4]:8.3f} {diff_row[5]:8.3f}")

# Published values (Table II from paper p.17)
print("\n--- Comparison with published Table 2 ---")
print_comparison("Pre-crisis realized corr fin", tab2[0, 0], 0.458, "")
print_comparison("Crisis realized corr fin", tab2[2, 0], 0.576, "")
print_comparison("Pre-crisis realized corr nonfin", tab2[0, 1], 0.337, "")
print_comparison("Crisis realized corr nonfin", tab2[2, 1], 0.568, "")

# =============================================================================
# TABLE 3: Black-Scholes Model Fit
# Uses implied vol from basket calls and realized correlation
# =============================================================================
print("\n" + "=" * 80)
print("TABLE 3: Black-Scholes Model Fit (CPDI, cents)")
print("=" * 80)

# Load pre-computed from ImpliedVols - already has putcpdibsk etc.
iv = load_implied_vols()
rc = load_realized_corr()

# BS model: index IV = basket IV * sqrt(correlation)
equicorr_9 = np.tile(rc['equicorrnf'], (1, 9))
equicorr_9[:, FIN_IDX] = rc['equicorrf'].flatten()
callbskiv = iv['callbskiv']
callixivfit = callbskiv * np.sqrt(equicorr_9)
putbskivfit = callbskiv
putixivfit = callixivfit

iv_datelist = iv['datelist'].flatten()
iv_T = len(iv_datelist)
iv_rf = iv['rf'].flatten()
iv_putstrbsk = iv['putstrbsk']
iv_callstrbsk = iv['callstrbsk']
iv_putstrix = iv['putstrix']
iv_callstrix = iv['callstrix']
iv_yldbsk = iv['yldbsk']
iv_yldix = iv['yldix']
iv_totme = iv['totme']

callbskfit = np.full((iv_T, 9), np.nan)
putbskfit = np.full((iv_T, 9), np.nan)
callixfit = np.full((iv_T, 9), np.nan)
putixfit = np.full((iv_T, 9), np.nan)

for i in range(9):
    valid = ~np.isnan(iv_callstrix[:, i] + iv_callstrbsk[:, i] + iv_putstrix[:, i] +
                      iv_putstrbsk[:, i] + callbskiv[:, i] + callixivfit[:, i] +
                      iv_yldbsk[:, i] + iv_yldix[:, i])
    loc = np.where(valid)[0]
    if len(loc) == 0:
        continue

    callbskfit[loc, i], _ = blsprice(1, iv_callstrbsk[loc, i], iv_rf[loc], 1,
                                      callbskiv[loc, i], iv_yldbsk[loc, i])
    _, putbskfit[loc, i] = blsprice(1, iv_putstrbsk[loc, i], iv_rf[loc], 1,
                                     putbskivfit[loc, i], iv_yldbsk[loc, i])
    callixfit[loc, i], _ = blsprice(1, iv_callstrix[loc, i], iv_rf[loc], 1,
                                     callixivfit[loc, i], iv_yldbsk[loc, i])
    _, putixfit[loc, i] = blsprice(1, iv_putstrix[loc, i], iv_rf[loc], 1,
                                    putixivfit[loc, i], iv_yldbsk[loc, i])

# Model CPDI
putcpdiix_model = 100 * putixfit / iv_putstrix
callcpdiix_model = 100 * callixfit / iv_callstrix
putcpdibsk_model = 100 * putbskfit / iv_putstrbsk
callcpdibsk_model = 100 * callbskfit / iv_callstrbsk

# Data CPDI (from ImpliedVols)
putcpdiix_data = 100 * iv['putcpdiix']
callcpdiix_data = 100 * iv['callcpdiix']
putcpdibsk_data = 100 * iv['putcpdibsk']
callcpdibsk_data = 100 * iv['callcpdibsk']

def model_fits_table(dl, putixf, putixnf, putbskf, putbsknf,
                     callixf, callixnf, callbskf, callbsknf):
    """Replicate ModelFitsTable.m"""
    pre_m = dl < PRE_CRISIS_END
    cri_m = dl > CRISIS_START

    # Row order: [pre, cri, change] x [putbsk, putix, putspread, callbsk, callix, callspread] x [fin, nonfin]
    result = np.zeros((3, 12))
    for col_offset, (bskf, ixf, bsknf, ixnf) in enumerate([
        (putbskf, putixf, putbsknf, putixnf),
        (callbskf, callixf, callbsknf, callixnf)
    ]):
        co = col_offset * 6
        result[0, co:co+3] = [np.nanmean(bskf[pre_m]), np.nanmean(ixf[pre_m]),
                                np.nanmean(bskf[pre_m]) - np.nanmean(ixf[pre_m])]
        result[1, co:co+3] = [np.nanmean(bskf[cri_m]), np.nanmean(ixf[cri_m]),
                                np.nanmean(bskf[cri_m]) - np.nanmean(ixf[cri_m])]
        result[0, co+3:co+6] = [np.nanmean(bsknf[pre_m]), np.nanmean(ixnf[pre_m]),
                                  np.nanmean(bsknf[pre_m]) - np.nanmean(ixnf[pre_m])]
        result[1, co+3:co+6] = [np.nanmean(bsknf[cri_m]), np.nanmean(ixnf[cri_m]),
                                  np.nanmean(bsknf[cri_m]) - np.nanmean(ixnf[cri_m])]
    result[2] = result[1] - result[0]
    return result

# Model fits
model_f_putix = putcpdiix_model[:, FIN_IDX]
model_f_putbsk = putcpdibsk_model[:, FIN_IDX]
model_f_callix = callcpdiix_model[:, FIN_IDX]
model_f_callbsk = callcpdibsk_model[:, FIN_IDX]

model_nf_putix = compute_vw_nonfin(putcpdiix_model, iv_totme)
model_nf_putbsk = compute_vw_nonfin(putcpdibsk_model, iv_totme)
model_nf_callix = compute_vw_nonfin(callcpdiix_model, iv_totme)
model_nf_callbsk = compute_vw_nonfin(callcpdibsk_model, iv_totme)

tab3_model = model_fits_table(iv_datelist, model_f_putix, model_nf_putix,
                               model_f_putbsk, model_nf_putbsk,
                               model_f_callix, model_nf_callix,
                               model_f_callbsk, model_nf_callbsk)

# Data fits
data_f_putix = putcpdiix_data[:, FIN_IDX]
data_f_putbsk = putcpdibsk_data[:, FIN_IDX]
data_f_callix = callcpdiix_data[:, FIN_IDX]
data_f_callbsk = callcpdibsk_data[:, FIN_IDX]

data_nf_putix = compute_vw_nonfin(putcpdiix_data, iv_totme)
data_nf_putbsk = compute_vw_nonfin(putcpdibsk_data, iv_totme)
data_nf_callix = compute_vw_nonfin(callcpdiix_data, iv_totme)
data_nf_callbsk = compute_vw_nonfin(callcpdibsk_data, iv_totme)

tab3_data = model_fits_table(iv_datelist, data_f_putix, data_nf_putix,
                              data_f_putbsk, data_nf_putbsk,
                              data_f_callix, data_nf_callix,
                              data_f_callbsk, data_nf_callbsk)

print("\nTable 3 - BS Model vs Data (Puts, Financials)")
print(f"{'':>12s}  {'PutBsk':>8s} {'PutIx':>8s} {'PutSpr':>8s}  {'CallBsk':>8s} {'CallIx':>8s} {'CallSpr':>8s}")
print("-" * 72)
for label, tab in [("Model", tab3_model), ("Data", tab3_data)]:
    for r, rl in enumerate(["Pre", "Cri", "Chg"]):
        print(f"{label+' '+rl:>12s}  {tab[r,0]:8.2f} {tab[r,1]:8.2f} {tab[r,2]:8.2f}  "
              f"{tab[r,6]:8.2f} {tab[r,7]:8.2f} {tab[r,8]:8.2f}")
    print()

# =============================================================================
# TABLES 7-8: Merton-Jump Model Fits (from pre-computed estimation results)
# =============================================================================
print("\n" + "=" * 80)
print("TABLES 7-8: Merton-Jump Model Estimated Fits")
print("=" * 80)

# Load all sector estimation results
params_bail = np.full((9, 4), np.nan)
params_nobail = np.full((9, 3), np.nan)

bail_full = np.full((1635, 6, 9), np.nan)
nobail_full = np.full((1635, 6, 9), np.nan)
data_full_est = np.full((1635, 6, 9), np.nan)
totme_full = np.full((1635, 9), np.nan)
cntfac_full = np.full((1635, 6, 9), np.nan)

for i, sec in enumerate(SECTOR_ABBREVS):
    # Bailout model
    d_bail = load_estimation_results(sec, bailout=True)
    oth_bail = d_bail['othinfo']
    params_bail[i] = d_bail['x'].flatten()
    bail_full[:, :, i] = oth_bail['seriesbailfull'][0, 0]
    cntfac_full[:, :, i] = oth_bail['seriesnobailfull'][0, 0]
    data_full_est[:, :, i] = oth_bail['data_full'][0, 0]
    totme_full[:, i] = oth_bail['totme_full'][0, 0].flatten()

    # No-bailout model
    d_nobail = load_estimation_results(sec, bailout=False)
    params_nobail[i] = d_nobail['x'].flatten()[:3]
    nobail_full[:, :, i] = d_nobail['othinfo']['seriesnobailfull'][0, 0]

est_datelist = oth_bail['datelist_full'][0, 0].flatten()
TT = len(est_datelist)

print("\nEstimated Parameters:")
print(f"{'Sector':>12s}  {'sigma_d':>8s} {'theta_r':>8s} {'delta_r':>8s} {'J_bar':>8s}  | {'sigma_d':>8s} {'theta_r':>8s} {'delta_r':>8s}")
print("-" * 90)
for i, sec in enumerate(SECTOR_NAMES):
    pb = params_bail[i]
    pnb = params_nobail[i]
    print(f"{sec:>12s}  {pb[0]:8.3f} {pb[1]:8.3f} {pb[2]:8.3f} {pb[3]:8.3f}  | {pnb[0]:8.3f} {pnb[1]:8.3f} {pnb[2]:8.3f}")

# Table 7: No-bailout model fits
def build_model_table(series_full, dl, me):
    """Build ModelFitsTable from 6-column series data."""
    putcpdiix_s = series_full[:, 1, :]
    putcpdibsk_s = series_full[:, 0, :]
    callcpdiix_s = series_full[:, 4, :]
    callcpdibsk_s = series_full[:, 3, :]

    f_putix = putcpdiix_s[:, FIN_IDX]
    f_putbsk = putcpdibsk_s[:, FIN_IDX]
    f_callix = callcpdiix_s[:, FIN_IDX]
    f_callbsk = callcpdibsk_s[:, FIN_IDX]

    nf_putix = compute_vw_nonfin(putcpdiix_s, me)
    nf_putbsk = compute_vw_nonfin(putcpdibsk_s, me)
    nf_callix = compute_vw_nonfin(callcpdiix_s, me)
    nf_callbsk = compute_vw_nonfin(callcpdibsk_s, me)

    return model_fits_table(dl, f_putix, nf_putix, f_putbsk, nf_putbsk,
                            f_callix, nf_callix, f_callbsk, nf_callbsk)

tab7 = build_model_table(nobail_full, est_datelist, totme_full)
tab8 = build_model_table(bail_full, est_datelist, totme_full)

for tnum, tab, title in [(7, tab7, "No-Bailout"), (8, tab8, "Bailout")]:
    print(f"\nTable {tnum}: {title} Model")
    print(f"{'':>12s}  {'F PutBsk':>8s} {'F PutIx':>8s} {'F PutSpr':>8s}  {'NF PutBsk':>9s} {'NF PutIx':>8s} {'NF PutSpr':>9s}")
    for r, rl in enumerate(["Pre-crisis", "Crisis", "Change"]):
        print(f"{rl:>12s}  {tab[r,0]:8.2f} {tab[r,1]:8.2f} {tab[r,2]:8.2f}  "
              f"{tab[r,3]:9.2f} {tab[r,4]:8.2f} {tab[r,5]:9.2f}")

# Data table (same for both)
tab_data_est = build_model_table(data_full_est, est_datelist, totme_full)
print(f"\nData (from estimation):")
print(f"{'':>12s}  {'F PutBsk':>8s} {'F PutIx':>8s} {'F PutSpr':>8s}  {'NF PutBsk':>9s} {'NF PutIx':>8s} {'NF PutSpr':>9s}")
for r, rl in enumerate(["Pre-crisis", "Crisis", "Change"]):
    print(f"{rl:>12s}  {tab_data_est[r,0]:8.2f} {tab_data_est[r,1]:8.2f} {tab_data_est[r,2]:8.2f}  "
          f"{tab_data_est[r,3]:9.2f} {tab_data_est[r,4]:8.2f} {tab_data_est[r,5]:9.2f}")

# =============================================================================
# TABLE 9A: Model Distance Metrics
# =============================================================================
print("\n" + "=" * 80)
print("TABLE 9A: Model Distance Tests")
print("=" * 80)

alldist = np.full(9, np.nan)
rawdist = np.full(9, np.nan)

for i, sec in enumerate(SECTOR_ABBREVS):
    d_bail = load_estimation_results(sec, bailout=True)
    d_nobail = load_estimation_results(sec, bailout=False)

    oth_bail = d_bail['othinfo']
    oth_nobail = d_nobail['othinfo']

    data_s = oth_bail['data'][0, 0]
    bail_s = oth_bail['seriesbail'][0, 0]
    nobail_s = oth_nobail['seriesnobail'][0, 0]
    dl_s = oth_bail['datelist'][0, 0].flatten()

    cols = [0, 2]  # putcpdibsk, putspread

    cri_mask = dl_s > CRISIS_START

    diff_b = (data_s - bail_s)[cri_mask][:, cols]
    diff_nb = (data_s - nobail_s)[cri_mask][:, cols]

    M_b = np.nanmean(diff_b, axis=0)
    M_nb = np.nanmean(diff_nb, axis=0)

    alldist[i] = 18 * oth_nobail['fval'][0, 0][0, 0] - 18 * oth_bail['fval'][0, 0][0, 0]
    rawdist[i] = M_nb @ M_nb - M_b @ M_b

order = [3, 0, 1, 2, 4, 5, 6, 7, 8]
print(f"\n{'':>12s}  ", end="")
for idx in order:
    print(f"{SECTOR_NAMES[idx]:>12s}", end="")
print()
print(f"{'All moments':>12s}  ", end="")
for idx in order:
    print(f"{alldist[idx]:12.2f}", end="")
print()
print(f"{'Crisis put':>12s}  ", end="")
for idx in order:
    print(f"{rawdist[idx]:12.4f}", end="")
print()

# =============================================================================
# TABLE 10: Tech Crash (2000-2002)
# =============================================================================
print("\n" + "=" * 80)
print("TABLE 10: Tech Crash Period (2000-2002)")
print("=" * 80)

try:
    tc = load_mat('ImpliedVolsTechCrash.mat')
    tc_rc = load_mat('RealizedCorrsTechCrash.mat')

    tc_datelist = tc['datelist'].flatten()
    tc_T = len(tc_datelist)
    startdate_tc = 20000300
    enddate_tc = 20020200

    # Pre-computed CPDI from the file
    tc_putcpdibsknf = tc.get('putcpdibsknf', np.full((tc_T, 1), np.nan))
    tc_putcpdiixnf = tc.get('putcpdiixnf', np.full((tc_T, 1), np.nan))
    tc_callcpdibsknf = tc.get('callcpdibsknf', np.full((tc_T, 1), np.nan))
    tc_callcpdiixnf = tc.get('callcpdiixnf', np.full((tc_T, 1), np.nan))

    if tc_putcpdibsknf.ndim > 1:
        tc_putcpdibsknf = tc_putcpdibsknf.flatten()
    if tc_putcpdiixnf.ndim > 1:
        tc_putcpdiixnf = tc_putcpdiixnf.flatten()
    if tc_callcpdibsknf.ndim > 1:
        tc_callcpdibsknf = tc_callcpdibsknf.flatten()
    if tc_callcpdiixnf.ndim > 1:
        tc_callcpdiixnf = tc_callcpdiixnf.flatten()

    loc_tc = (tc_datelist > startdate_tc) & (tc_datelist < enddate_tc)

    data_tc = 100 * np.array([
        np.nanmean(tc_putcpdibsknf[loc_tc]),
        np.nanmean(tc_putcpdiixnf[loc_tc]),
        np.nanmean(tc_putcpdibsknf[loc_tc]) - np.nanmean(tc_putcpdiixnf[loc_tc]),
        np.nanmean(tc_callcpdibsknf[loc_tc]),
        np.nanmean(tc_callcpdiixnf[loc_tc]),
        np.nanmean(tc_callcpdibsknf[loc_tc]) - np.nanmean(tc_callcpdiixnf[loc_tc])
    ])

    print("\nTech Crash Data (non-financials, cents):")
    print(f"  Put Basket: {data_tc[0]:.2f}, Put Index: {data_tc[1]:.2f}, Put Spread: {data_tc[2]:.2f}")
    print(f"  Call Basket: {data_tc[3]:.2f}, Call Index: {data_tc[4]:.2f}, Call Spread: {data_tc[5]:.2f}")

except Exception as e:
    print(f"  Could not load tech crash data: {e}")

# =============================================================================
# Summary
# =============================================================================
print("\n" + "=" * 80)
print("SUMMARY OF KEY REPLICATION RESULTS")
print("=" * 80)

print("\nTable 1 (key published values vs Python):")
pub = {
    'Pre mean put spread fin': (fin_tab[0, 0], 0.81),
    'Crisis mean put spread fin': (fin_tab[3, 0], 3.79),
    'Crisis max put spread fin': (fin_tab[5, 0], 12.46),
    'Crisis mean put spread nonfin': (nfin_tab[3, 0], 1.57),
    'Pre mean call spread fin': (fin_tab[0, 1], 0.25),
    'Crisis mean call spread fin': (fin_tab[3, 1], 0.06),
}
for k, (py_val, pub_val) in pub.items():
    print_comparison(k, py_val, pub_val, "cents")

print("\nTable 2 (key published values vs Python):")
pub2 = {
    'Pre realized corr fin': (tab2[0, 0], 0.458),
    'Crisis realized corr fin': (tab2[2, 0], 0.576),
    'Pre realized corr nonfin': (tab2[0, 1], 0.337),
    'Crisis realized corr nonfin': (tab2[2, 1], 0.568),
}
for k, (py_val, pub_val) in pub2.items():
    print_comparison(k, py_val, pub_val)

print("\n02_tables.py complete.")
