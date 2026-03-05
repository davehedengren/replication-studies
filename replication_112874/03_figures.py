"""
03_figures.py - Reproduce key figures from Kelly, Lustig & Van Nieuwerburgh (2016)
Figures 1-2 (empirical), Figures 3-6 (model), Figure 7 (announcement effects)
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from utils import *

# Load processed data
data = np.load(os.path.join(OUTPUT_DIR, 'spread_data.npz'))
datelist = data['datelist']
T = len(datelist)

# Convert datelist to datetime
def datelist_to_dates(dl):
    dates = []
    for d in dl:
        y = d // 10000
        m = (d % 10000) // 100
        d_day = d % 100
        if d_day == 0:
            d_day = 1
        try:
            dates.append(datetime(y, m, d_day))
        except:
            dates.append(datetime(y, 1, 1))
    return dates

dates = datelist_to_dates(datelist)

putcpdibskf = data['putcpdibskf']
putcpdiixf = data['putcpdiixf']
callcpdibskf = data['callcpdibskf']
callcpdiixf = data['callcpdiixf']
putcpdibsknf = data['putcpdibsknf']
putcpdiixnf = data['putcpdiixnf']
callcpdibsknf = data['callcpdibsknf']
callcpdiixnf = data['callcpdiixnf']

put_spread_f = data['put_spread_f']
put_spread_nf = data['put_spread_nf']
call_spread_f = data['call_spread_f']
call_spread_nf = data['call_spread_nf']

# =============================================================================
# Figure 1: Basket and Index Put/Call CPDI (Financial sector)
# =============================================================================
fig, axes = plt.subplots(2, 1, figsize=(14, 10))

# Panel A: Puts
ax = axes[0]
ax.plot(dates, 100 * putcpdibskf, 'gray', linewidth=1.5, linestyle=':', label='Basket')
ax.plot(dates, 100 * putcpdiixf, 'gray', linewidth=1.5, label='Index')
ax.plot(dates, 100 * (putcpdibskf - putcpdiixf), 'k', linewidth=2, label='Spread')
ax.set_ylabel('Cents per dollar insured')
ax.set_title('Panel A: Financial Sector Put Options')
ax.legend(loc='upper left')
ax.set_ylim([0, 30])
ax.grid(True, alpha=0.3)

# Panel B: Calls
ax = axes[1]
ax.plot(dates, 100 * callcpdibskf, 'gray', linewidth=1.5, linestyle=':', label='Basket')
ax.plot(dates, 100 * callcpdiixf, 'gray', linewidth=1.5, label='Index')
ax.plot(dates, 100 * (callcpdibskf - callcpdiixf), 'k', linewidth=2, label='Spread')
ax.set_ylabel('Cents per dollar insured')
ax.set_title('Panel B: Financial Sector Call Options')
ax.legend(loc='upper left')
ax.set_ylim([0, 8.5])
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'figure1_basket_index_cpdi.png'), dpi=150)
plt.close()
print("Figure 1 saved: Basket and Index CPDI (Financials)")

# =============================================================================
# Figure 2: Put Spread - Financial vs Non-Financial
# =============================================================================
fig, ax = plt.subplots(figsize=(14, 6))

# Focus on 2006 onwards
loc_2006 = datelist > 20060000
dates_2006 = [d for d, m in zip(dates, loc_2006) if m]

put_minus_call_f = (put_spread_f - call_spread_f)[loc_2006]
put_minus_call_nf = (put_spread_nf - call_spread_nf)[loc_2006]
diff = put_minus_call_f - put_minus_call_nf

ax.plot(dates_2006, 100 * put_minus_call_f, 'gray', linewidth=1.5, linestyle=':', label='Financials')
ax.plot(dates_2006, 100 * put_minus_call_nf, 'k', linewidth=1.5, linestyle=':', label='Non-financials')
ax.plot(dates_2006, 100 * diff, 'dimgray', linewidth=2, label='Difference')
ax.set_ylabel('Cents per dollar insured')
ax.set_title('Figure 2: Put-Call Spread Difference (Fin vs Non-Fin)')
ax.legend(loc='upper left')
ax.set_ylim([-1, 11])
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'figure2_put_call_spread.png'), dpi=150)
plt.close()
print("Figure 2 saved: Put-Call Spread Difference")

# =============================================================================
# Figure 3: Black-Scholes Model - Put Spread as function of correlation
# =============================================================================
fig, ax = plt.subplots(figsize=(10, 7))

# Simplified BS model: basket spread = f(correlation)
# Using representative parameters
N = 79  # number of firms
sigma_i = 0.30  # individual stock vol
K_frac = 0.80   # strike as fraction of spot (OTM put)
r = 0.03
ttm_val = 1.0

corrs = np.linspace(0.1, 0.9, 50)
put_spreads_model = []

for rho in corrs:
    sigma_index = sigma_i * np.sqrt(rho + (1 - rho) / N)
    _, put_index = blsprice(1, K_frac, r, ttm_val, sigma_index)
    _, put_indiv = blsprice(1, K_frac, r, ttm_val, sigma_i)
    cpdi_index = put_index / K_frac
    cpdi_basket = put_indiv / K_frac
    put_spreads_model.append(100 * (cpdi_basket - cpdi_index))

ax.plot(corrs, put_spreads_model, 'k-', linewidth=2)
ax.set_xlabel('Pairwise Correlation')
ax.set_ylabel('Put Spread (cents per dollar insured)')
ax.set_title('Figure 3: BS Model - Put Spread vs Correlation')
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'figure3_bs_correlation.png'), dpi=150)
plt.close()
print("Figure 3 saved: BS Model Put Spread vs Correlation")

# =============================================================================
# Figure 4: Realized Correlation Time Series
# =============================================================================
fig, ax = plt.subplots(figsize=(14, 6))

equicorrf = data['equicorrf']
equicorrnf = data['equicorrnf']

ax.plot(dates, equicorrf, 'k-', linewidth=1.5, label='Financials')
ax.plot(dates, equicorrnf, 'gray', linewidth=1.5, linestyle='--', label='Non-Financials')
ax.set_ylabel('Pairwise Equicorrelation')
ax.set_title('Figure 4: Realized Equity Return Correlations')
ax.legend(loc='upper left')
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'figure4_correlations.png'), dpi=150)
plt.close()
print("Figure 4 saved: Realized Correlations")

# =============================================================================
# Figure 5: Implied Volatility Skew (Basket minus Index)
# =============================================================================
# Load data for multiple deltas
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

iv = load_implied_vols()
iv_datelist = iv['datelist'].flatten()
pre_iv = iv_datelist < PRE_CRISIS_END
cri_iv = iv_datelist > CRISIS_START

# Put: basket IV minus index IV
putbskiv = iv['putbskivf'].flatten()
putixiv = iv['putixivf'].flatten()
callbskiv_f = iv['callbskivf'].flatten()
callixiv_f = iv['callixivf'].flatten()

iv_dates = datelist_to_dates(iv_datelist)

ax = axes[0]
ax.plot(iv_dates, 100 * (putbskiv - putixiv), 'k-', linewidth=1.5)
ax.set_title('Put IV: Basket minus Index (Fin)')
ax.set_ylabel('Percentage points')
ax.grid(True, alpha=0.3)

ax = axes[1]
ax.plot(iv_dates, 100 * (callbskiv_f - callixiv_f), 'k-', linewidth=1.5)
ax.set_title('Call IV: Basket minus Index (Fin)')
ax.set_ylabel('Percentage points')
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'figure5_iv_skew.png'), dpi=150)
plt.close()
print("Figure 5 saved: Implied Volatility Basket-Index Difference")

# =============================================================================
# Figure 6: MJ Model Fits - Implied Vol Spread by Sector
# =============================================================================
fig, axes = plt.subplots(3, 3, figsize=(18, 14))

# Load estimation results for each sector
for i, (sec, ax) in enumerate(zip(SECTOR_ABBREVS, axes.flat)):
    try:
        d_bail = load_estimation_results(sec, bailout=True)
        oth = d_bail['othinfo']

        dl_full = oth['datelist_full'][0, 0].flatten()
        data_s = oth['data_full'][0, 0]
        bail_s = oth['seriesbailfull'][0, 0]

        dl_dates = datelist_to_dates(dl_full)

        # Put spread: data vs model
        ax.plot(dl_dates, data_s[:, 2], 'k-', linewidth=1, alpha=0.7, label='Data')
        ax.plot(dl_dates, bail_s[:, 2], 'r-', linewidth=1, alpha=0.7, label='Model')
        ax.set_title(SECTOR_NAMES[i])
        ax.set_ylabel('Put Spread')
        if i == 0:
            ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
    except Exception as e:
        ax.set_title(f"{SECTOR_NAMES[i]} (error)")

plt.suptitle('Figure 6: MJ Model Put Spread Fits by Sector', fontsize=14, y=1.01)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'figure6_mj_fits_by_sector.png'), dpi=150, bbox_inches='tight')
plt.close()
print("Figure 6 saved: MJ Model Fits by Sector")

# =============================================================================
# Figure 7: Announcement Effects
# =============================================================================
fig, axes = plt.subplots(2, 1, figsize=(14, 10))

# Positive announcements (dates from paper p.19)
pos_dates = [20081003, 20081006, 20081125, 20090116, 20090202]
neg_dates = [20080303, 20080915, 20080929, 20081014, 20081107, 20081113]

# Financial put spread
put_spr_f = 100 * put_spread_f

# Map datelist to indices
dl_to_idx = {d: i for i, d in enumerate(datelist)}

# Find closest date for each announcement
def find_closest_idx(target, dl):
    diffs = np.abs(dl - target)
    return np.argmin(diffs)

window = 10  # +/- 10 business days

# Panel A: Positive announcements
ax = axes[0]
for j, ad in enumerate(pos_dates):
    idx = find_closest_idx(ad, datelist)
    start = max(0, idx - window)
    end = min(T, idx + window + 1)
    spread_window = put_spr_f[start:end]
    x_axis = np.arange(len(spread_window)) - (idx - start)
    ax.plot(x_axis, spread_window, '--', linewidth=1, alpha=0.6)

# Average
avg_pos = np.full(2 * window + 1, np.nan)
count = 0
for ad in pos_dates:
    idx = find_closest_idx(ad, datelist)
    if idx - window >= 0 and idx + window < T:
        if count == 0:
            avg_pos = put_spr_f[idx-window:idx+window+1].copy()
        else:
            avg_pos += put_spr_f[idx-window:idx+window+1]
        count += 1
if count > 0:
    avg_pos /= count
    ax.plot(np.arange(len(avg_pos)) - window, avg_pos, 'k-', linewidth=3, label='Average')

ax.axvline(0, color='red', linestyle='--', alpha=0.5)
ax.set_title('Panel A: Positive Announcements (Increase Bailout Probability)')
ax.set_xlabel('Business days from announcement')
ax.set_ylabel('Put Spread (cents)')
ax.legend()
ax.grid(True, alpha=0.3)

# Panel B: Negative announcements
ax = axes[1]
for j, ad in enumerate(neg_dates):
    idx = find_closest_idx(ad, datelist)
    start = max(0, idx - window)
    end = min(T, idx + window + 1)
    spread_window = put_spr_f[start:end]
    x_axis = np.arange(len(spread_window)) - (idx - start)
    ax.plot(x_axis, spread_window, '--', linewidth=1, alpha=0.6)

avg_neg = np.full(2 * window + 1, np.nan)
count = 0
for ad in neg_dates:
    idx = find_closest_idx(ad, datelist)
    if idx - window >= 0 and idx + window < T:
        if count == 0:
            avg_neg = put_spr_f[idx-window:idx+window+1].copy()
        else:
            avg_neg += put_spr_f[idx-window:idx+window+1]
        count += 1
if count > 0:
    avg_neg /= count
    ax.plot(np.arange(len(avg_neg)) - window, avg_neg, 'k-', linewidth=3, label='Average')

ax.axvline(0, color='red', linestyle='--', alpha=0.5)
ax.set_title('Panel B: Negative Announcements (Decrease Bailout Probability)')
ax.set_xlabel('Business days from announcement')
ax.set_ylabel('Put Spread (cents)')
ax.legend()
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'figure7_announcements.png'), dpi=150)
plt.close()
print("Figure 7 saved: Announcement Effects")

# =============================================================================
# Figure 8: Sector-by-sector basket-index spread
# =============================================================================
fig, axes = plt.subplots(3, 3, figsize=(18, 14))

putcpdibsk_all = data['putcpdibsk']
putcpdiix_all = data['putcpdiix']

for i, (sec, ax) in enumerate(zip(SECTOR_NAMES, axes.flat)):
    spread_i = 100 * (putcpdibsk_all[:, i] - putcpdiix_all[:, i])
    ax.plot(dates, spread_i, 'k-', linewidth=1)
    ax.set_title(sec)
    ax.set_ylabel('Cents')
    ax.grid(True, alpha=0.3)

plt.suptitle('Figure 8: Basket-Index Put Spread by Sector', fontsize=14, y=1.01)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'figure8_spread_by_sector.png'), dpi=150, bbox_inches='tight')
plt.close()
print("Figure 8 saved: Spread by Sector")

print("\n03_figures.py complete. All figures saved to output/")
