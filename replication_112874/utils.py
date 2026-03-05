"""
Utility functions for replication of Kelly, Lustig & Van Nieuwerburgh (2016)
"Too-Systemic-To-Fail: What Option Markets Imply About Sector-Wide Government Guarantees"
AER, Paper ID 112874
"""

import os
import numpy as np
import pandas as pd
import scipy.io as sio
from scipy.stats import norm

# =============================================================================
# Paths
# =============================================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, '..', '112874-V1', 'Data_Code_Final_AER', 'Data')
CODE_DIR = os.path.join(BASE_DIR, '..', '112874-V1', 'Data_Code_Final_AER', 'Code')
OUTPUT_DIR = os.path.join(BASE_DIR, 'output')
os.makedirs(OUTPUT_DIR, exist_ok=True)

# =============================================================================
# Sector definitions (order matches MATLAB ETFgiclist)
# =============================================================================
SECTOR_NAMES = [
    'ConsDisc', 'ConsStp', 'Energy', 'Financials', 'Healthcare',
    'Industrial', 'Materials', 'Technology', 'Utilities'
]
SECTOR_ABBREVS = [
    'consdisc', 'consstap', 'energy', 'financials', 'healthcare',
    'industrial', 'materials', 'technology', 'utilities'
]
FIN_IDX = 3  # Financials is index 3 (0-based)
NF_IDX = [0, 1, 2, 4, 5, 6, 7, 8]  # Non-financial sector indices

# ETF GICS codes (matching MATLAB ETFgiclist)
ETF_GICLIST = np.array([
    [25, np.nan], [30, np.nan], [10, np.nan], [40, np.nan], [35, np.nan],
    [20, np.nan], [15, np.nan], [45, 50], [55, np.nan]
])

# Date cutoffs
PRE_CRISIS_END = 20070800
CRISIS_START = 20070800

# =============================================================================
# Black-Scholes pricing (equivalent to MATLAB blsprice)
# =============================================================================
def blsprice(S, K, r, T, sigma, q=0.0):
    """
    Black-Scholes European option pricing.
    Equivalent to MATLAB's blsprice(Price, Strike, Rate, Time, Volatility, Yield).

    Returns: (call_price, put_price)
    """
    S = np.asarray(S, dtype=float)
    K = np.asarray(K, dtype=float)
    r = np.asarray(r, dtype=float)
    T = np.asarray(T, dtype=float)
    sigma = np.asarray(sigma, dtype=float)
    q = np.asarray(q, dtype=float)

    d1 = (np.log(S / K) + (r - q + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)

    call = S * np.exp(-q * T) * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    put = K * np.exp(-r * T) * norm.cdf(-d2) - S * np.exp(-q * T) * norm.cdf(-d1)

    return call, put


def bs_implied_vol(price, S, K, r, T, q=0.0, option_type='call', tol=1e-8, max_iter=100):
    """Compute Black-Scholes implied volatility using Newton-Raphson."""
    sigma = 0.3  # initial guess
    for _ in range(max_iter):
        call, put = blsprice(S, K, r, T, sigma, q)
        if option_type == 'call':
            diff = call - price
        else:
            diff = put - price

        d1 = (np.log(S / K) + (r - q + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
        vega = S * np.exp(-q * T) * norm.pdf(d1) * np.sqrt(T)

        if abs(vega) < 1e-12:
            return np.nan

        sigma = sigma - diff / vega
        if abs(diff) < tol:
            return sigma
    return sigma


# =============================================================================
# Weighted mean (equivalent to MATLAB wmean)
# =============================================================================
def wmean(values, weights):
    """Weighted mean, handling NaN values."""
    values = np.asarray(values, dtype=float)
    weights = np.asarray(weights, dtype=float)
    mask = ~np.isnan(values) & ~np.isnan(weights)
    if mask.sum() == 0:
        return np.nan
    return np.average(values[mask], weights=weights[mask])


# =============================================================================
# Data loading helpers
# =============================================================================
def load_mat(filename):
    """Load a .mat file from the data directory."""
    filepath = os.path.join(DATA_DIR, filename)
    return sio.loadmat(filepath)


def load_strike_matched_options():
    """Load the main strike-matched options dataset."""
    return load_mat('Strike-matched-options-delta25-365day.mat')


def load_implied_vols():
    """Load ImpliedVols.mat which contains pre-computed CPDI values."""
    return load_mat('ImpliedVols.mat')


def load_implied_corrs():
    """Load ImpliedCorrs.mat."""
    return load_mat('ImpliedCorrs.mat')


def load_realized_corr():
    """Load realized_correlation_20141220.mat."""
    return load_mat('realized_correlation_20141220.mat')


def load_estimation_results(sector, bailout=True):
    """Load sector estimation results."""
    bail_str = 'bail' if bailout else 'nobail'
    filename = f'Estimation_results_{sector}_{bail_str}_final.mat'
    return load_mat(filename)


def load_data_time_series_fits(sector):
    """Load data time series for fits."""
    filename = f'Data_time_series_for_fits_{sector}_150505.mat'
    return load_mat(filename)


def load_data_time_series_search(sector):
    """Load data time series for search."""
    filename = f'Data_time_series_for_search_{sector}_nobs_250_150505.mat'
    return load_mat(filename)


# =============================================================================
# Spread computation helpers
# =============================================================================
def compute_cpdi(option_prices, strikes):
    """Compute cost per dollar insured = option_price / strike."""
    return option_prices / strikes


def compute_spread(cpdi_basket, cpdi_index):
    """Compute basket-index spread in cents per dollar insured."""
    return 100 * (cpdi_basket - cpdi_index)


def compute_vw_nonfin(values_9sectors, weights_9sectors):
    """Compute value-weighted non-financial average across 8 non-fin sectors."""
    T = values_9sectors.shape[0]
    result = np.full(T, np.nan)
    for t in range(T):
        vals = values_9sectors[t, NF_IDX]
        wts = weights_9sectors[t, NF_IDX]
        mask = ~np.isnan(vals) & ~np.isnan(wts)
        if mask.sum() > 0:
            result[t] = np.average(vals[mask], weights=wts[mask])
    return result


# =============================================================================
# Summary statistics helpers
# =============================================================================
def spread_summary_stats(spread, datelist, pre_end=PRE_CRISIS_END, crisis_start=CRISIS_START):
    """Compute summary statistics for a spread series (pre-crisis and crisis)."""
    pre = datelist.flatten() < pre_end
    cri = datelist.flatten() > crisis_start

    stats = {}
    for label, mask in [('pre', pre), ('crisis', cri)]:
        s = spread[mask]
        s = s[~np.isnan(s)]
        stats[f'{label}_mean'] = np.mean(s) if len(s) > 0 else np.nan
        stats[f'{label}_std'] = np.std(s, ddof=1) if len(s) > 0 else np.nan
        stats[f'{label}_max'] = np.max(s) if len(s) > 0 else np.nan
        stats[f'{label}_min'] = np.min(s) if len(s) > 0 else np.nan

    return stats


def print_comparison(label, python_val, published_val, units=''):
    """Print a comparison between Python and published values."""
    if np.isnan(python_val) or np.isnan(published_val):
        print(f"  {label}: Python={python_val:.4f}, Published={published_val:.4f} {units}")
        return
    diff = python_val - published_val
    pct = 100 * diff / abs(published_val) if published_val != 0 else np.inf
    match = "MATCH" if abs(pct) < 5 else ("CLOSE" if abs(pct) < 15 else "DIFF")
    print(f"  {label}: Python={python_val:.4f}, Published={published_val:.4f}, "
          f"Diff={diff:.4f} ({pct:+.1f}%) [{match}] {units}")


if __name__ == '__main__':
    # Quick test
    c, p = blsprice(1, 0.8, 0.05, 1, 0.3, 0.02)
    print(f"BS test: call={c:.6f}, put={p:.6f}")

    # Test data loading
    d = load_strike_matched_options()
    print(f"Options data: {d['datelist'].shape[0]} dates, {d['N'][0,0]} stocks")
    print(f"Date range: {d['datelist'][0,0]} to {d['datelist'][-1,0]}")

    iv = load_implied_vols()
    print(f"ImpliedVols loaded: {len([k for k in iv.keys() if not k.startswith('_')])} variables")
