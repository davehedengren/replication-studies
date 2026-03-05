"""
05_robustness.py -- Descriptive and data-level robustness checks for
Federle et al. (2025) "The Price of War" (openICPSR 238484-V1)

Since the full LP estimation requires expensive panel construction
(trade_gravity.dta is ~80 MB with ~15.5M rows), this script focuses on
robustness checks that can be performed at the data/descriptive level:

  1. Exclude WWI/WWII: sample changes
  2. Post-WWII only: sample descriptives
  3. Casualty threshold sensitivity
  4. Interstate vs combined comparison
  5. Regional breakdown
  6. War duration sensitivity
  7. Temporal stability
  8. Winsorization impact
"""

import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Imports from project utils
# ---------------------------------------------------------------------------
sys.path.insert(0, str(Path(__file__).resolve().parent))
from utils import (
    load_sites,
    load_belligerents,
    load_macro,
    OUTPUT_DIR,
)

# ---------------------------------------------------------------------------
# Local output directory
# ---------------------------------------------------------------------------
REPL_OUTPUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'output')
os.makedirs(REPL_OUTPUT, exist_ok=True)


def section(title: str) -> None:
    """Print a section header."""
    print("\n" + "=" * 72)
    print(f"  {title}")
    print("=" * 72)


def subsection(title: str) -> None:
    """Print a subsection header."""
    print(f"\n  --- {title} ---")


def describe_sample(df: pd.DataFrame, label: str) -> dict:
    """
    Compute descriptive statistics for a sites DataFrame.

    Returns a dict of key statistics for comparison tables.
    """
    n_sites = len(df)
    n_wars = df["warname"].nunique()
    n_countries = df["iso"].nunique()

    cas = df["casualties"].dropna()
    cas_mean = cas.mean() if len(cas) > 0 else np.nan
    cas_median = cas.median() if len(cas) > 0 else np.nan
    cas_total = cas.sum() if len(cas) > 0 else 0

    caspop = df["shock_caspop_home"].dropna() if "shock_caspop_home" in df.columns else pd.Series(dtype=float)
    caspop_mean = caspop.mean() * 100 if len(caspop) > 0 else np.nan
    caspop_median = caspop.median() * 100 if len(caspop) > 0 else np.nan

    duration = df["end"] - df["start"] + 1
    dur_mean = duration.mean() if len(duration) > 0 else np.nan

    return {
        "label": label,
        "n_sites": n_sites,
        "n_wars": n_wars,
        "n_countries": n_countries,
        "cas_mean": cas_mean,
        "cas_median": cas_median,
        "cas_total": cas_total,
        "caspop_mean_pct": caspop_mean,
        "caspop_median_pct": caspop_median,
        "dur_mean": dur_mean,
    }


def print_comparison_table(stats_list: list, title: str = "") -> None:
    """Print a formatted comparison table from a list of stat dicts."""
    if title:
        print(f"\n  {title}")

    header = (
        f"  {'Sample':<30s}  {'Sites':>6s}  {'Wars':>5s}  {'Ctrys':>5s}  "
        f"{'MeanCas':>10s}  {'MedCas':>10s}  {'TotCas':>14s}  "
        f"{'Cas/Pop%':>8s}  {'Dur':>5s}"
    )
    print(header)
    print(f"  {'-'*30}  {'-'*6}  {'-'*5}  {'-'*5}  {'-'*10}  {'-'*10}  {'-'*14}  {'-'*8}  {'-'*5}")

    for s in stats_list:
        row = (
            f"  {s['label']:<30s}  {s['n_sites']:>6,}  {s['n_wars']:>5,}  {s['n_countries']:>5,}  "
            f"{s['cas_mean']:>10,.0f}  {s['cas_median']:>10,.0f}  {s['cas_total']:>14,.0f}  "
            f"{s['caspop_mean_pct']:>7.3f}%  {s['dur_mean']:>5.1f}"
        )
        print(row)


# ===================================================================
# Load data
# ===================================================================
print("Loading datasets...")

sites_all = load_sites("all")
sites_inter = load_sites("interstate")
bell = load_belligerents()
macro = load_macro()

print(f"  sites_all:   {sites_all.shape[0]:>6,} rows")
print(f"  sites_inter: {sites_inter.shape[0]:>6,} rows")
print(f"  belligerents:{bell.shape[0]:>6,} rows")
print(f"  macro:       {macro.shape[0]:>6,} rows")

# Baseline statistics
baseline_all = describe_sample(sites_all, "Baseline (all)")
baseline_inter = describe_sample(sites_inter, "Baseline (interstate)")


# ===================================================================
# CHECK 1: EXCLUDE WWI/WWII
# ===================================================================
section("CHECK 1: EXCLUDE WWI AND/OR WWII")

# Identify World War sites
ww1_names = sites_all[sites_all["warname"].str.contains("World War I", case=False, na=False)]["warname"].unique()
ww2_names = sites_all[sites_all["warname"].str.contains("World War II", case=False, na=False)]["warname"].unique()

print(f"  WWI war names found: {list(ww1_names)}")
print(f"  WWII war names found: {list(ww2_names)}")

ww_names = set(list(ww1_names) + list(ww2_names))

sites_no_ww = sites_all[~sites_all["warname"].isin(ww_names)]
sites_no_ww1 = sites_all[~sites_all["warname"].isin(ww1_names)]
sites_no_ww2 = sites_all[~sites_all["warname"].isin(ww2_names)]

stats_no_ww = describe_sample(sites_no_ww, "Excl. WWI+WWII")
stats_no_ww1 = describe_sample(sites_no_ww1, "Excl. WWI only")
stats_no_ww2 = describe_sample(sites_no_ww2, "Excl. WWII only")

print_comparison_table(
    [baseline_all, stats_no_ww1, stats_no_ww2, stats_no_ww],
    "Impact of excluding World Wars:"
)

# How much of total casualties are from World Wars?
ww_cas = sites_all[sites_all["warname"].isin(ww_names)]["casualties"].sum()
total_cas = sites_all["casualties"].sum()
print(f"\n  World War casualties: {ww_cas:,.0f} of {total_cas:,.0f} ({ww_cas/total_cas*100:.1f}%)")

ww_sites = sites_all[sites_all["warname"].isin(ww_names)]
print(f"  World War sites: {len(ww_sites)} of {len(sites_all)} ({len(ww_sites)/len(sites_all)*100:.1f}%)")


# ===================================================================
# CHECK 2: POST-WWII ONLY
# ===================================================================
section("CHECK 2: POST-WWII ONLY (1946+)")

sites_postww = sites_all[sites_all["start"] >= 1946]
sites_postww_inter = sites_inter[sites_inter["start"] >= 1946]

stats_postww = describe_sample(sites_postww, "Post-WWII (all)")
stats_postww_inter = describe_sample(sites_postww_inter, "Post-WWII (interstate)")

print_comparison_table(
    [baseline_all, stats_postww, baseline_inter, stats_postww_inter],
    "Full sample vs post-WWII:"
)

# How much of the sample is post-WWII?
print(f"\n  Share of sites post-1945 (all):       {len(sites_postww)/len(sites_all)*100:.1f}%")
print(f"  Share of sites post-1945 (interstate): {len(sites_postww_inter)/len(sites_inter)*100:.1f}%")


# ===================================================================
# CHECK 3: CASUALTY THRESHOLD SENSITIVITY
# ===================================================================
section("CHECK 3: CASUALTY THRESHOLD SENSITIVITY")

thresholds = [0, 100, 500, 1000, 2500, 5000, 10000, 50000]

subsection("All wars")
threshold_stats_all = []
for t in thresholds:
    df_t = sites_all[sites_all["casualties"] >= t]
    stats_t = describe_sample(df_t, f"casmin >= {t:,}")
    threshold_stats_all.append(stats_t)

print_comparison_table(threshold_stats_all, "Casualty threshold sensitivity (all wars):")

subsection("Interstate wars")
threshold_stats_inter = []
for t in thresholds:
    df_t = sites_inter[sites_inter["casualties"] >= t]
    stats_t = describe_sample(df_t, f"casmin >= {t:,}")
    threshold_stats_inter.append(stats_t)

print_comparison_table(threshold_stats_inter, "Casualty threshold sensitivity (interstate wars):")


# ===================================================================
# CHECK 4: INTERSTATE VS COMBINED
# ===================================================================
section("CHECK 4: INTERSTATE VS COMBINED COMPARISON")

# Load intrastate for comparison
try:
    sites_intra = load_sites("intrastate")
    stats_intra = describe_sample(sites_intra, "Intrastate/other")
except FileNotFoundError:
    sites_intra = None
    stats_intra = None
    print("  Warning: sites_intrastate.dta not found.")

comparison_list = [baseline_all, baseline_inter]
if stats_intra is not None:
    comparison_list.append(stats_intra)

print_comparison_table(comparison_list, "War type comparison:")

# Detailed comparison of key moments
if stats_intra is not None:
    print(f"\n  Interstate vs Intrastate differences:")
    print(f"    Mean casualties:    interstate={baseline_inter['cas_mean']:>12,.0f}, "
          f"intrastate={stats_intra['cas_mean']:>12,.0f}, "
          f"ratio={baseline_inter['cas_mean']/max(stats_intra['cas_mean'], 1):.2f}")
    print(f"    Mean cas/pop (%):   interstate={baseline_inter['caspop_mean_pct']:>8.3f}, "
          f"intrastate={stats_intra['caspop_mean_pct']:>8.3f}")
    print(f"    Mean duration (yr): interstate={baseline_inter['dur_mean']:>8.1f}, "
          f"intrastate={stats_intra['dur_mean']:>8.1f}")

# War type composition within "all"
if "type" in sites_all.columns:
    print(f"\n  War type composition in 'all' sample:")
    for wtype, count in sites_all["type"].value_counts().items():
        pct = count / len(sites_all) * 100
        mean_cas = sites_all.loc[sites_all["type"] == wtype, "casualties"].mean()
        print(f"    {wtype:<15s}: {count:>4} sites ({pct:>5.1f}%), mean cas={mean_cas:>12,.0f}")


# ===================================================================
# CHECK 5: REGIONAL BREAKDOWN
# ===================================================================
section("CHECK 5: REGIONAL BREAKDOWN")

# Continent mapping using COW/ISO codes
# This uses the Correlates of War state system codes common in the data
continent_map = {
    # Europe
    "AUS": "Europe", "AUH": "Europe", "BEL": "Europe", "BUL": "Europe",
    "CRO": "Europe", "CYP": "Europe", "CZE": "Europe", "DEN": "Europe",
    "EST": "Europe", "FIN": "Europe", "FRA": "Europe", "GMY": "Europe",
    "GRC": "Europe", "HUN": "Europe", "ICE": "Europe", "IRE": "Europe",
    "ITA": "Europe", "LAT": "Europe", "LIT": "Europe", "LUX": "Europe",
    "NTH": "Europe", "NOR": "Europe", "POL": "Europe", "POR": "Europe",
    "ROM": "Europe", "RUS": "Europe", "SER": "Europe", "SLO": "Europe",
    "SLV": "Europe", "SPN": "Europe", "SWD": "Europe", "SWZ": "Europe",
    "TUR": "Europe/Asia", "UKG": "Europe", "YUG": "Europe", "USR": "Europe",
    "GDR": "Europe",
    # Americas
    "USA": "Americas", "CAN": "Americas", "MEX": "Americas", "BRA": "Americas",
    "ARG": "Americas", "CHL": "Americas", "COL": "Americas", "VEN": "Americas",
    "PER": "Americas", "BOL": "Americas", "ECU": "Americas", "URU": "Americas",
    "PAR": "Americas", "CUB": "Americas", "DOM": "Americas", "HAI": "Americas",
    "GUA": "Americas", "HON": "Americas", "SAL": "Americas", "NIC": "Americas",
    "COS": "Americas", "PAN": "Americas", "JAM": "Americas",
    # Asia
    "CHN": "Asia", "JPN": "Asia", "ROK": "Asia", "PRK": "Asia",
    "TAW": "Asia", "IND": "Asia", "PAK": "Asia", "BNG": "Asia",
    "SRI": "Asia", "THI": "Asia", "VIE": "Asia", "DRV": "Asia",
    "RVN": "Asia", "CAM": "Asia", "LAO": "Asia", "MYA": "Asia",
    "MAL": "Asia", "INS": "Asia", "PHI": "Asia", "AFG": "Asia",
    "IRN": "Asia", "IRQ": "Asia", "SYR": "Asia", "LEB": "Asia",
    "JOR": "Asia", "ISR": "Asia", "SAU": "Asia", "KUW": "Asia",
    "YAR": "Asia", "YPR": "Asia", "OMA": "Asia", "UAE": "Asia",
    "BAH": "Asia", "QAT": "Asia", "MNG": "Asia", "NEP": "Asia",
    # Africa
    "EGY": "Africa", "LBY": "Africa", "TUN": "Africa", "ALG": "Africa",
    "MOR": "Africa", "SUD": "Africa", "ETH": "Africa", "ERI": "Africa",
    "DJI": "Africa", "SOM": "Africa", "KEN": "Africa", "UGA": "Africa",
    "TAZ": "Africa", "RWA": "Africa", "BUI": "Africa", "CON": "Africa",
    "DRC": "Africa", "ANG": "Africa", "ZAM": "Africa", "ZIM": "Africa",
    "MOZ": "Africa", "MAG": "Africa", "SAF": "Africa", "NAM": "Africa",
    "LES": "Africa", "SWA": "Africa", "BOT": "Africa", "MLI": "Africa",
    "SEN": "Africa", "GAM": "Africa", "GNB": "Africa", "GUI": "Africa",
    "SIE": "Africa", "LBR": "Africa", "CDI": "Africa", "GHA": "Africa",
    "TOG": "Africa", "BEN": "Africa", "NIR": "Africa", "NIG": "Africa",
    "CAO": "Africa", "CEN": "Africa", "CHA": "Africa", "GAB": "Africa",
    "EQG": "Africa", "MAW": "Africa",
    # Oceania
    "NEW": "Oceania", "AUL": "Oceania",
}

sites_all_reg = sites_all.copy()
sites_all_reg["continent"] = sites_all_reg["iso"].map(continent_map)

# Unmapped codes
unmapped = sites_all_reg[sites_all_reg["continent"].isna()]["iso"].unique()
if len(unmapped) > 0:
    print(f"  Unmapped ISO codes (assigned to 'Other'): {sorted(unmapped)}")
    sites_all_reg["continent"] = sites_all_reg["continent"].fillna("Other")

regional_stats = []
for cont in sorted(sites_all_reg["continent"].unique()):
    cont_df = sites_all_reg[sites_all_reg["continent"] == cont]
    stats = describe_sample(cont_df, cont)
    regional_stats.append(stats)

print_comparison_table(regional_stats, "Regional breakdown:")

# Share of total casualties by region
print(f"\n  Share of total casualties by region:")
total_cas = sites_all["casualties"].sum()
for cont in sorted(sites_all_reg["continent"].unique()):
    cont_cas = sites_all_reg.loc[sites_all_reg["continent"] == cont, "casualties"].sum()
    pct = cont_cas / total_cas * 100
    bar = "#" * int(pct / 2)
    print(f"    {cont:<15s}: {pct:>6.1f}%  {bar}")


# ===================================================================
# CHECK 6: WAR DURATION SENSITIVITY
# ===================================================================
section("CHECK 6: WAR DURATION SENSITIVITY")

sites_all_dur = sites_all.copy()
sites_all_dur["duration"] = sites_all_dur["end"] - sites_all_dur["start"] + 1

# Short vs long wars (threshold = 2 years)
short_wars = sites_all_dur[sites_all_dur["duration"] < 2]
long_wars = sites_all_dur[sites_all_dur["duration"] >= 2]

stats_short = describe_sample(short_wars, "Short (<2 yr)")
stats_long = describe_sample(long_wars, "Long (>=2 yr)")

print_comparison_table(
    [baseline_all, stats_short, stats_long],
    "Short vs long war comparison:"
)

# More granular duration breakdown
subsection("Duration bins")
dur_bins = [(1, 1, "1 year"), (2, 2, "2 years"), (3, 4, "3-4 years"),
            (5, 9, "5-9 years"), (10, 99, "10+ years")]

dur_stats = []
for lo, hi, label in dur_bins:
    df_bin = sites_all_dur[(sites_all_dur["duration"] >= lo) & (sites_all_dur["duration"] <= hi)]
    stats_bin = describe_sample(df_bin, label)
    dur_stats.append(stats_bin)

print_comparison_table(dur_stats, "Duration bin breakdown:")

# Do long wars have proportionally more casualties relative to population?
if "shock_caspop_home" in sites_all.columns:
    print(f"\n  Casualties relative to population by duration:")
    for lo, hi, label in dur_bins:
        df_bin = sites_all_dur[(sites_all_dur["duration"] >= lo) & (sites_all_dur["duration"] <= hi)]
        caspop = df_bin["shock_caspop_home"].dropna()
        if len(caspop) > 0:
            print(f"    {label:<12s}: mean cas/pop = {caspop.mean()*100:.3f}%, "
                  f"median = {caspop.median()*100:.3f}%, n={len(caspop)}")


# ===================================================================
# CHECK 7: TEMPORAL STABILITY
# ===================================================================
section("CHECK 7: TEMPORAL STABILITY")

subsection("Pre-1945 vs post-1945 casualty comparison")

pre_1945 = sites_all[sites_all["start"] < 1945]
post_1945 = sites_all[sites_all["start"] >= 1945]

stats_pre = describe_sample(pre_1945, "Pre-1945")
stats_post = describe_sample(post_1945, "Post-1945")

print_comparison_table(
    [baseline_all, stats_pre, stats_post],
    "Temporal split:"
)

# More granular: by era
subsection("By historical era")
eras = [
    (1800, 1913, "Pre-WWI (1800-1913)"),
    (1914, 1918, "WWI era (1914-1918)"),
    (1919, 1938, "Interwar (1919-1938)"),
    (1939, 1945, "WWII era (1939-1945)"),
    (1946, 1990, "Cold War (1946-1990)"),
    (1991, 2030, "Post-Cold War (1991+)"),
]

era_stats = []
for yr_lo, yr_hi, label in eras:
    df_era = sites_all[(sites_all["start"] >= yr_lo) & (sites_all["start"] <= yr_hi)]
    if len(df_era) > 0:
        stats_era = describe_sample(df_era, label)
        era_stats.append(stats_era)
    else:
        era_stats.append({
            "label": label, "n_sites": 0, "n_wars": 0, "n_countries": 0,
            "cas_mean": 0, "cas_median": 0, "cas_total": 0,
            "caspop_mean_pct": 0, "caspop_median_pct": 0, "dur_mean": 0,
        })

print_comparison_table(era_stats, "Era breakdown:")

# Trend in mean casualties per decade
subsection("Decadal trend in mean casualties")
sites_all_dec = sites_all.copy()
sites_all_dec["decade"] = (sites_all_dec["start"] // 10 * 10).astype(int)

decade_means = sites_all_dec.groupby("decade")["casualties"].agg(["mean", "median", "count"])
print(f"\n  {'Decade':<10s}  {'N sites':>8s}  {'Mean cas':>12s}  {'Median cas':>12s}")
print(f"  {'-'*10}  {'-'*8}  {'-'*12}  {'-'*12}")
for decade, row in decade_means.iterrows():
    print(f"  {decade}s     {row['count']:>8.0f}  {row['mean']:>12,.0f}  {row['median']:>12,.0f}")


# ===================================================================
# CHECK 8: WINSORIZATION IMPACT
# ===================================================================
section("CHECK 8: WINSORIZATION IMPACT")

if "shock_caspop_home" in sites_all.columns:
    caspop = sites_all["shock_caspop_home"].dropna()

    # Original distribution
    subsection("Original casualty/population distribution")
    print(f"  N = {len(caspop):,}")
    print(f"  Mean:   {caspop.mean():.6f} ({caspop.mean()*100:.3f}%)")
    print(f"  Median: {caspop.median():.6f} ({caspop.median()*100:.3f}%)")
    print(f"  Std:    {caspop.std():.6f}")
    print(f"  Min:    {caspop.min():.6f}")
    print(f"  Max:    {caspop.max():.6f}")
    print(f"  Skew:   {caspop.skew():.3f}")
    print(f"  Kurt:   {caspop.kurtosis():.3f}")

    # Winsorization at different levels
    subsection("Winsorization comparison")
    winsor_levels = [
        (1, 99, "1st/99th"),
        (2, 98, "2nd/98th"),
        (5, 95, "5th/95th"),
        (10, 90, "10th/90th"),
    ]

    print(f"\n  {'Level':<12s}  {'Lower':>10s}  {'Upper':>10s}  {'Mean':>10s}  {'Std':>10s}  "
          f"{'Skew':>8s}  {'Kurt':>8s}  {'N clipped':>10s}")
    print(f"  {'-'*12}  {'-'*10}  {'-'*10}  {'-'*10}  {'-'*10}  {'-'*8}  {'-'*8}  {'-'*10}")

    # Original (no winsorization)
    print(f"  {'None':<12s}  {caspop.min():>10.6f}  {caspop.max():>10.6f}  "
          f"{caspop.mean():>10.6f}  {caspop.std():>10.6f}  "
          f"{caspop.skew():>8.3f}  {caspop.kurtosis():>8.3f}  {'0':>10s}")

    for lo_pct, hi_pct, label in winsor_levels:
        lo_val = caspop.quantile(lo_pct / 100)
        hi_val = caspop.quantile(hi_pct / 100)
        winsorized = caspop.clip(lower=lo_val, upper=hi_val)
        n_clipped = ((caspop < lo_val) | (caspop > hi_val)).sum()

        print(f"  {label:<12s}  {lo_val:>10.6f}  {hi_val:>10.6f}  "
              f"{winsorized.mean():>10.6f}  {winsorized.std():>10.6f}  "
              f"{winsorized.skew():>8.3f}  {winsorized.kurtosis():>8.3f}  "
              f"{n_clipped:>10}")

    # Also examine raw casualties (not population-normalized)
    subsection("Raw casualty winsorization comparison")
    raw_cas = sites_all["casualties"].dropna()

    print(f"\n  {'Level':<12s}  {'Lower':>12s}  {'Upper':>12s}  {'Mean':>12s}  {'Std':>12s}  {'N clipped':>10s}")
    print(f"  {'-'*12}  {'-'*12}  {'-'*12}  {'-'*12}  {'-'*12}  {'-'*10}")

    print(f"  {'None':<12s}  {raw_cas.min():>12,.0f}  {raw_cas.max():>12,.0f}  "
          f"{raw_cas.mean():>12,.0f}  {raw_cas.std():>12,.0f}  {'0':>10s}")

    for lo_pct, hi_pct, label in winsor_levels:
        lo_val = raw_cas.quantile(lo_pct / 100)
        hi_val = raw_cas.quantile(hi_pct / 100)
        winsorized = raw_cas.clip(lower=lo_val, upper=hi_val)
        n_clipped = ((raw_cas < lo_val) | (raw_cas > hi_val)).sum()

        print(f"  {label:<12s}  {lo_val:>12,.0f}  {hi_val:>12,.0f}  "
              f"{winsorized.mean():>12,.0f}  {winsorized.std():>12,.0f}  "
              f"{n_clipped:>10}")
else:
    print("  shock_caspop_home not found in sites data; skipping winsorization checks.")


# ===================================================================
# SUMMARY TABLE
# ===================================================================
section("ROBUSTNESS SUMMARY TABLE")

# Collect all key robustness comparisons into one summary
summary_rows = [
    baseline_all,
    baseline_inter,
    stats_no_ww,
    stats_no_ww1,
    stats_no_ww2,
    stats_postww,
    stats_postww_inter,
    stats_short,
    stats_long,
    stats_pre,
    stats_post,
]

if stats_intra is not None:
    summary_rows.insert(2, stats_intra)

print_comparison_table(summary_rows, "Master robustness comparison:")

# Percentage changes relative to baseline
print(f"\n  Percentage change in mean casualties relative to baseline (all wars):")
baseline_cas = baseline_all["cas_mean"]
for s in summary_rows[1:]:
    if s["cas_mean"] > 0 and not np.isnan(s["cas_mean"]):
        pct_change = (s["cas_mean"] - baseline_cas) / baseline_cas * 100
        direction = "+" if pct_change >= 0 else ""
        print(f"    {s['label']:<30s}: {direction}{pct_change:.1f}%")

# Save summary to CSV
summary_df = pd.DataFrame(summary_rows)
summary_csv_path = os.path.join(REPL_OUTPUT, "robustness_summary.csv")
summary_df.to_csv(summary_csv_path, index=False)
print(f"\n  Summary table saved to: {summary_csv_path}")

print(f"\n  Output directory: {REPL_OUTPUT}")
print()
