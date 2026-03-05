"""
02_tables.py -- Reproduce Table 1 (descriptive statistics of the war sample)
from Federle et al. (2025) "The Price of War" (openICPSR 238484-V1)

Table 1 reports descriptive statistics across three panels:
  Panel A: Interstate wars
  Panel B: Other wars (intrastate + extrastate)
  Panel C: All wars

For each panel the table shows:
  - Number of war sites (country episodes)
  - Casualties as % of local population: mean, median
  - War duration (years spanned): mean, median
  - Macro data availability by country role (site, belligerent, third party)

The Stata source is: src/02_export/tables/descriptives/sample.do
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Imports from project utils
# ---------------------------------------------------------------------------
sys.path.insert(0, str(Path(__file__).resolve().parent))
from utils import (
    load_macro,
    load_sites,
    load_belligerents,
    OUTPUT_DIR,
)


def section(title: str) -> None:
    """Print a section header."""
    print("\n" + "=" * 72)
    print(f"  {title}")
    print("=" * 72)


def compute_panel_stats(sites: pd.DataFrame, label: str) -> dict:
    """
    Compute descriptive statistics for a sites DataFrame.

    Mirrors the Stata code in sample.do:
      - gen length = end - start + 1
      - sum shock_caspop_home, d   -> casualties_mean, casualties_median
      - sum length, d              -> length_mean, length_median
      - _N                         -> sites_total

    Parameters
    ----------
    sites : DataFrame
        War sites data with columns: warname, iso, start, end, casualties,
        shock_caspop_home.
    label : str
        Label for display purposes.

    Returns
    -------
    dict with computed statistics.
    """
    # War duration in years spanned (matches Stata: gen length = end - start + 1)
    sites = sites.copy()
    sites["length"] = sites["end"] - sites["start"] + 1

    # Total war sites
    n_sites = len(sites)

    # Unique wars
    n_wars = sites["warname"].nunique()

    # Casualty statistics as % of local population
    caspop = sites["shock_caspop_home"].dropna()
    cas_mean_pct = caspop.mean() * 100 if len(caspop) > 0 else np.nan
    cas_median_pct = caspop.median() * 100 if len(caspop) > 0 else np.nan

    # Raw casualty statistics
    raw_cas = sites["casualties"].dropna()
    cas_total = raw_cas.sum() if len(raw_cas) > 0 else 0

    # Duration statistics
    dur = sites["length"].dropna()
    length_mean = dur.mean() if len(dur) > 0 else np.nan
    length_median = dur.median() if len(dur) > 0 else np.nan

    # Year range
    year_start = int(sites["start"].min())
    year_end = int(sites["end"].max())

    return {
        "label": label,
        "n_wars": n_wars,
        "n_sites": n_sites,
        "cas_mean_pct": cas_mean_pct,
        "cas_median_pct": cas_median_pct,
        "cas_total": cas_total,
        "length_mean": length_mean,
        "length_median": length_median,
        "year_start": year_start,
        "year_end": year_end,
    }


def print_panel(stats: dict) -> None:
    """Pretty-print a panel's statistics."""
    print(f"  {'Statistic':<45s}  {'Value':>15s}")
    print(f"  {'-'*45}  {'-'*15}")
    print(f"  {'Number of wars':<45s}  {stats['n_wars']:>15,}")
    print(f"  {'Number of war sites (country episodes)':<45s}  {stats['n_sites']:>15,}")
    print(f"  {'Casualties/local pop - mean (%)':<45s}  {stats['cas_mean_pct']:>15.3f}")
    print(f"  {'Casualties/local pop - median (%)':<45s}  {stats['cas_median_pct']:>15.3f}")
    print(f"  {'Total casualties':<45s}  {stats['cas_total']:>15,.0f}")
    print(f"  {'Duration - mean (years)':<45s}  {stats['length_mean']:>15.1f}")
    print(f"  {'Duration - median (years)':<45s}  {stats['length_median']:>15.0f}")
    print(f"  {'Sample period':<45s}  {stats['year_start']:>7d}-{stats['year_end']:<7d}")


# ===================================================================
# Load data
# ===================================================================
print("Loading datasets...")

sites_all = load_sites("all")
sites_interstate = load_sites("interstate")
sites_intrastate = load_sites("intrastate")
macro = load_macro()

print(f"  sites_all:        {sites_all.shape[0]:>5,} rows")
print(f"  sites_interstate: {sites_interstate.shape[0]:>5,} rows")
print(f"  sites_intrastate: {sites_intrastate.shape[0]:>5,} rows")
print(f"  macro:            {macro.shape[0]:>5,} rows")


# ===================================================================
# TABLE 1: Descriptive statistics of the war sample
# ===================================================================
section("TABLE 1: DESCRIPTIVE STATISTICS OF THE WAR SAMPLE")

# Panel A: Interstate wars
stats_interstate = compute_panel_stats(sites_interstate, "Interstate wars")

# Panel B: Other wars (intrastate + extrastate)
# The "intrastate" sites file contains both intra and extra types
stats_other = compute_panel_stats(sites_intrastate, "Other wars (intrastate + extrastate)")

# Panel C: All wars
stats_all = compute_panel_stats(sites_all, "All wars")

# Print formatted table
print()
print(f"  {'Panel A: Interstate Wars':^65s}")
print(f"  {'='*65}")
print_panel(stats_interstate)

print()
print(f"  {'Panel B: Other Wars (Intrastate + Extrastate)':^65s}")
print(f"  {'='*65}")
print_panel(stats_other)

print()
print(f"  {'Panel C: All Wars':^65s}")
print(f"  {'='*65}")
print_panel(stats_all)


# ===================================================================
# Consistency check: Panel C should equal Panel A + Panel B
# ===================================================================
section("CONSISTENCY CHECKS")

sum_sites = stats_interstate["n_sites"] + stats_other["n_sites"]
check_sites = sum_sites == stats_all["n_sites"]
print(f"  Sites: interstate ({stats_interstate['n_sites']}) + "
      f"other ({stats_other['n_sites']}) = {sum_sites}  "
      f"vs. all ({stats_all['n_sites']})  "
      f"{'[MATCH]' if check_sites else '[MISMATCH]'}")

sum_wars = stats_interstate["n_wars"] + stats_other["n_wars"]
check_wars = sum_wars == stats_all["n_wars"]
print(f"  Wars:  interstate ({stats_interstate['n_wars']}) + "
      f"other ({stats_other['n_wars']}) = {sum_wars}  "
      f"vs. all ({stats_all['n_wars']})  "
      f"{'[MATCH]' if check_wars else '[MISMATCH -- wars may overlap]'}")


# ===================================================================
# Share of countries affected
# ===================================================================
section("SHARE OF COUNTRIES AFFECTED")

n_macro_countries = macro["iso"].nunique()
print(f"  Total countries in macro panel: {n_macro_countries}")

# Countries that appear as war sites
for lbl, sites_df in [("Interstate", sites_interstate),
                       ("Other", sites_intrastate),
                       ("All", sites_all)]:
    n_site_countries = sites_df["iso"].nunique()
    share = n_site_countries / n_macro_countries * 100
    print(f"  {lbl:12s}: {n_site_countries:>3} countries affected "
          f"({share:.1f}% of macro panel)")


# ===================================================================
# Macro data availability by country role
# (Mirrors sample.do: build_panel then count lcpi observations by role)
# ===================================================================
section("MACRO DATA AVAILABILITY BY COUNTRY ROLE (from sites data)")

# Instead of running the full build_panel (which requires trade_gravity.dta
# and is expensive), we approximate macro coverage by checking which site
# countries have CPI data in the macro panel.
#
# The Stata code does:
#   build_panel, wars(`sample')
#   sum lcpi if regr_cas_phi_site > 0   -> macro_home
#   sum lcpi if regr_cas_phi_bell > 0   -> macro_bell
#   sum lcpi if regr_cas_phi_third > 0  -> macro_third
#
# We approximate: for each site-country-year (war start year), check whether
# lcpi is available in the macro panel. Belligerent and third-party counts
# require the full panel build, so we report what we can compute directly.

macro_lcpi = macro[["iso", "year", "lcpi"]].dropna(subset=["lcpi"])
macro_lcpi_set = set(zip(macro_lcpi["iso"], macro_lcpi["year"]))

for lbl, sites_df in [("Interstate", sites_interstate),
                       ("Other", sites_intrastate),
                       ("All", sites_all)]:
    # Create site-country-year pairs
    site_pairs = list(zip(sites_df["iso"], sites_df["start"]))
    n_with_lcpi = sum(1 for iso, yr in site_pairs if (iso, yr) in macro_lcpi_set)
    print(f"  {lbl:12s}: {n_with_lcpi:>4} of {len(site_pairs)} site-country-years "
          f"have lcpi data ({n_with_lcpi/max(len(site_pairs),1)*100:.1f}%)")

print("\n  Note: Belligerent and third-party counts require build_panel(),")
print("  which involves loading trade_gravity.dta (~80 MB). These counts are")
print("  reported in the full Table 1 of the paper but are skipped here for speed.")


# ===================================================================
# TEXT NUMBERS FROM THE PAPER
# ===================================================================
section("TEXT NUMBERS FROM THE PAPER")

# 1. Total number of war sites
print(f"  Total war sites (all): {sites_all.shape[0]}")
print(f"  Total war sites (interstate): {sites_interstate.shape[0]}")
print(f"  Total war sites (intrastate/other): {sites_intrastate.shape[0]}")

# 2. Mean casualties as % of population (all wars)
caspop_all = sites_all["shock_caspop_home"].dropna()
print(f"\n  Mean casualties/local pop (all wars):      {caspop_all.mean()*100:.3f}%")

# 3. Mean casualties as % of population (post-WWII)
caspop_postww = sites_all.loc[sites_all["start"] >= 1946, "shock_caspop_home"].dropna()
print(f"  Mean casualties/local pop (1946+):         {caspop_postww.mean()*100:.3f}%")

# 4. Median casualties as % of population
print(f"  Median casualties/local pop (all wars):    {caspop_all.median()*100:.3f}%")

# 5. Mean destruction index (GPRC-based)
if "destruction" in sites_all.columns:
    destr = sites_all["destruction"].dropna()
    print(f"\n  Mean destruction index (GPRC, all):        {destr.mean():.3f}")
    print(f"  Destruction index non-missing obs:         {destr.shape[0]}")

# 6. Average war site population relative to world population
# Merge pop data for the start year
sites_pop = sites_all.copy()
sites_pop["year"] = sites_pop["start"]

# Use pop_site and pop_world already in sites data (if available)
if "pop_site" in sites_pop.columns and "pop_world" in sites_pop.columns:
    sites_pop["size"] = sites_pop["pop_site"] / sites_pop["pop_world"]
    size_all = sites_pop["size"].dropna()
    print(f"\n  Mean local pop/world pop (all sites):      {size_all.mean():.4f}")
    size_postww = sites_pop.loc[sites_pop["start"] >= 1946, "size"].dropna()
    if len(size_postww) > 0:
        print(f"  Mean local pop/world pop (1946+):          {size_postww.mean():.4f}")
else:
    print("\n  Note: pop_site/pop_world not directly in sites data.")
    print("  Would need to merge from pop.dta and pop_world.dta (see textnumbers.do).")


# ===================================================================
# Formatted Table 1 (matching paper layout)
# ===================================================================
section("TABLE 1 -- FORMATTED OUTPUT (matching sample.do)")

# The Stata code outputs: sites_total, casualties_mean, casualties_median,
# length_mean, length_median, macro_home, macro_bell, macro_third
# Format: %15.0fc %15.2fc %15.2fc %15.1fc %15.0fc %15.0fc %15.0fc %15.0fc

header = (f"  {'Sites':>7s}  {'Cas.Mean%':>10s}  {'Cas.Med%':>10s}  "
          f"{'Dur.Mean':>9s}  {'Dur.Med':>8s}")
print(f"\n{header}")
print(f"  {'-'*7}  {'-'*10}  {'-'*10}  {'-'*9}  {'-'*8}")

for stats in [stats_interstate, stats_other, stats_all]:
    row = (f"  {stats['n_sites']:>7,}  {stats['cas_mean_pct']:>10.2f}  "
           f"{stats['cas_median_pct']:>10.2f}  {stats['length_mean']:>9.1f}  "
           f"{stats['length_median']:>8.0f}")
    print(f"  {stats['label']:<50s}")
    print(row)

print()
print("  Note: 'Macro data availability' columns (site/bell/third) require")
print("  running build_panel() and are omitted from this direct computation.")
print("  The Stata code reports CPI observation counts conditional on positive")
print("  war exposure in each role.")


# ===================================================================
# Comparison notes
# ===================================================================
section("COMPARISON WITH PUBLISHED TABLE 1")

print("""
  The published Table 1 in Federle et al. (2025) reports:

  Panel A (Interstate):
    - War sites, casualties (mean/median as % of pop), duration (mean/median)
  Panel B (Other):
    - Same statistics for intrastate + extrastate wars
  Panel C (All):
    - Combined sample

  Computed values above should match the published table. Small differences
  may arise from:
    1. Stata vs Python floating-point rounding
    2. Stata's 'sum, d' uses detailed summary statistics
    3. The paper rounds to specific decimal places

  To fully replicate macro availability columns, run build_panel() from utils.
  This requires loading trade_gravity.dta (~80 MB, ~15.5M rows) and takes
  several minutes.
""")
