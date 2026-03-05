"""
04_data_audit.py -- Audit processed datasets for
Federle et al. (2025) "The Price of War" (openICPSR 238484-V1)

Performs six categories of audits:
  1. Macro panel: year coverage, gaps, missing patterns
  2. War sites: casualty distribution, outliers, war type composition
  3. Belligerents: multi-war countries, attacker/defender balance, outcomes
  4. Cross-dataset consistency: country overlap across datasets
  5. Population data: coverage gaps, consistency with macro
  6. Temporal patterns: wars per decade, geographic distribution
"""

import sys
from pathlib import Path
from collections import Counter

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
    load_pop,
    load_pop_world,
    load_territory,
)


def section(title: str) -> None:
    """Print a section header."""
    print("\n" + "=" * 72)
    print(f"  {title}")
    print("=" * 72)


def subsection(title: str) -> None:
    """Print a subsection header."""
    print(f"\n  --- {title} ---")


# ===================================================================
# Load all datasets
# ===================================================================
print("Loading datasets for audit...")

macro = load_macro()
sites = load_sites("all")
sites_inter = load_sites("interstate")
bell = load_belligerents()
pop = load_pop()
pop_world = load_pop_world()
territory = load_territory()

print(f"  macro:       {macro.shape[0]:>8,} rows x {macro.shape[1]:>3} cols")
print(f"  sites_all:   {sites.shape[0]:>8,} rows x {sites.shape[1]:>3} cols")
print(f"  sites_inter: {sites_inter.shape[0]:>8,} rows x {sites_inter.shape[1]:>3} cols")
print(f"  belligerents:{bell.shape[0]:>8,} rows x {bell.shape[1]:>3} cols")
print(f"  pop:         {pop.shape[0]:>8,} rows x {pop.shape[1]:>3} cols")
print(f"  pop_world:   {pop_world.shape[0]:>8,} rows x {pop_world.shape[1]:>3} cols")
print(f"  territory:   {territory.shape[0]:>8,} rows x {territory.shape[1]:>3} cols")


# ===================================================================
# AUDIT 1: MACRO PANEL
# ===================================================================
section("AUDIT 1: MACRO PANEL")

# 1a. Year coverage by country
subsection("1a. Year coverage by country")
macro_coverage = (
    macro.groupby("iso")["year"]
    .agg(["min", "max", "count"])
    .rename(columns={"min": "first_year", "max": "last_year", "count": "n_obs"})
)
macro_coverage["span"] = macro_coverage["last_year"] - macro_coverage["first_year"] + 1
macro_coverage["gap_years"] = macro_coverage["span"] - macro_coverage["n_obs"]

n_countries = len(macro_coverage)
n_balanced = (macro_coverage["gap_years"] == 0).sum()
n_with_gaps = (macro_coverage["gap_years"] > 0).sum()

print(f"  Total countries: {n_countries}")
print(f"  Fully balanced (no gaps): {n_balanced}")
print(f"  With year gaps: {n_with_gaps}")

if n_with_gaps > 0:
    gap_countries = macro_coverage[macro_coverage["gap_years"] > 0].sort_values(
        "gap_years", ascending=False
    )
    print(f"\n  Countries with most year gaps:")
    for iso, row in gap_countries.head(10).iterrows():
        print(
            f"    {iso}: {row['first_year']:.0f}-{row['last_year']:.0f}, "
            f"obs={row['n_obs']:.0f}, gap_years={row['gap_years']:.0f}"
        )

# Overall year range
print(f"\n  Overall year range: {int(macro['year'].min())}-{int(macro['year'].max())}")

# Check for duplicate country-year entries
dupes = macro.duplicated(subset=["iso", "year"], keep=False)
n_dupes = dupes.sum()
print(f"  Duplicate (iso, year) entries: {n_dupes}")
if n_dupes > 0:
    print(f"    Example duplicates:")
    for _, row in macro[dupes].head(5).iterrows():
        print(f"      iso={row['iso']}, year={row['year']}")

# 1b. Missing patterns for key variables
subsection("1b. Missing patterns for key variables")

key_vars = ["lgdp", "lcpi", "unemp", "inflation", "lcons", "ltfp",
            "imports", "exports", "milex", "milper",
            "ltrate", "strate", "cbrate",
            "institutions", "openness", "cinc"]
present_vars = [v for v in key_vars if v in macro.columns]

print(f"\n  {'Variable':<20s}  {'Total':>7s}  {'Non-NaN':>7s}  {'NaN':>7s}  {'% Miss':>7s}")
print(f"  {'-'*20}  {'-'*7}  {'-'*7}  {'-'*7}  {'-'*7}")

for v in present_vars:
    total = len(macro)
    non_nan = macro[v].notna().sum()
    n_nan = total - non_nan
    pct_miss = n_nan / total * 100
    print(f"  {v:<20s}  {total:>7,}  {non_nan:>7,}  {n_nan:>7,}  {pct_miss:>6.1f}%")

# 1c. Country-year completeness: which countries have lgdp, lcpi, unemp?
subsection("1c. Country-year completeness for core variables")

core_vars = ["lgdp", "lcpi", "unemp"]
core_present = [v for v in core_vars if v in macro.columns]

if core_present:
    macro_core = macro[["iso", "year"] + core_present].copy()
    macro_core["all_core_present"] = macro_core[core_present].notna().all(axis=1)

    core_by_country = macro_core.groupby("iso").agg(
        total_obs=("year", "count"),
        core_complete=("all_core_present", "sum"),
    )
    core_by_country["pct_complete"] = (
        core_by_country["core_complete"] / core_by_country["total_obs"] * 100
    )

    print(f"\n  Countries with ALL core vars ({core_present}) non-missing:")
    well_covered = (core_by_country["pct_complete"] >= 80).sum()
    partially_covered = (
        (core_by_country["pct_complete"] > 0)
        & (core_by_country["pct_complete"] < 80)
    ).sum()
    no_coverage = (core_by_country["pct_complete"] == 0).sum()
    print(f"    >= 80% complete: {well_covered} countries")
    print(f"    1-79% complete:  {partially_covered} countries")
    print(f"    0% complete:     {no_coverage} countries")

    # Bottom 10 by completeness (excluding 0%)
    bottom = core_by_country[core_by_country["core_complete"] > 0].sort_values(
        "pct_complete"
    )
    if len(bottom) > 0:
        print(f"\n  Least complete countries (with at least some data):")
        for iso, row in bottom.head(10).iterrows():
            print(
                f"    {iso}: {row['core_complete']:.0f}/{row['total_obs']:.0f} "
                f"complete ({row['pct_complete']:.1f}%)"
            )


# ===================================================================
# AUDIT 2: WAR SITES
# ===================================================================
section("AUDIT 2: WAR SITES")

# 2a. Casualty distribution and outliers
subsection("2a. Casualty distribution")

cas = sites["casualties"].dropna()
print(f"  Non-missing observations: {len(cas):,} of {len(sites):,}")
print(f"  Mean:     {cas.mean():>14,.0f}")
print(f"  Median:   {cas.median():>14,.0f}")
print(f"  Std dev:  {cas.std():>14,.0f}")
print(f"  Min:      {cas.min():>14,.0f}")
print(f"  Max:      {cas.max():>14,.0f}")

# Percentiles
print(f"\n  Percentile distribution:")
for pct in [1, 5, 10, 25, 50, 75, 90, 95, 99]:
    val = cas.quantile(pct / 100)
    print(f"    P{pct:>2d}: {val:>14,.0f}")

# Zero casualties
n_zero = (sites["casualties"] == 0).sum()
print(f"\n  Sites with zero casualties: {n_zero}")

# Extreme outliers (above 99th percentile)
p99 = cas.quantile(0.99)
extreme = sites[sites["casualties"] > p99][["warname", "iso", "casualties"]].sort_values(
    "casualties", ascending=False
)
print(f"\n  Sites above 99th percentile (>{p99:,.0f} casualties):")
for _, row in extreme.head(10).iterrows():
    print(f"    {row['warname']:<40s} {row['iso']}  {row['casualties']:>12,.0f}")

# 2b. War type composition
subsection("2b. War type composition")

if "type" in sites.columns:
    type_counts = sites["type"].value_counts()
    print(f"\n  War types:")
    for wtype, count in type_counts.items():
        pct = count / len(sites) * 100
        mean_cas = sites.loc[sites["type"] == wtype, "casualties"].mean()
        print(f"    {wtype:<15s}: {count:>4} sites ({pct:>5.1f}%), mean casualties={mean_cas:>12,.0f}")
else:
    print("  Column 'type' not found in sites data.")

# 2c. Temporal distribution of wars
subsection("2c. Temporal distribution of wars")

war_starts = sites.groupby("warname")["start"].min()
print(f"\n  Wars by decade (based on start year):")
decades = (war_starts // 10 * 10).astype(int)
decade_counts = decades.value_counts().sort_index()
for decade, count in decade_counts.items():
    bar = "#" * count
    print(f"    {decade}s: {count:>3} wars  {bar}")

# Number of countries affected per war
subsection("2d. Countries affected per war")

sites_per_war = sites.groupby("warname").agg(
    n_countries=("iso", "nunique"),
    total_casualties=("casualties", "sum"),
    start_year=("start", "min"),
).sort_values("n_countries", ascending=False)

print(f"\n  Wars affecting most countries:")
for warname, row in sites_per_war.head(10).iterrows():
    print(
        f"    {warname:<40s} {row['n_countries']:>3} countries, "
        f"{row['total_casualties']:>12,.0f} casualties ({row['start_year']:.0f})"
    )

# Single-country wars
n_single = (sites_per_war["n_countries"] == 1).sum()
print(f"\n  Wars affecting only 1 country: {n_single} of {len(sites_per_war)}")


# ===================================================================
# AUDIT 3: BELLIGERENTS
# ===================================================================
section("AUDIT 3: BELLIGERENTS")

# 3a. Multi-war countries
subsection("3a. Multi-war countries")

wars_per_country = bell.groupby("iso")["warname"].nunique().sort_values(ascending=False)
print(f"  Countries involved in most wars:")
for iso, n_wars in wars_per_country.head(15).items():
    print(f"    {iso}: {n_wars:>3} wars")

print(f"\n  Distribution of war involvement:")
for threshold in [1, 2, 5, 10, 20]:
    n_above = (wars_per_country >= threshold).sum()
    print(f"    >= {threshold:>2} wars: {n_above:>3} countries")

# 3b. Attacker/defender balance
subsection("3b. Attacker/defender balance")

if "initiator" in bell.columns:
    init_counts = bell["initiator"].value_counts().sort_index()
    total_with_init = bell["initiator"].notna().sum()
    print(f"  Initiator distribution (n={total_with_init:,}):")
    for val, count in init_counts.items():
        label = {1: "Attacker/Initiator", 2: "Defender/Target"}.get(val, f"Code {val}")
        pct = count / total_with_init * 100
        print(f"    {val} ({label:<22s}): {count:>5} ({pct:>5.1f}%)")

    # Countries that are predominantly attackers or defenders
    country_init = bell.groupby("iso")["initiator"].agg(["mean", "count"])
    mostly_attackers = country_init[
        (country_init["mean"] < 1.3) & (country_init["count"] >= 3)
    ].sort_values("mean")
    mostly_defenders = country_init[
        (country_init["mean"] > 1.7) & (country_init["count"] >= 3)
    ].sort_values("mean", ascending=False)

    if len(mostly_attackers) > 0:
        print(f"\n  Countries predominantly as attacker (mean init < 1.3, >=3 wars):")
        for iso, row in mostly_attackers.head(5).iterrows():
            print(f"    {iso}: mean={row['mean']:.2f} across {row['count']:.0f} wars")
    if len(mostly_defenders) > 0:
        print(f"\n  Countries predominantly as defender (mean init > 1.7, >=3 wars):")
        for iso, row in mostly_defenders.head(5).iterrows():
            print(f"    {iso}: mean={row['mean']:.2f} across {row['count']:.0f} wars")

# 3c. Outcome distribution
subsection("3c. Outcome distribution")

if "outcome" in bell.columns:
    outcome_counts = bell["outcome"].value_counts().sort_index()
    total_with_outcome = bell["outcome"].notna().sum()
    outcome_labels = {
        1: "Winner",
        2: "Loser",
        3: "Compromise/Stalemate",
        4: "Transformed into another war",
        5: "Ongoing",
        6: "Stalemate/Other",
        7: "Other",
    }
    print(f"  Outcome distribution (n={total_with_outcome:,}):")
    for val, count in outcome_counts.items():
        label = outcome_labels.get(int(val), f"Code {int(val)}")
        pct = count / total_with_outcome * 100
        print(f"    {int(val)} ({label:<35s}): {count:>5} ({pct:>5.1f}%)")

    # Win rate by initiator status
    if "initiator" in bell.columns:
        bell_valid = bell.dropna(subset=["outcome", "initiator"])
        bell_valid = bell_valid.copy()
        bell_valid["won"] = (bell_valid["outcome"] == 1).astype(int)
        win_by_init = bell_valid.groupby("initiator")["won"].agg(["mean", "count"])
        print(f"\n  Win rate by initiator status:")
        for init_val, row in win_by_init.iterrows():
            label = {1: "Attacker", 2: "Defender"}.get(init_val, f"Code {init_val}")
            print(f"    {label:<12s}: {row['mean']*100:.1f}% win rate (n={row['count']:.0f})")


# ===================================================================
# AUDIT 4: CROSS-DATASET CONSISTENCY
# ===================================================================
section("AUDIT 4: CROSS-DATASET CONSISTENCY")

# 4a. Do all site countries appear in macro?
subsection("4a. Site countries in macro panel")

site_isos = set(sites["iso"].unique())
macro_isos = set(macro["iso"].unique())

sites_in_macro = site_isos & macro_isos
sites_not_in_macro = site_isos - macro_isos

print(f"  Site countries: {len(site_isos)}")
print(f"  Macro countries: {len(macro_isos)}")
print(f"  Site countries ALSO in macro: {len(sites_in_macro)}")
print(f"  Site countries NOT in macro: {len(sites_not_in_macro)}")
if sites_not_in_macro:
    print(f"    Missing: {sorted(sites_not_in_macro)}")

# 4b. Do all belligerent countries appear in macro?
subsection("4b. Belligerent countries in macro panel")

bell_isos = set(bell["iso"].unique())
bell_in_macro = bell_isos & macro_isos
bell_not_in_macro = bell_isos - macro_isos

print(f"  Belligerent countries: {len(bell_isos)}")
print(f"  Belligerent countries ALSO in macro: {len(bell_in_macro)}")
print(f"  Belligerent countries NOT in macro: {len(bell_not_in_macro)}")
if bell_not_in_macro:
    print(f"    Missing: {sorted(bell_not_in_macro)}")

# 4c. War name consistency between sites and belligerents
subsection("4c. War name consistency")

site_wars = set(sites["warname"].unique())
bell_wars = set(bell["warname"].unique())

wars_in_both = site_wars & bell_wars
wars_only_sites = site_wars - bell_wars
wars_only_bell = bell_wars - site_wars

print(f"  Wars in sites:        {len(site_wars)}")
print(f"  Wars in belligerents: {len(bell_wars)}")
print(f"  Wars in BOTH:         {len(wars_in_both)}")
print(f"  Wars ONLY in sites:   {len(wars_only_sites)}")
if wars_only_sites:
    examples = sorted(wars_only_sites)[:5]
    print(f"    Examples: {examples}")
print(f"  Wars ONLY in bell:    {len(wars_only_bell)}")
if wars_only_bell:
    examples = sorted(wars_only_bell)[:5]
    print(f"    Examples: {examples}")

# 4d. For wars in both, check that site countries are also belligerents
subsection("4d. Site countries that are belligerents in the same war")

site_pairs = set(zip(sites["warname"], sites["iso"]))
bell_pairs = set(zip(bell["warname"], bell["iso"]))

site_also_bell = site_pairs & bell_pairs
site_not_bell = site_pairs - bell_pairs

print(f"  (war, country) pairs in sites: {len(site_pairs)}")
print(f"  (war, country) pairs in bell:  {len(bell_pairs)}")
print(f"  Site countries also belligerent: {len(site_also_bell)}")
print(f"  Site countries NOT belligerent:  {len(site_not_bell)}")
if site_not_bell:
    examples = sorted(site_not_bell)[:5]
    print(f"    Examples (war, iso): {examples}")


# ===================================================================
# AUDIT 5: POPULATION DATA
# ===================================================================
section("AUDIT 5: POPULATION DATA")

# 5a. Coverage gaps
subsection("5a. Population coverage")

pop_coverage = pop.groupby("iso")["year"].agg(["min", "max", "count"])
pop_coverage["span"] = pop_coverage["max"] - pop_coverage["min"] + 1
pop_coverage["gap_years"] = pop_coverage["span"] - pop_coverage["count"]

n_pop_countries = len(pop_coverage)
n_pop_balanced = (pop_coverage["gap_years"] == 0).sum()
n_pop_gaps = (pop_coverage["gap_years"] > 0).sum()

print(f"  Countries in pop data: {n_pop_countries}")
print(f"  Fully balanced: {n_pop_balanced}")
print(f"  With gaps: {n_pop_gaps}")

if n_pop_gaps > 0:
    gap_pop = pop_coverage[pop_coverage["gap_years"] > 0].sort_values(
        "gap_years", ascending=False
    )
    print(f"\n  Countries with most pop gaps:")
    for iso, row in gap_pop.head(5).iterrows():
        print(
            f"    {iso}: {row['min']:.0f}-{row['max']:.0f}, "
            f"obs={row['count']:.0f}, gaps={row['gap_years']:.0f}"
        )

# 5b. Consistency with macro panel -- do pop countries cover macro countries?
subsection("5b. Population vs macro country overlap")

pop_isos = set(pop["iso"].unique())

macro_in_pop = macro_isos & pop_isos
macro_not_in_pop = macro_isos - pop_isos

print(f"  Macro countries in pop:     {len(macro_in_pop)} of {len(macro_isos)}")
print(f"  Macro countries NOT in pop: {len(macro_not_in_pop)}")
if macro_not_in_pop:
    print(f"    Missing: {sorted(macro_not_in_pop)}")

# 5c. Pop world vs sum of country pops
subsection("5c. World population consistency check")

# Compare pop_world with sum of country populations for overlapping years
pop_totals = pop.groupby("year")["pop"].sum().reset_index()
pop_totals.columns = ["year", "pop_sum_countries"]

pw_compare = pop_world.merge(pop_totals, on="year", how="inner")
if len(pw_compare) > 0:
    pw_compare["ratio"] = pw_compare["pop_sum_countries"] / pw_compare["pop_world"]
    print(f"  Comparing sum-of-country-pop to pop_world (overlapping years: {len(pw_compare)}):")
    print(f"    Ratio mean:   {pw_compare['ratio'].mean():.4f}")
    print(f"    Ratio min:    {pw_compare['ratio'].min():.4f}")
    print(f"    Ratio max:    {pw_compare['ratio'].max():.4f}")
    print(f"    (Ratio ~1.0 means good agreement; < 1 means pop_world includes non-macro countries)")
else:
    print("  No overlapping years between pop and pop_world.")


# ===================================================================
# AUDIT 6: TEMPORAL PATTERNS
# ===================================================================
section("AUDIT 6: TEMPORAL PATTERNS")

# 6a. Wars per decade
subsection("6a. Wars per decade (unique wars by start year)")

# Use unique wars (not sites) for this count
unique_wars = sites.drop_duplicates(subset="warname")
unique_wars_decade = (unique_wars["start"] // 10 * 10).astype(int)
decade_war_counts = unique_wars_decade.value_counts().sort_index()

print(f"\n  {'Decade':<10s}  {'Wars':>5s}  {'Bar'}")
print(f"  {'-'*10}  {'-'*5}  {'-'*40}")
for decade, count in decade_war_counts.items():
    bar = "#" * count
    print(f"  {decade}s     {count:>5}  {bar}")

total_wars = unique_wars.shape[0]
print(f"\n  Total unique wars: {total_wars}")

# 6b. Geographic distribution
subsection("6b. Geographic distribution of war sites")

# Manual continent mapping based on ISO-3166 codes
# This is approximate; some codes may not map perfectly
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

sites_with_continent = sites.copy()
sites_with_continent["continent"] = sites_with_continent["iso"].map(continent_map)

# Count sites and unique wars by continent
print(f"\n  {'Continent':<15s}  {'Sites':>6s}  {'Wars':>6s}  {'Mean Cas':>12s}  {'Countries':>10s}")
print(f"  {'-'*15}  {'-'*6}  {'-'*6}  {'-'*12}  {'-'*10}")

for cont in sorted(sites_with_continent["continent"].dropna().unique()):
    cont_df = sites_with_continent[sites_with_continent["continent"] == cont]
    n_sites = len(cont_df)
    n_wars = cont_df["warname"].nunique()
    mean_cas = cont_df["casualties"].mean()
    n_ctries = cont_df["iso"].nunique()
    print(f"  {cont:<15s}  {n_sites:>6}  {n_wars:>6}  {mean_cas:>12,.0f}  {n_ctries:>10}")

# Countries not mapped
unmapped = sites_with_continent[sites_with_continent["continent"].isna()]["iso"].unique()
if len(unmapped) > 0:
    print(f"\n  Unmapped ISO codes: {sorted(unmapped)}")

# 6c. Pre-1945 vs post-1945 comparison
subsection("6c. Pre-1945 vs post-1945 comparison")

pre_1945 = sites[sites["start"] < 1945]
post_1945 = sites[sites["start"] >= 1945]

print(f"\n  {'Metric':<35s}  {'Pre-1945':>12s}  {'Post-1945':>12s}")
print(f"  {'-'*35}  {'-'*12}  {'-'*12}")
print(f"  {'Total sites':<35s}  {len(pre_1945):>12,}  {len(post_1945):>12,}")
print(f"  {'Unique wars':<35s}  {pre_1945['warname'].nunique():>12,}  {post_1945['warname'].nunique():>12,}")
print(f"  {'Unique countries':<35s}  {pre_1945['iso'].nunique():>12,}  {post_1945['iso'].nunique():>12,}")
print(f"  {'Mean casualties':<35s}  {pre_1945['casualties'].mean():>12,.0f}  {post_1945['casualties'].mean():>12,.0f}")
print(f"  {'Median casualties':<35s}  {pre_1945['casualties'].median():>12,.0f}  {post_1945['casualties'].median():>12,.0f}")
print(f"  {'Total casualties':<35s}  {pre_1945['casualties'].sum():>12,.0f}  {post_1945['casualties'].sum():>12,.0f}")

if "shock_caspop_home" in sites.columns:
    pre_caspop = pre_1945["shock_caspop_home"].dropna()
    post_caspop = post_1945["shock_caspop_home"].dropna()
    if len(pre_caspop) > 0 and len(post_caspop) > 0:
        print(f"  {'Mean cas/pop (%)':<35s}  {pre_caspop.mean()*100:>11.3f}%  {post_caspop.mean()*100:>11.3f}%")
        print(f"  {'Median cas/pop (%)':<35s}  {pre_caspop.median()*100:>11.3f}%  {post_caspop.median()*100:>11.3f}%")

if "type" in sites.columns:
    for wtype in sites["type"].unique():
        n_pre = (pre_1945["type"] == wtype).sum()
        n_post = (post_1945["type"] == wtype).sum()
        print(f"  {'Type: ' + str(wtype):<35s}  {n_pre:>12,}  {n_post:>12,}")


# ===================================================================
# SUMMARY
# ===================================================================
section("DATA AUDIT SUMMARY")

print(f"""
  1. MACRO PANEL
     - {n_countries} countries, {int(macro['year'].min())}-{int(macro['year'].max())}
     - Balanced: {n_balanced}, With gaps: {n_with_gaps}
     - Duplicate entries: {n_dupes}

  2. WAR SITES
     - {len(sites):,} total sites across {sites['warname'].nunique()} wars
     - Casualty range: {cas.min():,.0f} to {cas.max():,.0f}
     - Zero casualties: {n_zero} sites

  3. BELLIGERENTS
     - {len(bell):,} entries across {bell['warname'].nunique()} wars
     - {bell['iso'].nunique()} unique countries involved

  4. CROSS-DATASET CONSISTENCY
     - Site countries not in macro: {len(sites_not_in_macro)} {sorted(sites_not_in_macro) if sites_not_in_macro else ''}
     - Bell countries not in macro: {len(bell_not_in_macro)} {sorted(bell_not_in_macro) if bell_not_in_macro else ''}
     - War name mismatches: {len(wars_only_sites)} in sites only, {len(wars_only_bell)} in bell only

  5. POPULATION DATA
     - {n_pop_countries} countries, balanced: {n_pop_balanced}, gaps: {n_pop_gaps}
     - Macro countries missing from pop: {len(macro_not_in_pop)}

  6. TEMPORAL PATTERNS
     - Pre-1945: {len(pre_1945):,} sites, Post-1945: {len(post_1945):,} sites
     - Pre-1945 mean casualties: {pre_1945['casualties'].mean():,.0f}
     - Post-1945 mean casualties: {post_1945['casualties'].mean():,.0f}
""")
