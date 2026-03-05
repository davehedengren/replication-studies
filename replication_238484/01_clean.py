"""
01_clean.py -- Load and validate all processed datasets for
Federle et al. (2025) "The Price of War" (openICPSR 238484-V1)

For each dataset:
  - Print shape and column count
  - Summarize key variables
  - Validate expected sizes and ranges
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
    load_pop,
    load_pop_world,
    load_territory,
    OUTPUT_DIR,
)


def section(title: str) -> None:
    """Print a section header."""
    print("\n" + "=" * 72)
    print(f"  {title}")
    print("=" * 72)


def check(condition: bool, msg_pass: str, msg_fail: str) -> None:
    """Print a pass/fail validation message."""
    if condition:
        print(f"  [PASS] {msg_pass}")
    else:
        print(f"  [FAIL] {msg_fail}")


# ===================================================================
# 1. macro.dta
# ===================================================================
section("1. MACRO DATA (macro.dta)")

macro = load_macro()
print(f"  Shape: {macro.shape[0]:,} rows x {macro.shape[1]} columns")
print(f"  Columns ({macro.shape[1]}): {list(macro.columns[:20])} ...")

# Country and year coverage
n_countries = macro["iso"].nunique()
year_min = int(macro["year"].min())
year_max = int(macro["year"].max())
print(f"\n  Countries: {n_countries} unique ISO codes")
print(f"  Year range: {year_min} - {year_max}")

check(
    n_countries >= 55 and n_countries <= 70,
    f"Country count ({n_countries}) is in expected range [55, 70]",
    f"Country count ({n_countries}) outside expected range [55, 70]",
)
check(
    year_min <= 1870,
    f"Earliest year ({year_min}) is <= 1870",
    f"Earliest year ({year_min}) is > 1870 -- expected <= 1870",
)
check(
    year_max >= 2023,
    f"Latest year ({year_max}) is >= 2023",
    f"Latest year ({year_max}) is < 2023 -- expected >= 2023",
)

# Key macro variables
key_vars = ["lgdp", "lcpi", "unemp", "inflation", "pop", "gdp",
            "imports", "exports", "milex", "milper", "institutions",
            "ltrate", "strate", "cbrate", "openness", "cinc"]
present_vars = [v for v in key_vars if v in macro.columns]
missing_vars = [v for v in key_vars if v not in macro.columns]
print(f"\n  Key variables present ({len(present_vars)}/{len(key_vars)}): {present_vars}")
if missing_vars:
    print(f"  Key variables MISSING: {missing_vars}")

for v in present_vars[:6]:
    s = macro[v].dropna()
    print(f"    {v:20s}  obs={s.shape[0]:>6,}  mean={s.mean():>12.4f}  "
          f"min={s.min():>12.4f}  max={s.max():>12.4f}")

# Sample of ISO codes
print(f"\n  Sample ISO codes: {sorted(macro['iso'].unique())[:15]} ...")


# ===================================================================
# 2. sites_all.dta
# ===================================================================
section("2. WAR SITES -- ALL (sites_all.dta)")

sites_all = load_sites("all")
print(f"  Shape: {sites_all.shape[0]:,} rows x {sites_all.shape[1]} columns")
print(f"  Columns: {list(sites_all.columns)}")

n_wars = sites_all["warname"].nunique()
year_range_start = int(sites_all["start"].min())
year_range_end = int(sites_all["end"].max())
print(f"\n  Unique wars (warname): {n_wars}")
print(f"  Year range: start={year_range_start}, end={year_range_end}")

# War type breakdown
if "type" in sites_all.columns:
    print(f"\n  War type distribution:")
    for wtype, count in sites_all["type"].value_counts().items():
        print(f"    {wtype}: {count}")

# Casualty distribution
cas = sites_all["casualties"].dropna()
print(f"\n  Casualties (non-missing: {cas.shape[0]:,} of {sites_all.shape[0]:,}):")
print(f"    Mean:   {cas.mean():>14,.0f}")
print(f"    Median: {cas.median():>14,.0f}")
print(f"    Min:    {cas.min():>14,.0f}")
print(f"    Max:    {cas.max():>14,.0f}")
print(f"    Total:  {cas.sum():>14,.0f}")

# shock_caspop_home (casualties as share of local population)
if "shock_caspop_home" in sites_all.columns:
    caspop = sites_all["shock_caspop_home"].dropna()
    print(f"\n  Casualties/local pop (shock_caspop_home), non-missing: {caspop.shape[0]:,}")
    print(f"    Mean:   {caspop.mean():.6f}  ({caspop.mean()*100:.3f}%)")
    print(f"    Median: {caspop.median():.6f}  ({caspop.median()*100:.3f}%)")
    print(f"    Max:    {caspop.max():.6f}  ({caspop.max()*100:.3f}%)")

# War duration
length = sites_all["end"] - sites_all["start"] + 1
print(f"\n  War duration (years spanned):")
print(f"    Mean:   {length.mean():.1f}")
print(f"    Median: {length.median():.0f}")
print(f"    Min:    {length.min():.0f}")
print(f"    Max:    {length.max():.0f}")


# ===================================================================
# 3. sites_interstate.dta
# ===================================================================
section("3. WAR SITES -- INTERSTATE (sites_interstate.dta)")

sites_inter = load_sites("interstate")
print(f"  Shape: {sites_inter.shape[0]:,} rows x {sites_inter.shape[1]} columns")

n_inter_wars = sites_inter["warname"].nunique()
n_inter_sites = sites_inter.shape[0]
print(f"\n  Unique interstate wars: {n_inter_wars}")
print(f"  Total interstate war sites (country episodes): {n_inter_sites}")

check(
    70 <= n_inter_wars <= 100,
    f"Interstate war count ({n_inter_wars}) in expected range [70, 100] (COW, 1870+)",
    f"Interstate war count ({n_inter_wars}) outside expected range [70, 100]",
)

# Casualty statistics for interstate
cas_inter = sites_inter["casualties"].dropna()
print(f"\n  Casualties:")
print(f"    Mean:   {cas_inter.mean():>14,.0f}")
print(f"    Median: {cas_inter.median():>14,.0f}")
print(f"    Total:  {cas_inter.sum():>14,.0f}")

if "shock_caspop_home" in sites_inter.columns:
    caspop_inter = sites_inter["shock_caspop_home"].dropna()
    print(f"\n  Casualties/local pop:")
    print(f"    Mean:   {caspop_inter.mean()*100:.3f}%")
    print(f"    Median: {caspop_inter.median()*100:.3f}%")

# Duration
length_inter = sites_inter["end"] - sites_inter["start"] + 1
print(f"\n  Duration (years spanned):")
print(f"    Mean:   {length_inter.mean():.1f}")
print(f"    Median: {length_inter.median():.0f}")

inter_year_range = f"{int(sites_inter['start'].min())}-{int(sites_inter['end'].max())}"
print(f"  Sample period: {inter_year_range}")


# ===================================================================
# 4. sites_intrastate.dta
# ===================================================================
section("4. WAR SITES -- INTRASTATE (sites_intrastate.dta)")

sites_intra = load_sites("intrastate")
print(f"  Shape: {sites_intra.shape[0]:,} rows x {sites_intra.shape[1]} columns")

n_intra_wars = sites_intra["warname"].nunique()
print(f"\n  Unique intrastate wars: {n_intra_wars}")
print(f"  Total intrastate war sites: {sites_intra.shape[0]}")

# War type breakdown (intrastate file may contain 'intra' and 'extra')
if "type" in sites_intra.columns:
    print(f"\n  War type distribution:")
    for wtype, count in sites_intra["type"].value_counts().items():
        print(f"    {wtype}: {count}")

cas_intra = sites_intra["casualties"].dropna()
print(f"\n  Casualties:")
print(f"    Mean:   {cas_intra.mean():>14,.0f}")
print(f"    Median: {cas_intra.median():>14,.0f}")
print(f"    Total:  {cas_intra.sum():>14,.0f}")

if "shock_caspop_home" in sites_intra.columns:
    caspop_intra = sites_intra["shock_caspop_home"].dropna()
    print(f"\n  Casualties/local pop:")
    print(f"    Mean:   {caspop_intra.mean()*100:.3f}%")
    print(f"    Median: {caspop_intra.median()*100:.3f}%")


# ===================================================================
# 5. all_belligerents.dta
# ===================================================================
section("5. BELLIGERENTS (all_belligerents.dta)")

bell = load_belligerents()
print(f"  Shape: {bell.shape[0]:,} rows x {bell.shape[1]} columns")
print(f"  Columns: {list(bell.columns)}")

n_bell_wars = bell["warname"].nunique()
n_bell_countries = bell["iso"].nunique()
print(f"\n  Unique wars in belligerents data: {n_bell_wars}")
print(f"  Unique country ISOs: {n_bell_countries}")

# Validate linkage: all warnames in belligerents should also appear in sites_all
bell_wars_set = set(bell["warname"].unique())
sites_wars_set = set(sites_all["warname"].unique())
overlap = bell_wars_set & sites_wars_set
only_bell = bell_wars_set - sites_wars_set
only_sites = sites_wars_set - bell_wars_set

print(f"\n  Linkage to sites_all:")
print(f"    Wars in both: {len(overlap)}")
print(f"    Wars only in belligerents: {len(only_bell)}")
print(f"    Wars only in sites_all: {len(only_sites)}")
if only_bell:
    print(f"      Examples only in belligerents: {sorted(only_bell)[:5]}")
if only_sites:
    print(f"      Examples only in sites: {sorted(only_sites)[:5]}")

# Outcome and initiator distributions
if "outcome" in bell.columns:
    print(f"\n  Outcome distribution (1=winner, 2=loser, 3/4/6=other):")
    for val, count in bell["outcome"].value_counts().sort_index().items():
        print(f"    {val}: {count}")

if "initiator" in bell.columns:
    print(f"\n  Initiator distribution (1=attacker, 2=defender):")
    for val, count in bell["initiator"].value_counts().sort_index().items():
        print(f"    {val}: {count}")


# ===================================================================
# 6. pop.dta
# ===================================================================
section("6. POPULATION DATA (pop.dta)")

pop = load_pop()
print(f"  Shape: {pop.shape[0]:,} rows x {pop.shape[1]} columns")
print(f"  Columns: {list(pop.columns)}")

pop_year_min = int(pop["year"].min())
pop_year_max = int(pop["year"].max())
n_pop_countries = pop["iso"].nunique()
print(f"\n  Year range: {pop_year_min} - {pop_year_max}")
print(f"  Countries: {n_pop_countries}")

check(
    pop_year_min <= 1800,
    f"Earliest year ({pop_year_min}) is <= 1800",
    f"Earliest year ({pop_year_min}) is > 1800 -- expected <= 1800",
)
check(
    pop_year_max >= 2023,
    f"Latest year ({pop_year_max}) is >= 2023",
    f"Latest year ({pop_year_max}) is < 2023 -- expected >= 2023",
)

pop_vals = pop["pop"].dropna()
print(f"\n  Population values (non-missing: {pop_vals.shape[0]:,}):")
print(f"    Mean:   {pop_vals.mean():>14,.0f}")
print(f"    Median: {pop_vals.median():>14,.0f}")
print(f"    Min:    {pop_vals.min():>14,.0f}")
print(f"    Max:    {pop_vals.max():>14,.0f}")


# ===================================================================
# 7. pop_world.dta
# ===================================================================
section("7. WORLD POPULATION (pop_world.dta)")

pop_world = load_pop_world()
print(f"  Shape: {pop_world.shape[0]:,} rows x {pop_world.shape[1]} columns")
print(f"  Columns: {list(pop_world.columns)}")

pw_year_min = int(pop_world["year"].min())
pw_year_max = int(pop_world["year"].max())
print(f"\n  Year range: {pw_year_min} - {pw_year_max}")
print(f"  Latest world population: {pop_world.loc[pop_world['year'].idxmax(), 'pop_world']:,.0f}")


# ===================================================================
# 8. territory.dta
# ===================================================================
section("8. TERRITORIAL CHANGES (territory.dta)")

territory = load_territory()
print(f"  Shape: {territory.shape[0]:,} rows x {territory.shape[1]} columns")
print(f"  Columns: {list(territory.columns)}")

n_terr_changes = territory.shape[0]
n_terr_countries = territory["iso"].nunique()
print(f"\n  Total territorial change events: {n_terr_changes}")
print(f"  Countries with territorial changes: {n_terr_countries}")

terr_year_min = int(territory["year"].min())
terr_year_max = int(territory["year"].max())
print(f"  Year range: {terr_year_min} - {terr_year_max}")

print(f"\n  Top countries by number of territorial changes:")
for iso, count in territory["iso"].value_counts().head(10).items():
    print(f"    {iso}: {count}")


# ===================================================================
# SUMMARY
# ===================================================================
section("VALIDATION SUMMARY")

print(f"  macro.dta:            {macro.shape[0]:>8,} rows, {macro.shape[1]:>3} cols, "
      f"{n_countries} countries, {year_min}-{year_max}")
print(f"  sites_all.dta:        {sites_all.shape[0]:>8,} rows, {sites_all.shape[1]:>3} cols, "
      f"{n_wars} wars")
print(f"  sites_interstate.dta: {sites_inter.shape[0]:>8,} rows, {sites_inter.shape[1]:>3} cols, "
      f"{n_inter_wars} wars")
print(f"  sites_intrastate.dta: {sites_intra.shape[0]:>8,} rows, {sites_intra.shape[1]:>3} cols, "
      f"{n_intra_wars} wars")
print(f"  all_belligerents.dta: {bell.shape[0]:>8,} rows, {bell.shape[1]:>3} cols, "
      f"{n_bell_wars} wars, {n_bell_countries} countries")
print(f"  pop.dta:              {pop.shape[0]:>8,} rows, {pop.shape[1]:>3} cols, "
      f"{n_pop_countries} countries, {pop_year_min}-{pop_year_max}")
print(f"  pop_world.dta:        {pop_world.shape[0]:>8,} rows, {pop_world.shape[1]:>3} cols, "
      f"{pw_year_min}-{pw_year_max}")
print(f"  territory.dta:        {territory.shape[0]:>8,} rows, {territory.shape[1]:>3} cols, "
      f"{n_terr_changes} changes, {n_terr_countries} countries")

print(f"\n  OUTPUT_DIR: {OUTPUT_DIR}")
print(f"  OUTPUT_DIR exists: {OUTPUT_DIR.exists()}")
print()
