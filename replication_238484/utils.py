"""
Replication utilities for Federle et al. (2025) "The Price of War"
(openICPSR 238484-V1)

This module provides:
  - Path constants pointing to the replication package data directories
  - Data loader functions for each processed .dta file
  - build_panel(): Python translation of the Stata panel-building program (panel.do)
  - run_local_projection(): local projection estimation (LP-IV style, as in lp.do)
  - compute_irf(): impulse response computation from LP coefficient estimates

The paper uses local projections to estimate the dynamic effects of wars on
macroeconomic outcomes. War exposure is decomposed into three channels based on
each country's relationship to the conflict:
  - War site (direct destruction)
  - Belligerent (participated but war not on own territory)
  - Third party (no direct involvement)

Spillover channels are weighted by population share or trade intensity.

Dependencies: pandas, numpy, statsmodels
"""

import os
import warnings
from pathlib import Path
from typing import Optional, Union

import numpy as np
import pandas as pd
import statsmodels.api as sm
from linearmodels.panel import PanelOLS

# =============================================================================
# PATH CONSTANTS
# =============================================================================
# These mirror the global directory macros in main.do:
#   DIR_DATA_RAW       = ${DIR_PROJ}/data/01_raw
#   DIR_DATA_PROCESSED = ${DIR_PROJ}/data/02_processed
#   DIR_DATA_EXPORTS   = ${DIR_PROJ}/data/03_exports
# =============================================================================

BASE_DIR = Path("/Users/davehedengren/code/replication_studies/238484-V1")
DATA_DIR = BASE_DIR / "data" / "02_processed"   # processed Stata datasets
RAW_DIR = BASE_DIR / "data" / "01_raw"           # raw source data
OUTPUT_DIR = BASE_DIR / "data" / "03_exports"    # figures, tables, CSV exports


# =============================================================================
# DATA LOADERS
# =============================================================================
# Each loader reads one processed .dta file and returns a pandas DataFrame.
# Column names and types are preserved from the Stata originals.
# =============================================================================

def load_macro() -> pd.DataFrame:
    """
    Load the macroeconomic panel (macro.dta).

    Country-year panel (~9,240 rows x 90 columns) covering 1870-2024 for
    ~60 countries.  Key variables include:
      - iso, year, cid (country numeric id)
      - lgdp, lcpi, lcons, lcs_ppp, ltfp (log levels)
      - lgdp_dtrd, lgdp_dthp, lgdp_dtpl (detrended variants)
      - imports, exports, gdp, pop, milex, milper
      - institutions, judicial, electoral, medial
      - ltrate, strate, cbrate, unemp, inflation
      - nationalism, openness, cinc
    """
    return pd.read_stata(DATA_DIR / "macro.dta", convert_categoricals=False)


def load_sites(wars: str = "all") -> pd.DataFrame:
    """
    Load war-site data (sites_{wars}.dta).

    Parameters
    ----------
    wars : str
        One of 'all', 'interstate', 'intrastate', 'causality'.
        Determines which sites file to load.

    Returns
    -------
    DataFrame with columns:
      - warname, iso, start, end, casualties
      - start_alt, end_alt (alternative dating)
      - type ('inter' or 'intra')
      - destruction (GPRC-based destruction index, NaN before ~1900)
      - pop_site, pop_world
      - shock_dummy_home, shock_caspop_home, shock_dest_home
    """
    fname = f"sites_{wars}.dta"
    fpath = DATA_DIR / fname
    if not fpath.exists():
        raise FileNotFoundError(f"Sites file not found: {fpath}")
    return pd.read_stata(fpath, convert_categoricals=False)


def load_belligerents() -> pd.DataFrame:
    """
    Load belligerent status data (all_belligerents.dta).

    ~1,235 rows identifying each country's role in each war:
      - warname, iso
      - belligerent (always 1 in this file; non-belligerents are absent)
      - initiator (1=attacker, 2=defender)
      - outcome (1=winner, 2=loser, 3/4/6=other)
      - type ('inter' or 'intra')
    """
    return pd.read_stata(
        DATA_DIR / "all_belligerents.dta", convert_categoricals=False
    )


def load_pop() -> pd.DataFrame:
    """
    Load population data (pop.dta).

    ~59,700 rows with columns: iso, year, pop.
    Covers a wide range of countries from ~1800-2024.
    """
    return pd.read_stata(DATA_DIR / "pop.dta", convert_categoricals=False)


def load_pop_world() -> pd.DataFrame:
    """
    Load world population data (pop_world.dta).

    155 rows with columns: year, pop_world.
    Covers 1869-2023.
    """
    return pd.read_stata(DATA_DIR / "pop_world.dta", convert_categoricals=False)


def load_trade() -> pd.DataFrame:
    """
    Load bilateral trade/gravity data (trade_gravity.dta).

    WARNING: This file is ~80 MB and contains ~15.5 million rows.
    Columns: exporter, importer, year, trade_value_notimp, trade_value,
             distw, Contig, proximity.

    Consider filtering on load if you only need a subset of country pairs.
    """
    warnings.warn(
        "trade_gravity.dta is ~80 MB (~15.5M rows). Loading may take a moment.",
        stacklevel=2,
    )
    return pd.read_stata(
        DATA_DIR / "trade_gravity.dta", convert_categoricals=False
    )


def load_trade_national() -> pd.DataFrame:
    """
    Load national-level trade data (trade_national.dta).

    ~34,000 rows with columns: iso, year, imports, exports.
    """
    return pd.read_stata(
        DATA_DIR / "trade_national.dta", convert_categoricals=False
    )


def load_territory() -> pd.DataFrame:
    """
    Load territorial changes data (territory.dta).

    194 rows with columns: year, iso.
    Each row indicates a country-year where a territorial change occurred.
    """
    return pd.read_stata(DATA_DIR / "territory.dta", convert_categoricals=False)


def load_ltp() -> pd.DataFrame:
    """
    Load TFP and capital stock data (ltp.dta).

    ~59,700 rows with columns: year, iso, tfp, cs_ppp.
    """
    return pd.read_stata(DATA_DIR / "ltp.dta", convert_categoricals=False)


# =============================================================================
# PANEL CONSTRUCTION
# =============================================================================
# Python translation of the Stata `build_panel` program in panel.do.
#
# The panel construction follows these steps:
#   1. Load and filter war sites by parameters
#   2. Create a balanced country-year panel
#   3. Cross-join (joinby) to create all country-pair x year combinations
#   4. Merge belligerent status to classify country roles
#   5. Compute gamma coefficients (share of war impact by role)
#   6. Merge lagged population data -> compute epsilon_pop
#   7. Merge lagged trade data -> compute epsilon_trade and epsilon_prox
#   8. Generate regression variables: regr_{cas|gprc}_phi_{site|bell|third}
#      and regr_{cas|gprc}_psi_{pop|trade|prox}_{bell|third}
#   9. Collapse to country-year by summing regr_* variables
#  10. Merge with macro data and apply sample restrictions
# =============================================================================

def build_panel(
    wars: str = "all",
    casmin: float = 0,
    eww: bool = False,
    eww1: bool = False,
    eww2: bool = False,
    minlength: int = -1,
    maxlength: int = -1,
    altstart: bool = False,
    excludeUS: bool = False,
    excludeterrchange: bool = False,
    postww: bool = False,
    winsor_cas: float = -1,
    gprc_only: bool = False,
) -> pd.DataFrame:
    """
    Build the analysis panel from processed data.

    This is a Python translation of the Stata `build_panel` program in panel.do.
    It constructs a country-year panel with war exposure variables suitable for
    local projection estimation.

    Parameters
    ----------
    wars : str
        Type of wars to include: 'all', 'interstate', 'intrastate', 'causality'.
    casmin : float
        Minimum casualty threshold for including war sites (default 0).
    eww : bool
        Exclude both World Wars.
    eww1 : bool
        Exclude World War I only.
    eww2 : bool
        Exclude World War II only.
    minlength : int
        Minimum war duration in years (-1 = no restriction).
    maxlength : int
        Maximum war duration in years (-1 = no restriction).
    altstart : bool
        Use alternative war start dates (start_alt instead of start).
    excludeUS : bool
        Exclude the United States from the sample.
    excludeterrchange : bool
        Exclude countries with territorial changes in the following 8 years.
    postww : bool
        Restrict sample to post-WWII period (year >= 1946).
    winsor_cas : float
        Winsorize shock_caspop_home at this level (-1 = no winsorization).
    gprc_only : bool
        Restrict to countries with GPRC data (requires gprc.dta).

    Returns
    -------
    pd.DataFrame
        Country-year panel with war exposure regression variables merged
        with macroeconomic data.  Key generated columns include:
          regr_cas_phi_{site,bell,third}
          regr_cas_psi_{pop,trade,prox}_{bell,third}
          regr_gprc_phi_{site,bell,third}
          regr_gprc_psi_{pop,trade,prox}_{bell,third}
          (plus winner/loser and attacker/defender interaction variants)
        along with all macro variables from macro.dta.
    """
    # ------------------------------------------------------------------
    # STEP 1: Load and filter war sites
    # ------------------------------------------------------------------
    sites = load_sites(wars=wars)

    # Apply casualty threshold
    sites = sites[sites["casualties"] >= casmin].copy()

    # Exclude World Wars
    if eww:
        sites = sites[~sites["warname"].isin(["World War I", "World War II"])]
    if eww1:
        sites = sites[sites["warname"] != "World War I"]
    if eww2:
        sites = sites[sites["warname"] != "World War II"]

    # Filter by war duration
    if minlength >= 0:
        duration = sites["end"] - sites["start"] + 1
        sites = sites[duration >= minlength]
    if maxlength >= 0:
        duration = sites["end"] - sites["start"] + 1
        sites = sites[duration <= maxlength]

    # Use alternative start dates if requested
    if altstart:
        sites = sites.drop(columns=["start"])
        sites = sites.rename(columns={"start_alt": "start"})

    # Create year = start, deduplicate by iso-year
    sites = sites.copy()
    sites["year"] = sites["start"]
    sites = sites.drop_duplicates(subset=["iso", "year"], keep="first")

    # Winsorize casualty shock variable
    if winsor_cas > 0:
        sites["shock_caspop_home"] = sites["shock_caspop_home"].clip(
            upper=winsor_cas
        )

    # Keep a copy of sites for later merges (belligerent lookup, site indicator)
    sites_full = sites.copy()

    # ------------------------------------------------------------------
    # STEP 2: Create balanced country-year panel
    # ------------------------------------------------------------------
    year_min = int(sites["start"].min())
    year_max = int(sites["end"].max())

    # Combine all country codes from sites and macro datasets
    macro = load_macro()
    all_isos = pd.concat(
        [sites[["iso"]], macro[["iso"]]], ignore_index=True
    ).drop_duplicates()

    # Expand to balanced country-year panel
    years = pd.DataFrame({"year": range(year_min, year_max + 1)})
    panel = all_isos.merge(years, how="cross")

    # ------------------------------------------------------------------
    # STEP 3: Merge sites data and create country-pair panel
    # ------------------------------------------------------------------
    # Merge sites onto the panel by (iso, year) where iso is the site country
    # and year is the war start year.
    # In Stata: rename year start -> merge 1:m iso start using sites
    #           -> rename back -> joinby year using panel
    sites_merge = sites[
        ["iso", "year", "warname", "shock_caspop_home", "destruction"]
    ].copy()
    sites_merge = sites_merge.rename(
        columns={
            "shock_caspop_home": "sites_cas",
            "destruction": "sites_gprc",
        }
    )

    # Merge sites onto site countries: for each site-country x year, get war info
    site_panel = panel.merge(
        sites_merge,
        on=["iso", "year"],
        how="inner",  # keep only site-country-years that have wars
    )

    # Calculate total casualties per site-country-year (summing across wars)
    site_panel["sites_cas_tot"] = site_panel.groupby(["iso", "year"])[
        "sites_cas"
    ].transform("sum")

    # Rename iso -> iso_site (this is the country where the war occurred)
    site_panel = site_panel.rename(columns={"iso": "iso_site"})

    # Cross-join with all countries for each year (the expensive joinby step).
    # For efficiency, only keep the columns we need from the panel.
    panel_years = panel.rename(columns={"iso": "iso_exposed"})

    # joinby year: for each (iso_site, year, warname) row, merge with all
    # (iso_exposed, year) rows sharing the same year.
    country_pairs = site_panel.merge(
        panel_years, on="year", how="inner"
    )
    country_pairs = country_pairs.sort_values(
        ["iso_site", "year", "iso_exposed"]
    ).reset_index(drop=True)

    # ------------------------------------------------------------------
    # STEP 4: Identify country types and calculate gamma coefficients
    # ------------------------------------------------------------------
    # Merge belligerent information
    belligerents = load_belligerents()
    # The Stata code does: keepusing(bell outcome initiator)
    # The .dta file has 'belligerent' not 'bell'; Stata abbreviation matching
    # converts it. We rename explicitly.
    bell_merge = belligerents[
        ["iso", "warname", "belligerent", "outcome", "initiator"]
    ].rename(columns={"belligerent": "bell"})

    country_pairs = country_pairs.merge(
        bell_merge,
        left_on=["iso_exposed", "warname"],
        right_on=["iso", "warname"],
        how="left",
        suffixes=("", "_bell"),
    )
    # Drop the extra iso column from belligerents merge
    if "iso_bell" in country_pairs.columns:
        country_pairs = country_pairs.drop(columns=["iso_bell"])
    if "iso" in country_pairs.columns and "iso_site" in country_pairs.columns:
        # iso may have been brought in from bell_merge; drop it
        country_pairs = country_pairs.drop(
            columns=[c for c in ["iso"] if c in country_pairs.columns
                     and c not in ["iso_site", "iso_exposed"]],
            errors="ignore",
        )

    # Merge site indicator: does iso_exposed also appear as a site in this war?
    # In Stata: merge m:1 iso warname using `sites', keepusing(shock_dummy_home)
    #           rename shock_dummy_home site
    sites_site_indicator = sites_full[["iso", "warname", "shock_dummy_home"]].rename(
        columns={"shock_dummy_home": "site"}
    )
    country_pairs = country_pairs.merge(
        sites_site_indicator,
        left_on=["iso_exposed", "warname"],
        right_on=["iso", "warname"],
        how="left",
        suffixes=("", "_siteindicator"),
    )
    # Clean up extra iso column
    if "iso_siteindicator" in country_pairs.columns:
        country_pairs = country_pairs.drop(columns=["iso_siteindicator"])
    if "iso" in country_pairs.columns and "iso_site" in country_pairs.columns:
        country_pairs = country_pairs.drop(
            columns=[c for c in country_pairs.columns
                     if c == "iso" and c not in ["iso_site", "iso_exposed"]],
            errors="ignore",
        )

    # Clean up indicators: fill NaN with 0
    country_pairs["site"] = country_pairs["site"].fillna(0)
    # Bell = 0 if missing or if the country is a site (Stata: replace bell = 0 if bell == . | site == 1)
    country_pairs["bell"] = country_pairs["bell"].fillna(0)
    country_pairs.loc[country_pairs["site"] == 1, "bell"] = 0

    # Fill missing casualty/destruction values with 0
    country_pairs["sites_cas"] = country_pairs["sites_cas"].fillna(0)
    country_pairs["sites_gprc"] = country_pairs["sites_gprc"].fillna(0)

    # Calculate gamma coefficients
    # gamma_site: share for site countries (iso_site == iso_exposed AND positive casualties)
    country_pairs["gamma_site"] = np.where(
        (country_pairs["iso_site"] == country_pairs["iso_exposed"])
        & (country_pairs["sites_cas"] > 0)
        & (country_pairs["sites_cas"].notna()),
        country_pairs["sites_cas"] / country_pairs["sites_cas_tot"],
        np.nan,
    )
    # gamma_bell: share for belligerent countries
    country_pairs["gamma_bell"] = np.where(
        country_pairs["bell"] == 1,
        country_pairs["sites_cas"] / country_pairs["sites_cas_tot"],
        np.nan,
    )
    # gamma_third: share for third-party countries (not belligerent and not site)
    country_pairs["gamma_third"] = np.where(
        (country_pairs["bell"] == 0) & (country_pairs["site"] == 0),
        country_pairs["sites_cas"] / country_pairs["sites_cas_tot"],
        np.nan,
    )

    # Collapse to country-pair-year level:
    # (sum) gamma_* sites_cas (mean) sites_gprc outcome initiator
    agg_dict = {
        "gamma_site": "sum",
        "gamma_bell": "sum",
        "gamma_third": "sum",
        "sites_cas": "sum",
        "sites_gprc": "mean",
        "outcome": "mean",
        "initiator": "mean",
    }
    # Before collapse, replace NaN gamma values with 0 for summation
    # (Stata sum treats missing as 0 in collapse)
    for col in ["gamma_site", "gamma_bell", "gamma_third"]:
        country_pairs[col] = country_pairs[col].fillna(0)
    # Also fill outcome/initiator NaN for mean computation (Stata ignores missing)
    # but keep them NaN so mean only counts non-missing.
    # Actually Stata collapse (mean) ignores missing. pandas mean also ignores NaN
    # by default, so we leave outcome/initiator as-is.

    collapsed = (
        country_pairs.groupby(["iso_site", "iso_exposed", "year"])
        .agg(agg_dict)
        .reset_index()
    )

    # ------------------------------------------------------------------
    # STEP 5: Calculate epsilon coefficients (exposure weights)
    # ------------------------------------------------------------------
    # Merge population data (lagged: year - 1)
    pop = load_pop()
    pop_world = load_pop_world()

    # Create lagged year for merging population
    collapsed["year_lag"] = collapsed["year"] - 1

    # Merge site-country population (iso_site at year-1)
    collapsed = collapsed.merge(
        pop.rename(columns={"iso": "iso_site", "year": "year_lag"}),
        on=["iso_site", "year_lag"],
        how="left",
    )
    # Merge world population at year-1
    collapsed = collapsed.merge(
        pop_world.rename(columns={"year": "year_lag"}),
        on="year_lag",
        how="left",
    )

    # Population-weighted epsilon coefficients
    # epsilon_pop_site = 0 (sites don't have pop-weighted exposure to themselves)
    collapsed["epsilon_pop_site"] = 0.0
    collapsed["epsilon_pop_bell"] = (
        (collapsed["pop"] / collapsed["pop_world"]) * collapsed["gamma_bell"]
    )
    collapsed["epsilon_pop_third"] = (
        (collapsed["pop"] / collapsed["pop_world"]) * collapsed["gamma_third"]
    )

    # Merge trade data (lagged: year - 1)
    # Trade is keyed on (exporter=iso_site, importer=iso_exposed, year=year-1)
    # WARNING: trade_gravity.dta is very large (~15.5M rows).
    # We load only the columns we need and filter to relevant pairs.
    warnings.warn(
        "Loading trade_gravity.dta (~80 MB). This may take a moment.",
        stacklevel=2,
    )
    trade = pd.read_stata(
        DATA_DIR / "trade_gravity.dta",
        convert_categoricals=False,
        columns=["exporter", "importer", "year", "trade_value", "proximity"],
    )

    collapsed = collapsed.merge(
        trade.rename(
            columns={
                "exporter": "iso_site",
                "importer": "iso_exposed",
                "year": "year_lag",
            }
        ),
        on=["iso_site", "iso_exposed", "year_lag"],
        how="left",
    )

    # Merge GDP from macro data for exposed country at year-1
    macro_gdp = macro[["iso", "year", "gdp"]].copy()
    macro_gdp = macro_gdp.rename(
        columns={"iso": "iso_exposed", "year": "year_lag"}
    )
    collapsed = collapsed.merge(
        macro_gdp, on=["iso_exposed", "year_lag"], how="left"
    )

    # Fill missing proximity with 0 (no geographic connection)
    collapsed["proximity"] = collapsed["proximity"].fillna(0)

    # Trade-weighted epsilon coefficients
    collapsed["epsilon_trade_site"] = 0.0
    collapsed["epsilon_trade_bell"] = (
        (collapsed["trade_value"] / collapsed["gdp"]) * collapsed["gamma_bell"]
    )
    collapsed["epsilon_trade_third"] = (
        (collapsed["trade_value"] / collapsed["gdp"]) * collapsed["gamma_third"]
    )

    # Proximity-weighted epsilon coefficients
    collapsed["epsilon_prox_site"] = 0.0
    collapsed["epsilon_prox_bell"] = (
        collapsed["proximity"] * collapsed["gamma_bell"]
    )
    collapsed["epsilon_prox_third"] = (
        collapsed["proximity"] * collapsed["gamma_third"]
    )

    # ------------------------------------------------------------------
    # STEP 6: Generate regression variables
    # ------------------------------------------------------------------
    # For each shock type (cas, gprc) and each country group, generate:
    #   regr_{shock}_phi_{group}       = gamma_{group} * sites_{shock}
    #   regr_{shock}_psi_pop_{group}   = epsilon_pop_{group} * sites_{shock}
    #   regr_{shock}_psi_trade_{group} = epsilon_trade_{group} * sites_{shock}
    #   regr_{shock}_psi_prox_{group}  = epsilon_prox_{group} * sites_{shock}
    # Plus winner/loser and attacker/defender interactions for site and bell.
    for shock in ["cas", "gprc"]:
        shock_var = f"sites_{shock}"

        for group in ["site", "bell", "third"]:
            # Basic phi variable: gamma * shock
            collapsed[f"regr_{shock}_phi_{group}"] = (
                collapsed[f"gamma_{group}"] * collapsed[shock_var]
            )

        for group in ["bell", "third"]:
            # Weighted psi variables (only for belligerents and third parties)
            collapsed[f"regr_{shock}_psi_pop_{group}"] = (
                collapsed[f"epsilon_pop_{group}"] * collapsed[shock_var]
            )
            collapsed[f"regr_{shock}_psi_trade_{group}"] = (
                collapsed[f"epsilon_trade_{group}"] * collapsed[shock_var]
            )
            collapsed[f"regr_{shock}_psi_prox_{group}"] = (
                collapsed[f"epsilon_prox_{group}"] * collapsed[shock_var]
            )

        # Winner/loser and attacker/defender interactions
        # These are conditional on outcome/initiator values:
        #   outcome: 1=winner, 2=loser
        #   initiator: 1=attacker, 2=defender
        for group in ["site", "bell"]:
            gamma_col = f"gamma_{group}"

            # Winner/loser
            collapsed[f"regr_{shock}_phi_{group}_winner"] = np.where(
                collapsed["outcome"] == 1,
                collapsed[gamma_col] * collapsed[shock_var],
                np.nan,
            )
            collapsed[f"regr_{shock}_phi_{group}_loser"] = np.where(
                collapsed["outcome"] == 2,
                collapsed[gamma_col] * collapsed[shock_var],
                np.nan,
            )

            # Population-weighted winner/loser
            eps_col = f"epsilon_pop_{group}"
            collapsed[f"regr_{shock}_psi_pop_{group}_winner"] = np.where(
                collapsed["outcome"] == 1,
                collapsed[eps_col] * collapsed[shock_var],
                np.nan,
            )
            collapsed[f"regr_{shock}_psi_pop_{group}_loser"] = np.where(
                collapsed["outcome"] == 2,
                collapsed[eps_col] * collapsed[shock_var],
                np.nan,
            )

            # Attacker/defender
            collapsed[f"regr_{shock}_phi_{group}_attack"] = np.where(
                collapsed["initiator"] == 1,
                collapsed[gamma_col] * collapsed[shock_var],
                np.nan,
            )
            collapsed[f"regr_{shock}_phi_{group}_defend"] = np.where(
                collapsed["initiator"] == 2,
                collapsed[gamma_col] * collapsed[shock_var],
                np.nan,
            )

            # Population-weighted attacker/defender
            collapsed[f"regr_{shock}_psi_pop_{group}_attack"] = np.where(
                collapsed["initiator"] == 1,
                collapsed[eps_col] * collapsed[shock_var],
                np.nan,
            )
            collapsed[f"regr_{shock}_psi_pop_{group}_defend"] = np.where(
                collapsed["initiator"] == 2,
                collapsed[eps_col] * collapsed[shock_var],
                np.nan,
            )

    # ------------------------------------------------------------------
    # STEP 7: Collapse to country-year and merge with macro
    # ------------------------------------------------------------------
    # Identify all regr_* columns
    regr_cols = [c for c in collapsed.columns if c.startswith("regr_")]

    # Replace NaN with 0 for summation (Stata collapse (sum) treats missing as 0)
    for col in regr_cols:
        collapsed[col] = collapsed[col].fillna(0)

    # Collapse to (iso_exposed, year) by summing all regr_* variables
    final_panel = (
        collapsed.groupby(["iso_exposed", "year"])[regr_cols]
        .sum()
        .reset_index()
    )
    final_panel = final_panel.rename(columns={"iso_exposed": "iso"})
    final_panel = final_panel.sort_values(["iso", "year"]).reset_index(drop=True)

    # Merge with macro data (keep all macro observations, left join)
    # In Stata: merge 1:1 iso year using macro.dta, keep(matched using)
    # "matched using" means: keep everything in macro, plus matches from panel.
    # This is equivalent to a right join on macro.
    final_panel = macro.merge(
        final_panel, on=["iso", "year"], how="left"
    )

    # Fill missing regr_* values with 0 (country-years with no war exposure)
    for col in regr_cols:
        if col in final_panel.columns:
            final_panel[col] = final_panel[col].fillna(0)

    # ------------------------------------------------------------------
    # STEP 8: Apply sample restrictions
    # ------------------------------------------------------------------
    if excludeUS:
        final_panel = final_panel[final_panel["iso"] != "USA"]

    if excludeterrchange:
        territory = load_territory()
        # Expand territory changes forward 8 years (country cannot appear in
        # the sample if a territorial change occurs in any of the next 8 years)
        terr_expanded = []
        for _, row in territory.iterrows():
            for offset in range(8):
                terr_expanded.append(
                    {"iso": row["iso"], "year": row["year"] - offset}
                )
        terr_df = pd.DataFrame(terr_expanded).drop_duplicates()
        terr_df["terrchange"] = 1
        final_panel = final_panel.merge(
            terr_df, on=["iso", "year"], how="left"
        )
        final_panel = final_panel[final_panel["terrchange"] != 1]
        final_panel = final_panel.drop(columns=["terrchange"])

    if postww:
        final_panel = final_panel[final_panel["year"] >= 1946]

    if gprc_only:
        gprc_path = DATA_DIR / "gprc.dta"
        if gprc_path.exists():
            gprc = pd.read_stata(gprc_path, convert_categoricals=False)
            gprc_keys = gprc[["iso", "year"]].drop_duplicates()
            final_panel = final_panel.merge(
                gprc_keys, on=["iso", "year"], how="inner"
            )
        else:
            warnings.warn(
                "gprc.dta not found; gprc_only filter not applied.",
                stacklevel=2,
            )

    final_panel = final_panel.reset_index(drop=True)

    return final_panel


# =============================================================================
# LOCAL PROJECTIONS
# =============================================================================
# Python translation of the Stata `run_and_plot_lp` program in lp.do.
#
# The Stata code uses xtscc (Driscoll-Kraay standard errors) for inference.
# Here we use linearmodels PanelOLS with entity (country) fixed effects and
# clustered standard errors by country, which is a common approximation.
# =============================================================================

# Dependent variable transformation type mapping (from lp.do)
DEPVAR_DIFFTYPE = {
    "lgdp": "difference_long",
    "lcpi": "difference_long",
    "cpi": "change",
    "eq_tr_real": "level",
    "capital_tr_real": "level",
    "unemp": "difference_long",
    "trade": "over_preshock_gdp",
    "imports": "over_preshock_gdp",
    "cons": "over_preshock_gdp",
    "lcons": "difference_long",
    "lcons_dtrd": "difference_long",
    "exports": "over_preshock_gdp",
    "milex": "over_preshock_gdp",
    "milex_gdp": "level",
    "inflation": "level",
    "linflation": "level",
    "lcs_ppp": "difference_long",
    "lcs_ppp_dtrd": "difference_long",
    "ltfp": "difference_long",
    "lpop": "difference_long",
    "leqrtcum": "difference_long",
    "lcapital_tr_cum": "difference_long",
    "lfertility": "difference_long",
    "ldeaths_mp": "difference_long",
    "ldeaths_nmp": "difference_long",
    "deaths": "over_preshock_pop",
    "deaths_mp": "over_preshock_pop",
    "deaths_nmp": "over_preshock_pop",
    "milper": "over_preshock_pop",
    "institutions": "difference_long",
    "judicial": "difference_long",
    "medial": "difference_long",
    "electoral": "difference_long",
    "terrchange_pop": "over_preshock_pop",
    "cbrate": "difference_long",
    "strate": "difference_long",
    "ltrate": "difference_long_ppt",
    "ltrate_dthp": "difference_long",
    "ltrate_dtpl": "difference_long",
    "ca": "over_preshock_gdp",
    "lgdp_dthp": "difference_long",
    "lcpi_dthp": "difference_long",
    "lcons_dthp": "difference_long",
    "lcs_ppp_dthp": "difference_long",
    "leqrtcum_dthp": "difference_long",
    "ltfp_dthp": "difference_long",
    "lmilex_dthp": "difference_long",
    "lmilper_dthp": "difference_long",
    "lHPI": "difference_long",
    "govtax": "over_preshock_gdp",
    "lgdp_dtpl": "difference_long",
    "lcpi_dtpl": "difference_long",
    "lcons_dtpl": "difference_long",
    "lcs_ppp_dtpl": "difference_long",
    "leqrtcum_dtpl": "difference_long",
    "ltfp_dtpl": "difference_long",
    "lmilex_dtpl": "difference_long",
    "lmilper_dtpl": "difference_long",
}


def _create_depvar_transform(
    panel: pd.DataFrame,
    depvar: str,
    h: int,
    depvar_type: str,
) -> pd.Series:
    """
    Create the dependent variable Y_{t+h} transformation for local projections.

    Parameters
    ----------
    panel : DataFrame
        Must be sorted by (iso, year) and have 'iso', 'year', depvar columns.
        Also needs 'gdp' for 'over_preshock_gdp' and 'pop' for 'over_preshock_pop'.
    depvar : str
        Name of the dependent variable column.
    h : int
        Forecast horizon (0 = contemporaneous).
    depvar_type : str
        Transformation type. One of:
          'difference_long':     (Y_{t+h} - Y_{t-1}) * 100
          'difference_long_ppt': (Y_{t+h} - Y_{t-1}) * 100
          'over_preshock_gdp':   ((Y_{t+h} - Y_{t-1}) / GDP_{t-1}) * 100
          'over_preshock_pop':   ((Y_{t+h} - Y_{t-1}) / POP_{t-1}) * 100
          'change':              ((Y_{t+h} / Y_{t-1}) - 1) * 100
          'level':               Y_{t+h} * 100

    Returns
    -------
    pd.Series
        Transformed dependent variable aligned to the panel index.
    """
    # Ensure panel is sorted and we can use shift operations within groups
    # Forward h periods = lead h = shift(-h) within each country
    # Lag 1 period = shift(1) within each country
    y = panel[depvar]
    y_forward_h = panel.groupby("iso")[depvar].shift(-h)
    y_lag1 = panel.groupby("iso")[depvar].shift(1)

    if depvar_type in ("difference_long", "difference_long_ppt"):
        return (y_forward_h - y_lag1) * 100
    elif depvar_type == "over_preshock_gdp":
        gdp_lag1 = panel.groupby("iso")["gdp"].shift(1)
        return ((y_forward_h - y_lag1) / gdp_lag1) * 100
    elif depvar_type == "over_preshock_pop":
        pop_lag1 = panel.groupby("iso")["pop"].shift(1)
        return ((y_forward_h - y_lag1) / pop_lag1) * 100
    elif depvar_type == "change":
        return ((y_forward_h / y_lag1) - 1) * 100
    elif depvar_type == "level":
        return y_forward_h * 100
    else:
        raise ValueError(f"Unknown depvar_type: {depvar_type}")


def run_local_projection(
    panel: pd.DataFrame,
    depvar: str,
    xvars: list[str],
    lags: int = 4,
    h_max: int = 8,
    depvar_type: Optional[str] = None,
    cluster_var: str = "iso",
    time_fe: bool = False,
    custom_controls: Optional[list[str]] = None,
    confidence: float = 0.90,
) -> pd.DataFrame:
    """
    Run local projections for horizons h = 0, ..., h_max.

    For each horizon h:
      1. Create Y_{t+h} transformation based on depvar_type
      2. Regress Y_{h} on xvars (with lags 0..lags), lagged dep var, country FE
      3. Use clustered standard errors by country

    This is a Python translation of the estimation loop in lp.do. The Stata
    code uses xtscc (Driscoll-Kraay standard errors). Here we use
    linearmodels.PanelOLS with entity fixed effects and clustered SEs by country,
    which is a reasonable approximation.

    Parameters
    ----------
    panel : DataFrame
        Country-year panel from build_panel(). Must contain 'iso', 'year', 'cid',
        the depvar column, all xvar columns, and 'gdp'/'pop' if needed by the
        depvar transformation.
    depvar : str
        Name of the dependent variable (e.g., 'lgdp', 'lcpi').
    xvars : list of str
        List of contemporaneous regressor names (e.g.,
        ['regr_cas_phi_site', 'regr_cas_psi_pop_bell', ...]).
        Lags 0..lags of each xvar will be included automatically.
    lags : int
        Number of lags to include for each xvar and for the lagged dependent
        variable (default: 4, matching the Stata code).
    h_max : int
        Maximum impulse response horizon (default: 8).
    depvar_type : str or None
        Dependent variable transformation type. If None, looked up from the
        DEPVAR_DIFFTYPE dictionary. Options:
          'difference_long', 'difference_long_ppt', 'over_preshock_gdp',
          'over_preshock_pop', 'change', 'level'.
    cluster_var : str
        Variable to cluster standard errors on (default: 'iso').
    time_fe : bool
        Include year fixed effects in addition to country FE (default: False).
    custom_controls : list of str or None
        Additional control variable names to include (contemporaneous only;
        lags should be pre-generated in the panel if needed).
    confidence : float
        Confidence level for confidence intervals (default: 0.90, matching
        the Stata code's level(90)).

    Returns
    -------
    pd.DataFrame
        DataFrame with one row per horizon containing:
          - horizon (int): the forecast horizon h
          - estimate (float): point estimate (coefficient or lincom result)
          - se (float): standard error
          - ci_lower (float): lower confidence bound
          - ci_upper (float): upper confidence bound
          - N (int): number of observations
          - coef_dict (dict): full coefficient dictionary for this horizon
          - vcov (np.ndarray): variance-covariance matrix of coefficients

        The coef_dict and vcov are included so that compute_irf() can form
        arbitrary linear combinations of the estimated coefficients.
    """
    if depvar_type is None:
        depvar_type = DEPVAR_DIFFTYPE.get(depvar)
        if depvar_type is None:
            raise ValueError(
                f"Unknown depvar '{depvar}'. Specify depvar_type explicitly "
                f"or add it to DEPVAR_DIFFTYPE."
            )

    # Sort panel by (iso, year) for correct shift operations
    panel = panel.sort_values(["iso", "year"]).copy()

    # Create the h=0 transformation of the dependent variable (used as
    # lagged dependent variable control: l(1..lags) of depvar_0)
    panel[f"{depvar}_0"] = _create_depvar_transform(
        panel, depvar, h=0, depvar_type=depvar_type
    )

    # Pre-generate all lagged versions of xvars and the dep var control
    # For xvar lags: l(0/lags).xvar means current + lags 1..lags
    for xvar in xvars:
        for lag in range(1, lags + 1):
            col_name = f"L{lag}_{xvar}"
            if col_name not in panel.columns:
                panel[col_name] = panel.groupby("iso")[xvar].shift(lag)

    # Lagged dependent variable controls: l(1..lags) of depvar_0
    for lag in range(1, lags + 1):
        col_name = f"L{lag}_{depvar}_0"
        if col_name not in panel.columns:
            panel[col_name] = panel.groupby("iso")[f"{depvar}_0"].shift(lag)

    # Build the full list of RHS variables for each regression
    rhs_vars = []
    for xvar in xvars:
        rhs_vars.append(xvar)  # contemporaneous (lag 0)
        for lag in range(1, lags + 1):
            rhs_vars.append(f"L{lag}_{xvar}")
    # Lagged dependent variable
    for lag in range(1, lags + 1):
        rhs_vars.append(f"L{lag}_{depvar}_0")
    # Custom controls
    if custom_controls:
        rhs_vars.extend(custom_controls)

    # Results storage
    results_list = []

    from scipy import stats

    z_crit = stats.norm.ppf(1 - (1 - confidence) / 2)

    for h in range(h_max + 1):
        # Create dependent variable for horizon h
        lhs_col = f"{depvar}_{h}"
        panel[lhs_col] = _create_depvar_transform(
            panel, depvar, h=h, depvar_type=depvar_type
        )

        # Build regression DataFrame (drop missing)
        reg_cols = [lhs_col] + rhs_vars + ["iso", "year"]
        if time_fe:
            reg_cols.append("year")
        reg_df = panel[list(set(reg_cols))].dropna().copy()

        if len(reg_df) < len(rhs_vars) + 2:
            warnings.warn(
                f"Horizon {h}: insufficient observations ({len(reg_df)}). "
                "Skipping.",
                stacklevel=2,
            )
            results_list.append({
                "horizon": h,
                "estimate": np.nan,
                "se": np.nan,
                "ci_lower": np.nan,
                "ci_upper": np.nan,
                "N": len(reg_df),
                "coef_dict": {},
                "vcov": np.array([]),
            })
            continue

        # Set up PanelOLS with entity (country) fixed effects
        reg_df = reg_df.set_index(["iso", "year"])

        y = reg_df[lhs_col]
        X = reg_df[rhs_vars]

        # Add time fixed effects as dummies if requested
        if time_fe:
            model = PanelOLS(
                y, X, entity_effects=True, time_effects=True, drop_absorbed=True
            )
        else:
            model = PanelOLS(
                y, X, entity_effects=True, drop_absorbed=True
            )

        try:
            res = model.fit(cov_type="clustered", cluster_entity=True)
        except Exception as e:
            warnings.warn(
                f"Horizon {h}: estimation failed: {e}",
                stacklevel=2,
            )
            results_list.append({
                "horizon": h,
                "estimate": np.nan,
                "se": np.nan,
                "ci_lower": np.nan,
                "ci_upper": np.nan,
                "N": 0,
                "coef_dict": {},
                "vcov": np.array([]),
            })
            continue

        # Store full coefficient vector and variance-covariance matrix
        coef_dict = dict(res.params)
        vcov = res.cov.values
        param_names = list(res.params.index)

        # For the results table, report the first xvar coefficient as the
        # "headline" estimate. Users should use compute_irf() for proper
        # linear combinations.
        first_xvar = xvars[0]
        est = coef_dict.get(first_xvar, np.nan)
        idx = param_names.index(first_xvar) if first_xvar in param_names else None
        se = np.sqrt(vcov[idx, idx]) if idx is not None else np.nan

        results_list.append({
            "horizon": h,
            "estimate": est,
            "se": se,
            "ci_lower": est - z_crit * se,
            "ci_upper": est + z_crit * se,
            "N": int(res.nobs),
            "coef_dict": coef_dict,
            "vcov": vcov,
            "param_names": param_names,
        })

        # Clean up horizon-specific column
        panel.drop(columns=[lhs_col], inplace=True)

    # Clean up pre-generated columns
    panel.drop(columns=[f"{depvar}_0"], inplace=True, errors="ignore")

    return pd.DataFrame(results_list)


# =============================================================================
# IMPULSE RESPONSE COMPUTATION
# =============================================================================
# Given LP results from run_local_projection(), compute the impulse response
# for a specific shock specification by forming the appropriate linear
# combination of estimated coefficients.
# =============================================================================

def compute_irf(
    lp_results: pd.DataFrame,
    lincom_expr: dict[str, float],
    confidence: float = 0.90,
) -> pd.DataFrame:
    """
    Compute impulse response from LP estimates for a specific shock specification.

    The impulse response at each horizon h is computed as a linear combination
    of the estimated coefficients:

        IRF_h = sum_j (weight_j * beta_j_h)

    where weight_j and beta_j are specified by lincom_expr.

    Parameters
    ----------
    lp_results : DataFrame
        Output from run_local_projection(). Must contain columns:
        'horizon', 'coef_dict', 'vcov', 'param_names'.
    lincom_expr : dict
        Dictionary mapping coefficient names to weights. For example:
          - War site effect with cas=0.02:
            {'regr_cas_phi_site': 0.02}
          - Belligerent with cas=0.02, integration=0.03 (pop channel):
            {'regr_cas_phi_bell': 0.02,
             'regr_cas_psi_pop_bell': 0.02 * 0.03}
          - Third party with cas=0.02, integration=0.03 (pop channel):
            {'regr_cas_phi_third': 0.02,
             'regr_cas_psi_pop_third': 0.02 * 0.03}
          - Third party with cas=0.02, integration=0.03 (trade channel):
            {'regr_cas_phi_third': 0.02,
             'regr_cas_psi_trade_third': 0.02 * 0.03}

        These correspond to the Stata lincom expressions in lp.do, e.g.:
          lincom 0.02*regr_cas_phi_site
          lincom 0.02*0.03*regr_cas_psi_pop_bell + 0.02*regr_cas_phi_bell
    confidence : float
        Confidence level for confidence intervals (default: 0.90).

    Returns
    -------
    pd.DataFrame
        DataFrame with columns:
          - horizon (int): forecast horizon
          - irf (float): impulse response point estimate
          - se (float): standard error of the linear combination
          - ci_lower (float): lower confidence bound
          - ci_upper (float): upper confidence bound

    Examples
    --------
    Baseline "casroles" specification from the paper (Figure 4):

    >>> # War site effect (cas = 2% of local population)
    >>> irf_site = compute_irf(lp_results, {'regr_cas_phi_site': 0.02})

    >>> # Belligerent effect (cas = 2%, integration = 3%)
    >>> irf_bell = compute_irf(lp_results, {
    ...     'regr_cas_phi_bell': 0.02,
    ...     'regr_cas_psi_pop_bell': 0.02 * 0.03,
    ... })

    >>> # Third-party effect (cas = 2%, integration = 3%)
    >>> irf_third = compute_irf(lp_results, {
    ...     'regr_cas_phi_third': 0.02,
    ...     'regr_cas_psi_pop_third': 0.02 * 0.03,
    ... })
    """
    from scipy import stats

    z_crit = stats.norm.ppf(1 - (1 - confidence) / 2)

    irf_rows = []

    for _, row in lp_results.iterrows():
        h = row["horizon"]
        coef_dict = row.get("coef_dict", {})
        vcov = row.get("vcov", np.array([]))
        param_names = row.get("param_names", [])

        if not coef_dict or len(vcov) == 0:
            irf_rows.append({
                "horizon": h,
                "irf": np.nan,
                "se": np.nan,
                "ci_lower": np.nan,
                "ci_upper": np.nan,
            })
            continue

        # Compute the linear combination: sum of weight_j * beta_j
        irf_val = 0.0
        # Build weight vector aligned with param_names for variance calculation
        weight_vec = np.zeros(len(param_names))

        for var_name, weight in lincom_expr.items():
            if var_name in coef_dict:
                irf_val += weight * coef_dict[var_name]
                idx = param_names.index(var_name)
                weight_vec[idx] = weight
            else:
                # Variable not found in this horizon's estimates; treat as 0
                warnings.warn(
                    f"Horizon {h}: variable '{var_name}' not found in "
                    "coefficient estimates. Treating as zero.",
                    stacklevel=2,
                )

        # Standard error of linear combination: sqrt(w' V w)
        se = np.sqrt(weight_vec @ vcov @ weight_vec)

        irf_rows.append({
            "horizon": h,
            "irf": irf_val,
            "se": se,
            "ci_lower": irf_val - z_crit * se,
            "ci_upper": irf_val + z_crit * se,
        })

    return pd.DataFrame(irf_rows)


# =============================================================================
# CONVENIENCE: PRE-DEFINED SPECIFICATIONS
# =============================================================================
# These match the specification definitions in lp.do. Each returns the xvars
# list and a dict of lincom expressions (keyed by plot label) for use with
# run_local_projection() and compute_irf().
# =============================================================================

def spec_casroles(
    lags: int = 4,
    cas: float = 0.02,
    integration: float = 0.03,
) -> tuple[list[str], dict[str, dict[str, float]]]:
    """
    Casualty-roles specification (Figure 4 of the paper).

    Compares effects across war sites, belligerents, and third parties
    using population-weighted exposure.

    Returns
    -------
    xvars : list of str
        Regressor names for run_local_projection().
    lincom_specs : dict
        Keys are plot labels ('War site', 'Belligerent', 'Third'),
        values are lincom_expr dicts for compute_irf().
    """
    xvars = [
        "regr_cas_phi_site",
        "regr_cas_psi_pop_bell",
        "regr_cas_psi_pop_third",
        "regr_cas_phi_bell",
        "regr_cas_phi_third",
    ]
    lincom_specs = {
        "War site": {
            "regr_cas_phi_site": cas,
        },
        "Belligerent": {
            "regr_cas_psi_pop_bell": cas * integration,
            "regr_cas_phi_bell": cas,
        },
        "Third": {
            "regr_cas_psi_pop_third": cas * integration,
            "regr_cas_phi_third": cas,
        },
    }
    return xvars, lincom_specs


def spec_castrd(
    lags: int = 4,
    cas: float = 0.02,
    integration: float = 0.03,
) -> tuple[list[str], dict[str, dict[str, float]]]:
    """
    Casualty-trade specification (Figure 5 of the paper).

    Examines third-party effects through the trade channel, comparing
    zero trade exposure vs. the specified integration level.

    Returns
    -------
    xvars : list of str
    lincom_specs : dict
    """
    xvars = [
        "regr_cas_phi_site",
        "regr_cas_psi_trade_bell",
        "regr_cas_phi_bell",
        "regr_cas_phi_third",
        "regr_cas_psi_trade_third",
        "regr_cas_psi_prox_third",
        "regr_cas_psi_prox_bell",
    ]
    integration_pct = integration * 100
    lincom_specs = {
        f"Third (exposure = 0%)": {
            "regr_cas_phi_third": cas,
        },
        f"Third (exposure = {integration_pct}%)": {
            "regr_cas_phi_third": cas,
            "regr_cas_psi_trade_third": cas * integration,
        },
    }
    return xvars, lincom_specs


def spec_casprox(
    lags: int = 4,
    cas: float = 0.02,
) -> tuple[list[str], dict[str, dict[str, float]]]:
    """
    Casualty-proximity specification.

    Compares third-party effects for non-neighbors vs. neighbors.

    Returns
    -------
    xvars : list of str
    lincom_specs : dict
    """
    xvars = [
        "regr_cas_phi_site",
        "regr_cas_psi_trade_bell",
        "regr_cas_phi_bell",
        "regr_cas_phi_third",
        "regr_cas_psi_trade_third",
        "regr_cas_psi_prox_third",
        "regr_cas_psi_prox_bell",
    ]
    lincom_specs = {
        "Third (non-neighbors)": {
            "regr_cas_phi_third": cas,
        },
        "Third (neighbors)": {
            "regr_cas_phi_third": cas,
            "regr_cas_psi_prox_third": cas,
        },
    }
    return xvars, lincom_specs


def spec_destroles(
    lags: int = 4,
    gprc: float = 1.975,
    integration: float = 0.03,
) -> tuple[list[str], dict[str, dict[str, float]]]:
    """
    Destruction-roles specification (GPRC-based).

    Same structure as casroles but uses the GPRC destruction measure
    instead of casualties. Typically restricted to year >= 1900.

    Returns
    -------
    xvars : list of str
    lincom_specs : dict
    """
    xvars = [
        "regr_gprc_phi_site",
        "regr_gprc_psi_pop_bell",
        "regr_gprc_psi_pop_third",
        "regr_gprc_phi_bell",
        "regr_gprc_phi_third",
    ]
    lincom_specs = {
        "War site": {
            "regr_gprc_phi_site": gprc,
        },
        "Belligerent": {
            "regr_gprc_psi_pop_bell": gprc * integration,
            "regr_gprc_phi_bell": gprc,
        },
        "Third": {
            "regr_gprc_psi_pop_third": gprc * integration,
            "regr_gprc_phi_third": gprc,
        },
    }
    return xvars, lincom_specs


def spec_desttrd(
    lags: int = 4,
    gprc: float = 1.975,
    integration: float = 0.03,
) -> tuple[list[str], dict[str, dict[str, float]]]:
    """
    Destruction-trade specification (GPRC-based).

    Examines third-party effects through trade channel using destruction
    measure. Typically restricted to year >= 1900.

    Returns
    -------
    xvars : list of str
    lincom_specs : dict
    """
    xvars = [
        "regr_gprc_phi_site",
        "regr_gprc_psi_trade_bell",
        "regr_gprc_phi_bell",
        "regr_gprc_phi_third",
        "regr_gprc_psi_trade_third",
        "regr_gprc_psi_prox_third",
        "regr_gprc_psi_prox_bell",
    ]
    integration_pct = integration * 100
    lincom_specs = {
        f"Third (exposure = 0%)": {
            "regr_gprc_phi_third": gprc,
        },
        f"Third (exposure = {integration_pct}%)": {
            "regr_gprc_phi_third": gprc,
            "regr_gprc_psi_trade_third": gprc * integration,
        },
    }
    return xvars, lincom_specs
