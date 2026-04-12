"""Shared helpers for replication of Antman, Duncan, Trejo (JEP 2023, openICPSR 183164).

Data sources: IPUMS-USA 1970-2019 Census/ACS and IPUMS-CPS 2003-2019 monthly.
Cleaned Stata files under 183164-V1/data/analysis/ are used directly to avoid
re-running the (slow) variable construction; the source .do files are mirrored
in the loaders below so the logic is documented.
"""
from pathlib import Path
import numpy as np
import pandas as pd
import pyreadstat

REPO = Path(__file__).resolve().parents[1]
PKG  = REPO / "183164-V1"
DATA_ANALYSIS = PKG / "data" / "analysis"
DATA_COEF = PKG / "data" / "coef_data"
RESULTS = PKG / "results"
OUT = Path(__file__).resolve().parent / "out"
OUT.mkdir(exist_ok=True)

CENSUS_DTA = DATA_ANALYSIS / "Census1970-2000-ACS2006-2019.dta"
CPS_DTA    = DATA_ANALYSIS / "CPS2003-2019.dta"


# ---- variable lists kept small so we can fit the 4.4 GB file in memory ---- #
CENSUS_COLS = [
    "year", "sample", "age", "sex", "perwt", "bpl", "statefip",
    "race", "raced", "hispan", "hispand", "hisp1970",
    "educ", "educd", "incwage", "incfarm", "incbus", "incbus00",
]

CPS_COLS_BASE = [
    "year", "age", "sex", "mish", "hhintype", "wtfinl", "marst", "race", "hispan",
    "bpl", "citizen", "mbpl", "fbpl", "educ",
    "age_sp", "race_sp", "hispan_sp",
]


def load_census(cols=None):
    """Load the cleaned 1970-2019 Census/ACS data as a DataFrame.

    The .dta file already has all derived variables from the authors' do-file;
    we just rebuild the handful we use on top to keep the loader transparent.
    """
    path = CENSUS_DTA
    # pyreadstat is ~3x faster than pd.read_stata for this file
    df, _ = pyreadstat.read_dta(str(path), disable_datetime_conversion=True)
    if cols is not None:
        df = df[cols]
    return df


def load_cps(cols=None):
    df, _ = pyreadstat.read_dta(str(CPS_DTA), disable_datetime_conversion=True)
    if cols is not None:
        df = df[cols]
    return df


# ---------- sample filters matching the author do-files ---------- #
def restrict_table12_sample(df):
    """2019 ACS, age 25-59, FB+USB Hispanics + USB white + USB black."""
    d = df[df["sample"] == 201901].copy()
    d = d[(d["age"] >= 25) & (d["age"] <= 59)]
    d = d[(d["hisp"] == 1) | ((d["white"] == 1) & (d["USBorn"] == 1)) |
          ((d["black"] == 1) & (d["USBorn"] == 1))]
    return d


def restrict_figure1_sample(df, year):
    d = df[df["sample"].isin([197001, 198001, 199001, 200001, 201001, 201901])].copy()
    d = d[(d["age"] >= 25) & (d["age"] <= 59)]
    d = d[d["USBorn"] == 1]
    d = d[(d["hisp"] == 1) | (d["white"] == 1) | (d["black"] == 1)]
    d = d[d["sex"] == 1]              # men
    d = d[d["year"] == year]
    return d
