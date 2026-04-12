"""Shared paths and helpers for Sampson (2023) 'Technology Gaps, Trade and Income' replication."""
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parent
RAW = ROOT.parent / "182522-V1" / "data" / "raw"
OUT = ROOT / "output"
OUT.mkdir(exist_ok=True)

# 20 ISIC Rev 4 manufacturing industries used in R&D efficiency calculation
# (matching ind filter in 02_rd_efficiency.do lines 17)
MANUF_INDS = [
    "D10T12", "D13", "D14", "D15", "D16", "D17", "D18", "D19", "D20",
    "D21", "D22", "D23", "D24", "D25", "D26", "D27", "D28", "D29", "D30", "D31T33",
]

# 25 OECD countries in baseline sample (Section 4.2 of paper)
BASELINE_COUS = [
    "AUS", "AUT", "BEL", "CAN", "CHL", "CZE", "DEU", "DNK", "ESP", "FIN",
    "FRA", "GBR", "HUN", "IRL", "ITA", "JPN", "KOR", "MEX", "NLD", "NOR",
    "POL", "PRT", "SVN", "TUR", "USA",
]

# Country code -> human name (informational)
COU_NAMES = {
    "AUS": "Australia", "AUT": "Austria", "BEL": "Belgium", "CAN": "Canada",
    "CHL": "Chile", "CZE": "Czech Republic", "DEU": "Germany", "DNK": "Denmark",
    "ESP": "Spain", "FIN": "Finland", "FRA": "France", "GBR": "United Kingdom",
    "HUN": "Hungary", "IRL": "Ireland", "ITA": "Italy", "JPN": "Japan",
    "KOR": "Korea", "MEX": "Mexico", "NLD": "Netherlands", "NOR": "Norway",
    "POL": "Poland", "PRT": "Portugal", "SVN": "Slovenia", "TUR": "Turkey",
    "USA": "United States",
}

# Duplicate-reporter rules from 02_rd_efficiency.do lines 22-28:
# Belgium, France, UK: keep product field (drop _MA)
# Czech, Finland, Italy, Portugal: keep main activity (drop _PF)
DROP_SUFFIX = {"BEL_MA", "CZE_PF", "FIN_PF", "FRA_MA", "GBR_MA", "ITA_PF", "PRT_PF"}


def load_anberd():
    """Load OECD ANBERD R&D expenditure, filtered to 20 manufacturing industries.

    Mirrors 02_rd_efficiency.do section A:
      - keep CUR=='NATCUR'
      - keep 20 ISIC Rev 4 manufacturing industries
      - drop duplicate reporter series per paper's selection rules
      - collapse COU code to 3 letters
    """
    df = pd.read_csv(RAW / "ANBERD_REV4.csv", low_memory=False)
    df = df[df["CUR"] == "NATCUR"].copy()
    df = df[df["IND"].isin(MANUF_INDS)].copy()
    df = df[~df["COU"].isin(DROP_SUFFIX)].copy()
    df["cou"] = df["COU"].str[:3]
    df = df.rename(columns={"Value": "rnd", "TIME": "year"})
    return df[["cou", "IND", "year", "rnd"]].rename(columns={"IND": "ind"})


def load_stan():
    """Load OECD STAN value-added (current national currency), manufacturing industries.

    Value is in millions (per STAN convention). Matches 02_rd_efficiency.do section A.
    """
    df = pd.read_csv(RAW / "STANI4_2016.csv", low_memory=False)
    df = df[df["TIME"] >= 1987]
    df = df[df["VAR"] == "VALU"].copy()
    df = df.rename(columns={"Value": "value_added", "LOCATION": "cou", "TIME": "year"})
    return df[["cou", "IND", "year", "value_added"]].rename(columns={"IND": "ind"})


def compute_rd_intensity():
    """Merge ANBERD and STAN to compute R&D intensity and its log.

    R&D intensity = rnd / (1_000_000 * value_added), where STAN VA is in millions.
    Returns a long-format DataFrame with one row per (cou, ind, year).
    """
    rd = load_anberd()
    va = load_stan()
    m = rd.merge(va, on=["cou", "ind", "year"], how="inner")
    m["rd_intensity"] = m["rnd"] / (1_000_000.0 * m["value_added"])
    m = m[m["rd_intensity"].notna() & (m["rd_intensity"] > 0)].copy()
    import numpy as np
    m["lrd_intensity"] = np.log(m["rd_intensity"])
    return m
