"""Shared paths and helpers for Cai, Li & Santacreu (2022) replication."""
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parent
PKG = ROOT.parent / "120281-V1" / "CLS_ReplicationPackage" / "CLS_ReplicationPackage_v1"
RAW = PKG / "Raw_data"
RESULTS = PKG / "Results"
OUT = ROOT / "output"
OUT.mkdir(exist_ok=True)

COUNTRIES = [
    "AUS", "AUT", "BEL", "CAN", "DEU", "ESP", "FIN", "FRA", "GBR",
    "IRL", "ISR", "ITA", "JPN", "KOR", "NLD", "NOR", "NZL", "PRT", "USA",
]
OMIT_CODE = 15  # NLD — matches Stata `char exp[omit] 15`
N_TRADABLE = 18  # sectors 1..18; sector 19 = nontradable

DIST_BINS = ["d0_375", "d375_750", "d750_1500", "d1500_3000", "d3000_6000", "d6000_"]
OTHER_VARS = ["contig", "FTA", "CU"]


def load_trade():
    df = pd.read_stata(RAW / "trade19_00.dta", convert_categoricals=False)
    return df
