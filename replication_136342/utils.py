"""Shared paths and helpers for replication 136342.

Caliendo, Parro, Tsyvinski (2017/2021), "Distortions and the Structure of the
World Economy". Measurement step translated from genFrictions.do.
"""
from __future__ import annotations

from pathlib import Path
import pandas as pd

BASE = Path("/Users/davehedengren/code/replication-studies/136342-V1/AEJ_Macro_replication_revision")
CPT = BASE / "CPT_measurement"
INPUT = CPT / "input"
INTERM = CPT / "intermediate"
OUT = Path("/Users/davehedengren/code/replication-studies/replication_136342")
OUT.mkdir(exist_ok=True)

WIOT_DTA = INPUT / "wiot_full.dta"

# Structural parameters used in the paper
SIGMA = 4.0
THETA = 4.0

# WIOT row/col item codes (see Readme.pdf / WIOD documentation)
#   col_item 1-35: intermediate input sectors (destination sector)
#   col_item 37,38,39,41: final demand columns (C, G, I + inventories used in paper)
#   row_item 1-35: source sectors
#   row_item 60 = total intermediate consumption; 64 = value added
SECTOR_CODES = list(range(1, 36))
FINAL_DEMAND_COLS = {37, 38, 39, 41}
DROP_FINAL_ROWS = {60, 61, 62, 63, 64, 65, 69, 99}


def load_wiot() -> pd.DataFrame:
    return pd.read_stata(WIOT_DTA)


def read_stata(name: str) -> pd.DataFrame:
    return pd.read_stata(INTERM / name)
