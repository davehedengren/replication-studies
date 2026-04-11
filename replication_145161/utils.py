"""Shared paths + helpers for Dinkelman & Ngai (2022) JEP replication."""
from pathlib import Path
import pandas as pd
import numpy as np

ROOT = Path("/Users/davehedengren/code/replication-studies/145161-V1/DinkelmanNgai2021Data")
DATA = ROOT / "data"
SYNTAX = ROOT / "syntax"

OUT = Path("/Users/davehedengren/code/replication-studies/replication_145161/output")
OUT.mkdir(parents=True, exist_ok=True)

# 11 African countries for Figure 1
AFRICA11 = ["BWA", "ETH", "GHA", "KEN", "MWI", "MUS", "NGA", "SEN", "TZA", "ZAF", "ZMB"]

# Bridgman subset for Figure 2B
FIG2B_COUNTRIES = [
    ("Uganda", 2005),
    ("Tanzania", 2014),
    ("Ghana", 2009),
    ("South Africa", 2000),
    ("South Africa", 2010),
    ("Algeria", 2012),
]


def read_dta(path):
    """Read a Stata file without converting categoricals."""
    return pd.read_stata(path, convert_categoricals=False)


def wmean(x, w):
    x = np.asarray(x, dtype="float64")
    w = np.asarray(w, dtype="float64")
    m = ~np.isnan(x) & ~np.isnan(w) & (w > 0)
    if m.sum() == 0:
        return np.nan
    return float(np.sum(x[m] * w[m]) / np.sum(w[m]))
