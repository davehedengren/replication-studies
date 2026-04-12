"""Shared paths and helpers for replication of Halloran, Jack, Okun, Oster (2021).

Pandemic Schooling Mode and Student Test Scores: Evidence from US States.
NBER Working Paper 29497.
"""
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
PKG = ROOT.parent / "168843-V1"
DATA_CLEAN = PKG / "Data" / "Clean"
OUT = ROOT / "output"
OUT.mkdir(exist_ok=True)

STATE_SCORE_FILE = DATA_CLEAN / "state_score_data.dta"

# Controls matching analysis.do exhibit_4
CONTROLS = [
    "share_white", "share_black", "share_hisp",
    "missing_lunch", "share_lunch_updated",
    "missing_ELL", "share_ELL_updated",
    "unemployment", "participation", "participate_denom",
]


def load_state_scores():
    """Load district-year-subject panel used in main analysis."""
    df = pd.read_stata(STATE_SCORE_FILE, convert_categoricals=False)
    df = df.copy()
    df["state_gr"] = df["state"].astype("category").cat.codes
    df["district_unique"] = (
        df["state_gr"].astype(str) + "_" + df["district_id"].astype(str)
    )
    df["year"] = df["year"].astype(int)
    return df


def apply_main_treatment_replacement(df):
    """Replicate analysis.do: for year<2021 set share_inperson=1, share_hybrid=0, share_virtual=0.

    This is the main 'treatment' coding: pre-pandemic years are treated as
    fully in-person, so share_inperson in 2021 captures deviation from that.
    """
    df = df.copy()
    pre = df["year"] < 2021
    df.loc[pre, "share_inperson"] = 1.0
    df.loc[pre, "share_hybrid"] = 0.0
    df.loc[pre, "share_virtual"] = 0.0
    return df


def weighted_mean(x, w):
    x = np.asarray(x, dtype=float)
    w = np.asarray(w, dtype=float)
    mask = ~np.isnan(x) & ~np.isnan(w)
    if mask.sum() == 0:
        return np.nan
    return np.sum(x[mask] * w[mask]) / np.sum(w[mask])
