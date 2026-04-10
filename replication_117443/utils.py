"""Shared helpers for replication of Kueng & Yakovlev (AEJ:Pol 2020).

Paper: "The Long-Run Effects of a Public Policy on Alcohol Tastes and Mortality"
openICPSR 117443.
"""
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "117443-V1" / "Data"
OUT = Path(__file__).resolve().parent / "output"
OUT.mkdir(exist_ok=True)

CONTROLS = [
    "alcohol_intake",
    "price_beer_to_vodka",
    "logincome",
    "logincome_missing",
    "univ_educ",
    "wtself",
    "health_evaluation",
    "married",
    "seventeen_before_1970",
]


def load_base_sample():
    df = pd.read_stata(DATA / "base_sample_aej.dta", convert_categoricals=False)
    # DiD indicators per 02_Gorbachev.do
    age17 = df["birthy"] + 17
    df["gorbachev"] = ((age17 >= 1986) & (age17 <= 1990)).astype(int)
    df["beforegorbachev"] = (age17 < 1986).astype(int)
    df["aftergorbachev"] = ((age17 > 1990) & age17.notna()).astype(int)
    df["rural_gorbachev"] = df["rural"] * df["gorbachev"]
    df["urban_beforegorbachev"] = df["urban"] * df["beforegorbachev"]
    df["urban_aftergorbachev"] = df["urban"] * df["aftergorbachev"]
    df["seventeen_before_1970"] = (age17 < 1970).astype(int)

    # Drop alcohol non-consumers (share == NaN for any of the six categories)
    for var in ["beer", "vodka", "samogon", "dwine", "fwine", "other"]:
        df = df[df[f"share_{var}"].notna()]

    # Drop inconsistent birth-place and rural->urban movers
    df = df[~((df["bptype_max"] >= 2) & (df["bptype_min"] == 1))]
    df = df[~((df["city_max"] == 1) & (df["city_min"] == 0))]
    return df.reset_index(drop=True)


def table2_sample(df, year_min=2001, age_min=18, age_max=65):
    """Age 18-65, keep only years >= year_min."""
    return df[
        (df["year"] >= year_min)
        & (df["age"] >= age_min)
        & (df["age"] <= age_max)
    ].reset_index(drop=True)


def impose_common_sample(df, yvar, regressors):
    """Replicate Stata `e(sample)` trick: run the largest spec first, then
    restrict subsequent regressions to rows where it had non-missing values."""
    cols = [yvar] + regressors
    mask = df[cols].notna().all(axis=1)
    return df[mask].reset_index(drop=True)
