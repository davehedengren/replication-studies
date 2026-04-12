"""Shared helpers for Leonardi & Moretti (2022) Milan Restaurants replication."""
from pathlib import Path
import numpy as np
import pandas as pd
import statsmodels.api as sm

ROOT = Path(__file__).resolve().parent.parent
PKG = ROOT / "170981-V1" / "submission_revised"
DATA_FINAL = PKG / "data_final.dta"
OUT = Path(__file__).resolve().parent


def load_raw_neighborhood():
    """Load the neighborhood-level dataset as saved by replication_submit.do."""
    return pd.read_stata(DATA_FINAL, convert_categoricals=False)


def build_analysis_frame(df=None):
    """Replicate the post-save transformations in replication_submit.do:
    per-capita scaling of establishment counts, growth variables, and log prices.
    """
    if df is None:
        df = load_raw_neighborhood()
    df = df.copy()

    # for var lat20*: replace X = X*1000/day_pop
    lat_cols = [c for c in df.columns if c.startswith("lat20")]
    for c in lat_cols:
        df[c] = df[c] * 1000.0 / df["day_pop"]

    # growth variables (log ratios)
    df["d1204_r"] = np.log(df["lat2012"]) - np.log(df["lat2004"])
    df["d1204_d"] = np.log(df["lat2012_d"]) - np.log(df["lat2004_d"])
    df["d1204_a"] = np.log(df["lat2012_a"]) - np.log(df["lat2004_a"])
    df["d0004_d"] = np.log(df["lat2004_d"]) - np.log(df["lat2000_d"])
    df["d0004_a"] = np.log(df["lat2004_a"]) - np.log(df["lat2000_a"])

    # logs of prices
    for c in [
        "prezzo2004Abitaz",
        "prezzo2004Negozi",
        "prezzo2012Abitaz",
        "prezzo2012Negozi",
    ]:
        df[c] = np.log(df[c])

    return df


def ols(df, y, xs):
    """OLS with constant; homoskedastic SEs (Stata default reg X Y)."""
    sub = df[[y] + list(xs)].dropna()
    X = sm.add_constant(sub[list(xs)])
    model = sm.OLS(sub[y], X).fit()
    return model
