"""Shared paths and helpers for Greenstone, He, Jia, Liu (2020)
'Can Technology Solve the Principal-Agent Problem? Evidence from China's War
on Air Pollution' — NBER WP 27502 / paper id 125321."""

from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
PKG = ROOT / "125321-V1" / "China_Pollution_Monitoring"
DATA = PKG / "Data"
OUT = ROOT / "replication_125321" / "outputs"
OUT.mkdir(exist_ok=True, parents=True)

DEADLINE_WAVE1 = 19359  # 2013-01-01 as Stata date
DEADLINE_WAVE2 = 19724  # 2014-01-01

WEATHER_VARS = ["wind_speed", "rain", "temp", "rh"]


def load_dta(name):
    return pd.read_stata(DATA / f"{name}.dta", convert_categoricals=False)


def build_station_day():
    """Replicates Prepare_Data.do step 1: station-day panel with weather
    and station metadata."""
    poll = load_dta("pollution_1116").drop(columns=["code_city"])
    weather = load_dta("weather_1116")
    stations = load_dta("station_list")

    df = poll.merge(weather, on=["station_n", "date"], how="outer")
    df = df.rename(columns={"station_n": "pm10_n"})
    df = df.merge(stations.drop(columns=["station_lat", "station_lon"]),
                  on="pm10_n", how="left")
    return df


def residualize(df, y, fe_cols, controls):
    """Within-transform on fe_cols then partial out controls; mimic Stata's
    `reghdfe y controls, absorb(fe) res(resid)` on the non-missing sub-sample."""
    mask = df[[y] + fe_cols + controls].notna().all(axis=1)
    sub = df.loc[mask].copy()
    for fe in fe_cols:
        sub[y] = sub[y] - sub.groupby(fe)[y].transform("mean")
        for c in controls:
            sub[c] = sub[c] - sub.groupby(fe)[c].transform("mean")
    # Now OLS of y on controls (within)
    X = sub[controls].to_numpy()
    y_vec = sub[y].to_numpy()
    beta, *_ = np.linalg.lstsq(X, y_vec, rcond=None)
    sub["_resid"] = y_vec - X @ beta
    out = pd.Series(np.nan, index=df.index, dtype=float)
    out.loc[mask] = sub["_resid"].values
    return out


def format_rd(res):
    """Extract the conventional local-linear RD estimate and robust SE from
    an rdrobust result object (matches Stata rdrobust's default output)."""
    coef = res.coef.iloc[0, 0]          # Conventional
    se = res.se.iloc[0, 0]              # Conventional
    nl = int(res.N_h[0]) if hasattr(res, "N_h") else None
    nr = int(res.N_h[1]) if hasattr(res, "N_h") else None
    bw = res.bws.iloc[0, 0] if hasattr(res, "bws") else None
    return coef, se, nl, nr, bw
