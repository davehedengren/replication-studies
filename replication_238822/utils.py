"""Shared paths and helpers for replication of Aman-Rana & Minaudier (2025)."""

from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
PKG = ROOT.parent / "238822-V1"
DATA = PKG / "Data" / "Analysis"
OUT = ROOT / "output"
OUT.mkdir(exist_ok=True)


def load(name):
    return pd.read_stata(DATA / name, convert_categoricals=False)


def twfe(df, y, x, unit, time, cluster=None, add_const=True):
    """OLS with two-way FE via dummies. Returns statsmodels RegressionResults.

    Uses cov_type='cluster' with Stata-style small sample adjustment.
    """
    import statsmodels.api as sm

    d = df[[y, x, unit, time] + ([cluster] if cluster and cluster not in (unit, time) else [])].dropna()
    unit_d = pd.get_dummies(d[unit], prefix=unit, drop_first=True).astype(float)
    time_d = pd.get_dummies(d[time], prefix=time, drop_first=True).astype(float)
    X = pd.concat([d[[x]].astype(float), unit_d, time_d], axis=1)
    if add_const:
        X = sm.add_constant(X)
    if cluster is None:
        cluster = unit
    g = d[cluster].values
    model = sm.OLS(d[y].astype(float).values, X.values)
    res = model.fit(cov_type="cluster", cov_kwds={"groups": g})
    return res, len(d)


def fmt(b, se, n):
    return f"beta={b:.4f}  se={se:.4f}  N={n}"
