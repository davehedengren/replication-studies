"""Shared paths and helpers for replication 127341.

Fajgelbaum, Khandelwal, Kim, Mantovani & Schaal,
"Optimal Lockdown in a Commuting Network" (AER:Insights 2021).
"""
from pathlib import Path
import pandas as pd

PKG = Path("/Users/davehedengren/code/replication-studies/127341-V1/replication")
WORK = PKG / "data" / "work"
RAW = PKG / "data" / "raw"
RESULTS = PKG / "results" / "101"
OUT = Path(__file__).parent

# Published values (from results/101/*.tex)
PUB = {
    "kappa_sigma": 1.53,        # (sigma-1) * kappa_1  (distance elasticity)
    "kappa_sigma_se": 0.066,
    "epsilon_sigma": 0.45,      # (sigma-1) * epsilon   (commuting elasticity)
    "epsilon_sigma_se": 0.067,
    "sigma": 5,
    "own_share_pct": 55,        # Seoul own-district spending share, Mar 2020
    # Implied from (sigma-1)=4
    "kappa1": 1.53 / 4,
    "epsilon": 0.45 / 4,
}

def _stata_date(series: pd.Series) -> pd.Series:
    """Convert Stata %td integer days (days since 1960-01-01) to datetime."""
    return pd.to_datetime(series, unit="D", origin="1960-01-01")


def load_sel_cc() -> pd.DataFrame:
    """Seoul district-to-district daily credit-card expenditures."""
    df = pd.read_stata(WORK / "SEL_CC.dta", convert_categoricals=False)
    df["statadate"] = _stata_date(df["statadate"])
    return df


def load_sel_parms() -> pd.DataFrame:
    """Seoul OD parameter file with yod2 = chi (commute flow ratio)."""
    df = pd.read_stata(WORK / "SEL_parms_od.dta", convert_categoricals=False)
    # statadate already datetime in this file
    return df

def load_commute(cty: str) -> pd.DataFrame:
    """Bilateral commute distances. cty in {'KOR','NYM'}."""
    fn = "KOR_commute_obs.dta" if cty == "KOR" else "NYM_commute_obs.dta"
    return pd.read_stata(WORK / fn, convert_categoricals=False)
