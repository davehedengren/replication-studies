"""Shared paths, loaders, and helpers for replication 231881.

Paper: De los Santos, O'Brien, Wildenbeest (2025)
"Agency Pricing and Bargaining: Evidence from the E-Book Market"
"""
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
PKG = ROOT.parent / "231881-V1" / "replication"
DATA = PKG / "data"
OUT = ROOT / "output"
OUT.mkdir(exist_ok=True)

EBOOK2_0 = DATA / "ebook2_0.dta"
SALES = DATA / "sales.dta"

PUBLISHER_NAMES = {
    1: "Harper Collins",
    2: "Hachette",
    3: "Simon & Schuster",
    4: "Macmillan",
    5: "Penguin Random House",
    0: "Other publishers",
}

# Agency switch dates by publisher (from 2_ebooksbarg.do)
SWITCH_DATES = {
    1: "2015-04-15",   # Harper Collins
    2: "2015-02-01",   # Hachette
    3: "2015-01-01",   # Simon & Schuster
    4: "2015-01-05",   # Macmillan
    5: "2015-09-01",   # Penguin Random House
}


def stata_wkt(d):
    """Replicate Stata's (year-2010)*52 + week(d) encoding from 2_ebooksbarg.do.

    Stata's week() is NOT ISO week — it's floor((doy-1)/7) + 1.
    """
    d = pd.to_datetime(d)
    return (d.dt.year - 2010) * 52 + ((d.dt.dayofyear - 1) // 7 + 1).astype(int)


def stata_wkt_scalar(date_str):
    d = pd.Timestamp(date_str)
    return (d.year - 2010) * 52 + ((d.dayofyear - 1) // 7 + 1)


def load_ebook2_0():
    df = pd.read_stata(EBOOK2_0, convert_categoricals=False)
    return df


def load_sales():
    s = pd.read_stata(SALES, convert_categoricals=False)
    s["sales"] = pd.to_numeric(s["sales"], errors="coerce")
    s["salesrank"] = pd.to_numeric(s["salesrank"], errors="coerce")
    return s


def sales_rank_regression(sales_df):
    """Chevalier-Goolsbee style log-log sales/rank regression."""
    import statsmodels.api as sm
    s = sales_df.dropna(subset=["sales", "salesrank"]).copy()
    s = s[(s["sales"] > 0) & (s["salesrank"] > 0)]
    y = np.log(s["sales"])
    X = sm.add_constant(np.log(s["salesrank"]).rename("lsalesrank"))
    return sm.OLS(y, X).fit()
