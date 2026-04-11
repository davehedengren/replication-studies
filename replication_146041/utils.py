"""Shared paths, sample lists, and helpers for the Rossi (2020) replication."""
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
PKG = ROOT / "146041-V1" / "Replication"
TEMP = PKG / "temp"
DATA = PKG / "data"
OUT = Path(__file__).resolve().parent / "outputs"
OUT.mkdir(exist_ok=True)

MICRO_COUNTRIES = [
    "Brazil", "Canada", "India", "Indonesia", "Israel", "Jamaica",
    "Mexico", "Panama", "Trinidad and Tobago", "United States",
    "Uruguay", "Venezuela",
]

SIGMA = 1.5
SIGMA_HIGH = 2.0
SIGMA_LOW = 1.3
MINC = 0.1


def load_calib():
    """Load calibration file and merge with misc (country-level macro data)."""
    calib = pd.read_stata(TEMP / "calib_all_1990_2010.dta", convert_categoricals=False)
    misc = pd.read_stata(DATA / "misc" / "misc.dta", convert_categoricals=False)
    # amend year to 2000 for near-2000 samples (from AQ_main.do lines 16-17)
    remap = {
        "Canada 2001", "India 1999", "Indonesia 1995", "Israel 1995",
        "Jamaica 2001", "Uruguay 2006", "Venezuela 2001",
    }
    calib["year_orig"] = calib["year"]
    calib.loc[calib["sample"].isin(remap), "year"] = 2000
    df = calib.merge(misc, on=["country", "year"], how="outer", indicator=True)
    # keep matches + country-years from misc with year==2000 and l_y present (from AQ_main.do)
    df = df[(df["_merge"] == "both") | ((df["_merge"] == "right_only") & (df["year"] == 2000) & df["l_y"].notna())]
    df = df.drop(columns="_merge")
    # impute missing edu durations
    for c, v in [("dur_noedu", 0), ("dur_pri", 6), ("dur_sec", 6), ("dur_ter", 4),
                 ("dur_pri_inc", 3), ("dur_sec_inc", 3), ("dur_ter_inc", 2),
                 ("age_end_noedu", 6)]:
        df[c] = df[c].fillna(v)
    df["yrs_1"] = df["dur_pri_inc"]
    df["yrs_2"] = df["dur_pri"] + df["dur_sec_inc"]
    df["yrs_3"] = df["dur_pri"] + df["dur_sec"]
    df["yrs_4"] = df["dur_pri"] + df["dur_sec"] + df["dur_ter_inc"]
    df["yrs_5"] = df["dur_pri"] + df["dur_sec"] + df["dur_ter"]
    df["sample_micro"] = df["country"].isin(MICRO_COUNTRIES).astype(int)
    return df


def compute_wage_ratios(df, spec="dum_skti_secall"):
    """wrat_e3 = exp(l_w_e - l_w_3) for e in 1..5."""
    out = df.copy()
    for e in range(1, 6):
        out[f"wrat{e}3_{spec}"] = np.exp(out[f"l_w{e}_{spec}"])
    return out


def compute_H5L3(df, spec, share_suffix):
    """Relative labor stock H_5/L_3 using share{e}_{share_suffix}."""
    H5 = pd.Series(0.0, index=df.index)
    for e in [4, 5]:
        H5 = H5 + (df[f"wrat{e}3_{spec}"] / df[f"wrat53_{spec}"]) * df[f"share{e}_{share_suffix}"]
    L3 = pd.Series(0.0, index=df.index)
    for e in [1, 2, 3]:
        L3 = L3 + (df[f"wrat{e}3_{spec}"] / df[f"wrat33_{spec}"]) * df[f"share{e}_{share_suffix}"]
    return H5, L3, H5 / L3


def compute_irAQ(df, wrat53, H5L3, sigma, us_mask):
    """Relative skill efficiency (irAQ53), normalized to US 2000 = 1."""
    AQ = (wrat53 ** (sigma / (sigma - 1))) * (H5L3 ** (1.0 / (sigma - 1)))
    us_val = AQ[us_mask].mean()
    return AQ / us_val


def ols_log_elasticity(df, yvar, xvar="l_y"):
    """OLS of log(yvar) on log GDP p.w. Returns (coef, se, n)."""
    sub = df[[yvar, xvar]].dropna()
    sub = sub[sub[yvar] > 0]
    y = np.log(sub[yvar].values)
    x = sub[xvar].values
    X = np.column_stack([np.ones(len(x)), x])
    beta = np.linalg.lstsq(X, y, rcond=None)[0]
    resid = y - X @ beta
    n, k = X.shape
    s2 = (resid @ resid) / (n - k)
    cov = s2 * np.linalg.inv(X.T @ X)
    se = np.sqrt(np.diag(cov))
    return float(beta[1]), float(se[1]), int(n)
