"""Paths and loaders for replication 151521 (Cai & Heathcote 2020)."""
from pathlib import Path
import pandas as pd
import numpy as np
import pyreadstat

ROOT = Path(__file__).resolve().parent
DATA = ROOT.parent / "151521-V1"
OUT = ROOT / "output"
OUT.mkdir(exist_ok=True)


def load_mostrecent():
    """Load the College Scorecard CSV that `main.do` insheets.

    The shipped mostrecent.dta is a post-processed snapshot containing only
    private nonprofit colleges (control==2 for all rows) and with `faminc`
    blanked out, so it cannot reproduce Table 2. We use mostrecent.csv directly
    and lowercase column names to match main.do's naming.
    """
    df = pd.read_csv(DATA / "mostrecent.csv", low_memory=False, encoding="latin-1")
    df.columns = [c.lower() for c in df.columns]
    # force-numeric everything that isn't clearly text, mimicking Stata
    # "destring, replace force ignore(,)"
    for c in df.columns:
        if df[c].dtype == object:
            df[c] = pd.to_numeric(df[c].astype(str).str.replace(",", "", regex=False),
                                  errors="coerce")
    return df


def load_mrc_merged():
    df, _ = pyreadstat.read_dta(str(DATA / "mrc_merged.dta"))
    return df


def load_figure1():
    raw = pd.read_excel(DATA / "Figure_1_data.xlsx", sheet_name="FIGURE 1 in PAPER", header=None)
    # cols 3..7 are: year, public sticker, public net, private sticker, private net
    df = raw.iloc[1:, [3, 4, 5, 6, 7]].copy()
    df.columns = ["year", "public_sticker", "public_net", "private_sticker", "private_net"]
    df["year"] = df["year"].astype(str).str.strip()
    for c in ["public_sticker", "public_net", "private_sticker", "private_net"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df.reset_index(drop=True)


def load_scf_tabulated(year):
    """Load the tabulated income distribution (income, count) from the SCF sheet.

    The 'raw 2016' / 'raw 1989' sheets are SDA-style frequency tables with columns
    Bin center -> frequency weight. We find the numeric block robustly.
    """
    sheet = "raw 2016" if year == 2016 else "raw 1989"
    raw = pd.read_excel(DATA / "SCF_AER_submit.xlsx", sheet_name=sheet, header=None)
    # Scan for a 'Bin' / N-style header row; locate first column of numeric pairs.
    # Based on inspection, the table has columns: bin_idx, income (row_center), count
    # Use the 2016/1989 processed tabs instead — they have a clean (income, count) form.
    proc = pd.read_excel(DATA / "SCF_AER_submit.xlsx", sheet_name=str(year), header=None)
    # row 5 begins the data (0,1,2 = bin, income, count)
    tab = proc.iloc[5:, [1, 2]].copy()
    tab.columns = ["income", "count"]
    tab["income"] = pd.to_numeric(tab["income"], errors="coerce")
    tab["count"] = pd.to_numeric(tab["count"], errors="coerce")
    tab = tab.dropna()
    return tab.reset_index(drop=True)


def weighted_mean(x, w):
    w = np.asarray(w, dtype=float)
    x = np.asarray(x, dtype=float)
    return np.sum(x * w) / np.sum(w)


def weighted_std(x, w):
    w = np.asarray(w, dtype=float)
    x = np.asarray(x, dtype=float)
    m = weighted_mean(x, w)
    v = np.sum(w * (x - m) ** 2) / np.sum(w)
    return np.sqrt(v)


def weighted_corr(x, y, w):
    w = np.asarray(w, dtype=float)
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    mx, my = weighted_mean(x, w), weighted_mean(y, w)
    sx = np.sqrt(np.sum(w * (x - mx) ** 2) / np.sum(w))
    sy = np.sqrt(np.sum(w * (y - my) ** 2) / np.sum(w))
    cov = np.sum(w * (x - mx) * (y - my)) / np.sum(w)
    return cov / (sx * sy)
