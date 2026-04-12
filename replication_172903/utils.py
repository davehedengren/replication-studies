"""Shared paths and helpers for replication of 172903 (List, Petrie, Samek 2021 JEL)."""
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
PKG = ROOT / "172903-V1" / "JEL Replication_11.18.22"
INPUT = PKG / "input"
OUT = Path(__file__).resolve().parent / "output"
OUT.mkdir(parents=True, exist_ok=True)

PAPERS_DTA = INPUT / "papers_data.dta"
EXPER_DTA = INPUT / "experimental_dataset.dta"
TABLE_XLSX = PKG / "Table1_A1.xlsx"


def load_papers() -> pd.DataFrame:
    return pd.read_stata(PAPERS_DTA, convert_categoricals=False)


def load_experimental() -> pd.DataFrame:
    return pd.read_stata(EXPER_DTA, convert_categoricals=False)


def first_nonmissing(df: pd.DataFrame, cols) -> pd.Series:
    """Return the first non-missing value across `cols` for each row.

    Mirrors the Stata idiom:
        gen x = a
        replace x = b if x==. & b!=.
        replace x = c if x==. & c!=.
    """
    out = df[cols[0]].copy()
    for c in cols[1:]:
        out = out.where(out.notna(), df[c])
    return out


def cut_bins():
    """The histogram bin cuts used by the paper for time/risk/dictator categories."""
    return [0, 0.01, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.99, 1.01]


def egen_cut(s: pd.Series, bins) -> pd.Series:
    """Replicate Stata `egen cut(var), at(...)`: left-closed right-open bins,
    labeled by the lower edge of each bin. Values outside the range are NaN."""
    import numpy as np
    s = s.astype(float)
    out = pd.Series(np.nan, index=s.index, dtype="float64")
    for lo, hi in zip(bins[:-1], bins[1:]):
        mask = s.notna() & (s >= lo) & (s < hi)
        out.loc[mask] = lo
    return out
