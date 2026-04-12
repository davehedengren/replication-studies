"""Helpers for the Bandiera et al. (2022) JEP replication."""
from pathlib import Path
import pandas as pd
import numpy as np

ROOT = Path(__file__).resolve().parent
DATA = ROOT.parent / "154661-V1" / "data_replication"
OUT = ROOT / "output"
OUT.mkdir(exist_ok=True)


def load(name: str) -> pd.DataFrame:
    return pd.read_stata(DATA / f"{name}.dta", convert_categoricals=False)


def load_labelled(name: str) -> pd.DataFrame:
    return pd.read_stata(DATA / f"{name}.dta")


def mark_africa(df: pd.DataFrame) -> pd.DataFrame:
    """Add an `africa` indicator matching the paper: region==SSA OR country in {Morocco, Egypt}."""
    df = df.copy()
    reg = df["region"]
    if reg.dtype.name in ("category", "object"):
        reg_str = reg.astype(str)
        is_ssa = reg_str.str.contains("SSA", case=False, na=False) | reg_str.str.contains(
            "Sub-Saharan", case=False, na=False
        )
    else:
        is_ssa = reg == 6
    country_col = "country_string" if "country_string" in df.columns else "country_str"
    extra = df[country_col].isin(["Morocco", "Egypt"])
    df["africa"] = (is_ssa | extra).astype(int)
    return df


def prep_pooled_age18to24() -> pd.DataFrame:
    """Replicates the Stata sample filter in 1_main_figures.do for figure 2 / text stats."""
    df = load("pooled_age18to24")
    df = mark_africa(df)
    keepvars = [
        "paidwk_all",
        "unpaidwk_all",
        "inwork",
        "self_paid_agri",
        "self_paid_noagri",
        "employee_paid_agri",
        "employee_paid_noagri",
        "lgdppc",
    ]
    df = df.dropna(subset=keepvars)
    df = df[df["country_string"] != "Venezuela"]
    dup = df.duplicated(subset=["countrycode", "year", "age18to24"], keep=False)
    df = df[~(dup & (df["source"].astype(str) == "DHS"))]
    df["maxy"] = df.groupby("country_string")["year"].transform("max")
    df = df[df["year"] == df["maxy"]].copy()
    df["SE_agri"] = df["self_paid_agri"] + df["unpaidwk_agri"]
    df["SE_noagri"] = df["self_paid_noagri"] + df["unpaidwk_noagri"]
    df["outofLF"] = 1 - df["paidwk_all"] - df["unpaidwk_all"]
    return df


def prep_pooled_cross_country() -> pd.DataFrame:
    df = load("pooled_cross_country")
    df = mark_africa(df)
    df = df.dropna(subset=["paidwk_all", "unpaidwk_all", "inwork", "self_paid", "sector_agri", "lgdppc"])
    df = df[df["country_string"] != "Venezuela"]
    dup = df.duplicated(subset=["countrycode", "year"], keep=False)
    df = df[~(dup & (df["source"].astype(str) == "DHS"))]
    df["maxy"] = df.groupby("country_string")["year"].transform("max")
    df = df[df["year"] == df["maxy"]].copy()
    return df


def weighted_mean(x: pd.Series, w: pd.Series) -> float:
    w = w.astype(float)
    return float((x * w).sum() / w.sum())
