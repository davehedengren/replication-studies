"""Shared paths and helpers for replication of Bursztyn et al. (2022),
'Scapegoating During Crises' (AEA P&P, May 2022)."""

from pathlib import Path
import re
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
PKG = ROOT.parent / "164561-V1" / "replication_PP"
RAW = PKG / "raw" / "study1.sav"
OUTDIR = ROOT / "output"
OUTDIR.mkdir(parents=True, exist_ok=True)

RACISM_STEMS = ["xenophob", "racis", "intoler", "bias", "bigot"]
LABOR_STEMS = ["labor", "job", "unemploy", "work"]

CONTROLS = [
    "male", "college", "white", "hispanic",
    "northeast", "midwest", "west", "fulltime", "ln_income",
]


def clean_text(s):
    if pd.isna(s):
        return ""
    s = str(s).lower()
    # strip punctuation / non-alphanumeric, collapse whitespace
    s = re.sub(r"[^a-z0-9\s]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def contains_any_stem(text, stems):
    if not text:
        return 0
    return int(any(stem in text for stem in stems))


def load_clean():
    """Return the cleaned analysis sample, mirroring clean_study1.do."""
    import pyreadstat
    df, _ = pyreadstat.read_sav(str(RAW))
    df.columns = [c.lower() for c in df.columns]

    df = df.rename(columns={"prolific_pid": "prolific_pid", "duration__in_seconds_": "time_survey"})

    # drop missing prolific pid (none in practice)
    df = df.dropna(subset=["prolific_pid"])
    df["prolific_pid"] = df["prolific_pid"].astype(str)

    # sort by pid + startdate, keep first
    df = df.sort_values(["prolific_pid", "startdate"]).drop_duplicates(
        subset=["prolific_pid"], keep="first"
    )

    # consent
    df = df[df["consent"] == 1]

    # attention check: original code drops if attention != 1.
    # In SPSS import, blank -> NaN; "1" -> "1". Keep only attention == "1".
    df["attention"] = df["attention"].astype(str).str.strip()
    df = df[df["attention"] == "1"]

    # drop if missing treatment
    df["treatment"] = df["treatment"].astype(str).str.strip()
    df = df[df["treatment"].isin(["excuse", "noexcuse"])]

    # drop vote in {3,4}
    df = df[~df["vote"].isin([3, 4])]

    df = df.reset_index(drop=True)

    # party dummies
    df["republican"] = (df["pol"] == 1).astype(int)
    df["democrat"] = (df["pol"] == 2).astype(int)
    df["independent"] = 0  # matches Stata code which sets Independent = 0

    df["liberal"] = df["liberal"].apply(lambda x: 1 if x in (1, 2) else (0 if pd.notna(x) else np.nan))

    df["college"] = np.where(df["educ"].notna(), (df["educ"] > 4).astype(int), np.nan)
    df["white"] = np.where(df["race"].notna(), (df["race"] == 3).astype(int), np.nan)
    df["black"] = np.where(df["race"].notna(), (df["race"] == 1).astype(int), np.nan)
    df["fulltime"] = np.where(df["employment"].notna(), (df["employment"] == 1).astype(int), np.nan)

    income_midpoint_map = {1: 7500, 2: 20000, 3: 37500, 4: 62500, 5: 82500, 6: 125000, 7: 175000, 8: 22500}
    df["income_midpoint"] = df["inc"].map(income_midpoint_map)
    df["ln_income"] = np.log(df["income_midpoint"])
    df = df.rename(columns={"inc": "income"})

    df["vote_trump"] = df["vote"].map({1: 1, 2: 0})

    # region dummies
    df["northeast"] = (df["region"] == 1).astype(int)
    df["midwest"] = (df["region"] == 2).astype(int)
    df["south"] = (df["region"] == 3).astype(int)
    df["west"] = (df["region"] == 4).astype(int)

    # treatment indicator: Stata code makes Tr=1 for "excuse" (After), 0 for "noexcuse" (Before)
    df["Tr"] = (df["treatment"] == "excuse").astype(int)

    # text features: clean, then substring-match with stems
    df["open_ended"] = df["open_ended"].fillna("")
    df["story_cleaned"] = df["open_ended"].apply(clean_text)
    df["contain_racism_terms"] = df["story_cleaned"].apply(
        lambda t: contains_any_stem(t, RACISM_STEMS)
    )
    df["contain_labor_terms"] = df["story_cleaned"].apply(
        lambda t: contains_any_stem(t, LABOR_STEMS)
    )

    return df
