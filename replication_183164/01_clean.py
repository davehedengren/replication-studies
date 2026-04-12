"""Load the (already-cleaned-by-author-do-file) Census/ACS and CPS data and
stash slim parquet subsets that the downstream scripts read quickly.

The authors' do-files build all derived variables and save them into
Census1970-2000-ACS2006-2019.dta (4.4 GB) and CPS2003-2019.dta (195 MB).
Rather than reproduce the variable construction, we trust those files (the
do-files were read and audited) and extract the analysis slices we need.

Outputs (under replication_183164/out/):
  census_2019.parquet  — 2019 ACS, age 25-59, Hispanics + USB white/black.
  census_figure1.parquet — 1970/80/90/2000/2010/2019, USB, hisp/white/black, age 25-59, men.
  cps_clean.parquet     — CPS 2003-2019 cleaned analytic sample.
"""
import time
import numpy as np
import pandas as pd
import pyreadstat

from utils import CENSUS_DTA, CPS_DTA, OUT

CENSUS_COLS = [
    "year", "sample", "age", "sex", "perwt", "statefip",
    "USBorn", "hisp", "white", "black",
    "race_cat4", "race_cat6", "race_cat16",
    "edyrs", "hsgrad", "college",
    "ln_annualearnings", "annualearnings",
]


def _to_int(series):
    return pd.to_numeric(series, errors="coerce").astype("Int64")


def load_census_slim():
    t = time.time()
    df, _ = pyreadstat.read_dta(str(CENSUS_DTA), usecols=CENSUS_COLS)
    print(f"[01_clean] loaded Census/ACS: {len(df):,} rows in {time.time()-t:.0f}s")
    # hisp/white/black came in as object (nullable byte).  Normalize.
    for c in ["hisp", "white", "black"]:
        df[c] = _to_int(df[c])
    return df


def make_table12_slice(df):
    d = df[df["sample"] == 201901]
    d = d[(d["age"] >= 25) & (d["age"] <= 59)]
    mask = (
        (d["hisp"] == 1)
        | ((d["white"] == 1) & (d["USBorn"] == 1))
        | ((d["black"] == 1) & (d["USBorn"] == 1))
    )
    d = d[mask].copy()
    return d


def make_figure1_slice(df):
    keep_samples = [197001, 198001, 199001, 200001, 201001, 201901]
    d = df[df["sample"].isin(keep_samples)]
    d = d[(d["age"] >= 25) & (d["age"] <= 59)]
    d = d[d["USBorn"] == 1]
    d = d[(d["hisp"] == 1) | (d["white"] == 1) | (d["black"] == 1)]
    return d.copy()


def load_cps_slim():
    t = time.time()
    df, _ = pyreadstat.read_dta(str(CPS_DTA))
    print(f"[01_clean] loaded CPS: {len(df):,} rows in {time.time()-t:.0f}s")
    for c in ["gen_org_cat12", "gen_org_cat15", "male", "female", "edyrs"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    keep = [
        "year", "age", "sex", "male", "female", "wtfinl",
        "hisp", "mexican", "prican", "cuban", "oth_hisp",
        "white", "black", "edyrs",
        "gen_org_cat12", "gen_org_cat15",
    ]
    keep = [c for c in keep if c in df.columns]
    d = df[keep].copy()
    d = d[d["gen_org_cat12"].between(1, 9)]
    return d


def main():
    # ---- Census ---- #
    c = load_census_slim()

    t12 = make_table12_slice(c)
    print(f"[01_clean] table12 slice: {len(t12):,}")
    t12.to_parquet(OUT / "census_2019.parquet", index=False)

    f1 = make_figure1_slice(c)
    print(f"[01_clean] figure1 slice: {len(f1):,}")
    f1.to_parquet(OUT / "census_figure1.parquet", index=False)

    # ---- CPS ---- #
    cps = load_cps_slim()
    print(f"[01_clean] cps analytic slice: {len(cps):,}")
    cps.to_parquet(OUT / "cps_clean.parquet", index=False)

    print("[01_clean] done.")


if __name__ == "__main__":
    main()
