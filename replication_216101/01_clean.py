"""Load NLSY97 extract and HSLS:09 subset, produce cleaned parquet files."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pyreadstat

from utils import (
    HSLS_DTA,
    build_nlsy_clean,
    cpi_2016_deflator,
    load_nlsy,
)

OUT = Path(__file__).resolve().parent / "data"
OUT.mkdir(exist_ok=True)


def clean_hsls() -> pd.DataFrame:
    cols = [
        "STU_ID",
        "W5W1W2W3W4PSRECORDS", "W4W1W2W3STU",
        "X1PAR1EDU", "X2PAR1EDU", "X1PAR2EDU", "X2PAR2EDU",
        "X2P1RELATION", "X2P2RELATION",
        "X2SEX", "X2STUEDEXPCT", "X3TGPAWGT", "X3HSCOMPSTAT",
        "X1FAMINCOME", "X2FAMINCOME",
        "X5PFYDEGREE", "X5PFYSEC", "X5PFYENRLSTAT", "X5PFYEAR", "X5POSTHSAY",
        "X5AY1314ATT", "X5AY1415ATT", "X5AY1516ATT",
        "X5AY1314ERN", "X5AY1415ERN", "X5AY1516ERN",
        "X5BACCRED", "X5HIGHDEG",
        "X5T4XLNCUM",
    ]
    df, _ = pyreadstat.read_dta(str(HSLS_DTA), usecols=cols)
    # HS sample flag (lifted from HSLS_Cleaning.do)
    par1_rel = df["X2P1RELATION"].isin([1, 2])
    par2_rel = df["X2P2RELATION"].isin([1, 2])
    hs = (
        ((df["X2PAR1EDU"] >= 0) | (df["X1PAR1EDU"] >= 0)
         | (df["X2PAR2EDU"] >= 0) | (df["X1PAR2EDU"] >= 0))
        & (par1_rel | par2_rel)
        & df["X2SEX"].isin([1, 2])
        & (df["X2STUEDEXPCT"] >= 0)
        & (df["X3TGPAWGT"] >= 0)
        & ((df["X1FAMINCOME"] >= 0) | (df["X2FAMINCOME"] >= 0))
        & df["X3HSCOMPSTAT"].isin([1, 2])
    )
    df["flag_sample_hs"] = hs.astype(int)

    # 4yr BA program, full-time-ish, >=20 credits attempted in AY13-14,
    # records from fall 2013, etc.
    sample_college = (
        df["X5PFYDEGREE"].isin([4, -9])
        & df["X5PFYSEC"].isin([1, 2, 3, 4, 5, 6])
        & df["X5PFYENRLSTAT"].isin([1, 2, 3])
        & (df["X5AY1314ATT"] >= 20)
        & (df["X5PFYEAR"] == 2014)
    )
    # Drop if contradictory first year records
    contra = (df["X5PFYEAR"] != df["X5POSTHSAY"]) & (df["X5POSTHSAY"] > 0) & (df["X5PFYEAR"] > 0)
    sample_college = sample_college & ~contra
    # Drop if earned > attempted in any AY
    def gt_max(a, b):
        return a > np.maximum(b, 0)
    bad = (
        gt_max(df["X5AY1314ERN"], df["X5AY1314ATT"])
        | gt_max(df["X5AY1415ERN"], df["X5AY1415ATT"])
        | gt_max(df["X5AY1516ERN"], df["X5AY1516ATT"])
    )
    sample_college = sample_college & ~bad
    sample_college = sample_college & hs
    df["flag_sample_college"] = sample_college.astype(int)

    # Persisted through Y3
    credit_floor = (
        (df[["X5AY1314ATT", "X5AY1415ATT", "X5AY1516ATT"]].min(axis=1) >= 20)
    )
    earned_ba = (df["X5BACCRED"] == 1) | (df["X5HIGHDEG"] == 3)
    df["flag_persisted"] = (sample_college & (earned_ba | credit_floor)).astype(int)

    # Cumulative federal student loans (positive balances)
    fed = np.where(df["X5T4XLNCUM"] >= 0, df["X5T4XLNCUM"], 0.0)
    df["fed_SLcum"] = fed
    df["anyfed"] = (fed > 0).astype(int)

    # Deflate to 2016 dollars (X5T4XLNCUM already covers loans through 2016).
    return df


def main():
    print("Loading NLSY97 CSV...")
    raw = load_nlsy()
    print(f"  {len(raw)} respondents, {raw.shape[1]} variables")
    clean = build_nlsy_clean(raw)
    clean.to_parquet(OUT / "nlsy97_clean.parquet")
    n_sample = int(clean["flag_beliefs_sample"].sum())
    n_enr = int(((clean["flag_beliefs_sample"] == 1) & (clean["EnrBA_by_30"] == 1)).sum())
    print(f"  belief sample (HS grads by 30): {n_sample}  (target 2,222)")
    print(f"  belief sample (BA enrollees): {n_enr}  (target 1,200)")

    print("Loading HSLS:09...")
    hsls = clean_hsls()
    hsls.to_parquet(OUT / "hsls_clean.parquet")
    w = hsls["W5W1W2W3W4PSRECORDS"]
    obs_college = int(((hsls["flag_sample_college"] == 1) & (w > 0)).sum())
    print(f"  HSLS college sample: {obs_college}  (target 2,356)")
    persist = int(((hsls["flag_persisted"] == 1) & (w > 0)).sum())
    print(f"  HSLS persisted: {persist}  (target 1,855)")


if __name__ == "__main__":
    main()
