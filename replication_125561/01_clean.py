"""Load Elections.dta, restrict to 8 national presidential elections,
apply iterative singleton drop for the district + election×province×cat FE,
and save the main analysis panel."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import pandas as pd
from utils import load_elections, make_group_ids, iterative_singleton_drop, OUT

def main():
    df = load_elections()
    print("Raw (national elections only):", len(df), "districts:", df["ubigeo"].nunique())

    # Cast ids
    df["ubigeo"] = df["ubigeo"].astype(int)
    df["province_id"] = df["province_id"].astype(int)
    df["category_06"] = df["category_06"].astype(int)
    df["category_10"] = df["category_10"].astype(int)

    # Interaction FE: date × province × category_06 (paper's main absorbing group)
    df = make_group_ids(df, ["date_id", "province_id", "category_06"], "dpc_id")
    df = make_group_ids(df, ["date_id", "province_id"], "dp_id")

    # Iterative singleton drop on (ubigeo, dpc_id)
    main_sample = iterative_singleton_drop(df.copy(), ["ubigeo", "dpc_id"])
    # Rebuild integer codes so they are 0..K-1 (needed for bincount)
    for col in ["ubigeo", "dpc_id", "dp_id", "province_id", "date_id", "category_06"]:
        main_sample[col] = pd.Categorical(main_sample[col]).codes.astype(int)

    print("Main sample:", len(main_sample), "districts:", main_sample["ubigeo"].nunique())
    assert len(main_sample) == 13536, f"Expected 13,536; got {len(main_sample)}"
    assert main_sample["ubigeo"].nunique() == 1692

    main_sample.to_parquet(OUT / "main_sample.parquet")
    # Also save the full national panel (before singleton drop) for Table 1 col 2
    df_all = df.copy()
    for col in ["ubigeo", "dpc_id", "dp_id", "province_id", "date_id", "category_06"]:
        df_all[col] = pd.Categorical(df_all[col]).codes.astype(int)
    df_all.to_parquet(OUT / "full_national.parquet")
    print("Saved main_sample.parquet (13,536) and full_national.parquet (14,040)")

if __name__ == "__main__":
    main()
