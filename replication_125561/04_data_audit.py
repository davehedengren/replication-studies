"""Data audit for Elections.dta (125561)."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np
import pandas as pd
from utils import OUT


def main():
    df = pd.read_parquet(OUT / "main_sample.parquet")
    full = pd.read_parquet(OUT / "full_national.parquet")

    print("=" * 60)
    print("DATA AUDIT — Elections.dta (national presidential elections)")
    print("=" * 60)

    print("\n1. Coverage")
    print(f"  Full national panel: {len(full):,} obs, {full['ubigeo'].nunique()} districts")
    print(f"  Main regression sample (after singleton drop): {len(df):,} obs, "
          f"{df['ubigeo'].nunique()} districts")
    print(f"  Years: {sorted(df['year'].unique().tolist())}")
    print(f"  Election dates: {df['date_str'].nunique()} unique")
    print(f"  Provinces: {full['province_id'].nunique()}  Regions: {full['region'].nunique()}")
    print(f"  Poverty categories (category_06): {full['category_06'].value_counts().to_dict()}")

    print("\n2. Panel balance")
    per_district = full.groupby("ubigeo").size()
    print(f"  Obs per district — mean={per_district.mean():.1f}, "
          f"min={per_district.min()}, max={per_district.max()}")
    print(f"  Districts with full 8 elections: {(per_district == 8).sum()} / {len(per_district)}")

    print("\n3. Key variable distributions")
    for v in ["turnout", "ln_turnout", "fine_a", "ln_fine", "spoiled_elec",
              "ln_electores", "ln_votos_emitidos", "electores_01", "share_primaria"]:
        s = df[v].dropna()
        print(f"  {v:<22} n={len(s):>6}  mean={s.mean():+8.3f}  sd={s.std():6.3f}  "
              f"min={s.min():+8.3f}  max={s.max():+8.3f}")

    print("\n4. Logical checks")
    print(f"  turnout in [0,1]? min={df['turnout'].min():.3f}  max={df['turnout'].max():.3f}  "
          f"(out-of-range: {((df['turnout']<0)|(df['turnout']>1)).sum()})")
    print(f"  spoiled_elec in [0,1]? min={df['spoiled_elec'].min():.3f}  max={df['spoiled_elec'].max():.3f}")
    print(f"  fine_a values (S/100): unique={df['fine_a'].nunique()}  "
          f"range=[{df['fine_a'].min():.3f}, {df['fine_a'].max():.3f}]")
    fa = df.groupby(["year", "category_06"])["fine_a"].first().unstack()
    print("  fine_a by year × category_06 (first value seen):")
    print(fa.to_string())

    print("\n5. Missingness")
    vars_check = ["fine_a", "ln_fine", "turnout", "ln_turnout", "spoiled_elec",
                  "ln_electores", "ln_votos_emitidos", "share_primaria",
                  "ln_entre_18_20", "ln_mayores_75", "non_extreme", "extreme"]
    for v in vars_check:
        nmiss = df[v].isna().sum()
        if nmiss:
            print(f"  {v:<22}  missing={nmiss}")
    print("  (share_primaria missing in 2006 — no education data that year)")

    print("\n6. Duplicates")
    dups = df.duplicated(subset=["ubigeo", "date_str"]).sum()
    print(f"  Duplicate (ubigeo, date) rows: {dups}")

    print("\n7. Category transitions (2006 → 2010 reform)")
    cat = df[df["year"] == 2016][["category_06", "category_10"]]
    print(pd.crosstab(cat["category_06"], cat["category_10"], margins=True))

    print("\n8. Turnout by election")
    tout = full.groupby("date_str").apply(
        lambda g: np.average(g["turnout"], weights=g["electores_01"])
    )
    print(tout.to_string())

    print("\n9. Outliers (IQR × 1.5 rule)")
    for v in ["turnout", "spoiled_elec"]:
        s = df[v]
        q1, q3 = s.quantile(0.25), s.quantile(0.75)
        lo, hi = q1 - 1.5 * (q3 - q1), q3 + 1.5 * (q3 - q1)
        n_out = ((s < lo) | (s > hi)).sum()
        print(f"  {v}: {n_out} outliers ({n_out / len(s):.1%})")


if __name__ == "__main__":
    main()
