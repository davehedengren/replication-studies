"""Load and inspect the primary analytical sample (base_sample_aej.dta).

Reports sample size at each filter step so we can match it to the Stata pipeline.
"""
import pandas as pd
from utils import load_base_sample, table2_sample, DATA, OUT


def main():
    raw = pd.read_stata(DATA / "base_sample_aej.dta", convert_categoricals=False)
    print(f"Raw base_sample_aej.dta rows: {len(raw):,}")
    print(f"  Years: {int(raw['year'].min())}-{int(raw['year'].max())}")
    print(f"  Unique individuals: {raw['id'].nunique():,}")
    print(f"  Rounds present: {sorted(raw['round'].dropna().unique().tolist())}")

    df = load_base_sample()
    print(f"\nAfter keeping alcohol-consumer rows & movers filter: {len(df):,}")

    t2 = table2_sample(df)
    print(f"After sample restrictions (year>=2001 & 18<=age<=65): {len(t2):,}")
    print(f"  Target N for Table 2 cols 1-5: 29,083")
    print(f"  Unique id: {t2['id'].nunique():,}")
    print(f"  Unique identificator: {t2['identificator'].nunique():,}")

    print("\nKey variable means (Table 2 sample):")
    for v in ["share_vodka", "share_beer", "alcohol_intake", "rural", "gorbachev",
              "rural_gorbachev", "logincome", "married", "univ_educ"]:
        print(f"  {v}: {t2[v].mean():.4f}")

    t2.to_parquet(OUT / "table2_sample.parquet", index=False)
    df.to_parquet(OUT / "cleaned_full.parquet", index=False)
    print(f"\nSaved → {OUT}")


if __name__ == "__main__":
    main()
