"""Data audit for the Seoul gravity dataset used in equation (20)."""
import numpy as np
import pandas as pd
from utils import OUT


def main():
    df = pd.read_parquet(OUT / "sel_gravity.parquet")
    all_dates = sorted(df["statadate"].unique())

    print("=" * 60)
    print("COVERAGE")
    print("=" * 60)
    print(f"Total obs:        {len(df):,}")
    print(f"Obs in year 2020: {(df['year'] == 2020).sum():,}")
    print(f"# dates:          {len(all_dates)}")
    print(f"Date range:       {df['statadate'].min().date()} "
          f"to {df['statadate'].max().date()}")
    print(f"Unique RCo:       {df['RCo'].nunique()}")
    print(f"Unique RCd:       {df['RCd'].nunique()}")
    print(f"OD pairs:         {df[['RCo','RCd']].drop_duplicates().shape[0]}")

    # Panel balance: expect 25*25 = 625 pairs * days.
    n_pairs = df[["RCo", "RCd"]].drop_duplicates().shape[0]
    expected = n_pairs * len(all_dates)
    print(f"Balanced panel?:  {len(df):,} / {expected:,} "
          f"= {len(df) / expected * 100:.2f}%")

    # Check missing dates per OD pair
    pair_counts = df.groupby(["RCo", "RCd"]).size()
    print(f"Min obs/pair:     {pair_counts.min()}")
    print(f"Max obs/pair:     {pair_counts.max()}")

    print("\n" + "=" * 60)
    print("DISTRIBUTIONS")
    print("=" * 60)
    for col in ["eod", "lns", "dist", "lndist", "yod"]:
        s = df[col]
        print(f"{col:<10} n={s.count():>7,}  "
              f"mean={s.mean():>12.3f}  sd={s.std():>12.3f}  "
              f"min={s.min():>10.3f}  max={s.max():>10.3f}  "
              f"nmiss={s.isna().sum()}")

    # Zero / negative expenditures?
    zero_eod = (df["eod"] <= 0).sum()
    print(f"\nNon-positive eod: {zero_eod}  (would break log)")

    # Expenditure tails (top 5)
    top = df.nlargest(5, "eod")[["statadate", "RCo", "RCd", "eod", "yod", "dist"]]
    print("\nTop 5 expenditure rows:")
    print(top.to_string(index=False))

    # Same-district (diagonal) vs off-diagonal
    diag = df[df["RCo"] == df["RCd"]]
    off = df[df["RCo"] != df["RCd"]]
    print(f"\nDiagonal obs:      {len(diag):,}  mean eod={diag['eod'].mean():,.0f}")
    print(f"Off-diagonal obs:  {len(off):,}  mean eod={off['eod'].mean():,.0f}")

    # Commute flow ratio (yod2 = chi) — should mostly hover around 1 pre-pandemic
    pre = df[df["year"] < 2020]
    post = df[df["year"] == 2020]
    print(f"\nyod (chi) pre-2020:  mean={pre['yod'].mean():.3f}  "
          f"sd={pre['yod'].std():.3f}")
    print(f"yod (chi) 2020:      mean={post['yod'].mean():.3f}  "
          f"sd={post['yod'].std():.3f}")

    # Lockdown indicator
    if "ipat" in df.columns:
        print(f"\nipat (post-first-case flag) shares: "
              f"{df['ipat'].value_counts(normalize=True).to_dict()}")

    print("\n" + "=" * 60)
    print("2020 SUBSAMPLE (regression sample)")
    print("=" * 60)
    d20 = df[df["year"] == 2020]
    print(f"Obs:              {len(d20):,}  (published Figure-1 N = 75,625)")
    print(f"Unique dates:     {d20['statadate'].nunique()}")
    print(f"Obs per date:     {len(d20) / d20['statadate'].nunique():.0f}"
          f"  (=25^2 = 625 if fully balanced)")
    date_counts = d20.groupby("statadate").size()
    print(f"Min obs/date:     {date_counts.min()}")
    print(f"Max obs/date:     {date_counts.max()}")

    print("\nData audit complete.")


if __name__ == "__main__":
    main()
