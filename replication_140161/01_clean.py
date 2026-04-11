"""Build and cache the Wave-1 analysis sample.

Run first. Exports a parquet + CSV copy of the cleaned dataset used by the
later numbered scripts.
"""
from replication_140161.utils import build_all, OUT


def main():
    df = build_all()
    df.to_parquet(OUT / "wave1_clean.parquet", index=False)
    df.to_csv(OUT / "wave1_clean.csv", index=False)
    print(f"Wrote {len(df)} rows to {OUT/'wave1_clean.parquet'}")
    print("Per-treatment Ns:")
    print(df.groupby("survey").size())
    print("\nMain DV means by treatment (want_share_fb):")
    print(df.groupby("survey")["want_share_fb"].mean().round(4))


if __name__ == "__main__":
    main()
