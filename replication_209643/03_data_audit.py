"""Data audit for MORG 1980/2000 panel used in Table 1 / Section 2 stylized facts."""
import numpy as np
import pandas as pd
from utils import load_morg_pooled, apply_sample_filter, build_edu, merge_crosswalk, OUT


def main():
    raw = load_morg_pooled()
    print("=" * 60)
    print("COVERAGE (raw before filters)")
    print("=" * 60)
    print(raw.groupby("year").size())

    df = apply_sample_filter(raw)
    print("\nAfter age/mlr/ft/wage filter:")
    print(df.groupby("year").size())

    df = build_edu(df)
    print("\nAfter edu construction:")
    print(df.groupby("year").size())

    df = merge_crosswalk(df)
    df = df[df["occ1990dd"].notna()]
    print("\nAfter occupation crosswalk merge:")
    print(df.groupby("year").size())

    print("\n" + "=" * 60)
    print("WAGE DISTRIBUTION (hourly, nominal $)")
    print("=" * 60)
    ws = df.groupby("year")["wage"].describe(percentiles=[0.01, 0.05, 0.5, 0.95, 0.99])
    print(ws)

    print("\n" + "=" * 60)
    print("LOG WAGE DISTRIBUTION")
    print("=" * 60)
    df["logw"] = np.log(df["wage"])
    print(df.groupby("year")["logw"].describe(percentiles=[0.1, 0.5, 0.9]))

    print("\n" + "=" * 60)
    print("DEMOGRAPHIC COMPOSITION")
    print("=" * 60)
    for col in ["edu", "young", "male", "black", "other"]:
        print(f"\n{col}:")
        print(df.groupby("year")[col].mean().round(3))

    print("\n" + "=" * 60)
    print("OCCUPATION COVERAGE")
    print("=" * 60)
    print("Unique occ1990dd codes per year:")
    print(df.groupby("year")["occ1990dd"].nunique())
    # Any occs missing in one year?
    occs80 = set(df[df["year"] == 1980]["occ1990dd"].unique())
    occs00 = set(df[df["year"] == 2000]["occ1990dd"].unique())
    print(f"\nOccs only in 1980: {len(occs80 - occs00)}")
    print(f"Occs only in 2000: {len(occs00 - occs80)}")

    # Singleton cells - small occs with only one obs in a year
    cell = df.groupby(["year", "occ1990dd"]).size()
    print(f"\n(year, occ) cells with N=1: {(cell == 1).sum()}")
    print(f"(year, occ) cells with N<5: {(cell < 5).sum()}")

    print("\n" + "=" * 60)
    print("WEIGHT / PERWT CHECKS")
    print("=" * 60)
    df["perwt"] = np.floor(df["wgt_hrs"] / 100).astype(int)
    print(df.groupby("year")["perwt"].describe(percentiles=[0.5]))
    print(f"\nRows with perwt==0 dropped: {(df['perwt'] == 0).sum()}")

    print("\n" + "=" * 60)
    print("TOP-CODING / OUTLIERS")
    print("=" * 60)
    for yr in [1980, 2000]:
        g = df[df["year"] == yr]
        p99 = g["wage"].quantile(0.99)
        print(f"{yr}: 99th pctile ${p99:.2f}, max ${g['wage'].max():.2f}, "
              f"{(g['wage'] >= 1499).sum()} at top filter")

    print("\n" + "=" * 60)
    print("DUPLICATES")
    print("=" * 60)
    dup = raw.duplicated(subset=["hhid", "lineno", "year", "month"]).sum()
    print(f"Person-month duplicates in raw: {dup}")

    # Save summary to csv
    summary = {
        "n_raw_1980": int((raw["year"] == 1980).sum()),
        "n_raw_2000": int((raw["year"] == 2000).sum()),
        "n_analysis_1980": int((df["year"] == 1980).sum()),
        "n_analysis_2000": int((df["year"] == 2000).sum()),
        "occs_1980": int(df[df["year"] == 1980]["occ1990dd"].nunique()),
        "occs_2000": int(df[df["year"] == 2000]["occ1990dd"].nunique()),
        "mean_logw_1980": float(df[df["year"] == 1980]["logw"].mean()),
        "mean_logw_2000": float(df[df["year"] == 2000]["logw"].mean()),
    }
    pd.Series(summary).to_csv(OUT / "data_audit_summary.csv")
    print(f"\nSaved {OUT/'data_audit_summary.csv'}")


if __name__ == "__main__":
    main()
