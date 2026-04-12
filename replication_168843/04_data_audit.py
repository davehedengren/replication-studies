"""Data audit for state_score_data.dta."""
import numpy as np
import pandas as pd
from utils import load_state_scores, OUT


def main():
    df = load_state_scores()
    print(f"\nRaw shape: {df.shape}")
    print(f"Unique districts: {df.groupby(['state','district_id']).ngroups}")

    # Coverage: years and obs per state
    print("\n=== Obs per state × year ===")
    print(pd.crosstab(df["state"], df["year"]).to_string())

    print("\n=== Obs per state × subject ===")
    print(pd.crosstab(df["state"], df["subject"]).to_string())

    # Missingness of key variables
    print("\n=== Missingness ===")
    miss = df.isna().sum().sort_values(ascending=False)
    print(miss[miss > 0].to_string())

    # Summary stats for key variables
    print("\n=== Summary stats ===")
    cols = [
        "pass",
        "share_inperson",
        "share_hybrid",
        "share_virtual",
        "share_black",
        "share_hisp",
        "share_lunch",
        "share_ELL",
        "EnrollmentTotal",
        "participation",
        "trump_vote_share",
        "unemployment",
    ]
    print(df[cols].describe().T.round(3).to_string())

    # Sanity: shares in [0,1]
    print("\n=== Share plausibility ===")
    for c in [
        "pass",
        "share_inperson",
        "share_hybrid",
        "share_virtual",
        "share_black",
        "share_hisp",
        "share_lunch",
        "share_ELL",
    ]:
        v = df[c].dropna()
        print(
            f"{c}: min={v.min():.4f} max={v.max():.4f} "
            f"n_below_0={int((v < 0).sum())} n_above_1={int((v > 1).sum())}"
        )

    # Schooling shares should sum to roughly 1 in 2021
    d21 = df[df.year == 2021].copy()
    d21["share_sum"] = d21["share_inperson"] + d21["share_hybrid"] + d21["share_virtual"]
    print(
        f"\n=== 2021 share sum: mean={d21['share_sum'].mean():.4f} "
        f"min={d21['share_sum'].min():.4f} max={d21['share_sum'].max():.4f} "
        f"n_not_1 (|diff|>0.01)={int((d21['share_sum'] - 1).abs().gt(0.01).sum())}"
    )

    # Panel balance: how many years per district
    dm = df[df.subject == "math"]
    yrs_per_dist = dm.groupby(["state", "district_id"])["year"].nunique()
    print("\n=== Years per district (math) ===")
    print(yrs_per_dist.value_counts().sort_index().to_string())

    # Which districts have 2021 but not earlier?
    has_2021 = dm[dm.year == 2021].set_index(["state", "district_id"]).index
    has_pre = dm[dm.year < 2021].set_index(["state", "district_id"]).index
    only_2021 = len(has_2021.difference(has_pre))
    only_pre = len(has_pre.difference(has_2021))
    print(f"Districts only in 2021: {only_2021}")
    print(f"Districts only pre-2021: {only_pre}")

    # Duplicates on (district, year, subject)
    dup = df.duplicated(subset=["state", "district_id", "year", "subject"]).sum()
    print(f"\n=== Duplicate (state, district, year, subject) rows: {dup}")

    # Pre vs 2021 pass rate change by state (to check Figure 1 story)
    print("\n=== Pass rate change, 2019 -> 2021, enrollment weighted ===")
    for sub in ("math", "ela"):
        ds = df[df.subject == sub]
        out = []
        for st in sorted(ds.state.unique()):
            g = ds[ds.state == st]
            p19 = np.average(
                g[g.year == 2019]["pass"], weights=g[g.year == 2019]["EnrollmentTotal"]
            ) if (g.year == 2019).any() else np.nan
            p21 = np.average(
                g[g.year == 2021]["pass"], weights=g[g.year == 2021]["EnrollmentTotal"]
            ) if (g.year == 2021).any() else np.nan
            out.append((st, round(p19 * 100, 2), round(p21 * 100, 2),
                        round((p21 - p19) * 100, 2)))
        print(f"  {sub}: " + ", ".join(f"{s}:{d:+.1f}" for s, _, _, d in out))

    # Outliers in EnrollmentTotal
    print("\n=== Enrollment extremes ===")
    big = df.sort_values("EnrollmentTotal", ascending=False).head(5)[
        ["state", "district_id", "year", "EnrollmentTotal", "pass"]
    ]
    print(big.to_string(index=False))


if __name__ == "__main__":
    main()
