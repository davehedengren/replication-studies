"""Replicate Table 1 (Summary Statistics by State) and basic descriptives."""
import numpy as np
import pandas as pd
from utils import load_state_scores, weighted_mean, OUT


def main():
    df = load_state_scores()
    print(f"Total obs: {len(df):,}")
    print(f"Unique districts: {df.groupby(['state','district_id']).ngroups:,}")
    print(f"States: {sorted(df.state.unique())}")
    print(f"Years: {sorted(df.year.unique())}")
    print(f"Subjects: {df.subject.value_counts().to_dict()}")

    # Districts and average years observed (per state, one row per district)
    # Use subject==math to dedupe to district-year
    dm = df[df.subject == "math"].copy()

    # Number of districts per state
    n_dist = dm.groupby("state")["district_id"].nunique()

    # Average years of data per district per state
    yrs_per_dist = (
        dm.groupby(["state", "district_id"])["year"].nunique().groupby("state").mean()
    )

    # Pass rates by state x year (enrollment weighted), math
    def wmean_group(g, col):
        return weighted_mean(g[col], g["EnrollmentTotal"])

    pass19 = (
        dm[dm.year == 2019]
        .groupby("state")
        .apply(lambda g: wmean_group(g, "pass"), include_groups=False)
        * 100
    )
    pass21 = (
        dm[dm.year == 2021]
        .groupby("state")
        .apply(lambda g: wmean_group(g, "pass"), include_groups=False)
        * 100
    )

    # District-level (collapse firstnm): shares use 2021 values (only year with variation)
    d21 = dm[dm.year == 2021].copy()
    d21["share_blh"] = d21["share_black"] + d21["share_hisp"]

    def wm_state(col):
        return d21.groupby("state").apply(
            lambda g: wmean_group(g, col) * 100, include_groups=False
        )

    tbl = pd.DataFrame(
        {
            "Districts": n_dist,
            "Avg Years": yrs_per_dist.round(2),
            "Pass 2019": pass19.round(2),
            "Pass 2021": pass21.round(2),
            "% In-Person": wm_state("share_inperson").round(2),
            "% Hybrid": wm_state("share_hybrid").round(2),
            "% Virtual": wm_state("share_virtual").round(2),
            "% Black+Hisp": wm_state("share_blh").round(2),
            "% FRPL": wm_state("share_lunch").round(2),
        }
    )
    # Sort by state code matching Table 1 order
    order = ["CO", "CT", "MA", "MN", "MS", "OH", "RI", "VA", "WI", "WV", "WY"]
    tbl = tbl.reindex([s for s in order if s in tbl.index])
    print("\n=== Table 1: Summary statistics by state ===")
    print(tbl.to_string())

    tbl.to_csv(OUT / "table1_summary.csv")

    # Overall decline (abstract: math 14.2pp, ELA 6.3pp)
    for sub in ("math", "ela"):
        ds = df[df.subject == sub].copy()
        p19 = weighted_mean(
            ds.loc[ds.year == 2019, "pass"],
            ds.loc[ds.year == 2019, "EnrollmentTotal"],
        ) * 100
        p21 = weighted_mean(
            ds.loc[ds.year == 2021, "pass"],
            ds.loc[ds.year == 2021, "EnrollmentTotal"],
        ) * 100
        print(f"\n{sub.upper()} 2019 pass: {p19:.2f}, 2021 pass: {p21:.2f}, "
              f"decline: {p21 - p19:.2f} pp")


if __name__ == "__main__":
    main()
