"""Phase 3 — data audit of DataTeam1.dta (the main replication sample)."""

import numpy as np
import pandas as pd

from utils import load_team1, main_sample_mask


def main():
    df = load_team1()
    mask = main_sample_mask(df)
    sub = df.loc[mask].copy()
    print(f"Raw rows: {len(df):,}   main-sample rows: {len(sub):,}")

    # ---- Coverage ----
    print("\n[Coverage]")
    print(f"Years covered: {int(sub.yeardecision.min())}-{int(sub.yeardecision.max())}")
    print(f"Distinct districts: {sub.distric_encode.nunique()}  (paper: 16)")
    print(f"Distinct district-benches: {sub.districtXbench_enco.nunique()}  (paper: 64)")
    print(f"Distinct judges: {sub.judgename.nunique()}  (paper: 511)")
    print(f"Cases per year range: {sub.groupby('yeardecision').size().min()}-{sub.groupby('yeardecision').size().max()}")

    # ---- Outcome plausibility ----
    print("\n[Outcomes]")
    for col in ["StateWins", "Merit", "Process_Followed", "caselag", "correct"]:
        s = sub[col]
        print(f"  {col:<20}  mean={s.mean():7.3f}  sd={s.std():7.3f}  min={s.min():6.2f}  max={s.max():6.2f}")

    # ---- Shares/dummies in [0,1] ----
    print("\n[Dummy range checks]")
    for col in ["StateWins", "Merit", "constitutional", "criminal", "land_case", "HR",
                "Retirements_2010_Post2010", "Appointments_2010_Post2010"]:
        s = sub[col]
        if s.min() < 0 or s.max() > 1:
            print(f"  !! {col} has out-of-range values: min={s.min()} max={s.max()}")
        else:
            print(f"  OK  {col}  range=[{s.min():.3f},{s.max():.3f}]")

    # ---- Logical consistency ----
    print("\n[Logical consistency]")
    both = ((sub.constitutional == 1) & (sub.criminal == 1)).sum()
    neither = ((sub.constitutional == 0) & (sub.criminal == 0)).sum()
    print(f"  constitutional & criminal overlap: {both}")
    print(f"  neither constitutional nor criminal: {neither}")
    land_not_const = ((sub.land_case == 1) & (sub.constitutional == 0)).sum()
    hr_not_const = ((sub.HR == 1) & (sub.constitutional == 0)).sum()
    print(f"  land_case=1 but constitutional=0: {land_not_const}")
    print(f"  HR=1 but constitutional=0: {hr_not_const}")
    land_hr = ((sub.land_case == 1) & (sub.HR == 1)).sum()
    print(f"  land_case=1 & HR=1 (should be 0): {land_hr}")

    # ---- Sub-sample sizes for Table 5 ----
    print("\n[Subsample sizes used in Table 5]")
    print(f"  constitutional==1 (paper col 1): {(sub.constitutional == 1).sum()}  (paper: 6094)")
    print(f"  criminal==1      (paper col 3):  {(sub.criminal == 1).sum()}  (paper: 2368)")
    print(f"  land_case==1 & criminal==0      : {((sub.land_case == 1) & (sub.criminal == 0)).sum()}"
          f"  (paper Col 2 'Land' label: 2650; paper Col 1 'HR' label: 3428)")
    print(f"  HR==1 & criminal==0              : {((sub.HR == 1) & (sub.criminal == 0)).sum()}")

    # ---- Panel balance ----
    print("\n[Panel balance — cases by year]")
    by_year = sub.groupby("yeardecision").size()
    print(by_year.to_string())

    # ---- Treatment distribution ----
    print("\n[Treatment — Retirements_2010 share of judges per district-bench]")
    treat = df.groupby("districtXbench_enco")[
        ["Retirements_2010_Post2010", "Appointments_2010_Post2010"]
    ].first()  # identical within a bench pre-post interaction? no — this is time-varying
    # correct aggregation: max/mean is same within (dbt)
    post = df[df.yeardecision >= 2010].groupby("districtXbench_enco")[
        "Retirements_2010_Post2010"
    ].max()
    print(f"  Benches with zero 2010 retirements (post period): {(post == 0).sum()}/{len(post)}")
    print(f"  Mean share (post): {post.mean():.3f}, max: {post.max():.3f}")

    # ---- Missing data patterns ----
    print("\n[Missingness in raw data for key covariates]")
    for col in ["StateWins", "Retirements_2010_Post2010", "pagesjudgenum",
                "Area_sqkm", "Population", "TenureatDecision"]:
        miss = df[col].isna().sum()
        print(f"  {col:<28} missing={miss}  ({miss / len(df):.1%})")

    # ---- Outliers ----
    print("\n[Outlier ranges for continuous vars in sub-sample]")
    for col in ["pagesjudgenum", "caselag", "lawyer_number", "judge_number", "Population", "Area_sqkm"]:
        s = sub[col]
        q = s.quantile([0.01, 0.5, 0.99])
        print(f"  {col:<15}  p01={q.iloc[0]:.2f}  median={q.iloc[1]:.2f}  p99={q.iloc[2]:.2f}  max={s.max():.2f}")


if __name__ == "__main__":
    main()
