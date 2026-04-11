"""Phase 1 sanity check: reproduce Table 1 (descriptive statistics)
and verify the main-sample size against the published N = 8,446.
"""

import numpy as np
import pandas as pd

from utils import (
    load_team1, main_sample_mask,
    CONTROLS_CASE, CONTROLS_JUDGE, CONTROLS_DISTRICT, CONTROLS_BENCH,
)


def main():
    df = load_team1()
    print(f"Raw DataTeam1: {df.shape[0]} rows, {df.shape[1]} cols")

    mask = main_sample_mask(df)
    sub = df.loc[mask].copy()
    print(f"Main-spec e(sample): {len(sub)} rows (paper: 8,446)")

    # Table 1, Panel A: case characteristics
    panelA_vars = [
        "StateWins", "caselag", "Merit", "Process_Followed",
        "constitutional", "land_case", "HR", "criminal",
        "pagesjudgenum", "lawyer_number", "judge_number",
        "benchchiefjustice",
    ]
    pa = sub[panelA_vars].agg(["count", "mean", "std", "min", "max"]).T
    print("\nTable 1 Panel A (case):")
    print(pa.round(3).to_string())

    # Panel B: collapse treatment vars to district-bench level
    pb_vars = [
        "Retirements_2010_Post2010", "Appointments_2010_Post2010",
    ]
    pb = sub.groupby("districtXbench_enco")[pb_vars].mean().agg(["count", "mean", "std", "min", "max"]).T
    # collapse first on FULL df like the do file
    full_pb = df.groupby("districtXbench_enco")[
        ["Retirements_2010_Post2010", "Appointments_2010_Post2010"]
    ].mean()
    print("\nTable 1 Panel B (treatment, collapsed to 64 benches):")
    print(full_pb.agg(["count", "mean", "std", "min", "max"]).T.round(3).to_string())

    # Panel C: collapse to judge level, restricted to e(sample)
    jvars = [
        "TenureatDecision", "Gender", "ElevatedtoSupremeCourt",
        "Former_LowerCourtJudge", "Former_MemberBarAssociation",
        "Prior_Political_Office", "Former_Lawyer",
        "AfterReformJudge", "Punjabi", "Sindhi", "Balochi", "Pashtun", "Others",
    ]
    jcol = sub.groupby("judgename")[jvars].mean()
    print(f"\nTable 1 Panel C (judge-level, N judges = {len(jcol)}; paper: 511)")
    print(jcol.agg(["count", "mean", "std", "min", "max"]).T.round(3).to_string())

    # Panel D: collapse to bench-district-year
    dvars = ["Retirements_2010_Post2010", "Appointment_2010_Post2010",
             "TotalJudges_bench", "Criminal_count_bench",
             "Land_count_per_bench", "Human_count_per_bench"]
    dcol = sub.groupby(["distric_encode", "Specialization_bench", "yeardecision"])[dvars].mean()
    print(f"\nTable 1 Panel D (bench-district-year, N = {len(dcol)}; paper: 1,516)")
    print(dcol.agg(["count", "mean", "std", "min", "max"]).T.round(3).to_string())


if __name__ == "__main__":
    main()
