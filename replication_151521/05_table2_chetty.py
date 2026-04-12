"""Table 2 row 1: 'Family income enrolled / Mean' from Chetty et al. (2020).

Paper target: 1.56 (data column). Paper footnote 34 says "Parental income is
averaged over 1996–2000, a five-year period in which the child (and potential
college attendee) is aged 15–19."

The Chetty mobility data in the package (mrc_merged.dta) has par_mean by
college plus a population-wide row. We form a count-weighted mean of par_mean
over four-year colleges and divide by par_mean for all children (the pop avg).
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import pyreadstat

sys.path.insert(0, str(Path(__file__).parent))
from utils import DATA, OUT, weighted_mean


def main():
    df, _ = pyreadstat.read_dta(str(DATA / "mrc_table2.dta"))
    # mrc_table2 is one row per super_opeid with par_mean and count (enrollment
    # cohort size). tier identifies college type: 1-8 (4-year public/private),
    # 9 (two-year), 10 (less than two-year), 11 (public four-year graduate)
    print("mrc_table2 shape:", df.shape)
    print("tier distribution:")
    print(df["tier"].value_counts().sort_index())

    # Population average: the paper reports 1.56 as 'family income enrolled / mean'
    # so the denominator is the cohort-wide average family income (weighted by cohort
    # count). The Chetty database uses IRS population averages; we approximate the
    # denominator as the count-weighted mean across all tiers = 1..10 (all
    # post-secondary aged 19 cohort members).
    pop = df.dropna(subset=["par_mean", "count"])
    pop = pop[pop["tier"].between(1, 14)]
    mean_all = weighted_mean(pop["par_mean"], pop["count"])

    # Cai & Heathcote (2020) model = four-year universities (tiers 1-8).
    fouryear = pop[pop["tier"].between(1, 8)]
    mean_4yr = weighted_mean(fouryear["par_mean"], fouryear["count"])

    ratio = mean_4yr / mean_all
    print(f"\nCount-weighted par_mean, all colleges (tiers 1–14): ${mean_all:,.0f}")
    print(f"Count-weighted par_mean, 4-year (tiers 1–8):         ${mean_4yr:,.0f}")
    print(f"Ratio 4-year / all:                                    {ratio:.3f}")
    print("Paper Table 2 (Data col, row 1): 1.560")

    # Also use Chetty's full-population "par_mean" for all children aged in cohort:
    # there is a row with super_opeid = -99 which aggregates non-attenders (or a
    # similar catch-all). Include both in denominator.
    if (df["super_opeid"] == -99).any():
        mean_all_incl_nonatt = weighted_mean(df.dropna(subset=["par_mean", "count"])["par_mean"],
                                             df.dropna(subset=["par_mean", "count"])["count"])
        ratio2 = mean_4yr / mean_all_incl_nonatt
        print(f"Count-weighted par_mean incl nonattenders (-99/-1):   ${mean_all_incl_nonatt:,.0f}")
        print(f"Ratio 4-year / full incl non-attenders:               {ratio2:.3f}")

    pd.DataFrame([
        {"stat": "Family income enrolled / Mean (Table 2 row 1)",
         "paper": 1.560, "repl": round(ratio, 3)}
    ]).to_csv(OUT / "table2_row1_chetty.csv", index=False)


if __name__ == "__main__":
    main()
