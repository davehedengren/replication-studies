"""Replicate Table 3 of Antman, Duncan, Trejo (JEP 2023).

Panel A (ages 25-59) and Panel B (age cohorts 25-34, 50-59): weighted
means and standard errors of completed years of schooling by sex, race,
and immigrant generation (1st, 2nd, 3rd+), pooling CPS 2003-2019.

Stata's estpost tabstat ... , stats(mean semean) [aweight=wtfinl] returns:
  * mean = weighted mean of edyrs
  * semean = standard error of the mean treating weights as aweights
    (variance divided by N_effective = sum(w)^2 / sum(w^2))
"""
import numpy as np
import pandas as pd
from utils import OUT


def wmean_sem(values, weights):
    v = np.asarray(values, float)
    w = np.asarray(weights, float)
    m = np.sum(w * v) / np.sum(w)
    # Stata's semean with aweights: variance of the mean =
    # [sum(w * (v - m)^2) / sum(w)] / N_eff
    # where N_eff is the count of observations (not the weight sum),
    # matching Stata's aweight convention for tabstat, semean.
    var = np.sum(w * (v - m) ** 2) / np.sum(w)
    N = len(v)
    sem = np.sqrt(var / (N - 1))
    return m, sem


def table_row(df, mask, label):
    """Return (label, 1st_mean, 1st_se, 2nd_mean, 2nd_se, 3rd_mean, 3rd_se)."""
    d = df[mask]
    out = [label]
    for gen_col_val, gen_val in zip(["gen_org_cat12"], [None]):
        pass  # placeholder
    return out


def compute_group(df, col, codes):
    rows = {}
    for name, code in codes:
        sub = df[df[col] == code]
        m, s = wmean_sem(sub["edyrs"].values, sub["wtfinl"].values)
        rows[name] = (m, s, len(sub))
    return rows


PAPER_MALE = {
    # panel A
    ("A", "Hispanic"):      {"1st": (10.38, 0.02), "2nd": (13.00, 0.02), "3rd+": (12.85, 0.02)},
    ("A", "Mexican"):       {"1st": ( 9.61, 0.02), "2nd": (12.73, 0.02), "3rd+": (12.71, 0.02)},
    ("A", "NH White"):      {"1st": (14.43, 0.02), "2nd": (14.49, 0.02), "3rd+": (13.84, 0.00)},
    ("A", "NH Black"):      {"1st": (13.63, 0.03), "2nd": (13.99, 0.05), "3rd+": (13.00, 0.01)},
    # panel B: age cohorts
    ("B25-34", "Hispanic"): {"1st": (10.48, 0.03), "2nd": (12.97, 0.03), "3rd+": (12.85, 0.02)},
    ("B50-59", "Hispanic"): {"1st": (10.17, 0.04), "2nd": (12.82, 0.06), "3rd+": (12.74, 0.04)},
    ("B25-34", "Mexican"):  {"1st": (10.05, 0.03), "2nd": (12.76, 0.03), "3rd+": (12.71, 0.03)},
    ("B50-59", "Mexican"):  {"1st": ( 8.73, 0.05), "2nd": (12.45, 0.08), "3rd+": (12.55, 0.05)},
    ("B25-34", "NH White"): {"1st": (14.45, 0.04), "2nd": (14.41, 0.03), "3rd+": (13.86, 0.01)},
    ("B50-59", "NH White"): {"1st": (14.25, 0.04), "2nd": (14.49, 0.03), "3rd+": (13.81, 0.01)},
    ("B25-34", "NH Black"): {"1st": (13.57, 0.05), "2nd": (13.96, 0.07), "3rd+": (13.01, 0.02)},
    ("B50-59", "NH Black"): {"1st": (13.45, 0.07), "2nd": (13.57, 0.17), "3rd+": (12.84, 0.02)},
}


def run_for_sex(df, sex_code, label):
    print(f"\n==== {label.upper()} ====")
    d = df[df["sex"] == sex_code].copy()

    panels = [
        ("Panel A (ages 25-59)", None),
        ("Panel B ages 25-34", (25, 34)),
        ("Panel B ages 50-59", (50, 59)),
    ]

    for pname, agerange in panels:
        if agerange is None:
            sub = d
            tag = "A"
        else:
            sub = d[(d["age"] >= agerange[0]) & (d["age"] <= agerange[1])]
            tag = "B" + f"{agerange[0]}-{agerange[1]}"
        print(f"\n-- {pname} --")

        groups = [
            ("Hispanic", "gen_org_cat12", [("1st", 1), ("2nd", 2), ("3rd+", 3)]),
            ("Mexican",  "gen_org_cat15", [("1st", 1), ("2nd", 2), ("3rd+", 3)]),
            ("NH White", "gen_org_cat15", [("1st", 4), ("2nd", 5), ("3rd+", 6)]),
            ("NH Black", "gen_org_cat15", [("1st", 7), ("2nd", 8), ("3rd+", 9)]),
        ]
        for grp, col, codes in groups:
            rows = compute_group(sub, col, codes)
            line = f"  {grp:<10s}"
            for gname, _ in codes:
                m, s, n = rows[gname]
                # get paper
                paper = None
                if label == "male":
                    key = (tag, grp)
                    if key in PAPER_MALE and gname in PAPER_MALE[key]:
                        paper = PAPER_MALE[key][gname]
                if paper:
                    pm, ps = paper
                    line += f"  {gname:<4s} paper {pm:.2f}({ps:.2f}) repl {m:.2f}({s:.2f})"
                else:
                    line += f"  {gname:<4s} repl {m:.2f}({s:.2f})"
            print(line)


def main():
    df = pd.read_parquet(OUT / "cps_clean.parquet")
    print(f"[03] N={len(df):,}")
    run_for_sex(df, 1, "male")
    run_for_sex(df, 2, "female")
    print("\n[03] done.")


if __name__ == "__main__":
    main()
