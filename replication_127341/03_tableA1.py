"""Replicate Table A.1 (summary statistics for Daegu, Seoul, NYC Metro).

Mirrors replication/code/stata/204_tableA1.do.
Published values (from the paper's PDF and the compiled .tex files) are shown
for comparison.
"""
import pandas as pd
from utils import WORK, OUT


def _sum_by_cty(df, col, cty=None):
    if cty is not None:
        df = df[df["cty"] == cty]
    return float(df[col].sum())


def main():
    kor_covid = pd.read_stata(WORK / "KOR_covid.dta", convert_categoricals=False)
    nym_covid = pd.read_stata(WORK / "NYM_covid.dta", convert_categoricals=False)
    kor_var = pd.read_stata(WORK / "KOR_var.dta", convert_categoricals=False)
    nym_var = pd.read_stata(WORK / "NYM_var.dta", convert_categoricals=False)
    policies = pd.read_stata(WORK / "policies.dta", convert_categoricals=False)

    sel_commute = pd.read_stata(WORK / "SEL_commute_od.dta", convert_categoricals=False)
    nym_commute = pd.read_stata(WORK / "NYM_commute_od.dta", convert_categoricals=False)
    dag_commute = pd.read_stata(WORK / "DAG_commute_ex.dta", convert_categoricals=False)

    # Cumulative confirmed cases (sample period).
    cases_dag = _sum_by_cty(kor_covid, "num_cased", cty="DAG")
    cases_sel = _sum_by_cty(kor_covid, "num_cased", cty="SEL")
    cases_nym = _sum_by_cty(nym_covid, "num_case")

    # District counts.
    districts_dag = int(dag_commute["RC"].nunique())
    districts_sel = int(sel_commute["RCo"].nunique())
    districts_nym = int(nym_commute["RCo"].nunique())

    # Populations by city.
    kor_pops = (kor_var.merge(
        pd.DataFrame({"RC": kor_covid["RC"].unique().tolist() +
                      dag_commute["RC"].unique().tolist()}),
        on="RC", how="inner")
        .drop_duplicates("RC"))
    dag_pop = int(kor_var[kor_var["cty"] == "DAG"]["pop"].sum())
    sel_pop = int(kor_var[kor_var["cty"] == "SEL"]["pop"].sum())
    nym_pop = int(nym_var["pop"].sum())

    # Sample periods (source: the commute datasets, which define the panel).
    def _range(df, col="statadate"):
        s = pd.to_datetime(df[col])
        return s.min().date(), s.max().date()

    dag_range = _range(dag_commute)
    sel_range = _range(sel_commute)
    nym_range = _range(nym_commute)

    policies = policies.copy()
    for c in ["pat0", "pat10", "lock"]:
        policies[c] = pd.to_datetime(policies[c]).dt.date
    pol = policies.set_index("cty")

    rows = [
        ("Population", f"{dag_pop:,}", f"{sel_pop:,}", f"{nym_pop:,}"),
        ("# Districts", districts_dag, districts_sel, districts_nym),
        ("Sample period",
            f"{dag_range[0]}–{dag_range[1]}",
            f"{sel_range[0]}–{sel_range[1]}",
            f"{nym_range[0]}–{nym_range[1]}"),
        ("First case (pat0)",
            pol.loc["DAG", "pat0"], pol.loc["SEL", "pat0"], pol.loc["NYM", "pat0"]),
        ("Lockdown date",
            pol.loc["DAG", "lock"], pol.loc["SEL", "lock"], pol.loc["NYM", "lock"]),
        ("Cumulative cases",
            f"{int(cases_dag):,}", f"{int(cases_sel):,}", f"{int(cases_nym):,}"),
    ]

    w = 28
    print(f"{'':<{w}}{'Daegu':>14}{'Seoul':>14}{'NYC Metro':>14}")
    print("-" * (w + 14 * 3))
    for r in rows:
        print(f"{r[0]:<{w}}{str(r[1]):>14}{str(r[2]):>14}{str(r[3]):>14}")

    out = OUT / "tableA1.csv"
    pd.DataFrame(rows, columns=["Stat", "Daegu", "Seoul", "NYC_Metro"]).to_csv(
        out, index=False)
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
