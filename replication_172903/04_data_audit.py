"""Data audit for replication 172903.

Checks coverage, distributions, missingness, logical consistency for both
input datasets (papers_data.dta, experimental_dataset.dta).
"""
import numpy as np
import pandas as pd

from utils import load_papers, load_experimental, first_nonmissing, OUT


def audit_papers():
    papers = load_papers()
    lines = ["=== papers_data.dta ==="]
    lines.append(f"rows = {len(papers)}")
    lines.append(f"columns = {list(papers.columns)}")
    lines.append(f"year range = [{int(papers['year'].min())}, {int(papers['year'].max())}]")
    lines.append(f"n papers per year (sorted):")
    lines.append(papers["year"].value_counts().sort_index().to_string())
    lines.append("")
    lines.append(f"rows with year==1966: {(papers['year']==1966).sum()}")
    lines.append(f"rows after dropping 1966: {(papers['year']!=1966).sum()}")
    lines.append(f"paper appendix A.5 says Figure 1 uses 257 studies")
    lines.append(f"(released data gives 265 -- 8-paper discrepancy documented in writeup)")
    lines.append("")
    dup = papers.duplicated(subset=["papertitle"], keep=False)
    lines.append(f"duplicate papertitles: {dup.sum()}")
    if dup.any():
        lines.append(papers[dup].sort_values("papertitle").to_string(index=False))
    lines.append("")
    lines.append(f"any missing year? {papers['year'].isna().sum()}")
    lines.append(f"any missing title? {papers['papertitle'].isna().sum()}")
    return "\n".join(lines)


def audit_experimental():
    exp = load_experimental()
    lines = ["=== experimental_dataset.dta ==="]
    lines.append(f"rows = {len(exp)}")
    lines.append(f"columns = {list(exp.columns)}")
    lines.append("")
    lines.append("non-null counts:")
    lines.append(exp.notna().sum().to_string())
    lines.append("")
    lines.append("summary stats:")
    lines.append(exp.describe().round(4).to_string())
    lines.append("")

    # Range / plausibility checks
    for col in [
        "child_2010_all", "child_2012_all", "child_2013_all",
        "child_2010_risk_all", "child_2012_risk_all", "child_2013_risk_all",
    ]:
        s = exp[col].dropna()
        lines.append(f"{col}: min={s.min():.4f}, max={s.max():.4f}, "
                     f"n outside [0,1]={((s<0)|(s>1)).sum()}")
    s = exp["dictator_giving2012"].dropna()
    lines.append(f"dictator_giving2012: min={s.min()}, max={s.max()}, unique={sorted(s.unique())}")
    lines.append("(should be integer 0-6: children share from 0 to 6 stickers)")
    s = exp["age_in_month_at_exp_2012"].dropna()
    lines.append(f"age_in_month_at_exp_2012: min={s.min()}, max={s.max()}, "
                 f"median={s.median()}")
    lines.append("")

    # Overlap between waves
    has_2010 = exp["child_2010_all"].notna()
    has_2012 = exp["child_2012_all"].notna()
    has_2013 = exp["child_2013_all"].notna()
    lines.append("Wave coverage for time preference:")
    lines.append(f"  only 2010  : {(has_2010 & ~has_2012 & ~has_2013).sum()}")
    lines.append(f"  only 2012  : {(~has_2010 & has_2012 & ~has_2013).sum()}")
    lines.append(f"  only 2013  : {(~has_2010 & ~has_2012 & has_2013).sum()}")
    lines.append(f"  2010+2012  : {(has_2010 & has_2012 & ~has_2013).sum()}")
    lines.append(f"  2010+2013  : {(has_2010 & ~has_2012 & has_2013).sum()}")
    lines.append(f"  2012+2013  : {(~has_2010 & has_2012 & has_2013).sum()}")
    lines.append(f"  all 3 waves: {(has_2010 & has_2012 & has_2013).sum()}")
    lines.append(f"  none       : {(~has_2010 & ~has_2012 & ~has_2013).sum()}")
    lines.append("")

    # Which wave contributes each observation for the 'time' variable
    time = first_nonmissing(exp, ["child_2010_all", "child_2012_all", "child_2013_all"])
    src = pd.Series("none", index=exp.index)
    src[exp["child_2010_all"].notna()] = "2010"
    src[exp["child_2010_all"].isna() & exp["child_2012_all"].notna()] = "2012"
    src[exp["child_2010_all"].isna() & exp["child_2012_all"].isna() & exp["child_2013_all"].notna()] = "2013"
    lines.append("Source wave for 'time' (first non-missing across 2010→2012→2013):")
    lines.append(src.value_counts().to_string())
    lines.append("")

    # Mean time by source wave (to see if selection choice biases picture)
    tmp = pd.DataFrame({"time": time, "src": src}).dropna(subset=["time"])
    lines.append("Mean 'time' by source wave:")
    lines.append(tmp.groupby("src")["time"].agg(["mean", "std", "count"]).round(4).to_string())
    lines.append("")

    # Age buckets used in Fig 4B
    age = exp["age_in_month_at_exp_2012"] / 12
    bins = [4, 4.5, 5, 5.5, 6, 6.5, 10]
    cuts = pd.cut(age.dropna(), bins=bins, right=False)
    lines.append("Age bin counts (Figure 4B):")
    lines.append(cuts.value_counts().sort_index().to_string())
    lines.append(f"(Fig 4B only shows ages 4–6.5; children ≥6.5 fall into the [6.5,10) bucket)")
    lines.append("")

    # Sample size bounded by all three rows
    nrows = len(exp)
    max_valid = max(
        exp["child_2010_all"].notna().sum(),
        exp["child_2012_all"].notna().sum(),
        exp["child_2013_all"].notna().sum(),
    )
    lines.append(f"N rows in file = {nrows}; rows with *any* time/risk signal:")
    any_time = exp[["child_2010_all","child_2012_all","child_2013_all"]].notna().any(axis=1).sum()
    any_risk = exp[["child_2010_risk_all","child_2012_risk_all","child_2013_risk_all"]].notna().any(axis=1).sum()
    lines.append(f"  any time measurement: {any_time}")
    lines.append(f"  any risk measurement: {any_risk}")
    lines.append(f"  any dictator 2012    : {exp['dictator_giving2012'].notna().sum()}")
    lines.append(f"  any age  2012        : {exp['age_in_month_at_exp_2012'].notna().sum()}")
    return "\n".join(lines)


def main():
    a = audit_papers()
    b = audit_experimental()
    out = a + "\n\n" + b
    (OUT / "data_audit.txt").write_text(out)
    print(out)
    print()
    print("wrote:", OUT / "data_audit.txt")


if __name__ == "__main__":
    main()
