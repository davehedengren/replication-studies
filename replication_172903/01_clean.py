"""Load and sanity-check the two input datasets used for Figures 1, 3, 4."""
from utils import load_papers, load_experimental, first_nonmissing, OUT


def main():
    papers = load_papers()
    print("=== papers_data.dta ===")
    print("rows:", len(papers))
    print("columns:", list(papers.columns))
    print()
    print("year counts:")
    print(papers["year"].value_counts().sort_index().to_string())
    print()
    print("after dropping year==1966:", (papers["year"] != 1966).sum())
    print("paper (appendix A.5) claims: 257 studies meeting criteria")
    print()

    exp = load_experimental()
    print("=== experimental_dataset.dta ===")
    print("rows:", len(exp))
    print("columns:", list(exp.columns))
    print()
    print("non-null counts:")
    print(exp.notna().sum().to_string())
    print()

    time = first_nonmissing(exp, ["child_2010_all", "child_2012_all", "child_2013_all"])
    risk = first_nonmissing(
        exp, ["child_2010_risk_all", "child_2012_risk_all", "child_2013_risk_all"]
    )
    print("time non-null:", time.notna().sum())
    print("risk non-null:", risk.notna().sum())
    print("dictator_giving2012 non-null:", exp["dictator_giving2012"].notna().sum())
    print("age_in_month_at_exp_2012 non-null:", exp["age_in_month_at_exp_2012"].notna().sum())

    clean = exp.copy()
    clean["time"] = time
    clean["risk"] = risk
    clean["d2012"] = exp["dictator_giving2012"] / 6
    clean["age2012"] = exp["age_in_month_at_exp_2012"] / 12
    clean.to_parquet(OUT / "clean_experimental.parquet")
    papers.to_parquet(OUT / "clean_papers.parquet")
    print("\nwrote:", OUT / "clean_experimental.parquet")
    print("wrote:", OUT / "clean_papers.parquet")


if __name__ == "__main__":
    main()
