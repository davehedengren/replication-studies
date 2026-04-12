"""Data audit for the Jensen & Weigel (2025, JEL) sample."""
import numpy as np
import pandas as pd

from utils import OUT, load_final


def main():
    df = load_final()
    print(f"Final sample: {df.shape[0]} countries, {df.shape[1]} columns\n")

    # 1. Coverage / completeness
    print("== 1. Variable completeness ==")
    cov = pd.DataFrame({
        "n_nonnull": df.count(),
        "pct_nonnull": (df.count() / len(df) * 100).round(1),
    }).sort_values("n_nonnull")
    print(cov.to_string())
    print()

    # 2. Income group distribution
    print("== 2. Income group distribution ==")
    ig = df["income_group_wb"].value_counts(dropna=False).rename_axis("income_group_wb")
    print(ig.to_string())
    print()

    # 3. Distributions of key variables
    print("== 3. Key variable distributions ==")
    keep = ["tax_ex_sc", "tax_inc_sc", "fte_per_laborforce", "vdem_meritocratic_recruitment",
            "staff_index", "electronic_payments_pct_number", "vdem_executive",
            "vdem_rigorous_impartial", "gdp_pc", "population", "staff_pct_master", "staff_pct_20_more"]
    desc = df[keep].describe(percentiles=[.05, .25, .5, .75, .95]).T
    print(desc.to_string())
    print()

    # 4. Plausibility — staff_pct_* should be in [0, 100] (they're percentages)
    print("== 4. Implausible percentage values ==")
    for v in ["staff_pct_master", "staff_pct_20_more",
              "electronic_payments_pct_number", "electronic_payments_pct_value"]:
        bad = df[(df[v] > 100) | (df[v] < 0)]
        print(f"  {v}: {len(bad)} rows out of [0, 100]")
        if len(bad):
            print("   ", bad[["iso_code", v]].to_dict("records"))
    print()

    # 5. Top staff_index values — these are downstream of the implausible percentages
    print("== 5. Top 10 staff_index values (paper drops > 2) ==")
    top = df.nlargest(10, "staff_index")[["iso_code", "country_isora", "staff_pct_master",
                                          "staff_pct_20_more", "staff_index"]]
    print(top.to_string(index=False))
    print()

    # 6. Cross-source consistency: tax_ex_sc <= tax_inc_sc (excl. SS <= incl. SS)
    print("== 6. tax_ex_sc <= tax_inc_sc check ==")
    bad = df[(df["tax_ex_sc"] > df["tax_inc_sc"] + 1e-9) & df["tax_inc_sc"].notna()]
    print(f"  Violations: {len(bad)}")
    if len(bad):
        print(bad[["iso_code", "tax_ex_sc", "tax_inc_sc"]].head().to_string(index=False))
    print()

    # 7. Pairwise correlations among the six predictors
    print("== 7. Correlations among Figure 1 predictors ==")
    corr = df[["fte_per_laborforce", "vdem_meritocratic_recruitment", "staff_index",
               "electronic_payments_pct_number", "vdem_executive", "vdem_rigorous_impartial"]].corr()
    print(corr.round(2).to_string())
    print()

    # 8. Concentration: how much of the slope in Figure 1A is driven by tail observations?
    print("== 8. Top FTE-per-labor-force outliers (Figure 1A leverage) ==")
    print(df.nlargest(8, "fte_per_laborforce")[
        ["iso_code", "country_isora", "fte_per_laborforce", "tax_ex_sc"]
    ].to_string(index=False))
    print()

    # 9. GRD caution: any country with very small N years contributing?
    print("== 9. Sample by income group / region ==")
    print(df.groupby("income_group_wb")["tax_ex_sc"].agg(["count", "mean", "std", "min", "max"]).to_string())

    cov.to_csv(OUT / "audit_completeness.csv")
    desc.to_csv(OUT / "audit_descriptives.csv")
    corr.to_csv(OUT / "audit_predictor_correlations.csv")


if __name__ == "__main__":
    main()
