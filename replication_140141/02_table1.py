"""Phase 2 step 2: reproduce Table 1 (Population aging and sectoral shares)."""
import pandas as pd
import numpy as np
import statsmodels.formula.api as smf
from utils import OUT, ROOT


def run_areg(df, y, extras, cluster_var="code"):
    """Fixed-effects (absorb country) regression with cluster-robust SEs.

    Stata's `areg y x, absorb(code) vce(cl code)` dummies out country FEs but
    reports only the non-FE coefficients with cluster-robust SEs. We do the
    same by adding C(code) FEs in statsmodels and using CR1 clustering.
    """
    rhs = extras + ["C(code)"]
    formula = f"{y} ~ " + " + ".join(rhs)
    model = smf.ols(formula, data=df).fit(
        cov_type="cluster", cov_kwds={"groups": df[cluster_var]})
    return model


def fmt(m, name):
    if name not in m.params.index:
        return "          -"
    b = m.params[name]
    se = m.bse[name]
    p = m.pvalues[name]
    stars = "***" if p < 0.01 else "**" if p < 0.05 else "*" if p < 0.10 else ""
    return f"{b:+7.3f}{stars:<3} ({se:.3f})"


def main():
    df = pd.read_parquet(OUT / "macro_panel.parquet")
    # Stata: sample filter `if va_share_agr!=.` (non-missing VA)
    needed = ["va_share_agr", "va_share_man", "va_share_ser",
              "hw_share_agr", "hw_share_man", "hw_share_ser",
              "share_65plus", "log_gdp_pc", "log_gdp_pc2"]
    df = df.dropna(subset=needed).copy()
    print(f"Regression sample N={len(df)} (paper reports 707)")

    sectors = [("agr", "Agriculture"), ("man", "Manufacturing"), ("ser", "Services")]
    rows = []
    for measure, label in [("hw", "Panel A: Employment share"),
                            ("va", "Panel B: Value added share")]:
        rows.append([label, "", "", "", "", "", ""])
        for var in ["share_65plus", "log_gdp_pc", "log_gdp_pc2"]:
            row = [var]
            for s, _ in sectors:
                # Col 1: just share_65plus
                m1 = run_areg(df, f"{measure}_share_{s}", ["share_65plus"])
                # Col 2: share_65plus + log_gdp_pc + log_gdp_pc2
                m2 = run_areg(df, f"{measure}_share_{s}",
                              ["log_gdp_pc", "log_gdp_pc2", "share_65plus"])
                row.extend([fmt(m1, var), fmt(m2, var)])
            rows.append(row)
        r2row = ["R²"]
        for s, _ in sectors:
            m1 = run_areg(df, f"{measure}_share_{s}", ["share_65plus"])
            m2 = run_areg(df, f"{measure}_share_{s}",
                          ["log_gdp_pc", "log_gdp_pc2", "share_65plus"])
            r2row.extend([f"{m1.rsquared:.3f}", f"{m2.rsquared:.3f}"])
        rows.append(r2row)
        rows.append(["N"] + [f"{int(m2.nobs)}"] * 6)

    cols = ["Variable",
            "Agr (1)", "Agr (2)", "Man (3)", "Man (4)", "Ser (5)", "Ser (6)"]
    out = pd.DataFrame(rows, columns=cols)
    print("\n=== Table 1: Population aging and sectoral shares ===")
    print(out.to_string(index=False))
    out.to_csv(OUT / "table1_replication.csv", index=False)

    # Compact side-by-side comparison for share_65plus row
    pub = {
        "hw_agr_1": (-1.980, 0.440), "hw_agr_2": (-0.653, 0.285),
        "hw_man_1": (-1.351, 0.323), "hw_man_2": (-0.877, 0.381),
        "hw_ser_1": (3.330, 0.586),  "hw_ser_2": (1.530, 0.490),
        "va_agr_1": (-1.012, 0.261), "va_agr_2": (-0.0575, 0.105),
        "va_man_1": (-1.533, 0.297), "va_man_2": (-1.252, 0.381),
        "va_ser_1": (2.545, 0.353),  "va_ser_2": (1.309, 0.352),
    }
    print("\n=== Comparison: share_65plus coefficient vs published ===")
    print(f"{'Spec':<12} {'Repl β':>10} {'Repl SE':>10} {'Pub β':>10} {'Pub SE':>10}  Δβ")
    for measure in ["hw", "va"]:
        for s, _ in sectors:
            for i, extras in enumerate(
                    [["share_65plus"],
                     ["log_gdp_pc", "log_gdp_pc2", "share_65plus"]], start=1):
                m = run_areg(df, f"{measure}_share_{s}", extras)
                b = m.params["share_65plus"]
                se = m.bse["share_65plus"]
                pub_b, pub_se = pub[f"{measure}_{s}_{i}"]
                tag = f"{measure}_{s}_{i}"
                dlt = b - pub_b
                print(f"{tag:<12} {b:10.4f} {se:10.4f} {pub_b:10.4f} {pub_se:10.4f}  {dlt:+.4f}")


if __name__ == "__main__":
    main()
