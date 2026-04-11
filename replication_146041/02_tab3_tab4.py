"""Replicate Tables 3 and 4 of Rossi (2020).

Table 3: elasticities of various human-capital measures (Q) with respect to
log GDP p.w., plus the ratio between the Q and AQ elasticities. The Q
measures come from US migrants' wage regressions (Hendricks 2002-style).

Table 4: development accounting shares for India under multiple parameter
assumptions. Both tables pull heavily from intermediate .dta files produced
upstream by Q_main.do and devacc_main.do (which in turn depend on
migrants_estimates.dta and calib_all_1990_2010.dta).
"""
import numpy as np
import pandas as pd

from utils import TEMP, DATA, OUT, ols_log_elasticity


def table3():
    migrants = pd.read_stata(TEMP / "migrants_estimates.dta", convert_categoricals=False)
    aq2000   = pd.read_stata(TEMP / "AQ_2000.dta", convert_categoricals=False)

    us_pooled = migrants[(migrants["country_host"] == "United States")
                         & (migrants["Nu_o_edu3"] >= 50) & (migrants["Nu_o_edu5"] >= 50)].copy()
    us_pooled["irQ53_dum"] = np.exp(us_pooled["l_Q5_dum"])

    # country-level GDP from AQ_2000.dta (which carries l_y for every country)
    ly = aq2000[["country", "l_y"]].drop_duplicates("country")
    us_pooled = us_pooled.merge(ly, on="country", how="left")

    # Row 1 col 1: elasticity of log irQ53_dum wrt log GDP p.w. (US immigrants sample)
    b, se, n = ols_log_elasticity(us_pooled, "irQ53_dum")
    print(f"\n=== Table 3, Row 1 (US Immigrants), col 1 ===")
    print(f"  Q (US immigrants):   coef={b:.3f}  se={se:.3f}  n={n}")
    print("  Published:           0.105 (0.016)")

    # Col 5: irAQ53 elasticity on the US Pooled sample (not micro-only)
    # Need to merge irAQ columns from Q.dta or use AQ_2000 (which has them at country level).
    aq_cols = ["country",
               "irAQ53_blee_skti", "irAQ53rl_blee_skti", "irAQ53rh_blee_skti",
               "irAQ53_dum_skti_hrs_secall", "irAQ53rl_dum_skti_hrs_secall",
               "irAQ53rh_dum_skti_hrs_secall", "sample_micro"]
    aq_small = aq2000[aq_cols].drop_duplicates("country")
    us_pooled = us_pooled.merge(aq_small, on="country", how="left")

    # Column-by-column regressions
    # Cols 2-4: AQ (blee-based) on full sample
    # Cols 6-8: AQ (secall) on micro sample (because those only exist for micro countries)
    print("\n  AQ regressions on US-Pooled migrant sample (broad countries):")
    published = {
        "irAQ53_blee_skti":        (1.099, "blee σ=1.5 (col2)"),
        "irAQ53rl_blee_skti":      (1.833, "blee σ=1.3 (col3)"),
        "irAQ53rh_blee_skti":      (0.554, "blee σ=2.0 (col4)"),
    }
    for c, (_, label) in published.items():
        b, se, n = ols_log_elasticity(us_pooled, c)
        print(f"    {label:<22} coef={b:.3f}  se={se:.3f}  n={n}")

    print("\n  Re-run Q on micro-countries subset (col 5) + secall irAQ (cols 6-8):")
    micro_sub = us_pooled[us_pooled["sample_micro"] == 1].copy()
    b, se, n = ols_log_elasticity(micro_sub, "irQ53_dum")
    print(f"    Q (micro subset)         coef={b:.3f}  se={se:.3f}  n={n}  (published 0.043 (0.048))")
    for c, label in [("irAQ53_dum_skti_hrs_secall",   "secall σ=1.5 (col6)"),
                     ("irAQ53rl_dum_skti_hrs_secall", "secall σ=1.3 (col7)"),
                     ("irAQ53rh_dum_skti_hrs_secall", "secall σ=2.0 (col8)")]:
        b, se, n = ols_log_elasticity(us_pooled, c)
        print(f"    {label:<22} coef={b:.3f}  se={se:.3f}  n={n}")

    # Ratios theta_Q/theta_AQ (published values for row 1): 0.095 0.057 0.189 | 0.030 0.018 0.068
    b_q_full, _, _  = ols_log_elasticity(us_pooled, "irQ53_dum")
    b_q_micro, _, _ = ols_log_elasticity(us_pooled[us_pooled["sample_micro"] == 1], "irQ53_dum")
    ratios = []
    for c, numer in [
        ("irAQ53_blee_skti", b_q_full),
        ("irAQ53rl_blee_skti", b_q_full),
        ("irAQ53rh_blee_skti", b_q_full),
        ("irAQ53_dum_skti_hrs_secall",   b_q_micro),
        ("irAQ53rl_dum_skti_hrs_secall", b_q_micro),
        ("irAQ53rh_dum_skti_hrs_secall", b_q_micro),
    ]:
        b_aq, _, _ = ols_log_elasticity(us_pooled, c)
        ratios.append(numer / b_aq if b_aq != 0 else np.nan)
    print("\n  theta_Q / theta_AQ ratios (row 1):")
    print("    computed:  ", [f"{r:.3f}" for r in ratios])
    print("    published: ['0.095', '0.057', '0.189', '0.030', '0.018', '0.068']")


def table4():
    devacc = pd.read_stata(TEMP / "devacc.dta", convert_categoricals=False)
    india = devacc[devacc["country"] == "India"].iloc[0]

    methods = [
        ("Jensen (j)",     "_j"),
        ("Caselli-Ciccone (cc)", "_cc"),
        ("Mincerian (m)",  ""),
        ("Variant 2 (v2)", "_v2"),
    ]
    sigmas = [("σ=1.5", ""), ("σ=2",   "_s2"), ("σ=4", "_s4"), ("σ=∞", "_sinf")]
    published = {
        "Jensen (j)":            {"σ=1.5": 0.698, "σ=2": 0.289, "σ=4": 0.161, "σ=∞": 0.120},
        "Caselli-Ciccone (cc)":  {"σ=1.5": 0.104, "σ=2": 0.112, "σ=4": 0.126, "σ=∞": 0.140},
        "Mincerian (m)":         {"σ=1.5": 0.112, "σ=2": 0.123, "σ=4": 0.140, "σ=∞": 0.158},
        "Variant 2 (v2)":        {"σ=1.5": 0.112, "σ=2": 0.117, "σ=4": 0.127, "σ=∞": 0.139},
    }

    print("\n=== Table 4: Development accounting for India (y_PusmR) ===")
    print(f"{'method':<22} " + " ".join(f"{s:>10}" for s, _ in sigmas))
    rows = []
    for mlabel, msuf in methods:
        line = f"{mlabel:<22} "
        row_vals = {}
        for slabel, ssuf in sigmas:
            if msuf == "_v2":
                col = f"y_PusmR_v2{ssuf}"
            elif msuf:
                col = f"y_PusmR{ssuf}{msuf}"
            else:
                col = f"y_PusmR{ssuf}"
            val = float(india[col])
            pub = published[mlabel][slabel]
            mark = "" if abs(val - pub) < 0.0015 else "*"
            row_vals[slabel] = val
            line += f" {val:>6.3f}{mark:<3} "
        rows.append((mlabel, row_vals))
        print(line)

    print("\n=== Published values for reference ===")
    print(f"{'method':<22} " + " ".join(f"{s:>10}" for s, _ in sigmas))
    for mlabel, _ in methods:
        line = f"{mlabel:<22} "
        for slabel, _ in sigmas:
            line += f" {published[mlabel][slabel]:>6.3f}    "
        print(line)

    out = pd.DataFrame({m: r for m, r in rows}).T
    out.to_csv(OUT / "tab4_india_devacc.csv")


def main():
    table3()
    table4()


if __name__ == "__main__":
    main()
