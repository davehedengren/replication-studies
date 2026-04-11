"""Replicate Tables 1 and 2 of Rossi (2020).

Table 1 shows country-level values for the 12 micro-data countries in 2000:
skill premium w5/w3, relative labor stock H5/L3, and relative skill efficiency
irAQ53 under four measurement variants (hours, bodies, population, mincerian).
The table's bottom row gives elasticities of these quantities wrt log GDP p.w.

Table 2 runs the same elasticity regressions under alternative specifications
(baseline, gender+experience controls, self-employment, four sectors).
"""
import numpy as np
import pandas as pd

from utils import (
    load_calib, compute_wage_ratios, compute_H5L3, compute_irAQ,
    ols_log_elasticity, SIGMA, SIGMA_HIGH, SIGMA_LOW, MINC, OUT,
)


def main():
    df = load_calib()

    # baseline wage ratios (dum_skti_secall)
    df = compute_wage_ratios(df, "dum_skti_secall")
    # gender+experience spec
    df = compute_wage_ratios(df, "dumx_skti_secall")
    # self-employment spec
    df = compute_wage_ratios(df, "dumse_skti_secall")
    # sector specs
    for s in range(1, 5):
        df = compute_wage_ratios(df, f"dum_skti_sec{s}")

    # --- Baseline irAQ (hours weights) ---
    H5, L3, H5L3 = compute_H5L3(df, "dum_skti_secall", "hrs_secall")
    df["H5L3_hrs"] = H5L3
    us_mask = (df["country"] == "United States") & (df["year"] == 2000)
    df["irAQ53_hrs"] = compute_irAQ(df, df["wrat53_dum_skti_secall"], H5L3, SIGMA, us_mask)
    df["irAQ53rh_hrs"] = compute_irAQ(df, df["wrat53_dum_skti_secall"], H5L3, SIGMA_HIGH, us_mask)
    df["irAQ53rl_hrs"] = compute_irAQ(df, df["wrat53_dum_skti_secall"], H5L3, SIGMA_LOW, us_mask)

    # bodies weights
    H5b, L3b, H5L3b = compute_H5L3(df, "dum_skti_secall", "bod_secall")
    df["irAQ53_bod"] = compute_irAQ(df, df["wrat53_dum_skti_secall"], H5L3b, SIGMA, us_mask)

    # population weights
    H5p, L3p, H5L3p = compute_H5L3(df, "dum_skti_secall", "pop_secall")
    df["irAQ53_pop"] = compute_irAQ(df, df["wrat53_dum_skti_secall"], H5L3p, SIGMA, us_mask)

    # Mincerian variant: wage ratios and H5/L3 built from Barro-Lee-style exp(MINC*(y_e-y_ref))
    # wrat_e3_blee = exp(MINC*(yrs_e - yrs_3))
    wrat53_blee = np.exp(MINC * (df["yrs_5"] - df["yrs_3"]))
    H5m = np.exp(MINC * (df["yrs_4"] - df["yrs_5"])) * df["share4_pop_secall"] + \
          np.exp(MINC * (df["yrs_5"] - df["yrs_5"])) * df["share5_pop_secall"]
    L3m = np.exp(MINC * (df["yrs_1"] - df["yrs_3"])) * df["share1_pop_secall"] + \
          np.exp(MINC * (df["yrs_2"] - df["yrs_3"])) * df["share2_pop_secall"] + \
          np.exp(MINC * (df["yrs_3"] - df["yrs_3"])) * df["share3_pop_secall"]
    df["irAQ53_minc"] = compute_irAQ(df, wrat53_blee, H5m / L3m, SIGMA, us_mask)

    # gender+experience conditional irAQ
    H5x = pd.Series(0.0, index=df.index)
    L3x = pd.Series(0.0, index=df.index)
    for e in [4, 5]:
        for exp in range(1, 8):
            for g in [1, 2]:
                H5x += (df[f"wrat{e}3_dumx_skti_secall"] / df["wrat53_dumx_skti_secall"]) * \
                       np.exp(df[f"bh_exp{exp}_dumx_skti_secall"]) * \
                       np.exp(df[f"bh_g{g}_dumx_skti_secall"]) * \
                       df[f"share{e}_exp{exp}g{g}_hrs_secall"]
    for e in [1, 2, 3]:
        for exp in range(1, 8):
            for g in [1, 2]:
                L3x += (df[f"wrat{e}3_dumx_skti_secall"] / df["wrat33_dumx_skti_secall"]) * \
                       np.exp(df[f"bl_exp{exp}_dumx_skti_secall"]) * \
                       np.exp(df[f"bl_g{g}_dumx_skti_secall"]) * \
                       df[f"share{e}_exp{exp}g{g}_hrs_secall"]
    df["H5L3_x"] = H5x / L3x
    df["irAQ53_x"] = compute_irAQ(df, df["wrat53_dumx_skti_secall"], H5x / L3x, SIGMA, us_mask)
    df["irAQ53rh_x"] = compute_irAQ(df, df["wrat53_dumx_skti_secall"], H5x / L3x, SIGMA_HIGH, us_mask)
    df["irAQ53rl_x"] = compute_irAQ(df, df["wrat53_dumx_skti_secall"], H5x / L3x, SIGMA_LOW, us_mask)

    # self-employment variant
    H5se, L3se, H5L3se = compute_H5L3(df, "dumse_skti_secall", "hrs_secall")
    df["H5L3_se"] = H5L3se
    df["irAQ53_se"] = compute_irAQ(df, df["wrat53_dumse_skti_secall"], H5L3se, SIGMA, us_mask)
    df["irAQ53rh_se"] = compute_irAQ(df, df["wrat53_dumse_skti_secall"], H5L3se, SIGMA_HIGH, us_mask)
    df["irAQ53rl_se"] = compute_irAQ(df, df["wrat53_dumse_skti_secall"], H5L3se, SIGMA_LOW, us_mask)

    # sector variants — use sector-transformed elasticity sigma_sec (see AQ_main.do lines 358-393)
    # chi is a heterogeneity index computed at US 2000 and then applied to all countries.
    w_levels = {(e, s): np.exp(df[f"l_w{e}lev_dum_skti_sec{s}"]) for e in range(1, 6) for s in range(1, 5)}
    a_sec = {}
    theta_term = {}
    for s in range(1, 5):
        aH = w_levels[(4, s)] * df[f"share4_hrs_sec{s}"] + w_levels[(5, s)] * df[f"share5_hrs_sec{s}"]
        aL = (w_levels[(1, s)] * df[f"share1_hrs_sec{s}"]
              + w_levels[(2, s)] * df[f"share2_hrs_sec{s}"]
              + w_levels[(3, s)] * df[f"share3_hrs_sec{s}"])
        a_sec[s] = aH / (aH + aL)
        theta_term[s] = (aH + aL) * df[f"share_hrs_sec{s}"]
    theta_sum = sum(theta_term[s] for s in range(1, 5))
    theta_sec = {s: theta_term[s] / theta_sum for s in range(1, 5)}
    a_avg = sum(theta_sec[s] * a_sec[s] for s in range(1, 5))
    chi = sum(((a_sec[s] - a_avg) ** 2) * theta_sec[s] / (a_avg * (1 - a_avg)) for s in range(1, 5))
    chi_us = float(chi[us_mask].mean())
    sigma_sec    = (SIGMA      - chi_us / 3) / (1 - chi_us)
    sigma_sec_rh = (SIGMA_HIGH - chi_us / 3) / (1 - chi_us)
    sigma_sec_rl = (SIGMA_LOW  - chi_us / 3) / (1 - chi_us)
    print(f"chi_US_2000={chi_us:.4f}  sigma_sec={sigma_sec:.4f}  "
          f"sigma_sec_rh={sigma_sec_rh:.4f}  sigma_sec_rl={sigma_sec_rl:.4f}")

    for s in range(1, 5):
        H5s, L3s, H5L3s = compute_H5L3(df, f"dum_skti_sec{s}", f"hrs_sec{s}")
        df[f"H5L3_sec{s}"] = H5L3s
        df[f"irAQ53_sec{s}"]   = compute_irAQ(df, df[f"wrat53_dum_skti_sec{s}"], H5L3s, sigma_sec,    us_mask)
        df[f"irAQ53rh_sec{s}"] = compute_irAQ(df, df[f"wrat53_dum_skti_sec{s}"], H5L3s, sigma_sec_rh, us_mask)
        df[f"irAQ53rl_sec{s}"] = compute_irAQ(df, df[f"wrat53_dum_skti_sec{s}"], H5L3s, sigma_sec_rl, us_mask)

    # --- Table 1: country-level levels, 2000, micro sample ---
    t1_sample = df[(df["sample_micro"] == 1) & (df["year"] == 2000) & df["irAQ53_hrs"].notna()].copy()
    t1_sample = t1_sample.sort_values("l_y")
    tab1 = t1_sample[["country", "wrat53_dum_skti_secall", "H5L3_hrs",
                      "irAQ53_hrs", "irAQ53_bod", "irAQ53_pop", "irAQ53_minc"]].copy()
    tab1.columns = ["country", "w5/w3", "H5/L3", "irAQ(hrs)", "irAQ(bod)", "irAQ(pop)", "irAQ(minc)"]
    tab1.to_csv(OUT / "tab1_levels.csv", index=False)

    print("\n=== Table 1: Relative skill efficiency, 2000 ===")
    print(tab1.to_string(index=False, float_format=lambda x: f"{x:.3f}"))

    # --- Table 1 elasticities (6 cols) ---
    t1_elas = []
    for col in ["wrat53_dum_skti_secall", "H5L3_hrs", "irAQ53_hrs",
                "irAQ53_bod", "irAQ53_pop", "irAQ53_minc"]:
        b, se, n = ols_log_elasticity(t1_sample, col)
        t1_elas.append((col, b, se, n))
    print("\n=== Table 1 elasticity row ===")
    print(f"{'col':<30} {'coef':>8} {'se':>8} {'n':>4}")
    for col, b, se, n in t1_elas:
        print(f"{col:<30} {b:>8.3f} [{se:.3f}] {n:>4}")
    pd.DataFrame(t1_elas, columns=["col", "coef", "se", "n"]).to_csv(
        OUT / "tab1_elasticity.csv", index=False)

    # --- Table 2: elasticities under alternative specifications ---
    # Col sequence (5 cols): w5/w3, H5/L3, irAQ(sigma=1.5), irAQ(sigma_high=2), irAQ(sigma_low=1.3)
    # per tab_2.do (stores ey1..ey5 in that order via rl/rh) — but printed order:
    # esttab prints coeflabels, so ey1 (baseline wage ratio), ey2 (H5L3), ey3 (irAQ),
    # ey4 (irAQ rl=1.3), ey5 (irAQ rh=2). Published row 1: -0.138 0.911 1.408 2.439 0.635
    # which matches sigma=1.5->1.408, sigma=1.3->2.439 (very elastic, blows up),
    # sigma=2->0.635.
    rows = []

    def run_row(label, sub, cols):
        line = [label]
        for c in cols:
            b, se, _ = ols_log_elasticity(sub, c)
            line.append(f"{b:.3f}")
            line.append(f"[{se:.3f}]")
        rows.append(line)

    base_cols = ["wrat53_dum_skti_secall", "H5L3_hrs", "irAQ53_hrs", "irAQ53rl_hrs", "irAQ53rh_hrs"]
    x_cols    = ["wrat53_dumx_skti_secall", "H5L3_x",    "irAQ53_x",   "irAQ53rl_x",   "irAQ53rh_x"]
    se_cols   = ["wrat53_dumse_skti_secall","H5L3_se",   "irAQ53_se",  "irAQ53rl_se",  "irAQ53rh_se"]
    base_on_se_cols = ["wrat53_dum_skti_secall", "H5L3_hrs", "irAQ53_hrs", "irAQ53rl_hrs", "irAQ53rh_hrs"]

    t2_sample = t1_sample.copy()
    run_row("Baseline", t2_sample, base_cols)
    run_row("Experience and Gender", t2_sample, x_cols)
    se_sample = t2_sample[t2_sample["irAQ53_se"].notna()]
    run_row("Baseline (Self-Empl Sample)", se_sample, base_on_se_cols)
    run_row("Self-Employment", se_sample, se_cols)
    sec_labels = ["Agriculture", "Manufacturing", "Low-Skill Services", "High-Skill Services"]
    for s, label in enumerate(sec_labels, start=1):
        sec_cols = [f"wrat53_dum_skti_sec{s}", f"H5L3_sec{s}", f"irAQ53_sec{s}",
                    f"irAQ53rl_sec{s}", f"irAQ53rh_sec{s}"]
        run_row(label, t2_sample, sec_cols)

    print("\n=== Table 2: elasticities wrt log GDP p.w. ===")
    header = ["spec", "w5/w3", "se", "H5/L3", "se", "irAQ(1.5)", "se", "irAQ(1.3)", "se", "irAQ(2.0)", "se"]
    print(" | ".join(f"{h:>18}" for h in header))
    for r in rows:
        print(" | ".join(f"{c:>18}" for c in r))
    pd.DataFrame(rows, columns=header).to_csv(OUT / "tab2_elasticities.csv", index=False)


if __name__ == "__main__":
    main()
