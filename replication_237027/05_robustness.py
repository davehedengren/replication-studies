"""Robustness checks for Jensen & Weigel (2025, JEL).

The paper makes a series of bivariate descriptive claims via Figure 1 (six
positive correlations) and conditional claims via Figure 2 (interactions with
state legitimacy). We probe:
  1. Drop the Tajikistan data-quality outlier on staff %s
  2. Drop the top-3 FTE outliers (Slovenia, Belgium, Poland)
  3. Use tax_inc_sc (incl. social security) as the dependent variable
  4. Restrict to non-high-income countries
  5. Restrict to non-low-income countries
  6. Drop EU members (capacity-driven Eurozone correlation)
  7. Add log GDP per capita as a control (does admin add anything beyond GDP?)
  8. Multivariate horse race: all six predictors together
  9. Spearman (rank) correlation instead of Pearson/OLS
 10. Bootstrap 95% CIs on each Figure 1 slope (1,000 resamples)
 11. Recompute Figure 2C without the staff_index <= 2 restriction
 12. Use HC3 robust SEs everywhere instead of classical
"""
import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats

from utils import OUT, load_final, ols_fit


PREDS = [
    ("fte_per_laborforce",            "Fig1A FTE/labor force"),
    ("vdem_meritocratic_recruitment", "Fig1B Meritocratic"),
    ("staff_index",                   "Fig1C Staff index"),
    ("electronic_payments_pct_number","Fig1D E-payments %"),
    ("vdem_executive",                "Fig1E Constraints exec"),
    ("vdem_rigorous_impartial",       "Fig1F Impartial admin"),
]

EU_ISO = {"AUT","BEL","BGR","HRV","CYP","CZE","DNK","EST","FIN","FRA","DEU","GRC",
          "HUN","IRL","ITA","LVA","LTU","LUX","MLT","NLD","POL","PRT","ROU","SVK",
          "SVN","ESP","SWE"}


def slope_only(df, y, x, restrict=None):
    sub = df.copy()
    if restrict is not None:
        sub = restrict(sub)
    fit = ols_fit(sub[y], sub[x])
    if fit is None:
        return (np.nan, np.nan, 0)
    return (fit["slope"], fit["se_slope"], fit["n"])


def main():
    df = load_final()
    print("== Robustness checks for Figure 1 slopes ==\n")
    rows = []
    base = []
    for var, label in PREDS:
        b0, se0, n0 = slope_only(df, "tax_ex_sc", var,
                                 restrict=(lambda d, v=var: d[d["staff_index"] <= 2]) if var == "staff_index" else None)
        base.append((var, label, b0, se0, n0))

    # 1 - drop TJK
    print("1. Drop TJK (data outlier on staff_pct_*)")
    for var, label, b0, se0, n0 in base:
        b, se, n = slope_only(df, "tax_ex_sc", var,
                              restrict=lambda d, v=var: (d[d["iso_code"] != "TJK"] if v != "staff_index"
                                                          else d[(d["iso_code"] != "TJK") & (d["staff_index"] <= 2)]))
        rows.append({"check": "1_drop_TJK", "var": var, "slope": b, "se": se, "n": n,
                     "base_slope": b0, "delta_pct": (b - b0) / b0 * 100 if b0 else np.nan})

    print("2. Drop top-3 FTE outliers (SVN, BEL, POL)")
    drop3 = {"SVN", "BEL", "POL"}
    for var, label, b0, se0, n0 in base:
        b, se, n = slope_only(df, "tax_ex_sc", var,
                              restrict=lambda d, v=var: (d[~d["iso_code"].isin(drop3)] if v != "staff_index"
                                                          else d[(~d["iso_code"].isin(drop3)) & (d["staff_index"] <= 2)]))
        rows.append({"check": "2_drop_top3_fte", "var": var, "slope": b, "se": se, "n": n,
                     "base_slope": b0, "delta_pct": (b - b0) / b0 * 100 if b0 else np.nan})

    print("3. Use tax_inc_sc (incl. social security)")
    for var, label, b0, se0, n0 in base:
        b, se, n = slope_only(df, "tax_inc_sc", var,
                              restrict=(lambda d, v=var: d[d["staff_index"] <= 2]) if var == "staff_index" else None)
        rows.append({"check": "3_tax_inc_sc", "var": var, "slope": b, "se": se, "n": n,
                     "base_slope": b0, "delta_pct": (b - b0) / b0 * 100 if b0 else np.nan})

    print("4. Drop high-income countries")
    for var, label, b0, se0, n0 in base:
        b, se, n = slope_only(df, "tax_ex_sc", var,
                              restrict=lambda d, v=var: (d[d["income_group_wb"] != "High income"] if v != "staff_index"
                                                          else d[(d["income_group_wb"] != "High income") & (d["staff_index"] <= 2)]))
        rows.append({"check": "4_drop_high_income", "var": var, "slope": b, "se": se, "n": n,
                     "base_slope": b0, "delta_pct": (b - b0) / b0 * 100 if b0 else np.nan})

    print("5. Drop low-income countries")
    for var, label, b0, se0, n0 in base:
        b, se, n = slope_only(df, "tax_ex_sc", var,
                              restrict=lambda d, v=var: (d[d["income_group_wb"] != "Low income"] if v != "staff_index"
                                                          else d[(d["income_group_wb"] != "Low income") & (d["staff_index"] <= 2)]))
        rows.append({"check": "5_drop_low_income", "var": var, "slope": b, "se": se, "n": n,
                     "base_slope": b0, "delta_pct": (b - b0) / b0 * 100 if b0 else np.nan})

    print("6. Drop EU members")
    for var, label, b0, se0, n0 in base:
        b, se, n = slope_only(df, "tax_ex_sc", var,
                              restrict=lambda d, v=var: (d[~d["iso_code"].isin(EU_ISO)] if v != "staff_index"
                                                          else d[(~d["iso_code"].isin(EU_ISO)) & (d["staff_index"] <= 2)]))
        rows.append({"check": "6_drop_EU", "var": var, "slope": b, "se": se, "n": n,
                     "base_slope": b0, "delta_pct": (b - b0) / b0 * 100 if b0 else np.nan})

    print("7. Control for log(GDP per capita)")
    df["log_gdp"] = np.log(df["gdp_pc"])
    rows7 = []
    for var, label, b0, se0, n0 in base:
        sub = df.copy()
        if var == "staff_index":
            sub = sub[sub["staff_index"] <= 2]
        sub = sub.dropna(subset=["tax_ex_sc", var, "log_gdp"])
        X = sm.add_constant(sub[[var, "log_gdp"]])
        res = sm.OLS(sub["tax_ex_sc"], X).fit()
        rows.append({"check": "7_control_log_gdp", "var": var, "slope": res.params[var],
                     "se": res.bse[var], "n": int(res.nobs), "base_slope": b0,
                     "delta_pct": (res.params[var] - b0) / b0 * 100})

    print("8. Multivariate horse race")
    sub = df[df["staff_index"] <= 2].copy()
    keep = ["tax_ex_sc"] + [v for v, _ in PREDS]
    sub = sub.dropna(subset=keep)
    X = sm.add_constant(sub[[v for v, _ in PREDS]])
    res = sm.OLS(sub["tax_ex_sc"], X).fit()
    print(f"   N = {int(res.nobs)}, R² = {res.rsquared:.3f}")
    horse = pd.DataFrame({
        "var": [v for v, _ in PREDS],
        "coef": [res.params[v] for v, _ in PREDS],
        "se":   [res.bse[v]    for v, _ in PREDS],
        "t":    [res.tvalues[v] for v, _ in PREDS],
        "p":    [res.pvalues[v] for v, _ in PREDS],
    })
    print(horse.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    horse.to_csv(OUT / "robust_horserace.csv", index=False)

    print("\n9. Spearman (rank) correlation")
    for var, label, b0, se0, n0 in base:
        sub = df.copy()
        if var == "staff_index":
            sub = sub[sub["staff_index"] <= 2]
        sub = sub.dropna(subset=["tax_ex_sc", var])
        rho, p = stats.spearmanr(sub["tax_ex_sc"], sub[var])
        rows.append({"check": "9_spearman_rho", "var": var, "slope": rho, "se": np.nan,
                     "n": len(sub), "base_slope": b0, "delta_pct": np.nan, "p": p})

    print("10. Bootstrap 95% CIs (1,000 resamples)")
    rng = np.random.default_rng(20240411)
    for var, label, b0, se0, n0 in base:
        sub = df.copy()
        if var == "staff_index":
            sub = sub[sub["staff_index"] <= 2]
        sub = sub.dropna(subset=["tax_ex_sc", var]).reset_index(drop=True)
        slopes = np.empty(1000)
        n = len(sub)
        for i in range(1000):
            idx = rng.integers(0, n, n)
            x = sub[var].iloc[idx].to_numpy()
            y = sub["tax_ex_sc"].iloc[idx].to_numpy()
            slopes[i], _ = np.polyfit(x, y, 1)
        lo, hi = np.percentile(slopes, [2.5, 97.5])
        rows.append({"check": "10_bootstrap_ci", "var": var, "slope": b0, "se": slopes.std(),
                     "n": n, "base_slope": b0, "delta_pct": np.nan, "ci_lo": lo, "ci_hi": hi})

    print("11. Figure 2C without staff_index <= 2 restriction")
    sub = df.dropna(subset=["tax_ex_sc", "staff_index"]).copy()
    sub["z_staff"] = (sub["staff_index"] - sub["staff_index"].mean()) / sub["staff_index"].std()
    sub["z_exec"] = (sub["vdem_executive"] - sub["vdem_executive"].mean()) / sub["vdem_executive"].std()
    sub["z_imp"] = (sub["vdem_rigorous_impartial"] - sub["vdem_rigorous_impartial"].mean()) / sub["vdem_rigorous_impartial"].std()
    sub["legit"] = sub[["z_exec", "z_imp"]].mean(axis=1)
    sub["interaction"] = sub["z_staff"] * sub["legit"]
    X = sm.add_constant(sub[["z_staff", "legit", "interaction"]])
    res2 = sm.OLS(sub["tax_ex_sc"], X).fit()
    print(f"   N = {int(res2.nobs)}, b_int = {res2.params['interaction']:.4f}, p = {res2.pvalues['interaction']:.4f}")

    print("12. HC3 robust SEs vs classical for Figure 1 slopes")
    for var, label, b0, se0, n0 in base:
        sub = df.copy()
        if var == "staff_index":
            sub = sub[sub["staff_index"] <= 2]
        sub = sub.dropna(subset=["tax_ex_sc", var])
        X = sm.add_constant(sub[[var]])
        r_classic = sm.OLS(sub["tax_ex_sc"], X).fit()
        r_hc3 = sm.OLS(sub["tax_ex_sc"], X).fit(cov_type="HC3")
        rows.append({"check": "12_HC3_se", "var": var, "slope": r_classic.params[var],
                     "se": r_hc3.bse[var], "n": int(r_classic.nobs), "base_slope": b0,
                     "delta_pct": (r_hc3.bse[var] - se0) / se0 * 100,
                     "se_classic": r_classic.bse[var]})

    out = pd.DataFrame(rows)
    out.to_csv(OUT / "robustness_results.csv", index=False)
    print(f"\nWrote {OUT/'robustness_results.csv'}  ({len(out)} rows)\n")

    print("\n== Compact summary: each predictor's slope across checks ==")
    pivot = out[out["check"].isin(["1_drop_TJK", "2_drop_top3_fte", "3_tax_inc_sc",
                                   "4_drop_high_income", "5_drop_low_income", "6_drop_EU",
                                   "7_control_log_gdp"])].pivot(index="var", columns="check", values="slope")
    base_df = pd.DataFrame({"baseline": [b for _, _, b, _, _ in base]},
                           index=[v for v, _, _, _, _ in base])
    pivot = base_df.join(pivot)
    print(pivot.round(4).to_string())


if __name__ == "__main__":
    main()
