"""Replicate Table 2: Determinants of Schooling Mode.

Regresses 2021 share_inperson on district demographics and case rates,
weighted by enrollment, with and without state FE.
"""
import numpy as np
import pandas as pd
import statsmodels.api as sm
from utils import load_state_scores, OUT


def run_reg(df, y, xs, add_state_fe=False):
    d = df.copy()
    # Previous pass rate (average of pre-2021 pass rate, within-state demeaned)
    if "old_pass" in xs and "old_pass" not in d.columns:
        raise ValueError
    data = d.dropna(subset=[y] + xs + ["EnrollmentTotal"]).copy()
    X = data[xs].copy()
    if add_state_fe:
        state_dum = pd.get_dummies(data["state"], drop_first=True, dtype=float)
        X = pd.concat([X, state_dum], axis=1)
    X = sm.add_constant(X)
    X = X.astype(float)
    model = sm.WLS(
        data[y].astype(float), X, weights=data["EnrollmentTotal"].astype(float)
    )
    res = model.fit(cov_type="HC1")
    return res, data


def main():
    df = load_state_scores()
    # Build district-level cross section using 2021 rows (one per district per subject).
    # Use math subject to avoid double counting.
    df_m = df[df.subject == "math"].copy()

    # old_pass = mean pre-2021 pass, within-state demeaned (per analysis.do exhibit_2)
    df_m["temp1"] = np.where(df_m["year"] < 2021, df_m["pass"], np.nan)
    df_m["old_pass"] = df_m.groupby(["state", "district_id"])["temp1"].transform("mean")
    state_mean = df_m.groupby("state")["temp1"].transform("mean")
    df_m["old_pass"] = df_m["old_pass"] - state_mean

    cs = df_m[df_m.year == 2021].copy()
    cs["case_rate"] = cs["case_rate_per100k_zip"] / 100.0
    cs["share_blh"] = cs["share_black"] + cs["share_hisp"]

    rows = []
    # Table 2 in paper runs one regression per covariate (univariate + controls?)
    # Looking at Stata code, exhibit_2 runs `reg share_inperson <var>` for each var.
    for var, label in [
        ("old_pass", "Prev Pass"),
        ("share_black", "Share Black"),
        ("share_hisp", "Share Hispanic"),
        ("share_lunch", "Share FRPL"),
        ("share_ELL", "Share ELL"),
        ("case_rate", "Case Rate"),
        ("trump_vote_share", "Trump Vote"),
    ]:
        try:
            res1, _ = run_reg(cs, "share_inperson", [var], add_state_fe=False)
            res2, _ = run_reg(cs, "share_inperson", [var], add_state_fe=True)
            rows.append(
                {
                    "var": label,
                    "b_no_fe": res1.params.get(var),
                    "se_no_fe": res1.bse.get(var),
                    "b_state_fe": res2.params.get(var),
                    "se_state_fe": res2.bse.get(var),
                    "N": int(res1.nobs),
                }
            )
        except Exception as e:
            rows.append({"var": label, "error": str(e)})

    tbl = pd.DataFrame(rows)
    pd.options.display.float_format = "{:.4f}".format
    print("\n=== Table 2-ish: Determinants of share_inperson (univariate) ===")
    print(tbl.to_string(index=False))
    tbl.to_csv(OUT / "table2_determinants.csv", index=False)


if __name__ == "__main__":
    main()
