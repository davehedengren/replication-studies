"""Replicate Table 3 (main result) and Appendix Table 1.

Discovery: Table 3 in the paper drops `share_hybrid` from the regressors
(the note says "Districts that were not in-person were either hybrid or
virtual" — i.e., the baseline is hybrid+virtual combined). Appendix Table 1
keeps both.

Spec (matches analysis.do exhibit_4, with share_hybrid removed for Table 3):
  areg pass share_inperson i.year <controls> i.year*i.state_gr
       [aw=EnrollmentTotal] if subject==X, absorb(district_unique)
       cluster(district_unique)
"""
import warnings
import numpy as np
import pandas as pd
from linearmodels.panel import PanelOLS
from utils import (
    load_state_scores,
    apply_main_treatment_replacement,
    CONTROLS,
    OUT,
)

warnings.filterwarnings("ignore")


def make_panel(df, subject, extra_treatments, extra_controls=None):
    """Return (y, X, w, n) panel ready for PanelOLS with entity_effects."""
    d = df[df.subject == subject].copy()
    keep = ["pass", "EnrollmentTotal"] + list(extra_treatments) + CONTROLS
    if extra_controls:
        keep += extra_controls
    d = d.dropna(subset=keep)
    d["year_state"] = d["year"].astype(str) + "_" + d["state"]
    ysd = pd.get_dummies(d["year_state"], drop_first=True, dtype=float)
    yd = pd.get_dummies(d["year"], prefix="y", drop_first=True, dtype=float)
    X_cols = list(extra_treatments) + CONTROLS + (extra_controls or [])
    X = d[X_cols].astype(float)
    X = pd.concat([X, yd, ysd], axis=1)
    d_idx = d.set_index(["district_unique", "year"])
    X.index = d_idx.index
    y = d_idx["pass"].astype(float)
    w = d_idx["EnrollmentTotal"].astype(float)
    return y, X, w, len(d)


def fit(y, X, w):
    m = PanelOLS(
        y, X, entity_effects=True, weights=w, drop_absorbed=True, check_rank=False
    )
    return m.fit(cov_type="clustered", cluster_entity=True)


def print_row(label, res, key_params):
    parts = [f"N={int(res.nobs):>6}"]
    for p in key_params:
        try:
            if p in res.params.index:
                b = res.params[p]
                s = res.std_errors[p]
                parts.append(f"{p}={b:.4f} ({s:.5f})")
        except Exception as e:
            parts.append(f"{p}=ERR({type(e).__name__})")
    print(f"  {label:<30}" + "  ".join(parts))


def main():
    df = load_state_scores()
    df = apply_main_treatment_replacement(df)
    df["share_blh"] = df["share_black"] + df["share_hisp"]
    df["treat21"] = (df["year"] == 2021).astype(float)
    df["inperson_2021"] = df["share_inperson"] * df["treat21"]
    df["hybrid_2021"] = df["share_hybrid"] * df["treat21"]

    rows = []
    print("\n=== Table 3 col 1 & 4: main (share_inperson only) ===")
    print("Paper targets: math 0.101 (0.00314) N=11772;  ela 0.0317 (0.00236) N=11795")
    for sub in ("math", "ela"):
        y, X, w, _ = make_panel(df, sub, ["share_inperson"])
        res = fit(y, X, w)
        print_row(f"Col 1/4 {sub}", res, ["share_inperson"])
        rows.append(
            {
                "table": "T3",
                "col": "1" if sub == "math" else "4",
                "subject": sub,
                "n": int(res.nobs),
                "b_ip": float(res.params["share_inperson"]),
                "se_ip": float(res.std_errors["share_inperson"]),
            }
        )

    print("\n=== Appendix Table 1: main + share_hybrid ===")
    print("Paper targets: math IP=0.127(0.00561) H=0.0408(0.00716) N=11772;"
          " ela IP=0.0516(0.00420) H=0.0307(0.00535) N=11795")
    for sub in ("math", "ela"):
        y, X, w, _ = make_panel(df, sub, ["share_inperson", "share_hybrid"])
        res = fit(y, X, w)
        print_row(f"AppT1 {sub}", res, ["share_inperson", "share_hybrid"])
        rows.append(
            {
                "table": "AppT1",
                "subject": sub,
                "n": int(res.nobs),
                "b_ip": float(res.params["share_inperson"]),
                "se_ip": float(res.std_errors["share_inperson"]),
                "b_hyb": float(res.params["share_hybrid"]),
                "se_hyb": float(res.std_errors["share_hybrid"]),
            }
        )

    # Stata code uses share_black (not combined) but paper labels it "Black-Hisp".
    # We run both variants to see which matches.
    for interact_var, label, paper_tgt in [
        ("share_black", "Black", "math IP=0.143 Bl*IP=-0.0386; ela IP=0.0425 Bl*IP=0.0475"),
        ("share_blh", "Black+Hisp", "(combined variant)"),
    ]:
        print(f"\n=== Table 3 interaction: {label} × InPerson × 2021 ===")
        print(f"Paper targets: {paper_tgt}")
        for sub in ("math", "ela"):
            d = df.copy()
            d["c_x_yr_ip"] = d[interact_var] * d["inperson_2021"]
            d["c_x_yr_hyb"] = d[interact_var] * d["hybrid_2021"]
            d["year_x_v"] = d["year"] * d[interact_var]
            y, X, w, _ = make_panel(
                d,
                sub,
                [
                    "inperson_2021",
                    "hybrid_2021",
                    "c_x_yr_ip",
                    "c_x_yr_hyb",
                ],
                extra_controls=["year_x_v"],
            )
            res = fit(y, X, w)
            print_row(f"{label} {sub}", res, ["inperson_2021", "c_x_yr_ip"])

    # FRPL interaction. Stata uses `share_lunch` (with NAs) in the interaction,
    # but that yields a singular covariance matrix here. We use
    # `share_lunch_updated` (which is the control version) instead.
    print("\n=== Table 3 interaction cols 3,6 (FRPL × InPerson × 2021) ===")
    print("Paper targets: math IP=0.117(0.00578) FRPL*IP=0.0262(0.0122);"
          " ela IP=0.0290(0.00430) FRPL*IP=0.0735(0.00910)")
    for sub in ("math", "ela"):
        d = df.copy()
        d["c_lun_yr_ip"] = d["share_lunch_updated"] * d["inperson_2021"]
        d["c_lun_yr_hyb"] = d["share_lunch_updated"] * d["hybrid_2021"]
        d["year_x_lun"] = d["year"] * d["share_lunch_updated"]
        try:
            y, X, w, _ = make_panel(
                d,
                sub,
                [
                    "inperson_2021",
                    "hybrid_2021",
                    "c_lun_yr_ip",
                    "c_lun_yr_hyb",
                ],
                extra_controls=["year_x_lun"],
            )
            res = fit(y, X, w)
            print_row(f"FRPL {sub}", res, ["inperson_2021", "c_lun_yr_ip"])
        except Exception as e:
            print(f"  FRPL {sub}: FAILED ({type(e).__name__}: {str(e)[:80]})")

    pd.DataFrame(rows).to_csv(OUT / "table3_main.csv", index=False)


if __name__ == "__main__":
    main()
