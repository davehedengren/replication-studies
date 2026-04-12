"""Robustness checks for the main Table 3 result.

All checks use the main spec: pass ~ share_inperson + controls, with district
FE, year FE, year×state FE, WLS by enrollment, clustered SEs on district.
"""
import warnings
import numpy as np
import pandas as pd
from linearmodels.panel import PanelOLS
from utils import load_state_scores, apply_main_treatment_replacement, CONTROLS, OUT

warnings.filterwarnings("ignore")


def run(df, subject, treat_cols=("share_inperson",), extra=None, cluster_var=None):
    d = df[df.subject == subject].copy()
    d = d.dropna(subset=["pass", "EnrollmentTotal"] + list(treat_cols) + CONTROLS)
    d["year_state"] = d["year"].astype(str) + "_" + d["state"]
    ysd = pd.get_dummies(d["year_state"], drop_first=True, dtype=float)
    yd = pd.get_dummies(d["year"], prefix="y", drop_first=True, dtype=float)
    X_cols = list(treat_cols) + CONTROLS + (extra or [])
    X = d[X_cols].astype(float)
    X = pd.concat([X, yd, ysd], axis=1)
    d_idx = d.set_index(["district_unique", "year"])
    X.index = d_idx.index
    y = d_idx["pass"].astype(float)
    w = d_idx["EnrollmentTotal"].astype(float)
    m = PanelOLS(
        y, X, entity_effects=True, weights=w, drop_absorbed=True, check_rank=False
    )
    res = m.fit(cov_type="clustered", cluster_entity=True)
    return res


def summarize(name, res, params):
    out = {"check": name, "n": int(res.nobs)}
    for p in params:
        if p in res.params.index:
            out[f"{p}_b"] = round(float(res.params[p]), 4)
            out[f"{p}_se"] = round(float(res.std_errors[p]), 5)
    return out


def main():
    df = load_state_scores()
    df = apply_main_treatment_replacement(df)
    df["share_blh"] = df["share_black"] + df["share_hisp"]

    rows = []

    # --- 1. Baseline (replicate main)
    for sub in ("math", "ela"):
        r = run(df, sub)
        rows.append({**summarize("1. Baseline", r, ["share_inperson"]), "subject": sub})

    # --- 2. Leave-one-state out
    for st in sorted(df.state.unique()):
        sub_df = df[df.state != st]
        for sub in ("math", "ela"):
            r = run(sub_df, sub)
            rows.append(
                {
                    **summarize(f"2. Drop {st}", r, ["share_inperson"]),
                    "subject": sub,
                }
            )

    # --- 3. Drop VA (large outlier: -32pp math decline)
    for sub in ("math", "ela"):
        r = run(df[df.state != "VA"], sub)
        rows.append(
            {**summarize("3. Drop VA outlier", r, ["share_inperson"]), "subject": sub}
        )

    # --- 4. Drop top 1% enrollment districts (very large)
    cut = df["EnrollmentTotal"].quantile(0.99)
    sub_df = df[df["EnrollmentTotal"] < cut]
    for sub in ("math", "ela"):
        r = run(sub_df, sub)
        rows.append(
            {
                **summarize("4. Drop top 1% enrollment", r, ["share_inperson"]),
                "subject": sub,
            }
        )

    # --- 5. Unweighted (drop enrollment weights)
    def run_unweighted(df, sub):
        d = df[df.subject == sub].copy().dropna(
            subset=["pass", "share_inperson"] + CONTROLS
        )
        d["year_state"] = d["year"].astype(str) + "_" + d["state"]
        ysd = pd.get_dummies(d["year_state"], drop_first=True, dtype=float)
        yd = pd.get_dummies(d["year"], prefix="y", drop_first=True, dtype=float)
        X = d[["share_inperson"] + CONTROLS].astype(float)
        X = pd.concat([X, yd, ysd], axis=1)
        d_idx = d.set_index(["district_unique", "year"])
        X.index = d_idx.index
        y = d_idx["pass"].astype(float)
        m = PanelOLS(
            y, X, entity_effects=True, drop_absorbed=True, check_rank=False
        )
        return m.fit(cov_type="clustered", cluster_entity=True)

    for sub in ("math", "ela"):
        r = run_unweighted(df, sub)
        rows.append(
            {**summarize("5. Unweighted OLS", r, ["share_inperson"]), "subject": sub}
        )

    # --- 6. Drop district-specific year×state, replace with just year FE
    def run_no_yearstate(df, sub):
        d = df[df.subject == sub].copy().dropna(
            subset=["pass", "share_inperson", "EnrollmentTotal"] + CONTROLS
        )
        yd = pd.get_dummies(d["year"], prefix="y", drop_first=True, dtype=float)
        X = d[["share_inperson"] + CONTROLS].astype(float)
        X = pd.concat([X, yd], axis=1)
        d_idx = d.set_index(["district_unique", "year"])
        X.index = d_idx.index
        y = d_idx["pass"].astype(float)
        w = d_idx["EnrollmentTotal"].astype(float)
        m = PanelOLS(
            y, X, entity_effects=True, weights=w, drop_absorbed=True, check_rank=False
        )
        return m.fit(cov_type="clustered", cluster_entity=True)

    for sub in ("math", "ela"):
        r = run_no_yearstate(df, sub)
        rows.append(
            {
                **summarize("6. No year×state FE", r, ["share_inperson"]),
                "subject": sub,
            }
        )

    # --- 7. Include share_hybrid (App Table 1 spec)
    for sub in ("math", "ela"):
        r = run(df, sub, treat_cols=("share_inperson", "share_hybrid"))
        rows.append(
            {
                **summarize("7. Add share_hybrid", r, ["share_inperson", "share_hybrid"]),
                "subject": sub,
            }
        )

    # --- 8. Placebo: drop 2021, predict pre-pandemic pass on share_inperson.
    #    Since share_inperson is forced to 1 pre-2021, this is a degenerate regressor.
    #    Instead use the raw share (undo the replacement) on pre-period only — no
    #    pandemic effect should show up.
    df_raw = load_state_scores()  # without the replacement
    df_raw = df_raw[df_raw.year < 2021]
    for sub in ("math", "ela"):
        d = df_raw[df_raw.subject == sub].copy().dropna(
            subset=["pass", "share_inperson", "EnrollmentTotal"] + CONTROLS
        )
        d["year_state"] = d["year"].astype(str) + "_" + d["state"]
        ysd = pd.get_dummies(d["year_state"], drop_first=True, dtype=float)
        yd = pd.get_dummies(d["year"], prefix="y", drop_first=True, dtype=float)
        X = d[["share_inperson"] + CONTROLS].astype(float)
        X = pd.concat([X, yd, ysd], axis=1)
        d_idx = d.set_index(["district_unique", "year"])
        X.index = d_idx.index
        y = d_idx["pass"].astype(float)
        w = d_idx["EnrollmentTotal"].astype(float)
        try:
            m = PanelOLS(
                y, X, entity_effects=True, weights=w, drop_absorbed=True,
                check_rank=False,
            )
            r = m.fit(cov_type="clustered", cluster_entity=True)
            rows.append(
                {
                    **summarize("8. Placebo (pre-period raw)", r, ["share_inperson"]),
                    "subject": sub,
                }
            )
        except Exception as e:
            rows.append({"check": "8. Placebo", "subject": sub, "error": str(e)[:80]})

    # --- 9. Random permutation of share_inperson within state × year
    rng = np.random.default_rng(42)
    df_perm = df.copy()
    df_perm["share_inperson"] = df_perm.groupby(["state", "year"])[
        "share_inperson"
    ].transform(lambda s: pd.Series(rng.permutation(s.values), index=s.index))
    for sub in ("math", "ela"):
        r = run(df_perm, sub)
        rows.append(
            {
                **summarize("9. Permuted treatment", r, ["share_inperson"]),
                "subject": sub,
            }
        )

    # --- 10. Use share_virtual as treatment (should give opposite sign)
    for sub in ("math", "ela"):
        r = run(df, sub, treat_cols=("share_virtual",))
        rows.append(
            {
                **summarize("10. Treatment=virtual", r, ["share_virtual"]),
                "subject": sub,
            }
        )

    # --- 11. Subsample: drop small districts (<500 enrollment)
    sub_df = df[df["EnrollmentTotal"] >= 500]
    for sub in ("math", "ela"):
        r = run(sub_df, sub)
        rows.append(
            {
                **summarize("11. Drop small (<500)", r, ["share_inperson"]),
                "subject": sub,
            }
        )

    # --- 12. Cluster at state level instead of district
    def run_state_cluster(df, sub):
        d = df[df.subject == sub].copy().dropna(
            subset=["pass", "share_inperson", "EnrollmentTotal"] + CONTROLS
        )
        d["year_state"] = d["year"].astype(str) + "_" + d["state"]
        ysd = pd.get_dummies(d["year_state"], drop_first=True, dtype=float)
        yd = pd.get_dummies(d["year"], prefix="y", drop_first=True, dtype=float)
        X = d[["share_inperson"] + CONTROLS].astype(float)
        X = pd.concat([X, yd, ysd], axis=1)
        d_idx = d.set_index(["district_unique", "year"])
        X.index = d_idx.index
        y = d_idx["pass"].astype(float)
        w = d_idx["EnrollmentTotal"].astype(float)
        clusters = pd.Series(
            d["state"].values, index=d_idx.index, name="cluster"
        )
        m = PanelOLS(
            y, X, entity_effects=True, weights=w, drop_absorbed=True, check_rank=False
        )
        return m.fit(cov_type="clustered", clusters=clusters)

    for sub in ("math", "ela"):
        r = run_state_cluster(df, sub)
        rows.append(
            {
                **summarize("12. Cluster by state", r, ["share_inperson"]),
                "subject": sub,
            }
        )

    out = pd.DataFrame(rows)
    pd.options.display.float_format = "{:.4f}".format
    print("\n=== Robustness checks ===")
    print(out.to_string(index=False))
    out.to_csv(OUT / "robustness.csv", index=False)


if __name__ == "__main__":
    main()
