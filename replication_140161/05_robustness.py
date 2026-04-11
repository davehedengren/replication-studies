"""Robustness checks for the main ATE on sharing alt-facts (Table 2, col 1).

Base specification:
    want_share_fb ~ Imposed + Voluntary  (robust SEs)
Baseline result: Imposed = -0.045 (0.016), Voluntary = -0.038 (0.016),
                 control mean = 0.147, N = 2534.

Each check reports how Imposed + Voluntary coefficients move.
"""
import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats

from replication_140161.utils import build_all, OUT


def fit(y, X):
    Xc = sm.add_constant(X, has_constant="add")
    return sm.OLS(y, Xc, missing="drop").fit(cov_type="HC1")


def row(name, m):
    return {
        "check": name,
        "N": int(m.nobs),
        "imposed_b": m.params.get("imposed", np.nan),
        "imposed_se": m.bse.get("imposed", np.nan),
        "imposed_p": m.pvalues.get("imposed", np.nan),
        "voluntary_b": m.params.get("voluntary", np.nan),
        "voluntary_se": m.bse.get("voluntary", np.nan),
        "voluntary_p": m.pvalues.get("voluntary", np.nan),
        "control_mean": m.params.get("const", np.nan),
        "r2": m.rsquared,
    }


def main():
    df = build_all()
    results = []

    # ---- 0. Baseline -------------------------------------------------------
    base = fit(df["want_share_fb"], df[["imposed", "voluntary"]])
    results.append(row("0. baseline", base))

    # ---- 1. Probit instead of LPM -----------------------------------------
    Xc = sm.add_constant(df[["imposed", "voluntary"]])
    probit = sm.Probit(df["want_share_fb"], Xc).fit(disp=False)
    mfx = probit.get_margeff(at="overall")
    results.append({
        "check": "1. probit (avg marginal effects)",
        "N": int(probit.nobs),
        "imposed_b": mfx.margeff[0], "imposed_se": mfx.margeff_se[0],
        "imposed_p": mfx.pvalues[0],
        "voluntary_b": mfx.margeff[1], "voluntary_se": mfx.margeff_se[1],
        "voluntary_p": mfx.pvalues[1],
        "control_mean": df.loc[df["survey"] == 1, "want_share_fb"].mean(),
        "r2": probit.prsquared,
    })

    # ---- 2. Control for age, educ, religion, second_mlp, male (proxy) ----
    c = df.dropna(subset=["age", "educ", "second_mlp"]).copy()
    m = fit(c["want_share_fb"],
            c[["imposed", "voluntary", "age", "educ", "second_mlp"]])
    results.append(row("2. + basic demographics", m))

    # ---- 3. Drop observations with duration > 99th percentile -------------
    # proxy for satisficing: very long surveys may be distracted respondents.
    # We don't have duration in the cleaned df, re-load it via indicator
    from replication_140161.utils import load_survey
    dur = pd.concat([load_survey(i)[["durationinseconds"]] for i in (1, 2, 3)],
                    ignore_index=True)
    dur["durationinseconds"] = pd.to_numeric(dur["durationinseconds"])
    p99 = dur["durationinseconds"].quantile(0.99)
    keep = dur["durationinseconds"] <= p99
    m = fit(df.loc[keep.values, "want_share_fb"],
            df.loc[keep.values, ["imposed", "voluntary"]])
    results.append(row(f"3. trim duration > p99 ({p99:.0f}s)", m))

    # ---- 4. Combine imposed & voluntary into "any fact-check" -------------
    df["any_fc"] = df["imposed"] | df["voluntary"]
    m = fit(df["want_share_fb"], df[["any_fc"]])
    results.append({
        "check": "4. pooled any-factcheck effect",
        "N": int(m.nobs),
        "imposed_b": m.params["any_fc"], "imposed_se": m.bse["any_fc"],
        "imposed_p": m.pvalues["any_fc"],
        "voluntary_b": np.nan, "voluntary_se": np.nan, "voluntary_p": np.nan,
        "control_mean": m.params["const"], "r2": m.rsquared,
    })

    # ---- 5. Permutation / placebo: shuffle treatment labels -------------
    rng = np.random.default_rng(20260410)
    perm_imp = []
    perm_vol = []
    for _ in range(2000):
        perm_survey = rng.permutation(df["survey"].values)
        pi = (perm_survey == 2).astype(int)
        pv = (perm_survey == 3).astype(int)
        X = sm.add_constant(np.column_stack([pi, pv]))
        b = np.linalg.lstsq(X, df["want_share_fb"].values, rcond=None)[0]
        perm_imp.append(b[1])
        perm_vol.append(b[2])
    obs_imp = base.params["imposed"]
    obs_vol = base.params["voluntary"]
    p_imp = (np.abs(perm_imp) >= abs(obs_imp)).mean()
    p_vol = (np.abs(perm_vol) >= abs(obs_vol)).mean()
    results.append({
        "check": "5. permutation test (2000 draws, seed 20260410)",
        "N": int(base.nobs),
        "imposed_b": obs_imp, "imposed_se": np.std(perm_imp),
        "imposed_p": p_imp,
        "voluntary_b": obs_vol, "voluntary_se": np.std(perm_vol),
        "voluntary_p": p_vol,
        "control_mean": np.nan, "r2": np.nan,
    })

    # ---- 6. Heterogeneity: men vs women -----------------------------------
    # male_raw: in the raw Qualtrics file Q5 is gender (1=male, 2=female) —
    # verify sign via the baseline DV for men vs women.
    if "male" in df.columns:
        male = df["male"] == 1
        for label, mask in [("men", male), ("women", ~male)]:
            sub = df[mask]
            m = fit(sub["want_share_fb"], sub[["imposed", "voluntary"]])
            results.append(row(f"6. subgroup: {label}", m))

    # ---- 7. Drop respondents who voted for Le Pen 2nd round ---------------
    m = fit(df.loc[df["second_mlp"] == 0, "want_share_fb"],
            df.loc[df["second_mlp"] == 0, ["imposed", "voluntary"]])
    results.append(row("7. exclude MLP voters", m))

    # ---- 8. Alternative outcome: prob(share) × 100 instead of 0/1 --------
    # (equivalent to LPM but re-scaled so sanity check on scale)
    y100 = df["want_share_fb"] * 100
    m = fit(y100, df[["imposed", "voluntary"]])
    results.append(row("8. outcome rescaled ×100 (pp)", m))

    # ---- 9. HC3 standard errors -------------------------------------------
    Xc = sm.add_constant(df[["imposed", "voluntary"]])
    m = sm.OLS(df["want_share_fb"], Xc).fit(cov_type="HC3")
    results.append({
        "check": "9. HC3 robust SEs",
        "N": int(m.nobs),
        "imposed_b": m.params["imposed"], "imposed_se": m.bse["imposed"],
        "imposed_p": m.pvalues["imposed"],
        "voluntary_b": m.params["voluntary"], "voluntary_se": m.bse["voluntary"],
        "voluntary_p": m.pvalues["voluntary"],
        "control_mean": m.params["const"], "r2": m.rsquared,
    })

    # ---- 10. Winsorized DV (already binary; skip) + sharp Fisher bound ---
    # Exact two-sided Fisher test: Alt-Facts vs pooled fact-check
    alt = df.loc[df["survey"] == 1, "want_share_fb"]
    fc = df.loc[df["survey"].isin([2, 3]), "want_share_fb"]
    ct = np.array([[int(alt.sum()), int((alt == 0).sum())],
                   [int(fc.sum()),  int((fc == 0).sum())]])
    _, p_fisher = stats.fisher_exact(ct, alternative="two-sided")
    results.append({
        "check": "10. Fisher exact: alt-facts vs pooled FC",
        "N": int(len(alt) + len(fc)),
        "imposed_b": alt.mean() - fc.mean(), "imposed_se": np.nan,
        "imposed_p": p_fisher,
        "voluntary_b": np.nan, "voluntary_se": np.nan, "voluntary_p": np.nan,
        "control_mean": alt.mean(), "r2": np.nan,
    })

    # ---- 11. Placebo outcome: pre-treatment variable that should not move -
    # Use "voted Le Pen 2nd round" — measured pre-treatment — as a placebo DV.
    m = fit(df["second_mlp"], df[["imposed", "voluntary"]])
    results.append(row("11. placebo DV = MLP voter (pre-treat)", m))

    # ---- 12. Leave-one-bin-out jackknife across 30 time bins -------------
    # Assign to 30 roughly-equal-size chronological bins within each treatment
    # to test that the result isn't driven by one recruitment wave.
    from replication_140161.utils import load_survey
    raws = pd.concat([load_survey(i)[["startdate", "survey"]] for i in (1, 2, 3)],
                     ignore_index=True)
    df2 = df.copy()
    df2["start"] = pd.to_datetime(raws["startdate"].values, errors="coerce")
    df2 = df2.dropna(subset=["start"])
    df2["bin"] = pd.qcut(df2["start"].astype("int64"), 30, labels=False, duplicates="drop")
    deltas = []
    for b in df2["bin"].unique():
        sub = df2[df2["bin"] != b]
        m = fit(sub["want_share_fb"], sub[["imposed", "voluntary"]])
        deltas.append(m.params["imposed"])
    results.append({
        "check": "12. jackknife by 30 time bins (imposed coef range)",
        "N": int(len(df2)),
        "imposed_b": float(np.mean(deltas)),
        "imposed_se": float(np.std(deltas)),
        "imposed_p": np.nan,
        "voluntary_b": float(np.min(deltas)), "voluntary_se": float(np.max(deltas)),
        "voluntary_p": np.nan,
        "control_mean": np.nan, "r2": np.nan,
    })

    out = pd.DataFrame(results).set_index("check")
    pd.options.display.float_format = "{:.4f}".format
    print(out[["N", "imposed_b", "imposed_se", "imposed_p",
               "voluntary_b", "voluntary_se", "voluntary_p",
               "control_mean", "r2"]])
    out.to_csv(OUT / "robustness.csv")
    print(f"\nWrote {OUT/'robustness.csv'}")


if __name__ == "__main__":
    main()
