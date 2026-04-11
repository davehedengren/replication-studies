"""Robustness checks for the two headline specifications:

H1 (first stage): unemp_nextrecession on Del_unempincr + controls  (β̂ = 0.489)
H2 (reduced form): forecast_recession on Del_unempincr + controls  (β̂ = 0.006)
H3 (2SLS):         forecast_recession on unemp_nextrecession  (β̂ = 0.012)

Each check reports the key coefficient under a perturbation of the baseline
specification. The goal is to see whether the paper's conclusions are fragile.
"""
import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats
from linearmodels.iv import IV2SLS

from utils import (
    load_experiment, build_design, INFO_REGS, BASE_CONTROLS, OCC_VAR, OUT, fmt,
)

rng = np.random.default_rng(0)
df = load_experiment()


def fit(d, outcome, controls_list, extras=INFO_REGS, cov="HC1",
        cluster_col=None, include_occ=True):
    work = d.dropna(subset=extras + controls_list + [outcome]).copy()
    cols = [work[extras].astype(float).reset_index(drop=True),
            work[controls_list].astype(float).reset_index(drop=True)]
    if include_occ and OCC_VAR in work.columns:
        occ = pd.get_dummies(work[OCC_VAR].astype(int), prefix="occ", drop_first=True).astype(float)
        occ = occ.loc[:, occ.sum(axis=0) > 0]
        cols.append(occ.reset_index(drop=True))
    X = pd.concat(cols, axis=1)
    X = X.loc[:, X.std() > 0]
    X = sm.add_constant(X, has_constant="add")
    y = work[outcome].astype(float).reset_index(drop=True)
    if cluster_col is not None:
        groups = work[cluster_col].astype(int).reset_index(drop=True)
        res = sm.OLS(y, X).fit(cov_type="cluster", cov_kwds={"groups": groups})
    else:
        res = sm.OLS(y, X).fit(cov_type=cov)
    return res


def fit_iv(d, outcome, cov="robust"):
    work = d.dropna(subset=INFO_REGS + BASE_CONTROLS + [outcome, "unemp_nextrecession"]).copy()
    occ = pd.get_dummies(work[OCC_VAR].astype(int), prefix="occ", drop_first=True).astype(float)
    occ = occ.loc[:, occ.sum(axis=0) > 0]
    exog = pd.concat(
        [work[["unempincr_alt", "Del_unempbase", "unempbase_alt"] + BASE_CONTROLS]
            .astype(float).reset_index(drop=True),
         occ.reset_index(drop=True)],
        axis=1,
    )
    exog = exog.loc[:, exog.std() > 0]
    exog = sm.add_constant(exog, has_constant="add")
    y = work[outcome].astype(float).reset_index(drop=True)
    endog = work[["unemp_nextrecession"]].astype(float).reset_index(drop=True)
    instr = work[["Del_unempincr"]].astype(float).reset_index(drop=True)
    return IV2SLS(y, exog, endog, instr).fit(cov_type=cov, debiased=True)


def beta(res, name):
    b = res.params[name]
    se = res.std_errors[name] if hasattr(res, "std_errors") else res.bse[name]
    p = res.pvalues[name]
    n = int(res.nobs)
    return b, se, p, n


rows = []
def log(label, h1, h2, h3=None):
    r = {"check": label}
    for key, (b, se, p, n) in [("H1", h1), ("H2", h2)] + ([("H3", h3)] if h3 else []):
        r[f"{key}_b"] = b
        r[f"{key}_se"] = se
        r[f"{key}_p"] = p
        r[f"{key}_N"] = n
    rows.append(r)
    s = f"{label:42s}  H1: {fmt(h1[0])} ({fmt(h1[1])}) N={h1[3]}   "
    s += f"H2: {fmt(h2[0])} ({fmt(h2[1])}) N={h2[3]}"
    if h3:
        s += f"   H3: {fmt(h3[0])} ({fmt(h3[1])}) N={h3[3]}"
    print(s)


# Baseline
r1 = fit(df, "unemp_nextrecession", BASE_CONTROLS)
r2 = fit(df, "forecast_recession", BASE_CONTROLS)
r3 = fit_iv(df, "forecast_recession")
log("0. Baseline (paper spec)",
    beta(r1, "Del_unempincr"), beta(r2, "Del_unempincr"),
    beta(r3, "unemp_nextrecession"))

# 1. HC3 standard errors
r1 = fit(df, "unemp_nextrecession", BASE_CONTROLS, cov="HC3")
r2 = fit(df, "forecast_recession", BASE_CONTROLS, cov="HC3")
log("1. HC3 robust SEs",
    beta(r1, "Del_unempincr"), beta(r2, "Del_unempincr"))

# 2. No controls at all (only Del_unempincr + alt signals)
r1 = fit(df, "unemp_nextrecession", [], include_occ=False)
r2 = fit(df, "forecast_recession", [], include_occ=False)
log("2. No demographic controls, no occ FE",
    beta(r1, "Del_unempincr"), beta(r2, "Del_unempincr"))

# 3. Drop occupation FE only
r1 = fit(df, "unemp_nextrecession", BASE_CONTROLS, include_occ=False)
r2 = fit(df, "forecast_recession", BASE_CONTROLS, include_occ=False)
log("3. Drop occupation FE",
    beta(r1, "Del_unempincr"), beta(r2, "Del_unempincr"))

# 4. Cluster SEs on occupation_1
r1 = fit(df, "unemp_nextrecession", BASE_CONTROLS, cluster_col=OCC_VAR)
r2 = fit(df, "forecast_recession", BASE_CONTROLS, cluster_col=OCC_VAR)
log("4. Cluster SEs on occupation",
    beta(r1, "Del_unempincr"), beta(r2, "Del_unempincr"))

# 5. Winsorize perceived risk at 1%/99%
d5 = df.copy()
lo, hi = np.percentile(d5["unemp_nextrecession"], [1, 99])
d5["unemp_nextrecession"] = d5["unemp_nextrecession"].clip(lo, hi)
r1 = fit(d5, "unemp_nextrecession", BASE_CONTROLS)
log("5. Winsorize outcome (1%, 99%)",
    beta(r1, "Del_unempincr"), beta(fit(d5, "forecast_recession", BASE_CONTROLS), "Del_unempincr"))

# 6. Drop top and bottom 1% of Del_unempincr (signal outliers)
d6 = df[(df["Del_unempincr"] > df["Del_unempincr"].quantile(0.01)) &
        (df["Del_unempincr"] < df["Del_unempincr"].quantile(0.99))].copy()
r1 = fit(d6, "unemp_nextrecession", BASE_CONTROLS)
r2 = fit(d6, "forecast_recession", BASE_CONTROLS)
log("6. Trim |Del_unempincr| at 1/99 pct",
    beta(r1, "Del_unempincr"), beta(r2, "Del_unempincr"))

# 7. Restrict to ACS arm
d7 = df[df["tr_acs"] == 1].copy()
r1 = fit(d7, "unemp_nextrecession", BASE_CONTROLS)
r2 = fit(d7, "forecast_recession", BASE_CONTROLS)
log("7. ACS arm only",
    beta(r1, "Del_unempincr"), beta(r2, "Del_unempincr"))

# 8. Restrict to CPS arm
d8 = df[df["tr_cps"] == 1].copy()
r1 = fit(d8, "unemp_nextrecession", BASE_CONTROLS)
r2 = fit(d8, "forecast_recession", BASE_CONTROLS)
log("8. CPS arm only",
    beta(r1, "Del_unempincr"), beta(r2, "Del_unempincr"))

# 9. Placebo: shuffle the treatment signal (1000 draws) — report rejection rate
def placebo(outcome, n_draws=1000, true_beta=None):
    bs = np.empty(n_draws)
    base = df.copy()
    for i in range(n_draws):
        base["Del_unempincr"] = rng.permutation(df["Del_unempincr"].values)
        base["unempincr_alt"] = rng.permutation(df["unempincr_alt"].values)
        r = fit(base, outcome, BASE_CONTROLS)
        bs[i] = r.params["Del_unempincr"]
    p_emp = (np.abs(bs) >= abs(true_beta)).mean()
    return bs.mean(), bs.std(), p_emp

pb1_mean, pb1_sd, p1 = placebo("unemp_nextrecession", 500, 0.489)
pb2_mean, pb2_sd, p2 = placebo("forecast_recession", 500, 0.006)
print(f"9. Placebo shuffle (500 draws):                  "
      f"H1 placebo β̄={pb1_mean:+.3f} (sd={pb1_sd:.3f}) emp-p={p1:.3f}   "
      f"H2 placebo β̄={pb2_mean:+.4f} (sd={pb2_sd:.4f}) emp-p={p2:.3f}")
rows.append(dict(check="9. Placebo shuffle (500 draws)",
                 H1_b=pb1_mean, H1_se=pb1_sd, H1_p=p1, H1_N=1008,
                 H2_b=pb2_mean, H2_se=pb2_sd, H2_p=p2, H2_N=1008))

# 10. Drop high-confidence respondents (paper's heterogeneity result)
d10 = df[df["high_conf"] == 0].copy()
r1 = fit(d10, "unemp_nextrecession", [c for c in BASE_CONTROLS if c != "high_conf"])
r2 = fit(d10, "forecast_recession", [c for c in BASE_CONTROLS if c != "high_conf"])
log("10. Low-confidence subsample only",
    beta(r1, "Del_unempincr"), beta(r2, "Del_unempincr"))

# 11. Drop respondents who answered 0 or 100 to perceived next-recession risk
d11 = df[(df["unemp_nextrecession"] > 0) & (df["unemp_nextrecession"] < 100)].copy()
r1 = fit(d11, "unemp_nextrecession", BASE_CONTROLS)
r2 = fit(d11, "forecast_recession", BASE_CONTROLS)
log("11. Drop extreme (0 or 100) risk responses",
    beta(r1, "Del_unempincr"), beta(r2, "Del_unempincr"))

# 12. IHS transform of perceived risk (log-like, handles zeros)
d12 = df.copy()
d12["ihs_risk"] = np.arcsinh(d12["unemp_nextrecession"])
r1 = fit(d12, "ihs_risk", BASE_CONTROLS)
log("12. IHS transform of outcome H1",
    beta(r1, "Del_unempincr"),
    beta(fit(df, "forecast_recession", BASE_CONTROLS), "Del_unempincr"))

pd.DataFrame(rows).to_csv(OUT / "05_robustness.csv", index=False)
print("\nSaved:", OUT / "05_robustness.csv")
