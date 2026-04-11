"""Shared helpers for replication of Huh & Reif (2021), 'Teenage Driving,
Mortality, and Risky Behaviors'."""
from pathlib import Path
import numpy as np
import pandas as pd

PKG = Path(__file__).resolve().parent.parent / "133501-V1"
MORT = PKG / "data" / "mortality" / "derived"
ADDH = PKG / "data" / "add_health" / "derived"
OUT = Path(__file__).resolve().parent / "output"
OUT.mkdir(exist_ok=True)

MORTALITY_OUTCOMES = [
    "cod_any", "cod_internal", "cod_external",
    "cod_MVA", "cod_sa", "cod_sa_firearms",
    "cod_sa_poisoning", "cod_sa_poisoning_subst", "cod_sa_poisoning_gas",
    "cod_sa_drowning", "cod_sa_other", "cod_homicide", "cod_extother",
]


def load_mortality(scenario="none"):
    """Load monthly mortality counts aggregated to age-months-from-MDA and
    convert to deaths per 100,000 person-years (matching prep_data_rd in
    Stata). Scenario is 'none', 'male', or 'female' (case-insensitive)."""
    p = MORT / f"mortality_{scenario.lower()}.dta"
    df = pd.read_stata(p)
    for col in df.columns:
        if col != "agemo_mda":
            df[col] = df[col].astype(float)
    for col in df.columns:
        if col.startswith("cod_"):
            df[col] = 100_000.0 * df[col] / (df["pop"] / 12.0)
    df["post"] = (df["agemo_mda"] >= 0).astype(float)
    df["firstmonth"] = (df["agemo_mda"] == 0).astype(float)
    return df


def load_addhealth(scenario="none"):
    p = ADDH / f"addhealth_{scenario.lower()}.dta"
    df = pd.read_stata(p)
    for col in df.columns:
        if col != "agemo_mda":
            df[col] = df[col].astype(float)
    df["post"] = (df["agemo_mda"] >= 0).astype(float)
    df["firstmonth"] = (df["agemo_mda"] == 0).astype(float)
    return df


def tri_weights(agemo, bw):
    """Triangular kernel weights, matching the Stata prep_data_rd loop.

    Weights are (bw - |x|)/bw for 0<=x<=bw-1, and (bw - (|x|-1+1))/bw for
    x<0 down to -(bw-1). That replicates the exact Stata recipe; a proper
    triangular kernel would be symmetric, but this matches the Stata code
    used in the paper."""
    w = np.zeros(len(agemo))
    for x in range(bw):
        w = np.where(agemo == x, (bw - x) / bw, w)
    for x in range(2, bw + 1):
        w = np.where(agemo == -(x - 1), (bw - x + 1) / bw, w)
    return w


def rd_ols_fixed_bw(df, y, bw=13):
    """Replicate Stata's 'reg y i.post##c.agemo_mda i.firstmonth
    [aweight=tri_wgt], robust' with bandwidth 13. Returns (beta, se, ci,
    pval, n_used)."""
    import statsmodels.api as sm

    w = tri_weights(df["agemo_mda"].values, bw)
    mask = w > 0
    Y = df.loc[mask, y].values
    post = df.loc[mask, "post"].values
    age = df.loc[mask, "agemo_mda"].values
    fm = df.loc[mask, "firstmonth"].values
    ww = w[mask]

    X = np.column_stack([
        np.ones_like(post),
        post,
        age,
        post * age,
        fm,
    ])
    # WLS + HC1 = Stata aweight robust
    model = sm.WLS(Y, X, weights=ww).fit(cov_type="HC1")
    idx = 1  # post
    beta = model.params[idx]
    se = model.bse[idx]
    ci_l, ci_u = model.conf_int()[idx]
    return {
        "beta": beta, "se": se, "ci": (ci_l, ci_u),
        "pval": model.pvalues[idx], "n": int(mask.sum()),
    }


def rd_mse_opt(df, y, covs=True):
    """Run rdrobust with MSE-optimal bandwidth to match the paper's main
    specification. Returns dict with conv estimate and robust CI."""
    from rdrobust import rdrobust
    Y = df[y].values.astype(float)
    X = df["agemo_mda"].values.astype(float)
    kwargs = dict(y=Y, x=X, c=0, p=1, kernel="triangular", all=True)
    if covs:
        kwargs["covs"] = df[["firstmonth"]].values.astype(float)
    r = rdrobust(**kwargs)
    coef = np.asarray(r.coef.values).flatten()
    ci = np.asarray(r.ci.values)
    bws = np.asarray(r.bws.values)
    return {
        "conv": float(coef[0]),
        "bc": float(coef[1]),
        "robust_ci": (float(ci[2, 0]), float(ci[2, 1])),
        "conv_se": float(np.asarray(r.se.values).flatten()[0]),
        "robust_pval": float(np.asarray(r.pv.values).flatten()[2]),
        "h": float(bws[0, 0]),
    }


def mean_before(df, y):
    pre = df[(df["agemo_mda"] >= -12) & (df["agemo_mda"] <= -1)]
    return float(pre[y].mean())
