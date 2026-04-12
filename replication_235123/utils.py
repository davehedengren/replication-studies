"""Shared helpers for replication of Artiles (2025), AER, paper 235123.

Main dataset: parish-level data (N=336) with ethnic diversity, crop exchange
measures, and contemporary development outcomes.
"""
from pathlib import Path
import numpy as np
import pandas as pd
import statsmodels.api as sm
from linearmodels.iv.absorbing import AbsorbingLS

PKG = Path(__file__).resolve().parent.parent / "235123-V1" / "Z1_Replication"
DATA_DIR = PKG / "02_output" / "02b_final" / "data"
TAB_DIR = PKG / "02_output" / "02b_final" / "tables_figures"

DOCTRINAS_DTA = DATA_DIR / "data_doctrinas.dta"

CONTROLS_ALL = [
    "mean_el_hwsd", "std_el_hwsd",
    "mean_av_pre1500_all", "std_av_pre1500_all",
    "l_dist_river", "l_dist_shrine", "l_dist_mine",
    "l_tasa_doctrina_imp_int", "x", "y",
]

OUTCOMES_1 = [
    "l_mean_nl_pc00_03", "CN94_d_ua_venta",
    "CS93_sh_hig_red", "CS93_sh_agua_red",
]
OUTCOMES_2 = [
    "l_nl_pc13_viirs", "CN12_d_ua_venta",
    "CS17_sh_hig_red", "CS17_sh_agua_red",
    "n_f_1013_p100",
]

PANEL_A_OUTCOMES = [
    "l_nl_pc13_viirs",
    "l_mean_nl_pc10_13",
    "n_f_1013_p100",
    "CN12_d_ua_venta",
    "CN12_sh_ua_venta",
]
PANEL_B_PARISH_OUTCOMES = ["CS17_sh_hig_red_std", "CS17_sh_agua_red_std"]
PANEL_B_HOUSEHOLD_OUTCOMES = ["l_gross_inc_per_capita"]


def load_doctrinas() -> pd.DataFrame:
    return pd.read_stata(DOCTRINAS_DTA, convert_categoricals=False)


def absorb_fe_ols(y: pd.Series, X: pd.DataFrame, group: pd.Series, cluster: pd.Series | None = None):
    """OLS absorbing a single set of fixed effects by within-demean.

    Equivalent to reghdfe y X, absorb(group) vce(cluster cluster) when group
    has K levels and we adjust degrees of freedom appropriately.

    Parameters
    ----------
    y : dependent variable
    X : design matrix (NO constant, NO absorbed dummies)
    group : FE variable
    cluster : optional cluster variable (defaults to group)
    """
    if cluster is None:
        cluster = group
    df = pd.concat(
        [y.rename("__y"), X.reset_index(drop=True), group.rename("__g").reset_index(drop=True),
         cluster.rename("__c").reset_index(drop=True)],
        axis=1,
    )
    df = df.dropna().reset_index(drop=True)
    # Within transform
    g = df["__g"]
    ybar = df.groupby(g)["__y"].transform("mean")
    yt = df["__y"] - ybar
    cols = list(X.columns)
    Xt = df[cols].copy()
    for c in cols:
        Xt[c] = df[c] - df.groupby(g)[c].transform("mean")
    # OLS on within-transformed variables (no constant — absorbed by FE demean)
    model = sm.OLS(yt, Xt)
    # Standard cluster SEs from OLS on transformed — but we must correct DoF
    # for absorbed FEs. Mimic Stata reghdfe: df_resid = N - K - (G - 1)
    K = len(cols)
    G = int(df["__g"].nunique())
    N = len(df)
    res = model.fit(cov_type="cluster", cov_kwds={"groups": df["__c"].values})
    # reghdfe adjusts SEs by sqrt((N-1)/(N-K-(G-1))) * sqrt(Gc/(Gc-1)) ... the
    # cluster adjustment is handled by statsmodels; the only missing piece is
    # the absorbed FE df. Manually rescale.
    df_ols = N - K  # statsmodels used this
    df_true = N - K - (G - 1)
    scale = np.sqrt(df_ols / df_true)
    # Rebuild covariance
    cov = res.cov_params().values * (scale ** 2)
    se = np.sqrt(np.diag(cov))
    return {
        "params": res.params,
        "se": pd.Series(se, index=res.params.index),
        "N": N,
        "G": G,
        "K": K,
        "ymean": df["__y"].mean(),
        "result": res,
        "cov": cov,
    }


def reghdfe(df: pd.DataFrame, y: str, xvars: list[str], absorb: str, cluster: str | None = None):
    """Equivalent to Stata `reghdfe y xvars, absorb(absorb) vce(cluster cluster)`.

    Uses linearmodels AbsorbingLS which matches reghdfe small-sample cluster SEs
    closely.
    """
    if cluster is None:
        cluster = absorb
    cols = [y] + xvars + [absorb, cluster]
    d = df[list(dict.fromkeys(cols))].dropna().reset_index(drop=True)
    mod = AbsorbingLS(d[y], d[xvars], absorb=d[[absorb]].astype("category"))
    res = mod.fit(cov_type="clustered", clusters=d[cluster])
    return {
        "params": res.params,
        "se": res.std_errors,
        "pvalues": res.pvalues,
        "N": int(res.nobs),
        "ymean": float(d[y].mean()),
        "result": res,
    }


def ols_robust(y: pd.Series, X: pd.DataFrame):
    """OLS with HC1 (Stata `, r`) robust SEs."""
    df = pd.concat([y.rename("__y"), X.reset_index(drop=True)], axis=1).dropna().reset_index(drop=True)
    yv = df["__y"]
    Xv = sm.add_constant(df[list(X.columns)])
    res = sm.OLS(yv, Xv).fit(cov_type="HC1")
    return res


def fmt_coef(b, se, pval):
    stars = ""
    if pval < 0.01:
        stars = "***"
    elif pval < 0.05:
        stars = "**"
    elif pval < 0.1:
        stars = "*"
    return f"{b:.3f}{stars} [{se:.3f}]"
