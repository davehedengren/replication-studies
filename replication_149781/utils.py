"""Shared utilities for Derenoncourt (2022 AER) replication — paper 149781."""
from pathlib import Path
import numpy as np
import pandas as pd
import pyreadstat

ROOT = Path(__file__).resolve().parent.parent
PKG = ROOT / "149781-V1" / "replication_AER"
DATA_DTA = PKG / "data" / "randomization_inference" / "GM_cz_final_dataset_randomization_inference.dta"
PUB_TEXT = PKG / "figures_tables" / "text"

# Main specification from 5_main_figures_tables.do
X_OLS = "GM"                     # percentile of 1940-1970 Black pop change
X_IV = "GM_hat2"                 # percentile of predicted Black pop change (shift-share)
BASELINE_CONTROLS = [
    "frac_all_upm1940",
    "mfg_lfshare1940",
    "v2_blackmig3539_share1940",
    "reg2", "reg3", "reg4",
]


def load_cz():
    df, _ = pyreadstat.read_dta(str(DATA_DTA))
    # Drop the 1000 randomization columns we don't need
    df = df[[c for c in df.columns if not c.startswith("GM_hatr")]].copy()
    return df


def _add_constant(X):
    X = np.asarray(X, dtype=float)
    return np.column_stack([np.ones(len(X)), X])


def ols(y, X, w=None):
    """Plain OLS with analytic SEs. Returns dict of b, se, t, p, n, r2, resid."""
    from scipy import stats
    y = np.asarray(y, dtype=float)
    X = np.asarray(X, dtype=float)
    n, k = X.shape
    if w is None:
        w = np.ones(n)
    W = np.sqrt(w)
    Xw = X * W[:, None]
    yw = y * W
    b, *_ = np.linalg.lstsq(Xw, yw, rcond=None)
    resid = y - X @ b
    # classical (homoskedastic) SEs, analogous to ivreg2 default (small-sample)
    rss = float((w * resid**2).sum())
    dof = n - k
    sigma2 = rss / dof
    XtWX_inv = np.linalg.pinv(Xw.T @ Xw)
    vcv = sigma2 * XtWX_inv
    se = np.sqrt(np.diag(vcv))
    tss = float((w * (y - (w*y).sum()/w.sum())**2).sum())
    r2 = 1.0 - rss / tss if tss > 0 else np.nan
    return dict(b=b, se=se, n=n, k=k, r2=r2, resid=resid, vcv=vcv, dof=dof)


def tsls(y, x_endog, z_inst, X_exog, w=None):
    """2SLS: y = x_endog*beta + X_exog*gamma + eps, with z_inst replacing x_endog.

    Returns dict with b (for [const, x_endog, X_exog]), se, first-stage F, etc.
    """
    y = np.asarray(y, dtype=float)
    x_endog = np.asarray(x_endog, dtype=float).reshape(-1)
    z_inst = np.asarray(z_inst, dtype=float).reshape(-1)
    X_exog = np.asarray(X_exog, dtype=float)
    n = len(y)
    if w is None:
        w = np.ones(n)
    W = np.sqrt(w)

    # Matrices with constant
    const = np.ones(n)
    # First stage: x_endog on [const, z_inst, X_exog]
    Z = np.column_stack([const, z_inst, X_exog])
    fs = ols(x_endog, Z, w=w)
    x_hat = Z @ fs["b"]

    # F-stat on z_inst = 0 in first stage (classical, weighted)
    # Use restricted model: x_endog on [const, X_exog]
    Zr = np.column_stack([const, X_exog])
    fs_r = ols(x_endog, Zr, w=w)
    rss_r = float((w * fs_r["resid"]**2).sum())
    rss_u = float((w * fs["resid"]**2).sum())
    q = 1
    f_stat = ((rss_r - rss_u) / q) / (rss_u / (n - Z.shape[1]))

    # Second stage: regress y on [const, x_hat, X_exog]
    X2 = np.column_stack([const, x_hat, X_exog])
    ss = ols(y, X2, w=w)
    # Correct residual-based SEs: need to use original x_endog in residuals
    X_full = np.column_stack([const, x_endog, X_exog])
    resid_2sls = y - X_full @ ss["b"]
    dof = n - X2.shape[1]
    Ww = w
    sigma2 = float((Ww * resid_2sls**2).sum()) / dof
    Xw2 = X2 * np.sqrt(Ww)[:, None]
    vcv = sigma2 * np.linalg.inv(Xw2.T @ Xw2)
    se = np.sqrt(np.diag(vcv))
    return dict(b=ss["b"], se=se, n=n, f_stat=f_stat, dof=dof,
                first_stage=fs, fs_coef=fs["b"][1], fs_se=fs["se"][1])


def design(df, extra_cols=None, dropna=True):
    """Build design matrix X = baseline controls (no constant; constant added by estimators)."""
    cols = list(BASELINE_CONTROLS) + (extra_cols or [])
    sub = df[cols].copy()
    return sub


def format_row(label, b, se, n=None, extra=""):
    star = ""
    if se and se > 0:
        t = b / se
        from scipy import stats
        p = 2 * (1 - stats.norm.cdf(abs(t)))
        if p < 0.01: star = "***"
        elif p < 0.05: star = "**"
        elif p < 0.1: star = "*"
    return f"{label:<30s}  b={b:>10.4f}{star:<3s}  se={se:>8.4f}  N={n}  {extra}"
