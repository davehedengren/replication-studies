"""Shared paths and helpers for replication of Jensen & Weigel (2025, JEL)."""
from pathlib import Path
import numpy as np
import pandas as pd

PKG = Path("/Users/davehedengren/code/replication-studies/237027-V1/replication_package")
RAW = PKG / "data" / "raw"
INTERMEDIATE = PKG / "data" / "intermediate"
FINAL_DTA = PKG / "data" / "final" / "df_ISORA_GRD_VDEM_WB.dta"

OUT = Path("/Users/davehedengren/code/replication-studies/replication_237027/output")
OUT.mkdir(parents=True, exist_ok=True)


def zscore(s: pd.Series) -> pd.Series:
    """Stata-style z-score using population sd? No — Stata's sum reports sample sd (N-1)."""
    return (s - s.mean()) / s.std(ddof=1)


def load_final() -> pd.DataFrame:
    return pd.read_stata(FINAL_DTA, convert_categoricals=False)


def ols_fit(y: pd.Series, x: pd.Series):
    """Simple OLS y = a + b*x. Returns dict with slope, intercept, n, r2."""
    m = (~y.isna()) & (~x.isna())
    yv, xv = y[m].to_numpy(), x[m].to_numpy()
    n = len(yv)
    if n < 3:
        return None
    X = np.column_stack([np.ones(n), xv])
    beta, *_ = np.linalg.lstsq(X, yv, rcond=None)
    yhat = X @ beta
    resid = yv - yhat
    ss_res = (resid ** 2).sum()
    ss_tot = ((yv - yv.mean()) ** 2).sum()
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else np.nan
    sigma2 = ss_res / (n - 2)
    cov = sigma2 * np.linalg.inv(X.T @ X)
    se = np.sqrt(np.diag(cov))
    from scipy import stats
    t = beta / se
    p = 2 * (1 - stats.t.cdf(np.abs(t), df=n - 2))
    return {
        "intercept": beta[0],
        "slope": beta[1],
        "se_int": se[0],
        "se_slope": se[1],
        "t_slope": t[1],
        "p_slope": p[1],
        "n": n,
        "r2": r2,
    }
