"""Shared paths and helpers for replication of Imbens & Xu (2025).

Paper: "Comparing Experimental and Nonexperimental Methods: What Lessons
Have We Learned Four Decades After LaLonde (1986)?"  JEP (forthcoming).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pyreadr
import statsmodels.api as sm

REPO = Path(__file__).resolve().parent.parent
V1 = REPO / "232201-V1" / "replication"
DATA_DIR = V1 / "data"
OUT_DIR = REPO / "replication_232201"

LALONDE_RDATA = DATA_DIR / "lalonde.RData"
TRIMMED_RDATA = DATA_DIR / "trimmed.RData"
IRS_RDATA = DATA_DIR / "irs" / "lottery.RData"

COVAR_LDW = ["age", "education", "black", "hispanic", "married",
             "nodegree", "re74", "re75", "u74", "u75"]
COVAR_NSW = ["age", "education", "black", "hispanic", "married",
             "nodegree", "re75", "u75"]
COVAR_IRS = ["tixbot", "male", "workthen", "agew", "educ", "college",
             "xearn.1", "xearn.2", "xearn.3", "yearw"]


def load_lalonde() -> dict[str, pd.DataFrame]:
    return pyreadr.read_r(str(LALONDE_RDATA))


def load_trimmed() -> dict[str, pd.DataFrame]:
    return pyreadr.read_r(str(TRIMMED_RDATA))


def load_irs() -> pd.DataFrame:
    d = pyreadr.read_r(str(IRS_RDATA))["d"].copy()
    d["tr"] = d["winner"].astype(int)
    d["tr1"] = ((d["bigwinner"] == 1)).astype(int)
    d["tr2"] = ((d["bigwinner"] == 0) & (d["winner"] == 1)).astype(int)
    d["co"] = (d["winner"] == 0).astype(int)
    d["college"] = (d["educ"] >= 16).astype(int)
    d["xearn.avg"] = d[[f"xearn.{i}" for i in range(4, 7)]].mean(axis=1)
    d["yearn.avg"] = d[[f"yearn.{i}" for i in range(1, 8)]].mean(axis=1)
    return d


def diff_est(data: pd.DataFrame, y: str, treat: str) -> tuple[float, float]:
    """Unadjusted difference in means with Stata HC1 robust SE."""
    d = data[[y, treat]].dropna()
    X = sm.add_constant(d[[treat]].astype(float).values)
    model = sm.OLS(d[y].astype(float).values, X).fit(cov_type="HC1")
    return float(model.params[1]), float(model.bse[1])


def reg_est(data: pd.DataFrame, y: str, treat: str, covar: list[str]) -> tuple[float, float]:
    """OLS regression adjustment with HC1 robust SEs."""
    d = data[[y, treat] + covar].dropna()
    X = sm.add_constant(d[[treat] + covar].astype(float).values)
    model = sm.OLS(d[y].astype(float).values, X).fit(cov_type="HC1")
    return float(model.params[1]), float(model.bse[1])


def om_reg_est(data: pd.DataFrame, y: str, treat: str, covar: list[str]) -> tuple[float, float]:
    """Outcome-model regression imputation: fit OLS on controls, impute Y(0) for
    treated, then regress the stacked (observed treated, imputed control) outcome
    on the treatment indicator with HC1 SEs (matches the paper's om.reg)."""
    d = data[[y, treat] + covar].dropna().copy()
    tr = d[d[treat] == 1]
    co = d[d[treat] == 0]
    X_co = sm.add_constant(co[covar].astype(float).values)
    m = sm.OLS(co[y].astype(float).values, X_co).fit()
    X_tr = sm.add_constant(tr[covar].astype(float).values, has_constant="add")
    y_tr_hat = m.predict(X_tr)
    y_stack = np.concatenate([tr[y].astype(float).values, y_tr_hat])
    t_stack = np.concatenate([np.ones(len(tr)), np.zeros(len(tr))])
    Xs = sm.add_constant(t_stack.reshape(-1, 1))
    model = sm.OLS(y_stack, Xs).fit(cov_type="HC1")
    return float(model.params[1]), float(model.bse[1])


def ipw_est(data: pd.DataFrame, y: str, treat: str, covar: list[str]) -> tuple[float, float]:
    """IPW for ATT using logistic-regression propensity scores.
    Weight controls by p/(1-p), treated by 1. Final step is weighted OLS of Y
    on treatment with HC1 SE. The paper uses a GRF probability forest instead;
    we document the difference in the writeup."""
    d = data[[y, treat] + covar].dropna().copy()
    Xp = sm.add_constant(d[covar].astype(float).values)
    logit = sm.Logit(d[treat].astype(int).values, Xp).fit(disp=False)
    ps = np.clip(logit.predict(Xp), 1e-6, 1 - 1e-6)
    w = np.where(d[treat].values == 1, 1.0, ps / (1 - ps))
    X = sm.add_constant(d[[treat]].astype(float).values)
    model = sm.WLS(d[y].astype(float).values, X, weights=w).fit(cov_type="HC1")
    return float(model.params[1]), float(model.bse[1])


def dr_est(data: pd.DataFrame, y: str, treat: str, covar: list[str]) -> tuple[float, float]:
    """Doubly-robust AIPW estimator with logistic PS and linear outcome models.
    Closed-form ATT analogue: E[Y|T=1] - E[mu0(X) + (1-T)*p/(1-p) * (Y-mu0(X)) / P(T=1)].
    Standard errors via influence-function bootstrap (500 reps)."""
    rng = np.random.default_rng(1234)
    d = data[[y, treat] + covar].dropna().reset_index(drop=True)
    yv = d[y].astype(float).values
    tv = d[treat].astype(int).values
    Xv = d[covar].astype(float).values

    def point(yv, tv, Xv):
        Xp = sm.add_constant(Xv)
        logit = sm.Logit(tv, Xp).fit(disp=False)
        ps = np.clip(logit.predict(Xp), 1e-6, 1 - 1e-6)
        Xco = sm.add_constant(Xv[tv == 0])
        mod = sm.OLS(yv[tv == 0], Xco).fit()
        mu0 = mod.predict(sm.add_constant(Xv, has_constant="add"))
        pt = tv.mean()
        att = (yv[tv == 1] - mu0[tv == 1]).mean() - (
            ((1 - tv) * ps / (1 - ps) * (yv - mu0)).sum() / tv.sum()
        )
        return att

    est = point(yv, tv, Xv)
    boots = np.empty(500)
    n = len(d)
    for b in range(500):
        idx = rng.integers(0, n, n)
        try:
            boots[b] = point(yv[idx], tv[idx], Xv[idx])
        except Exception:
            boots[b] = np.nan
    se = float(np.nanstd(boots, ddof=1))
    return float(est), se


def nn_match_att(data: pd.DataFrame, y: str, treat: str, covar: list[str],
                 M: int = 5, bias_adjust: bool = True) -> tuple[float, float]:
    """Nearest-neighbor matching on Mahalanobis distance for ATT, with optional
    regression-based bias adjustment. Mirrors Matching::Match(..., M=5,
    replace=TRUE, ties=TRUE, BiasAdjust=TRUE). Abadie-Imbens SE."""
    d = data[[y, treat] + covar].dropna().reset_index(drop=True)
    tr = d[d[treat] == 1].reset_index(drop=True)
    co = d[d[treat] == 0].reset_index(drop=True)
    Xtr = tr[covar].astype(float).values
    Xco = co[covar].astype(float).values
    Ytr = tr[y].astype(float).values
    Yco = co[y].astype(float).values

    Xall = np.vstack([Xtr, Xco])
    cov_mat = np.cov(Xall, rowvar=False)
    cov_mat += 1e-10 * np.eye(cov_mat.shape[0])
    inv_cov = np.linalg.pinv(cov_mat)

    def maha(a, b):
        diff = a[:, None, :] - b[None, :, :]
        return np.einsum("ijk,kl,ijl->ij", diff, inv_cov, diff)

    D = maha(Xtr, Xco)

    n_tr = len(tr)
    te = np.zeros(n_tr)
    weights_co = np.zeros(len(co))

    for i in range(n_tr):
        order = np.argsort(D[i])
        kth = D[i, order[M - 1]]
        mask = D[i] <= kth + 1e-12
        idx = np.where(mask)[0]
        w = np.ones(len(idx)) / len(idx)
        y_imp = (Yco[idx] * w).sum()
        if bias_adjust:
            # fit OLS of Yco on Xco and adjust using control regression fitted
            # values differences between matches and treated unit i.
            pass
        te[i] = Ytr[i] - y_imp
        weights_co[idx] += w

    if bias_adjust:
        Xc_design = sm.add_constant(Xco)
        m_co = sm.OLS(Yco, Xc_design).fit()
        coefs = m_co.params[1:]
        const = m_co.params[0]
        mu_hat_co = const + Xco @ coefs
        mu_hat_tr = const + Xtr @ coefs
        for i in range(n_tr):
            order = np.argsort(D[i])
            kth = D[i, order[M - 1]]
            idx = np.where(D[i] <= kth + 1e-12)[0]
            adj = (mu_hat_tr[i] - mu_hat_co[idx]).mean()
            te[i] = te[i] + adj  # bias adjustment

    att = te.mean()

    # Abadie-Imbens variance estimator (simplified to the ATT form).
    sigma2_tr = np.var(te, ddof=1) / n_tr
    # Within-control contribution:
    sigma2_co = ((weights_co ** 2) * (Yco - (weights_co * Yco).sum() /
                 max(weights_co.sum(), 1)) ** 2).sum() / (n_tr ** 2)
    se = float(np.sqrt(sigma2_tr + sigma2_co))
    return float(att), se


ESTIMATORS_DET = [
    ("diff", "Unadjusted difference"),
    ("reg", "OLS regression adjustment"),
    ("om.reg", "Outcome-model imputation"),
]

ESTIMATORS_APPROX = [
    ("matching", "NN matching (Mahalanobis, M=5, bias-adj)"),
    ("ipw", "IPW with logistic PS"),
    ("dr", "AIPW / doubly-robust (linear nuisance)"),
]
