"""Helpers for replication of Cormun & De Leo (AER): Shocks and Exchange Rates in SOEs.

Implements the Uhlig (2003) max-FEV identification the authors use in 3-variable VARs:
y_t = [FFR, HomeRate, log FX], with the "domestic" shock minimising the FEV share
of the FFR over horizons 0..H, and the "external" (main external) shock maximising
the FEV share of log FX over the same horizons, projected orthogonal to the domestic
shock. The third shock is the residual (null space of the other two).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import scipy.io as sio

BASE = Path(__file__).resolve().parent.parent / "212383-V1"
DATA_XLS = BASE / "Data" / "CDL_DATA.xls"
COUNTRY_MAT = BASE / "Data" / "COUNTRYNAME_DATA.mat"
VIX_MAT = BASE / "Data" / "lVIX_DATA.mat"
OUT = Path(__file__).resolve().parent / "outputs"
OUT.mkdir(exist_ok=True)

# Countries used in the baseline analysis (MATLAB set_parameters.m, 1-indexed).
# N = [16,17,19,12,3,2,1,7,4,5,9,8,46,48,10,11,57,6]
N_MATLAB = [16, 17, 19, 12, 3, 2, 1, 7, 4, 5, 9, 8, 46, 48, 10, 11, 57, 6]

# Per-country AR lag order choice (mirrors decompose.m block using aikbic).
# aikbic returns the AIC-selected order; we hard-code the maxlag used so that
# we can solve the criterion identically. The comment in the source states:
#   if ord==4 or 12: maxlag=10
#   elif ord==7:     maxlag=3
#   elif ord==9,10:  maxlag=2
#   elif ord==11:    maxlag=6
#   elif ord==46:    maxlag=10
#   else:            maxlag=6
# plus two "info sufficiency" corrections: ord==1 or ord==16 → aik += 1 (baseline 1).
LAG_MAXLAG = {4: 10, 12: 10, 7: 3, 9: 2, 10: 2, 11: 6, 46: 10}
LAG_DEFAULT = 6
INFO_SUFF_BUMP = {1, 16}

H_MAX = 12            # horizon for max-FEV target
NIR = 100             # IRF horizons stored
KK = 12               # h=12 months for excess returns


def load_country_names() -> list[str]:
    m = sio.loadmat(COUNTRY_MAT)
    return [x[0][0] for x in m["COUNTRYNAME_DATA"]]


def load_cdl() -> pd.DataFrame:
    raw = pd.read_excel(DATA_XLS, header=None)
    # row 0 is group label, row 1 is variable label, data from row 2
    return raw.iloc[2:].reset_index(drop=True)


def load_vix() -> np.ndarray:
    m = sio.loadmat(VIX_MAT)
    return np.asarray(m["VIX_DATA"]).squeeze()


def country_block(cdl: pd.DataFrame, ord1: int) -> np.ndarray:
    """Return the 3-column [FFR, HomeRate, log FX] block for a given MATLAB ord.

    In MATLAB, jj = ord*3 (1-indexed). ord=1 -> cols 3,4,5; ord=2 -> 6,7,8; ...
    In 0-indexed pandas that maps to columns jj-1..jj+1.
    """
    jj0 = ord1 * 3 - 1
    block = cdl.iloc[:, jj0 : jj0 + 3].to_numpy(dtype=float)
    return block


def drop_nan_rows(block: np.ndarray) -> np.ndarray:
    mask = ~np.any(np.isnan(block), axis=1)
    return block[mask]


def estimate_var(Y: np.ndarray, nlag: int):
    """Return (beta, A1, mu, omega) matching MATLAB quick_VAR conventions.

    beta   : (nvar*nlag, nvar) slope matrix so that y_t' = [y_{t-1}',...,y_{t-p}'] beta + const
    A1     : Cholesky factor of Σ_u (lower triangular)
    mu     : residuals (T-p, nvar)
    omega  : residual covariance (divided by T, not T-k, like MATLAB's quick_VAR)
    """
    Y = np.asarray(Y, dtype=float)
    T_full, nvar = Y.shape
    T = T_full - nlag
    X = np.zeros((T, nlag * nvar))
    for j in range(1, nlag + 1):
        X[:, (j - 1) * nvar : j * nvar] = Y[nlag - j : T_full - j, :]
    X = np.hstack([X, np.ones((T, 1))])
    YY = Y[nlag:, :]
    beta_full = np.linalg.solve(X.T @ X, X.T @ YY)  # (nlag*nvar+1, nvar)
    mu = YY - X @ beta_full
    omega = mu.T @ mu / T
    A1 = np.linalg.cholesky(omega)  # lower-triangular, matches MATLAB chol(omega)'
    beta = beta_full[:-1, :]
    return beta, A1, mu, omega


def companion(beta: np.ndarray, nvar: int, nlag: int) -> np.ndarray:
    F = np.zeros((nvar * nlag, nvar * nlag))
    F[:nvar, :] = beta.T
    for j in range(1, nlag):
        F[j * nvar : (j + 1) * nvar, (j - 1) * nvar : j * nvar] = np.eye(nvar)
    return F


def reduced_impulses(beta: np.ndarray, A1: np.ndarray, nlag: int, H: int):
    """R_l = top (nvar x nvar) block of F^l times A1, for l=0..H."""
    nvar = A1.shape[0]
    F = companion(beta, nvar, nlag)
    Rt = np.zeros((H + 1, nvar, nvar))
    Fp = np.eye(nvar * nlag)
    for l in range(H + 1):
        Rt[l] = Fp[:nvar, :nvar] @ A1
        Fp = Fp @ F
    return Rt


def uhlig_weight_matrix(Rt: np.ndarray, target_idx: int) -> np.ndarray:
    H1 = Rt.shape[0]
    nvar = Rt.shape[1]
    E = np.zeros((nvar, nvar))
    E[target_idx, target_idx] = 1.0
    S = np.zeros((nvar, nvar))
    for l in range(H1):
        S += (H1 - l) * Rt[l].T @ E @ Rt[l]
    return S


def domestic_shock(beta: np.ndarray, A1: np.ndarray, nlag: int, H: int) -> np.ndarray:
    """Returns unit-norm vector minimising FEV of the first variable (FFR)."""
    Rt = reduced_impulses(beta, A1, nlag, H)
    S = uhlig_weight_matrix(Rt, target_idx=0)
    vals, vecs = np.linalg.eig(S)
    vals = np.real(vals)
    order = np.argsort(-vals)  # descending
    return np.real(vecs[:, order[-1]])  # smallest eigenvalue -> minimum FEV


def external_shock(beta: np.ndarray, A1: np.ndarray, nlag: int, H: int, fx_idx: int, d: np.ndarray) -> np.ndarray:
    """Returns unit-norm vector maximising FEV of variable `fx_idx`, orthogonal to d."""
    Rt = reduced_impulses(beta, A1, nlag, H)
    G = uhlig_weight_matrix(Rt, target_idx=fx_idx)
    vals, vecs = np.linalg.eig(G)
    vals = np.real(vals)
    idx = int(np.argmax(vals))
    v = np.real(vecs[:, idx])
    # project onto the orthogonal complement of d
    orth = v - (v @ d) * d
    return orth / np.linalg.norm(orth)


def build_structural(beta, A1, nlag, H, fx_idx=2):
    d = domestic_shock(beta, A1, nlag, H)
    g = external_shock(beta, A1, nlag, H, fx_idx, d)
    # third shock orthogonal to [g, d] (null space of the stacked pair)
    M = np.column_stack([g, d])  # nvar x 2
    # null space = right singular vectors of M^T with singular value 0
    u, s, vh = np.linalg.svd(M.T)
    # last column of vh.T corresponds to smallest singular value of M^T
    resid = vh.T[:, -1]
    D = np.column_stack([resid, g, d])  # columns [extra, external, domestic]
    return D


def structural_impulses(beta, A1, D, nlag, nir):
    nvar = A1.shape[0]
    F = companion(beta, nvar, nlag)
    S = A1 @ D
    Rt = np.zeros((nir, nvar, nvar))
    Fp = np.eye(nvar * nlag)
    for l in range(nir):
        Rt[l] = Fp[:nvar, :nvar] @ S
        Fp = Fp @ F
    # Rt[l][:, k] is response of all vars at horizon l to a unit shock in structural k
    return Rt


def forecast_error_variance_share(Rt: np.ndarray, variable_idx: int, horizons):
    """Return matrix of FEV shares at each requested horizon, shape (H, nshocks)."""
    nvar = Rt.shape[1]
    out = np.zeros((len(horizons), nvar))
    for h_i, h in enumerate(horizons):
        contrib = np.zeros(nvar)
        total = 0.0
        for l in range(h + 1):
            for k in range(nvar):
                c = Rt[l, variable_idx, k] ** 2
                contrib[k] += c
                total += c
        out[h_i] = contrib / total
    return out


def country_order_of_lag(ord1: int, nobs_guess: int | None = None) -> int:
    """Return the AIC-selected lag order matching decompose.m's aikbic call."""
    return LAG_MAXLAG.get(ord1, LAG_DEFAULT)


def aic_select(Y: np.ndarray, maxlag: int) -> int:
    """Select VAR order minimising AIC, matching aikbic.m exactly.

    aikbic.m uses T=full sample length (not T-p), penalty 2*p*N^2/T.
    """
    T_full, nvar = Y.shape
    best = (np.inf, 1)
    maxlag = max(1, min(maxlag, T_full // 3))
    for p in range(1, maxlag + 1):
        if T_full - p <= nvar * p + 1:
            continue
        try:
            _, _, mu, _ = estimate_var(Y, p)
        except np.linalg.LinAlgError:
            continue
        Sigma = (mu.T @ mu) / T_full
        logdet = np.log(np.linalg.det(Sigma))
        aic = logdet + 2 * p * nvar ** 2 / T_full
        if aic < best[0]:
            best = (aic, p)
    return best[1]


def select_lag(ord1: int, Y: np.ndarray) -> int:
    maxlag = country_order_of_lag(ord1)
    p = aic_select(Y, maxlag)
    if ord1 in INFO_SUFF_BUMP:
        p += 1  # matches decompose.m info-sufficiency bump for baseline=1
    return p


def impulse_responses(beta, S, nlag, nir):
    """Generic IRF routine given structural impact matrix S (nvar x k)."""
    nvar = S.shape[0]
    F = companion(beta, nvar, nlag)
    out = np.zeros((nir, nvar, S.shape[1]))
    Fp = np.eye(nvar * nlag)
    for l in range(nir):
        out[l] = Fp[:nvar, :nvar] @ S
        Fp = Fp @ F
    return out
