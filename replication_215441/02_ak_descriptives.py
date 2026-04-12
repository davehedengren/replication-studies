"""Andrews & Kasy (2019) publication-bias MLE with step-function selection.

Python port of source/funcs/metastudies.R. Implements the t-distribution and
normal variants used in Table 1 (descriptives) and Table 2 (AK robustness).
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import t as tdist, norm

from utils import load_long, OUT


# ---------- likelihood ----------
def _tpowers(T, cutoffs, symmetric):
    if symmetric:
        T = np.abs(T)
    n = len(T)
    k = len(cutoffs) + 1
    M = np.zeros((n, k))
    M[:, 0] = (T < cutoffs[0]).astype(float)
    for m in range(1, len(cutoffs)):
        M[:, m] = (T < cutoffs[m]).astype(float) * (T >= cutoffs[m - 1]).astype(float)
    M[:, -1] = (T >= cutoffs[-1]).astype(float)
    return M


def _nll(psi, X, sigma, cutoffs, symmetric, model, Tpowers):
    lam, tau = psi[0], psi[1]
    if model == "t":
        df = psi[2]
        beta_tail = psi[3:]
    else:
        df = np.inf
        beta_tail = psi[2:]
    betap = np.concatenate([beta_tail, [1.0]])

    if tau <= 0 or (model == "t" and df <= 0) or np.any(betap[:-1] <= 0):
        return 1e12

    phat = Tpowers @ betap  # (n,)
    sigma_tilde = np.sqrt(sigma**2 + tau**2)

    if np.isinf(df):
        fX = norm.pdf((X - lam) / sigma_tilde) / sigma_tilde
        cdf = norm.cdf
    else:
        fX = tdist.pdf((X - lam) / sigma_tilde, df) / sigma_tilde
        cdf = lambda z: tdist.cdf(z, df)

    n = len(X)
    normcuts = (sigma / sigma_tilde)[:, None] * np.array(cutoffs)[None, :] \
        - (lam / sigma_tilde)[:, None]
    if symmetric:
        negcuts = -(sigma / sigma_tilde)[:, None] * np.array(cutoffs)[None, :] \
            - (lam / sigma_tilde)[:, None]
        cdfs_mid = cdf(normcuts) - cdf(negcuts)
    else:
        cdfs_mid = cdf(normcuts)
    cdfs = np.column_stack([np.zeros(n), cdfs_mid, np.ones(n)])
    cell_probs = np.diff(cdfs, axis=1)
    normconst = cell_probs @ betap

    L = phat * fX / normconst
    if np.any(L <= 0) or np.any(~np.isfinite(L)):
        return 1e12
    return -np.sum(np.log(L))


def ak_fit(X, sigma, cutoffs=(1.96,), symmetric=False, model="t"):
    X = np.asarray(X, float)
    sigma = np.asarray(sigma, float)
    cutoffs = list(cutoffs)
    T = X / sigma
    Tpowers = _tpowers(T, cutoffs, symmetric)

    if model == "t":
        x0 = np.array([0.0, 1.0, 10.0] + [1.0] * len(cutoffs))
    else:
        x0 = np.array([0.0, 1.0] + [1.0] * len(cutoffs))

    obj = lambda p: _nll(p, X, sigma, cutoffs, symmetric, model, Tpowers)
    res = minimize(obj, x0, method="Nelder-Mead",
                   options={"xatol": 1e-8, "fatol": 1e-8, "maxiter": 200000, "maxfev": 200000})
    # polish with L-BFGS-B
    bounds = [(None, None), (1e-4, None)]
    if model == "t":
        bounds.append((0.5, None))
    bounds += [(1e-6, None)] * len(cutoffs)
    res2 = minimize(obj, res.x, method="L-BFGS-B", bounds=bounds,
                    options={"ftol": 1e-12, "gtol": 1e-10, "maxiter": 50000})
    if res2.fun < res.fun:
        res = res2
    return res


def main():
    df = load_long()  # Hunt retained for baseline
    rows = []
    for margin in ["RR", "PBD"]:
        sub = df[df["pbd_vs_rr"] == margin]
        X = sub["elasticity"].values
        s = sub["se"].values
        mean_e = X.mean()

        fit = ak_fit(X, s, cutoffs=(1.96,), symmetric=False, model="t")
        lam, tau, dfree, beta = fit.x
        rows.append({
            "margin": margin, "n": len(X), "mean_epsilon": mean_e,
            "beta_p": beta, "theta": lam, "tau": tau, "df": dfree,
            "nll": fit.fun,
        })

    out = pd.DataFrame(rows)
    print("Table 1 descriptives — AK t-distribution, one cutoff (1.96):")
    print(out.to_string(index=False, float_format=lambda v: f"{v:.4f}"))

    published = pd.DataFrame([
        {"margin": "RR",  "n_pub": 42, "epsilon_pub": 0.43, "beta_p_pub": 0.12, "theta_pub": 0.21, "tau_pub": 0.26, "df_pub": 3.80},
        {"margin": "PBD", "n_pub": 49, "epsilon_pub": 0.46, "beta_p_pub": 0.08, "theta_pub": 0.09, "tau_pub": 0.25, "df_pub": 2.92},
    ])
    print("\nPublished values:")
    print(published.to_string(index=False))

    out.to_csv(OUT / "ak_descriptives.csv", index=False)
    print(f"\nSaved {OUT/'ak_descriptives.csv'}")


if __name__ == "__main__":
    main()
