"""Conditional Fama-regression coefficients, Figure 2 equivalent.

For each country and each identified shock (domestic / external), we
construct the shock-conditional series of the VAR variables, form an excess
return at horizon KK=12, and regress the NEGATIVE excess return on the 12-
month interest-rate differential. A value near unity (the UIP prediction)
means the shock is UIP-consistent; values away from unity signal UIP
deviations associated with the shock.

Mirrors Main_Scripts/RUN_Figure2.m but without the bootstrap bands.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

import utils as u


def conditional_series(beta, A1D, shocks_one_col, nlag):
    """Return the VAR variable path driven only by shock `shocks_one_col`.

    `shocks_one_col` has shape (T,), where positions before initial condition
    are NaN or zero. We start from zeros and march the VAR forward with the
    impact vector A1D[:, k] (the k-th structural shock's impact).
    """
    T = shocks_one_col.shape[0]
    nvar = A1D.shape[0]
    ss = np.zeros((T, nvar))
    # zero initial nlag values
    for t in range(nlag, T):
        lag_vec = ss[t - 1 : t - nlag - 1 : -1].reshape(-1)  # [y_{t-1}', ..., y_{t-p}']
        ss[t] = lag_vec @ beta + shocks_one_col[t] * A1D  # impact vector is column k
    return ss


def fama_beta(er: np.ndarray, diff: np.ndarray) -> float:
    """OLS slope of (-er) on diff, no constant (mirrors `regress` in MATLAB)."""
    num = np.dot(diff, -er)
    den = np.dot(diff, diff)
    return num / den if den != 0 else np.nan


def main() -> None:
    cdl = u.load_cdl()
    names = u.load_country_names()
    shocks_all = np.load(u.OUT / "country_shocks.npz")
    impact_mats = np.load(u.OUT / "impact_mats.npz")

    rows = []
    kk = u.KK
    for ord1 in u.N_MATLAB:
        block = u.drop_nan_rows(u.country_block(cdl, ord1))
        p = u.select_lag(ord1, block)
        beta, A1, _, _ = u.estimate_var(block, p)
        A1D = impact_mats[str(ord1)]
        shocks = shocks_all[str(ord1)]  # (T-p, 3)

        # extend shocks to T-length, padding the first p with zeros for init
        T = block.shape[0]
        shock_full = np.zeros((T, 3))
        shock_full[p:, :] = shocks

        betas = {}
        for k, label in enumerate(["resid", "ext", "dom"]):
            # conditional path with only shock k
            one = np.zeros(T)
            one[p:] = shocks[:, k]
            ss = np.zeros((T, 3))
            for t in range(p, T):
                lag_vec = np.concatenate([ss[t - j] for j in range(1, p + 1)])
                ss[t] = lag_vec @ beta + one[t] * A1D[:, k]
            # build 12-month excess return on the conditional series
            ffr = ss[:, 0]
            hr = ss[:, 1]
            fx = ss[:, 2]
            head = slice(0, T - kk)
            fwd = slice(kk, T)
            er = hr[head] / 12 * kk - ffr[head] / 12 * kk - (fx[fwd] - fx[head])
            diff = hr[head] / 12 * kk - ffr[head] / 12 * kk
            betas[label] = fama_beta(er, diff)

        rows.append(dict(ord1=ord1, country=names[ord1 - 1], **betas))

    df = pd.DataFrame(rows)
    df.to_csv(u.OUT / "figure2_fama.csv", index=False)
    print(df.to_string(index=False, float_format=lambda v: f"{v:7.3f}"))
    print()
    print(f"Median Fama β, domestic shock: {df['dom'].median():.3f}")
    print(f"Median Fama β, external shock: {df['ext'].median():.3f}")
    print("Paper (Figure 2): median domestic β ~1 (UIP holds), external β strongly negative")


if __name__ == "__main__":
    main()
