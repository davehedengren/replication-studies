"""Replicate Figure 1a/1b — variance share of excess returns explained by
domestic vs external shocks at horizon h=12, country by country.

Mirrors Main_Scripts/RUN_Figure1a.m. The construction used by the authors:
  1. Compute IRFs of the three structural shocks to [FFR, HomeRate, log FX] at
     horizons 0..nir-KK-1.
  2. Append a 4th "variable" = h-month excess return on the SOE currency:
        er_h(t) = HomeRate(t) - FFR(t) - [log FX(t+KK) - log FX(t)]
     where H is KK=12 months ahead and rates are in annualised decimal.
  3. Compute VSHARE_{shock}(h) = cumsum_{l=0..h} IRF_shock(l, var4)^2 /
     sum over all three shocks of the same. The figure reports the share at
     h = HORIZON (=12 in the code — note 1-indexed, so python index 11).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

import utils as u


def excess_return_irf(irf_level: np.ndarray, kk: int = u.KK) -> np.ndarray:
    """Build the h-step excess-return IRF for a single shock.

    irf_level shape: (nir, 3) for one shock. Returns shape (nir - kk,).
    Matches the formula in RUN_Figure1a.m with KK=12.
    """
    ffr = irf_level[:, 0]
    hr = irf_level[:, 1]
    fx = irf_level[:, 2]
    nir = irf_level.shape[0]
    head = slice(0, nir - kk)
    fwd = slice(kk, nir)
    return hr[head] / 12 * kk - ffr[head] / 12 * kk - (fx[fwd] - fx[head])


def compute_shares(irfs: np.ndarray, kk: int, horizon_idx: int):
    """Given irfs of shape (nir, 3 vars, 3 shocks), return shares of the three
    shocks explaining variance of excess returns at horizon_idx.
    """
    er = np.zeros((irfs.shape[0] - kk, 3))
    for k in range(3):
        er[:, k] = excess_return_irf(irfs[:, :, k], kk)
    vshare_all = np.cumsum(np.sum(er ** 2, axis=1))
    vshare_by_shock = np.cumsum(er ** 2, axis=0) / vshare_all[:, None]
    return vshare_by_shock[horizon_idx]  # (3,) shares for [resid, ext, dom]


def main() -> None:
    names = u.load_country_names()
    irfs = np.load(u.OUT / "country_results.npz")

    rows = []
    for ord1 in u.N_MATLAB:
        arr = irfs[str(ord1)]
        shares = compute_shares(arr, kk=u.KK, horizon_idx=11)
        rows.append(
            dict(
                ord1=ord1,
                country=names[ord1 - 1],
                share_resid=shares[0],
                share_ext=shares[1],
                share_dom=shares[2],
            )
        )

    df = pd.DataFrame(rows)
    df["share_ext_total"] = df["share_ext"] + df["share_resid"]  # both external-type
    df.to_csv(u.OUT / "figure1_vshares.csv", index=False)

    print("Variance share of excess returns at h=12 (1-indexed) by country:")
    print(df.to_string(index=False, float_format=lambda v: f"{v:6.3f}"))
    print()
    print(f"Median domestic share : {df['share_dom'].median():.3f}")
    print(f"Median external share : {df['share_ext'].median():.3f}")
    print(f"Median ext+resid share: {df['share_ext_total'].median():.3f}")
    print()
    print("Paper claim (levels, Figure 1a): median external ≈ 0.80, domestic ≈ 0.20")


if __name__ == "__main__":
    main()
