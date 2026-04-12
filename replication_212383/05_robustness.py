"""Robustness checks for Cormun & De Leo (2019)'s headline results:

    H1: domestic shocks account for ~20% of currency excess-return variance at
        the 12-month horizon, external shocks the remaining ~80%.
    H2: the Fama beta is close to 1 conditional on domestic shocks but is
        strongly negative conditional on external shocks.

For each robustness scenario we recompute (a) the country-median variance
share of excess returns explained by the domestic shock at h=12 and (b) the
country-median Fama beta on each conditional series. The baseline is the
point estimate from `01_decompose.py` / `02_figure1.py` / `03_fama.py`.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

import utils as u
from importlib import import_module

fig1 = import_module("02_figure1")  # reuse helpers
fama = import_module("03_fama")


def run_pipeline(cdl, ord1, *, H=u.H_MAX, p_override=None, post_year=None, winsorize=0.0):
    block = u.country_block(cdl, ord1)
    clean = u.drop_nan_rows(block)
    if post_year is not None:
        # year column is first col of cdl
        year = pd.to_numeric(cdl.iloc[:, 0], errors='coerce').to_numpy()
        mask = ~np.any(np.isnan(block), axis=1)
        yy = year[mask]
        keep = yy >= post_year
        clean = clean[keep]
    if winsorize > 0:
        dfx = np.diff(clean[:, 2])
        lo, hi = np.quantile(dfx, [winsorize, 1 - winsorize])
        dfx_w = np.clip(dfx, lo, hi)
        clean = clean.copy()
        clean[1:, 2] = clean[0, 2] + np.cumsum(dfx_w)
    if clean.shape[0] < 25:
        return None
    p = p_override if p_override else u.select_lag(ord1, clean)
    p = max(1, min(p, (clean.shape[0] - 2) // 4))
    try:
        beta, A1, mu, _ = u.estimate_var(clean, p)
        D = u.build_structural(beta, A1, p, H, fx_idx=2)
    except Exception:
        return None
    irf = u.structural_impulses(beta, A1, D, p, u.NIR)
    shares = fig1.compute_shares(irf, kk=u.KK, horizon_idx=11)
    # Fama betas
    A1D = A1 @ D
    shocks = np.linalg.solve(A1D, mu.T).T
    T = clean.shape[0]
    fama_b = {}
    for k, lab in enumerate(["resid", "ext", "dom"]):
        one = np.zeros(T)
        one[p:] = shocks[:, k]
        ss = np.zeros((T, 3))
        for t in range(p, T):
            lag_vec = np.concatenate([ss[t - j] for j in range(1, p + 1)])
            ss[t] = lag_vec @ beta + one[t] * A1D[:, k]
        ffr, hr, fx = ss[:, 0], ss[:, 1], ss[:, 2]
        head = slice(0, T - u.KK)
        fwd = slice(u.KK, T)
        er = hr[head] / 12 * u.KK - ffr[head] / 12 * u.KK - (fx[fwd] - fx[head])
        diff = hr[head] / 12 * u.KK - ffr[head] / 12 * u.KK
        fama_b[lab] = fama.fama_beta(er, diff)
    return dict(
        share_dom=shares[2],
        share_ext=shares[1] + shares[0],
        fama_dom=fama_b["dom"],
        fama_ext=fama_b["ext"],
    )


def aggregate(cdl, label: str, **kwargs):
    rows = []
    for ord1 in u.N_MATLAB:
        out = run_pipeline(cdl, ord1, **kwargs)
        if out is not None:
            rows.append(out)
    df = pd.DataFrame(rows)
    return dict(
        scenario=label,
        n_countries=len(rows),
        median_share_dom=df["share_dom"].median(),
        median_share_ext=df["share_ext"].median(),
        median_fama_dom=df["fama_dom"].median(),
        median_fama_ext=df["fama_ext"].median(),
    )


def leave_one_out(cdl):
    """Country leave-one-out: compute median robustness bounds across LOO samples."""
    med_doms = []
    med_fams = []
    for drop in u.N_MATLAB:
        rows = [run_pipeline(cdl, ord1) for ord1 in u.N_MATLAB if ord1 != drop]
        rows = [r for r in rows if r is not None]
        df = pd.DataFrame(rows)
        med_doms.append(df["share_dom"].median())
        med_fams.append(df["fama_ext"].median())
    return dict(
        scenario="leave-one-country-out",
        n_countries=len(u.N_MATLAB) - 1,
        median_share_dom=np.median(med_doms),
        median_share_ext=1 - np.median(med_doms),
        median_share_dom_range=f"[{min(med_doms):.3f}, {max(med_doms):.3f}]",
        median_fama_ext=np.median(med_fams),
        median_fama_ext_range=f"[{min(med_fams):.3f}, {max(med_fams):.3f}]",
    )


def main() -> None:
    cdl = u.load_cdl()

    results = []
    results.append(aggregate(cdl, "baseline"))
    results.append(aggregate(cdl, "fixed p=4", p_override=4))
    results.append(aggregate(cdl, "fixed p=6", p_override=6))
    results.append(aggregate(cdl, "H=6 (shorter max-share horizon)", H=6))
    results.append(aggregate(cdl, "H=24 (longer max-share horizon)", H=24))
    results.append(aggregate(cdl, "post-1990 sample", post_year=1990))
    results.append(aggregate(cdl, "winsorize Δfx 1%", winsorize=0.01))
    results.append(aggregate(cdl, "winsorize Δfx 5%", winsorize=0.05))

    # Leave-one-out (computed separately because structure differs)
    loo = leave_one_out(cdl)

    df = pd.DataFrame(results)
    df.to_csv(u.OUT / "robustness.csv", index=False)
    print("Robustness scenarios (country-level medians):")
    print(df.to_string(index=False, float_format=lambda v: f"{v:7.3f}"))
    print()
    print("Leave-one-country-out:")
    for k, v in loo.items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
