"""Run country-by-country VARs and identify domestic + external shocks.

Mirrors Main_Scripts/decompose.m with baseline=1 (levels, 3-variable VAR on
[FFR, Home rate, log FX]). Bootstrap is omitted here; robustness script
adds block-bootstrap uncertainty bands for the FEV shares.

Outputs:
- outputs/country_results.npz  — IRFs and structural impact matrices
- outputs/country_shocks.npz   — identified structural shocks per country
- outputs/decompose_summary.csv — per-country N, lag, FEV shares
"""
from __future__ import annotations

import numpy as np
import pandas as pd

import utils as u


def main() -> None:
    cdl = u.load_cdl()
    names = u.load_country_names()

    summaries = []
    all_irf = {}
    shocks_out = {}
    impact_mats = {}

    for ord1 in u.N_MATLAB:
        block = u.country_block(cdl, ord1)
        clean = u.drop_nan_rows(block)
        T, nvar = clean.shape
        p = u.select_lag(ord1, clean)
        beta, A1, mu, omega = u.estimate_var(clean, p)
        D = u.build_structural(beta, A1, p, u.H_MAX, fx_idx=2)
        # IRFs at horizon NIR to the three structural shocks (cols = [resid, ext, dom])
        irf = u.structural_impulses(beta, A1, D, p, u.NIR)
        all_irf[str(ord1)] = irf
        impact_mats[str(ord1)] = A1 @ D

        # FEV shares on log FX at horizon 12
        fev_fx12 = u.forecast_error_variance_share(irf, variable_idx=2, horizons=[11])[0]
        fev_ffr12 = u.forecast_error_variance_share(irf, variable_idx=0, horizons=[11])[0]
        fev_hr12 = u.forecast_error_variance_share(irf, variable_idx=1, horizons=[11])[0]

        # structural residuals: shocks_t = (A1*D)^{-1} mu_t
        S = A1 @ D
        shocks = np.linalg.solve(S, mu.T).T  # (T-p, 3)
        shocks_out[str(ord1)] = shocks

        summaries.append(
            dict(
                ord1=ord1,
                country=names[ord1 - 1],
                T=T,
                lag=p,
                fev_fx_extra=fev_fx12[0],
                fev_fx_ext=fev_fx12[1],
                fev_fx_dom=fev_fx12[2],
                fev_ffr_extra=fev_ffr12[0],
                fev_ffr_ext=fev_ffr12[1],
                fev_ffr_dom=fev_ffr12[2],
                fev_hr_ext=fev_hr12[1],
                fev_hr_dom=fev_hr12[2],
            )
        )

    df = pd.DataFrame(summaries)
    df.to_csv(u.OUT / "decompose_summary.csv", index=False)

    print(df.to_string(index=False, float_format=lambda v: f"{v:7.3f}"))
    print()
    med_ext = df["fev_fx_ext"].median()
    med_dom = df["fev_fx_dom"].median()
    print(
        f"Median FEV share (log FX, h=12): external={med_ext:.3f}  domestic={med_dom:.3f}  "
        f"residual={(df['fev_fx_extra']).median():.3f}"
    )

    np.savez(u.OUT / "country_results.npz", **all_irf)
    np.savez(u.OUT / "country_shocks.npz", **shocks_out)
    np.savez(u.OUT / "impact_mats.npz", **impact_mats)


if __name__ == "__main__":
    main()
