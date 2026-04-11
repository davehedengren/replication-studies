"""Robustness checks for the California Prop 99 SDID replication."""
from __future__ import annotations
import json
import numpy as np
from utils import (
    load_california, panel_matrices, HERE,
    synthdid_estimate, sc_estimate, did_estimate,
    placebo_se, jackknife_se,
)

OUT = HERE / "outputs"


def build_panel():
    df = load_california()
    setup = panel_matrices(df)
    return setup


def _tau(Y, N0, T0, **kw):
    return synthdid_estimate(Y, N0, T0, **kw)["tau"]


def main():
    setup = build_panel()
    Y, N0, T0 = setup["Y"], setup["N0"], setup["T0"]
    units = setup["units"]
    times = setup["times"]

    lines = []
    def log(msg=""):
        print(msg); lines.append(msg)

    base_est = synthdid_estimate(Y, N0, T0)
    base_tau = base_est["tau"]
    log(f"Baseline SDID τ = {base_tau:+.3f}")
    log("")

    # ---- 1. Alternative SEs (jackknife vs placebo) ----
    placebo_val, _ = placebo_se(base_est, replications=200, seed=12345)
    placebo_500, _ = placebo_se(base_est, replications=500, seed=12345)
    jk = jackknife_se(base_est)
    log("[1] Standard error comparison")
    log(f"    Placebo SE (200 reps, Alg. 4):   {placebo_val:.2f}")
    log(f"    Placebo SE (500 reps, Alg. 4):   {placebo_500:.2f}")
    if isinstance(jk, float) and np.isnan(jk):
        log(f"    Fixed-weights jackknife: N/A (only 1 treated unit, Alg. 3 returns NaN)")
    else:
        log(f"    Fixed-weights jackknife (Alg. 3): {jk[0]:.2f}")
    log("")

    # ---- 2. Leave-one-top-donor-out ----
    omega = base_est["omega"]
    top_donors = np.argsort(-omega)[:5]
    log("[2] Leave-one-top-donor-out (5 biggest control weights)")
    for i in top_donors:
        keep = [j for j in range(Y.shape[0]) if j != i]
        Y2 = Y[keep]
        # Re-identify N0: we removed a control (i < N0)
        new_N0 = N0 - 1 if i < N0 else N0
        t = _tau(Y2, new_N0, T0)
        log(f"    drop {units[i]:<15} (ω={omega[i]:.3f}): τ = {t:+.2f}  (Δ={t-base_tau:+.2f})")
    log("")

    # ---- 3. Leave-one-year-out of pre-period (sensitivity to lambda) ----
    log("[3] Leave-one-pre-year-out")
    drops = []
    for j in range(T0):
        keep = [k for k in range(Y.shape[1]) if k != j]
        Y2 = Y[:, keep]
        t = _tau(Y2, N0, T0 - 1)
        drops.append((times[j], t))
    drops_sorted = sorted(drops, key=lambda x: abs(x[1] - base_tau), reverse=True)
    log(f"    worst drops:")
    for yr, t in drops_sorted[:5]:
        log(f"      drop {yr}: τ = {t:+.2f} (Δ={t-base_tau:+.2f})")
    log(f"    range of τ across drops: [{min(t for _,t in drops):+.2f}, "
        f"{max(t for _,t in drops):+.2f}]")
    log("")

    # ---- 4. Placebo-in-time (pretend treatment started earlier) ----
    log("[4] Placebo-in-time: shift treatment date earlier (no data change)")
    log("    Use only pre-treatment data, split at a fake cutoff")
    for fake_T0 in [10, 12, 14, 16]:
        # Use Y[:, :T0] only, pretend fake_T0 of those are pre and rest are post
        Y_pre = Y[:, :T0]
        t = _tau(Y_pre, N0, fake_T0)
        fake_year = times[fake_T0]
        log(f"    fake treatment at {fake_year}: τ = {t:+.2f}")
    log("")

    # ---- 5. Regularization sensitivity (varying eta_omega) ----
    log("[5] Regularization sensitivity (varying η_ω)")
    noise = base_est["noise_level"]
    N1 = Y.shape[0] - N0; T1 = Y.shape[1] - T0
    default_eta = (N1 * T1) ** 0.25
    for mult in [0.25, 0.5, 1.0, 2.0, 4.0]:
        eta = default_eta * mult
        t = _tau(Y, N0, T0, eta_omega=eta)
        log(f"    η_ω = {mult:4.2f} × default ({eta:.3f}): τ = {t:+.2f}")
    log("")

    # ---- 6. No sparsify ----
    log("[6] Disable sparsify step")
    t = _tau(Y, N0, T0, sparsify=False)
    log(f"    τ = {t:+.2f}  (Δ={t-base_tau:+.2f})")
    log("")

    # ---- 7. No omega intercept ----
    log("[7] Disable omega intercept (pure simplex, no constant)")
    t = _tau(Y, N0, T0, omega_intercept=False)
    log(f"    τ = {t:+.2f}  (Δ={t-base_tau:+.2f})")
    log("")

    # ---- 8. No lambda intercept ----
    log("[8] Disable lambda intercept")
    t = _tau(Y, N0, T0, lambda_intercept=False)
    log(f"    τ = {t:+.2f}  (Δ={t-base_tau:+.2f})")
    log("")

    # ---- 9. Placebo outcome: pretend Nevada is treated ----
    log("[9] In-space placebo (pretend each control is 'treated')")
    from utils import sc_estimate  # noqa
    placebo_taus = []
    for i in range(N0):
        order = [j for j in range(Y.shape[0]) if j != i and j < N0] + [i]
        # Drop actual California (last row) to keep a single treated placebo unit
        # Actually — put control i as last, keep all other N0-1 controls as control, drop CA.
        Y2 = Y[order]
        t = _tau(Y2, N0 - 1, T0)
        placebo_taus.append(t)
    placebo_taus = np.array(placebo_taus)
    share_exceeding = float(np.mean(np.abs(placebo_taus) >= abs(base_tau)))
    log(f"    placebo τ mean: {placebo_taus.mean():+.2f}, sd: {placebo_taus.std():.2f}")
    log(f"    share of |placebo τ| ≥ |baseline|: {share_exceeding:.3f}  "
        f"(one-sided p ≈ {share_exceeding:.3f})")
    log("")

    # ---- 10. Shorter post-period ----
    log("[10] Shorter post-period (effect of later years on the average)")
    # Only use years up through various post-period cutoffs
    for last_yr in [1993, 1995, 1997, 2000]:
        last_idx = times.index(last_yr) + 1
        Y2 = Y[:, :last_idx]
        t = _tau(Y2, N0, T0)
        log(f"    end at {last_yr} (T1={last_idx-T0}): τ = {t:+.2f}")
    log("")

    # ---- 11. Drop Nevada (the largest donor) and re-run full Table 1 ----
    log("[11] Drop Nevada (largest donor) and re-run Table 1-like comparison")
    nv = units.index("Nevada")
    keep = [j for j in range(Y.shape[0]) if j != nv]
    Y2 = Y[keep]
    new_N0 = N0 - 1
    sdid_t = synthdid_estimate(Y2, new_N0, T0)["tau"]
    sc_t   = sc_estimate(Y2, new_N0, T0)["tau"]
    did_t  = did_estimate(Y2, new_N0, T0)["tau"]
    log(f"    SDID: {sdid_t:+.2f}  SC: {sc_t:+.2f}  DID: {did_t:+.2f}")
    log(f"    (baseline:  {base_tau:+.2f},  "
        f"{sc_estimate(Y, N0, T0)['tau']:+.2f},  "
        f"{did_estimate(Y, N0, T0)['tau']:+.2f})")
    log("")

    with open(OUT / "robustness.txt", "w") as f:
        f.write("\n".join(lines))
    print(f"Saved {OUT}/robustness.txt")


if __name__ == "__main__":
    main()
