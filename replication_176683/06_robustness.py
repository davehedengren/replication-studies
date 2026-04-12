"""Robustness checks tailored to a theoretical calibration paper.

Since the paper has no regressions, the natural robustness exercises are:
  1. Sweep over β (ideas-harder-to-find) for both calibrations
  2. Sweep over λ (spillover exponent)
  3. Sweep over n_eq (equilibrium population growth rate)
  4. Sweep over ε (preference for offspring) via ellEqm
  5. Sweep over δ (death rate / expected lifetime)
  6. Reproduce the alternative SANDBOX cases commented in SetParameters.m
  7. Check Empty-Planet GDP factor exp(gA0/eta) for alternative (gA0, eta) grids
  8. Check semi-endogenous long-run factor (1 + β*gA0*η⁻¹)^(1/β) for alt grids
  9. Check that the High/Middle roots disappear when b²-4ac < 0 (degenerate case)
 10. Monotonicity of steady-state n*_sp and g^sp in β and in λ
"""
import numpy as np
from utils import set_parameters_nber, solvemodel


def try_solve(bbar, rho, epsilon, delta, lam, beta, gamma):
    """Return (nH, gAH, ellH) or (nan,nan,nan) if no real SS."""
    try:
        nEqm = bbar - delta - rho / epsilon
        if nEqm >= 0:
            return (np.nan, np.nan, np.nan)
        nu = (epsilon + gamma) / rho
        btilde = bbar - delta - (epsilon + lam) / (lam * nu)
        b24ac = 0.25 * btilde**2 + epsilon * nEqm / (lam * nu)
        if b24ac < 0:
            return (np.nan, np.nan, np.nan)
        nH = btilde / 2 + np.sqrt(b24ac)
        gAH = lam * nH / beta
        ellH = (nH + delta) / bbar
        return (nH, gAH, ellH)
    except Exception:
        return (np.nan, np.nan, np.nan)


def recompute(p):
    """Given a parameter dict with base inputs, derive bbar,epsilon,gamma and solve."""
    bbar = (p["nEqm"] + p["delta"]) / p["ellEqm"]
    if bbar <= 0 or p["ellEqm"] >= 1:
        return (np.nan, np.nan, np.nan)
    epsilon = p["rho"] / (bbar * (1 - p["ellEqm"]))
    gamma = p["lam"] * p["sigma"] / p["beta"]
    return try_solve(bbar, p["rho"], epsilon, p["delta"], p["lam"], p["beta"], gamma)


def sweep(base, key, values):
    print(f"\n--- Sweep over {key} ---")
    print(f"{key:>10s}  {'nH%':>10s}  {'gAH%':>10s}  {'ellH':>10s}")
    for v in values:
        p = dict(base)
        p[key] = v
        nH, gAH, ellH = recompute(p)
        print(f"{v:10.4f}  {100*nH:10.4f}  {100*gAH:10.4f}  {ellH:10.4f}")


def main():
    base = set_parameters_nber()  # Use NBER Table 2 as baseline

    print("=" * 72)
    print("ROBUSTNESS / SENSITIVITY CHECKS  (baseline: NBER Table 2)")
    print("=" * 72)
    nH0, gAH0, ellH0 = recompute(base)
    print(f"\nBaseline: nH={100*nH0:.4f}%, gAH={100*gAH0:.4f}%, ellH={ellH0:.4f}")

    # 1. beta sweep
    sweep(base, "beta", [0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 2.5])
    # 2. lambda sweep
    sweep(base, "lam", [0.5, 0.75, 1.0, 1.25])
    # 3. n_eq sweep
    sweep(base, "nEqm", [-0.01, -0.0075, -0.005, -0.0025, 0.0])
    # 4. ellEqm sweep
    sweep(base, "ellEqm", [1 / 12, 1 / 10, 1 / 8, 1 / 6, 1 / 4])
    # 5. delta sweep (=rho, per code)
    print("\n--- Sweep over delta (with rho=delta) ---")
    print(f"{'delta':>10s}  {'nH%':>10s}  {'gAH%':>10s}  {'ellH':>10s}")
    for d in [1 / 120, 1 / 100, 1 / 90, 1 / 80]:
        p = dict(base)
        p["delta"] = d
        p["rho"] = d
        nH, gAH, ellH = recompute(p)
        print(f"{d:10.5f}  {100*nH:10.4f}  {100*gAH:10.4f}  {ellH:10.4f}")

    # 6. SANDBOX cases (from SetParameters.m comments)
    print("\n--- SANDBOX cases from SetParameters.m comments ---")
    sandboxes = [
        ("Current Benchmark",  dict(sigma=1, lam=3/4,  ellEqm=1/8,  delta=1/90, nEqm=-.005, rho=1/90,  beta=2.0)),
        ("Lower ellEqm=1/10", dict(sigma=1, lam=3/4,  ellEqm=1/10, delta=1/90, nEqm=-.005, rho=1/90,  beta=2.0)),
        ("Life exp 80yrs",     dict(sigma=1, lam=3/4,  ellEqm=1/12, delta=1/80, nEqm=-.005, rho=1/80,  beta=3.0)),
        ("Higher bbar-delta",  dict(sigma=1, lam=3/4,  ellEqm=1/10, delta=1/90, nEqm=-.005, rho=1/90,  beta=2.0)),
        ("NBER WP v1.0",       dict(sigma=1, lam=1.0,  ellEqm=1/8,  delta=0.01, nEqm=-.005, rho=0.01,  beta=1.25)),
    ]
    print(f"{'Case':25s}  {'nH%':>8s}  {'gAH%':>8s}  {'ellH':>8s}")
    for name, p in sandboxes:
        nH, gAH, ellH = recompute(p)
        print(f"{name:25s}  {100*nH:8.3f}  {100*gAH:8.3f}  {ellH:8.3f}")

    # 7. Empty-Planet GDP factor (Romer case): y*/y0 = exp(gy0/eta)
    print("\n--- Check 7: Empty-Planet factor (Romer), grid of (gy0, eta) ---")
    print(f"{'gy0':>6s} {'eta':>6s} {'factor':>10s}")
    for gy0, eta in [(0.02, 0.01), (0.02, 0.005), (0.01, 0.005), (0.015, 0.01)]:
        print(f"{gy0:6.3f} {eta:6.3f} {np.exp(gy0/eta):10.4f}")

    # 8. Semi-endogenous long-run factor
    print("\n--- Check 8: Semi-endog factor (1+3*gy0/eta)^(1/beta) ---")
    print(f"{'gy0':>6s} {'eta':>6s} {'beta':>6s} {'factor':>10s}")
    for gy0, eta, beta in [(0.02, 0.01, 3), (0.02, 0.01, 1.25), (0.02, 0.01, 0.5), (0.01, 0.005, 3)]:
        m = beta * gy0 / eta
        print(f"{gy0:6.3f} {eta:6.3f} {beta:6.2f} {(1+m)**(1/beta):10.4f}")

    # 9. Degenerate case (discriminant negative)
    print("\n--- Check 9: Degenerate (no-real-SS) regime as gamma shrinks ---")
    # Reducing γ (= λ/β) pushes toward spiral / no real SS; try very large β
    print(f"{'beta':>6s} {'b2-4ac':>12s} {'nH%':>10s}")
    for beta in [1.25, 2, 3, 4, 5, 6, 8, 10]:
        p = dict(base)
        p["beta"] = beta
        bbar = (p["nEqm"] + p["delta"]) / p["ellEqm"]
        epsilon = p["rho"] / (bbar * (1 - p["ellEqm"]))
        gamma = p["lam"] * p["sigma"] / p["beta"]
        nu = (epsilon + gamma) / p["rho"]
        btilde = bbar - p["delta"] - (epsilon + p["lam"]) / (p["lam"] * nu)
        nEqm = bbar - p["delta"] - p["rho"] / epsilon
        b24ac = 0.25 * btilde**2 + epsilon * nEqm / (p["lam"] * nu)
        nH, gAH, _ = recompute(p)
        print(f"{beta:6.2f} {b24ac:12.6e} {100*nH:10.4f}")

    # 10. Monotonicity test
    print("\n--- Check 10: Monotonicity of n*_sp in β (β ↑ ⇒ n*_sp ↓?) ---")
    last_n = np.inf
    mono = True
    for b in np.linspace(0.5, 2.5, 21):
        p = dict(base); p["beta"] = b
        nH, _, _ = recompute(p)
        if not np.isnan(nH) and nH > last_n + 1e-9:
            mono = False
        if not np.isnan(nH):
            last_n = nH
    print(f"  n*_sp monotone decreasing in β over [0.5, 2.5]: {mono}")


if __name__ == "__main__":
    main()
