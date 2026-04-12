"""Reproduces OptimalDynamics.m — ODE transition dynamics (Figure 6).

Uses scipy's ode45-equivalent (solve_ivp with RK45) with reverse shooting:
  - Start on the stable manifold very close to the High SS
  - Integrate BACKWARDS in time in (n, x) space
  - Produce the upper and lower arms of the saddle path

This reproduces Figure 6 and the steady-state values printed at the end of
OptimalDynamics.m.
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from utils import set_parameters, solvemodel, transit_dZ, OUT


def main():
    p = set_parameters()
    bbar, rho, epsilon, delta, lam, beta = (
        p["bbar"], p["rho"], p["epsilon"], p["delta"], p["lam"], p["beta"]
    )
    gamma, nu = p["gamma"], p["nu"]

    # Steady states
    high = solvemodel("High", bbar, rho, epsilon, delta, lam, beta, gamma)
    mid = solvemodel("Medium", bbar, rho, epsilon, delta, lam, beta, gamma)
    nH, xH, gAH = float(high["n"]), float(high["x"]), float(high["gA"])
    nM, xM, gAM = float(mid["n"]), float(mid["x"]), float(mid["gA"])
    nEqm_code = bbar - delta - rho / epsilon

    # Stable manifold direction near High SS
    stable_evec = high["stable_evec"].real
    stable_eval = high["stable_eval"].real

    print(f"High SS: nH={100*nH:.4f}%, xH={xH:.4f}, gAH={100*gAH:.4f}%")
    print(f"Mid  SS: nM={100*nM:.4f}%, xM={xM:.4f}, gAM={100*gAM:.4f}%")
    print(f"Low  SS: nL={100*nEqm_code:.4f}% (Empty Planet)")
    print(f"Stable eigenvalue: {stable_eval:.6f}")
    print(f"Stable eigenvector: {stable_evec}")

    def rhs(t, Z):
        return transit_dZ(t, Z, bbar, rho, epsilon, delta, lam, beta, nu, gamma)

    tt = 500
    Zstar = np.array([nH, xH])

    # --- Top region (x < xH): integrate backwards toward lower x ---
    Zclose_top = Zstar - stable_evec * np.exp(stable_eval * tt)
    t0_top = -940  # Matches t0_TopRegion in SetParameters.m
    sol_top = solve_ivp(
        rhs, t_span=(tt, t0_top), y0=Zclose_top,
        dense_output=True, rtol=1e-10, atol=1e-10, max_step=1.0, method="RK45",
    )
    t_top = np.arange(tt, t0_top - 0.5, -1)
    t_top = t_top[(t_top >= min(sol_top.t)) & (t_top <= max(sol_top.t))]
    Z_top = sol_top.sol(t_top)
    n_HH, x_HH = Z_top[0], Z_top[1]

    # --- Middle region (xH to xM): integrate backwards toward higher x ---
    Zclose_mid = Zstar + stable_evec * np.exp(stable_eval * tt)
    sol_mid = solve_ivp(
        rhs, t_span=(tt, -4000), y0=Zclose_mid,
        dense_output=True, rtol=1e-10, atol=1e-10, max_step=1.0, method="RK45",
    )
    t_mid = np.arange(tt, -4000 - 0.5, -1)
    t_mid = t_mid[(t_mid >= min(sol_mid.t)) & (t_mid <= max(sol_mid.t))]
    Z_mid = sol_mid.sol(t_mid)
    n_HM, x_HM = Z_mid[0], Z_mid[1]

    # --- Lower arm of spiral: start from (nEqm, large x) ---
    xclose = 26000.0
    Zclose_low = np.array([nEqm_code, xclose])
    sol_low = solve_ivp(
        rhs, t_span=(1000, -4000), y0=Zclose_low,
        dense_output=True, rtol=1e-10, atol=1e-10, max_step=1.0, method="RK45",
    )
    t_low = np.arange(1000, -4000 - 0.5, -1)
    t_low = t_low[(t_low >= min(sol_low.t)) & (t_low <= max(sol_low.t))]
    Z_low = sol_low.sol(t_low)
    n_ML, x_ML = Z_low[0], Z_low[1]

    # Combine all three segments and sort by x (Figure 6 format)
    n_all = np.concatenate([n_HH, n_HM, n_ML])
    x_all = np.concatenate([x_HH, x_HM, x_ML])
    gA_all = 1 / x_all
    order = np.argsort(x_all)
    x_all, n_all, gA_all = x_all[order], n_all[order], gA_all[order]

    # "Between when population growth falls to zero and the Empty Planet,
    #  A(t) will increase by a factor of X"
    idx_neg = np.where(n_ML < 0)[0]
    if len(idx_neg) > 1:
        t_seg = t_low[idx_neg]
        g_seg = gA_all_at = 1 / x_ML[idx_neg]
        dt = np.abs(np.diff(t_seg))
        g1 = np.sum(g_seg[:-1] * dt)
        g2 = np.sum(g_seg[1:] * dt)
        cumulative_g = 0.5 * (g1 + g2)
        factor_increase = np.exp(cumulative_g)
        print(f"\nBetween n=0 and Empty Planet, A(t) rises by factor ≈ {factor_increase:.3f}")

    # --- Figure 6: Transition Dynamics ---
    fig, ax = plt.subplots(figsize=(9, 6))
    ax.semilogx(x_all, 100 * n_all, "b-", linewidth=2, label="Optimal path")
    ax.axhline(100 * nEqm_code, linestyle="--", color="gray", label="Empty Planet n*_eq")
    ax.axhline(0, linestyle=":", color="black", alpha=0.5)
    ax.plot(xH, 100 * nH, "o", color="green", markersize=12, label="High SS (Expanding Cosmos)")
    ax.plot(xM, 100 * nM, "o", color="orange", markersize=12, label="Middle SS (unstable)")
    ax.set_xlabel("Knowledge per person, x ≡ A^β/N^λ (log scale)")
    ax.set_ylabel("Population growth n, %")
    ax.set_title("Figure 6: Transition Dynamics for the Optimal Allocation")
    ax.set_xlim(20, 10000)
    ax.set_ylim(-0.8, 2.5)
    ax.legend(loc="upper right", fontsize=9)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT / "figure6_optimal_dynamics.png", dpi=150)
    plt.close(fig)

    # Print the Table-2-style row
    print("\n" + "=" * 72)
    print("STEADY-STATE ROW (current code baseline):")
    print("=" * 72)
    print(f"    nH  = {100*nH:6.4f}%")
    print(f"    xH  = {xH:6.4f}")
    print(f"    gAH = {100*gAH:6.4f}%")
    print(f"    nM  = {100*nM:6.4f}%")
    print(f"    xM  = {xM:6.4f}")
    print(f"    gAM = {100*gAM:6.4f}%")
    print(f"\nSaved: {OUT/'figure6_optimal_dynamics.png'}")


if __name__ == "__main__":
    main()
