"""Reproduces steady-state values from solvemodel.m (Table 2 in paper).

Solves the three candidate steady states (Empty Planet = Low, Middle, High/Expanding Cosmos)
for the optimal allocation in the endogenous-fertility endogenous-growth model.
"""
import numpy as np
from utils import set_parameters, set_parameters_nber, solvemodel


def report(label, p):
    print("\n" + "=" * 72)
    print(f"Calibration: {label}")
    print("=" * 72)
    print(
        f"  σ={p['sigma']}, λ={p['lam']:.4f}, β={p['beta']}, ρ={p['rho']:.6f}, "
        f"δ={p['delta']:.6f}, n_eq={p['nEqm']:.4f}, ℓ_eq={p['ellEqm']:.4f}"
    )
    print(
        f"  Implied: b̄={p['bbar']:.4f}, ε={p['epsilon']:.4f}, γ={p['gamma']:.4f}, ν={p['nu']:.4f}"
    )

    # Equilibrium (Empty Planet / low) values
    nEqm_code = p["bbar"] - p["delta"] - p["rho"] / p["epsilon"]
    print(f"\n  Equilibrium (Empty Planet / low SS):")
    print(f"    n*_eq = {100*nEqm_code:7.4f}%   (should = 100*nEqm = {100*p['nEqm']:.4f}%)")
    print(f"    ℓ*_eq = {p['ellEqm']:.4f}")

    # High SS
    high = solvemodel("High", p["bbar"], p["rho"], p["epsilon"], p["delta"],
                      p["lam"], p["beta"], p["gamma"])
    print(f"\n  High SS (Expanding Cosmos):")
    print(f"    n*_H   = {100*high['n']:7.4f}%")
    print(f"    ℓ*_H   = {high['ell']:7.4f}")
    print(f"    x*_H   = {high['x']:7.4f}")
    print(f"    g_A^H  = {100*high['gA']:7.4f}%   (=g_y^H with σ=1)")
    print(f"    eigenvalues: {high['all_evals']}")

    # Middle SS
    mid = solvemodel("Medium", p["bbar"], p["rho"], p["epsilon"], p["delta"],
                     p["lam"], p["beta"], p["gamma"])
    print(f"\n  Middle SS:")
    print(f"    n*_M   = {100*mid['n']:7.4f}%")
    print(f"    ℓ*_M   = {mid['ell']:7.4f}")
    print(f"    x*_M   = {mid['x']:7.4f}")
    print(f"    g_A^M  = {100*mid['gA']:7.4f}%")
    print(f"    eigenvalues: {mid['all_evals']}")

    # Idea share of social value: λz*/(ε + λz*) with z* = σgA/(ρ+β*gA)
    gA_H = high["gA"]
    z_star = p["sigma"] * gA_H / (p["rho"] + p["beta"] * gA_H)
    idea_share = p["lam"] * z_star / (p["epsilon"] + p["lam"] * z_star)
    print(f"\n  Social value of people (at High SS):")
    print(f"    z* = {z_star:.4f}")
    print(f"    Idea share λz*/(ε+λz*) = {100*idea_share:.2f}%")

    return {"high": high, "mid": mid, "idea_share": idea_share}


def main():
    print("\n\n##################################################################")
    print("STEADY STATE REPLICATION — solvemodel.m")
    print("##################################################################")

    # The benchmark currently in the replication code:
    p_code = set_parameters()
    res_code = report("SetParameters.m CURRENT BENCHMARK (λ=3/4, β=2)", p_code)

    # The NBER-version calibration (matches the PDF we have):
    p_nber = set_parameters_nber()
    res_nber = report("NBER WP 26651 Table 2 (λ=1, β=1.25)", p_nber)

    print("\n\n" + "=" * 72)
    print("COMPARISON TO PUBLISHED TABLE 2 (NBER WP 26651 values)")
    print("=" * 72)
    published = {
        "b̄": 0.04, "ε": 0.286, "n*_sp": 1.74, "ℓ*_sp": 0.68,
        "g_y^sp = g_A^sp": 1.39, "idea_share": 64.0,
    }
    computed = {
        "b̄": p_nber["bbar"],
        "ε": p_nber["epsilon"],
        "n*_sp": 100 * res_nber["high"]["n"].real if np.iscomplexobj(res_nber["high"]["n"]) else 100 * res_nber["high"]["n"],
        "ℓ*_sp": res_nber["high"]["ell"].real if np.iscomplexobj(res_nber["high"]["ell"]) else res_nber["high"]["ell"],
        "g_y^sp = g_A^sp": 100 * (res_nber["high"]["gA"].real if np.iscomplexobj(res_nber["high"]["gA"]) else res_nber["high"]["gA"]),
        "idea_share": 100 * res_nber["idea_share"],
    }
    print(f"{'Quantity':20s} {'Published':>12s} {'Computed':>12s} {'Diff':>12s}")
    for k, pub in published.items():
        c = computed[k]
        diff = c - pub
        print(f"{k:20s} {pub:12.4f} {c:12.4f} {diff:12.4f}")

    print("\n\n" + "=" * 72)
    print("NOTE: The replication code's CURRENT baseline (λ=3/4, β=2) differs from")
    print("the NBER Table 2 baseline (λ=1, β=1.25). The current code is labeled")
    print("'2022-January-11 Sandbox' and reports expected nH~1.15%, gAH~0.4%.")
    print("=" * 72)


if __name__ == "__main__":
    main()
