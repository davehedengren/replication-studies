"""Reproduces halflife.m — closed-form half-life calculations.

Computes how long it takes for:
  - the growth rate gA to fall halfway to zero
  - A(t) to reach (A*+A0)/2

for three calibrations: (beta=3, lambda=1), (beta=2, lambda=3/4), (beta=0 Romer, lambda=1).
"""
import numpy as np


def gethalf_growth(lam, eta, gA0, beta):
    """Half-life for the growth rate of gA. Eq: t* = 1/(xi) log((2+m)/(1+m))."""
    m = beta * gA0 / (lam * eta)
    return 1.0 / (lam * eta) * np.log((2 + m) / (1 + m))


def halflevel(lam, eta, gA0, beta):
    """Half-life for A(t) to reach (A0+A*)/2."""
    if beta > 0:
        m = beta * gA0 / (lam * eta)
        AstarA0 = (1 + m) ** (1 / beta)
        tstar = -1 / (lam * eta) * np.log(1 + 1 / m * (1 - (0.5 * (1 + AstarA0)) ** beta))
    else:  # Romer case (beta = 0)
        m = gA0 / (lam * eta)
        AstarA0 = np.exp(m)
        tstar = -1 / (lam * eta) * np.log(1 - 1 / m * np.log(0.5 * (1 + AstarA0)))
    return tstar


def main():
    eta = 0.005

    print("=" * 72)
    print("Half-life replication (halflife.m)")
    print("  eta = 0.005 (population shrinks at 0.5%/year)")
    print("=" * 72)

    results = []
    for label, beta, lam in [
        ("beta=3, lambda=1", 3, 1),
        ("beta=2, lambda=3/4", 2, 3 / 4),
        ("beta=0 (Romer), lambda=1", 0, 1),
    ]:
        print(f"\n--- {label} ---")
        for gA0 in [0.01, 0.005]:
            if beta > 0:
                tg = gethalf_growth(lam, eta, gA0, beta)
                print(f"  gA0={gA0:5.3f}  growth halflife = {tg:7.2f} years")
                results.append((label, "growth", gA0, tg))
            tl = halflevel(lam, eta, gA0, beta)
            print(f"  gA0={gA0:5.3f}  level  halflife = {tl:7.2f} years")
            results.append((label, "level", gA0, tl))

    # Paper text p.6: "if gy0=2% and eta=1%, long-run GDP per person = e^2 ≈ 7.4 times current"
    # This uses the Romer formula A* = A0 * exp(gA0/eta).
    factor = np.exp(0.02 / 0.01)
    print("\n" + "=" * 72)
    print("Paper p.6 calibration: gy0=2%, eta=1% (Romer fully-endogenous case)")
    print(f"  Long-run GDP/person factor = exp(0.02/0.01) = {factor:.4f}")
    print(f"  Paper reports: e^2 ≈ 7.4                       Match: {abs(factor-7.4)<0.05}")

    # Paper p.8: semi-endogenous calibration, gy0=2%, eta=1%, beta=3, gamma=1/3
    # A*/A0 = (1 + beta*gA0/(lam*eta))^(1/beta). With sigma=1, lam=1, beta=3:
    # y*/y0 = (1 + 3*2)^(1/3) = 7^(1/3) ≈ 1.91. Paper says (1+3·2)^(1/3) ≈ 1.9.
    ratio_semi = (1 + 3 * 2) ** (1 / 3)
    print(f"\nPaper p.8 semi-endogenous (beta=3, lam=1, gamma=1/3):")
    print(f"  y*/y0 = (1+3*2)^(1/3) = {ratio_semi:.4f}")
    print(f"  Paper reports: ≈ 1.9                             Match: {abs(ratio_semi-1.9)<0.05}")


if __name__ == "__main__":
    main()
