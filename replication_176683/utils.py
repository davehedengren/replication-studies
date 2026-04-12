"""Shared parameters and helper functions for Jones (2022) replication.

Paper: "The End of Economic Growth? Unintended Consequences of a Declining Population"
Author: Charles I. Jones
Original code: MATLAB (see 176683-V1/)
"""
import numpy as np
from pathlib import Path

ROOT = Path(__file__).parent
OUT = ROOT / "output"
OUT.mkdir(exist_ok=True)


def set_parameters():
    """Replicates SetParameters.m — the current benchmark calibration in the code.

    Code comment: '2022-January-11 Sandbox No Spiral Case that works'
    Produces: nH~1.15%, gAH~0.4%, nL=0.3, gAL=0.1.
    """
    sigma = 1.0
    lam = 3 / 4
    ellEqm = 1 / 8
    delta = 1 / 90
    nEqm = -0.005
    rho = delta
    beta = 2.0

    phi = 1 - beta
    bbar = (nEqm + delta) / ellEqm
    epsilon = rho / (bbar * (1 - ellEqm))
    gamma = lam * sigma / beta
    nu = (epsilon + gamma) / rho

    return dict(
        sigma=sigma, lam=lam, ellEqm=ellEqm, delta=delta, nEqm=nEqm,
        rho=rho, beta=beta, phi=phi, bbar=bbar, epsilon=epsilon,
        gamma=gamma, nu=nu,
    )


def set_parameters_nber():
    """Parameter set matching the NBER WP 26651 (Jan 2020) Table 2.

    These are the values reported in the published PDF we have on disk.
    Commented out in SetParameters.m as 'NBER Working Paper Version 1.0 values'.
    """
    sigma = 1.0
    lam = 1.0
    ellEqm = 1 / 8
    delta = 0.01
    nEqm = -0.005
    rho = delta  # = 0.01 matches published ρ=0.01
    beta = 1.25

    phi = 1 - beta
    bbar = (nEqm + delta) / ellEqm
    epsilon = rho / (bbar * (1 - ellEqm))
    gamma = lam * sigma / beta
    nu = (epsilon + gamma) / rho

    return dict(
        sigma=sigma, lam=lam, ellEqm=ellEqm, delta=delta, nEqm=nEqm,
        rho=rho, beta=beta, phi=phi, bbar=bbar, epsilon=epsilon,
        gamma=gamma, nu=nu,
    )


def solvemodel(which_root, bbar, rho, epsilon, delta, lam, beta, gamma):
    """Python translation of solvemodel.m — solves steady state BGP.

    Returns: dict with nSP, ellSP, xSP, gA, stable eigenvalue/eigenvector,
    and full list of linearized-system eigenvalues.
    """
    sign_root = +1 if which_root == "High" else -1

    nu = (epsilon + gamma) / rho
    btilde = bbar - delta - (epsilon + lam) / (lam * nu)
    nEqm = bbar - delta - rho / epsilon

    if nEqm > 0:
        raise ValueError("nEqm>0, so no possibility for multiple SS.")

    b24ac = 0.25 * btilde**2 + epsilon * nEqm / (lam * nu)

    nSP = btilde / 2 + sign_root * np.sqrt(b24ac + 0j)
    nSP = nSP.real if abs(nSP.imag) < 1e-12 else nSP
    gA = lam * nSP / beta
    xSP = 1 / gA
    ellSP = (nSP + delta) / bbar

    # Stability: eigenvalues of linearized system around SS
    gx = beta * ((bbar - delta - nSP) / xSP) ** 2 * (nu - 1 / (bbar - delta - nSP))
    gn = 2 * (bbar - delta - nSP) * (epsilon + lam * nu * nSP) - rho - lam * nSP
    B = np.array([[gn, gx], [-lam * xSP, -lam * nSP]])
    evals, evecs = np.linalg.eig(B)

    idx_neg = np.argwhere(np.real(evals) < 0).flatten()
    if len(idx_neg) > 0:
        stable_eval = evals[idx_neg[0]]
        stable_evec = evecs[:, idx_neg[0]]
    else:
        stable_eval = None
        stable_evec = None

    return dict(
        n=nSP, ell=ellSP, x=xSP, gA=gA,
        stable_eval=stable_eval, stable_evec=stable_evec,
        all_evals=evals, B=B,
    )


def transit_dZ(t, Z, bbar, rho, epsilon, delta, lam, beta, nu, gamma):
    """ODE system from OptimalDynamics.m (transit_dZ).

    Z = [n, x]. dn and dx from Hamiltonian first-order conditions.
    """
    n, x = Z
    dn = (bbar - delta - n) ** 2 * (gamma - (rho + beta / x) * (nu - 1 / (bbar - delta - n)))
    dx = beta - lam * n * x
    return np.array([dn, dx])


# UN World Population Prospects 2019 TFR data — hard-coded from FertilityData.m
# Columns: WORLD, HighInc, USA, China, India, Japan, Italy, Spain, France, Germany
FERTILITY_DATA = np.array([
    [4.967, 2.986, 3.311, 6.110, 5.903, 2.960, 2.355, 2.530, 2.763, 2.128],
    [4.897, 3.015, 3.582, 5.480, 5.896, 2.170, 2.290, 2.700, 2.703, 2.274],
    [5.018, 2.945, 3.234, 6.150, 5.890, 2.030, 2.504, 2.810, 2.845, 2.474],
    [4.926, 2.652, 2.543, 6.300, 5.723, 2.040, 2.499, 2.840, 2.652, 2.361],
    [4.471, 2.344, 2.029, 4.850, 5.410, 2.134, 2.323, 2.850, 2.308, 1.708],
    [3.861, 2.058, 1.772, 3.010, 4.974, 1.831, 1.886, 2.550, 1.861, 1.508],
    [3.588, 1.938, 1.804, 2.520, 4.682, 1.762, 1.525, 1.880, 1.865, 1.464],
    [3.439, 1.854, 1.915, 2.730, 4.266, 1.650, 1.350, 1.460, 1.804, 1.430],
    [3.005, 1.817, 2.030, 1.830, 3.833, 1.476, 1.272, 1.280, 1.715, 1.301],
    [2.777, 1.733, 1.997, 1.620, 3.483, 1.369, 1.224, 1.187, 1.762, 1.346],
    [2.651, 1.708, 2.042, 1.610, 3.141, 1.298, 1.312, 1.277, 1.882, 1.351],
    [2.584, 1.763, 2.055, 1.620, 2.796, 1.339, 1.436, 1.454, 1.977, 1.362],
    [2.517, 1.723, 1.875, 1.640, 2.396, 1.409, 1.421, 1.325, 1.981, 1.427],
    [2.468, 1.675, 1.776, 1.690, 2.240, 1.370, 1.330, 1.330, 1.852, 1.586],
])
FERTILITY_COLS = ["World", "HighInc", "USA", "China", "India",
                  "Japan", "Italy", "Spain", "France", "Germany"]
FERTILITY_YEARS = np.arange(1950, 2016, 5)
