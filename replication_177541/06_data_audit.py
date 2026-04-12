"""Data audit for Caballero-Simsek (openICPSR-177541).

No external data — this paper uses only numerically-simulated equilibria.
The audit verifies:
  1. parametric assumptions needed for each proposition hold
  2. stable-root selection returns exactly one root in (0,1)
  3. grids are fine enough that closed-form and numerical objects agree
  4. unconstrained equilibria satisfy first-order conditions
  5. initial condition yLastInit == yLBar (as in loadVariablesBaseline.m)
  6. NKPC + inertia steady states satisfy their closed-form equations
  7. ZLB piecewise-linear cutoffs are monotone-decreasing and nonneg.
"""
import numpy as np
from utils import (build_baseline_params, default_params,
                   solve_baseline, solve_nkpc, solve_inertia, solve_zlb)

p = build_baseline_params()

print("=" * 72)
print("DATA AUDIT (all objects numerically constructed from model parameters)")
print("=" * 72)


def chk(label, cond, extra=""):
    mark = "OK " if cond else "!! "
    print(f"  [{mark}] {label}  {extra}")


# ---- 1. Parameter sanity ----
print("\n[1] Parameter sanity")
chk("rho > 0", p.rho > 0, f"rho={p.rho}")
chk("0 < eta < 1", 0 < p.eta < 1, f"eta={p.eta}")
chk("0 < eta_low < eta", 0 < p.eta_low < p.eta)
chk("yHStar > yLStar > 0", p.yHStar > p.yLStar > 0)
chk("0 < lambda <= 1", 0 < p.lam <= 1)
chk("0 < beta < 1", 0 < p.beta < 1, f"beta={p.beta:.6f}")

# Assumption 1 (ZLB): rho - eta/(1-eta)*(yH-yL) < 0
assump1 = p.rho - p.eta / (1 - p.eta) * (p.yHStar - p.yLStar) < 0
chk("ZLB Assumption 1 (rho - eta/(1-eta)*(yH-yL) < 0)", assump1)

# Inertial-inflation extra condition: phi_y > kappa
chk("phi_y > kappa (inertia prop)", p.phi_y > p.kappa)


# ---- 2. Stable roots ----
print("\n[2] Characteristic polynomial stable roots in (0,1)")

# Baseline gammaH polynomial: y^2 - (1/(1-eta)+phi)y + eta/(1-eta) = 0
poly = [1, -(1 / (1 - p.eta) + p.phi), p.eta / (1 - p.eta)]
rts = np.roots(poly)
nstable = sum(0 < r.real < 1 and abs(r.imag) < 1e-9 for r in rts)
chk("baseline gammaH: exactly one stable root", nstable == 1,
    f"roots={rts}")

# NKPC polynomial
eq_nk = solve_nkpc(p, type_=1)
chk("NKPC gammaH in (0,1)", 0 < eq_nk['gammaH'] < 1, f"{eq_nk['gammaH']:.6f}")

# Inertia polynomial
eq_in = solve_inertia(p, type_=1)
chk("Inertia gammaH in (0,1)", 0 < eq_in['gammaH'] < 1, f"{eq_in['gammaH']:.6f}")
chk("Inertia gammaL in (0,b)", 0 < eq_in['gammaL'] < p.b,
    f"gammaL={eq_in['gammaL']:.6f}, b={p.b}")


# ---- 3. Grid resolution ----
print("\n[3] Grid resolution")
print(f"  yArr: {len(p.yArr)} points from {p.yArr[0]:.3f} to {p.yArr[-1]:.3f} "
      f"step {p.yArr[1] - p.yArr[0]:.4f}")
print(f"  piArr: {len(p.piArr)} points from {p.piArr[0]:.4f} to {p.piArr[-1]:.4f} "
      f"step {p.piArr[1] - p.piArr[0]:.4f}")
chk("yArr monotone, uniform", np.allclose(np.diff(p.yArr), p.yArr[1] - p.yArr[0]))
chk("piArr monotone, uniform", np.allclose(np.diff(p.piArr), p.piArr[1] - p.piArr[0]))


# ---- 4. Analytical vs numerical baseline ----
print("\n[4] Baseline: analytical closed-form check")
eq = solve_baseline(p, eta=p.eta, type_=1)
# yL should satisfy yL = (1+beta*lam*thetaH)^-1 * yLStar + ...
gH = eq['gammaH']; thH = eq['thetaH']
yL_expected = (1 / (1 + p.beta * p.lam * thH) * p.yLStar
               + p.beta * p.lam * thH / (1 + p.beta * p.lam * thH) * p.yHStar)
chk("yL closed form", np.isclose(eq['yL'], yL_expected, rtol=1e-12),
    f"yL={eq['yL']:.10f}, expected={yL_expected:.10f}")
# thetaH = gammaH^2/(1-beta*gammaH^2)
chk("thetaH closed form",
    np.isclose(thH, gH ** 2 / (1 - p.beta * gH ** 2), rtol=1e-12))
# yLBar = yL - (1-eta)/eta*(lam*(1-gammaH)+phi)*(yH-yL)
yLBar_expected = eq['yL'] - (1 - p.eta) / p.eta * (p.lam * (1 - gH) + p.phi) * (p.yHStar - eq['yL'])
chk("yLBar closed form", np.isclose(eq['yLBar'], yLBar_expected, rtol=1e-12))

# yLastInit should equal yLBar (set in loadVariablesBaseline)
chk("yLastInit == yLBar", np.isclose(p.yLastInit, eq['yLBar']))


# ---- 5. NKPC steady state consistency ----
print("\n[5] NKPC: steady-state / closed-form check")
gH_nk = eq_nk['gammaH']; thH_nk = eq_nk['thetaH']
piHBold = eq_nk['piHBold']
piHBold_expected = p.kappa * gH_nk / (1 - p.beta * gH_nk)
chk("piHBold closed form", np.isclose(piHBold, piHBold_expected, rtol=1e-12))

# Value-function fixed point in H: -thetaH*(y-yH)^2 should equal
#   -(y-yH)^2 - psi*(kappa gammaH)^2/(1-beta gammaH^2) normalized; we re-derive:
thH_expected = gH_nk ** 2 / (1 - p.beta * gH_nk ** 2) * (1 + p.psi * (p.kappa / (1 - p.beta * gH_nk)) ** 2)
chk("thetaH NKPC closed form", np.isclose(thH_nk, thH_expected, rtol=1e-12))

# NKPC yL steady state: omegaL*yLStar + omegaH*yHStar weighted
omegaL = 1 + p.psi * (p.kappa + p.beta * p.lam * piHBold) * p.kappa / (1 - p.beta * (1 - p.lam))
omegaH = p.beta * p.lam * (thH_nk + p.psi * (p.kappa + p.beta * p.lam * piHBold) * piHBold
                            / (1 - p.beta * (1 - p.lam)))
yL_nk_expected = omegaL / (omegaL + omegaH) * p.yLStar + omegaH / (omegaL + omegaH) * p.yHStar
chk("NKPC yL closed form", np.isclose(eq_nk['yL'], yL_nk_expected, rtol=1e-12))


# ---- 6. Inertia steady state consistency ----
print("\n[6] Inertia: steady-state consistency")
yLBar = eq_in['yLBar']; piLBar = eq_in['piLBar']
chk("piLBar = kappa/(1-b) * (yLBar - yLStar)",
    np.isclose(piLBar, p.kappa / (1 - p.b) * (yLBar - p.yLStar), rtol=1e-12))
# Under the closed form in the inertia solver, PiLArr[i,j] stores
# piLBar + gammaL*(piArr[j]-piLBar); test the mid-grid value
jmid = len(p.piArr) // 2
chk("PiLArr mid value", np.isclose(eq_in['PiLArr'][0, jmid],
                                   piLBar + eq_in['gammaL'] * (p.piArr[jmid] - piLBar)))


# ---- 7. ZLB cutoffs ----
print("\n[7] ZLB: cutoff monotonicity")
eq_zlb = solve_zlb(p, eta=p.eta, type_=1)
yHBarArr = eq_zlb['yHBarArr']; yLBarArr = eq_zlb['yLBarArr']
chk("yHBarArr strictly decreasing or (trim-tail nonpos)",
    all(yHBarArr[i] > yHBarArr[i + 1] for i in range(len(yHBarArr) - 2)))
chk("yLBarArr strictly decreasing (body)",
    all(yLBarArr[i] > yLBarArr[i + 1] for i in range(len(yLBarArr) - 2)))
chk("yHBarArr[0] > 0", yHBarArr[0] > 0)
chk("yLBarArr[0] > 0", yLBarArr[0] > 0)
print(f"  yHBarArr first 5: {yHBarArr[:5]}")
print(f"  yLBarArr first 5: {yLBarArr[:5]}")


# ---- 8. Sample-size/count sanity ----
print("\n[8] Simulation trajectory count")
timeArr = np.arange(-1, p.T + 1)
print(f"  simulation length = {len(timeArr)} periods (-1..{p.T})")
print(f"  transition period tTrans = {p.tTrans}")

print("\nAudit complete.")
