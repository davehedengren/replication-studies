"""Robustness checks for Caballero-Simsek (openICPSR-177541).

The paper's core numerical claim is qualitative:
  - Output in the low-supply state exceeds potential (y_L > yL*) under the
    optimal Taylor-rule policy with inertia, whereas a myopic policy sets
    y_L = yL*.
  - Weaker inertia (smaller eta) -> y_L closer to yL*.
  - After transition, y_t converges to yH* from below at rate gammaH.
  - With inflation (NKPC or inertia), the "over-stimulation" of the low state
    is muted relative to the no-inflation baseline.

We sweep each structural parameter and verify these qualitative patterns hold.
"""
import numpy as np
import copy
from utils import (build_baseline_params, solve_baseline, simulate_baseline,
                   solve_nkpc, simulate_nkpc, solve_inertia, simulate_inertia,
                   solve_zlb)


def base():
    return build_baseline_params()


p0 = base()
eqBase = solve_baseline(p0)
yL0 = eqBase['yL']
gH0 = eqBase['gammaH']
print("Baseline reference:")
print(f"  gammaH = {gH0:.6f}, yL = {yL0:.6f}, yLBar = {eqBase['yLBar']:.6f}")

results = []


def record(name, gammaH, yL, note=""):
    results.append((name, gammaH, yL, note))
    print(f"  [{name}]  gammaH={gammaH:.6f}  yL={yL:.6f}  {note}")


# -------------------------------------------------------------------------
# Check 1: Sweep inertia eta in [0.1, 0.95]
# -------------------------------------------------------------------------
print("\n[1] eta sweep: yL should rise monotonically with eta")
etas = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95]
prev_yL = -np.inf
mono = True
for eta in etas:
    p = base(); p.eta = eta
    try:
        eq = solve_baseline(p, eta=eta)
    except Exception as e:
        print(f"  eta={eta}: FAIL {e}")
        continue
    if eq['yL'] < prev_yL - 1e-10:
        mono = False
    prev_yL = eq['yL']
    record(f"eta={eta:.2f}", eq['gammaH'], eq['yL'])
print(f"  monotonicity yL rising in eta: {'YES' if mono else 'NO'}")


# -------------------------------------------------------------------------
# Check 2: Sweep lambda (recovery probability) in [0.1, 0.9]
# -------------------------------------------------------------------------
print("\n[2] lambda sweep: larger lambda -> yL closer to yH* (expansionary boost from future recovery)")
for lam in [0.1, 0.25, 0.5, 0.75, 0.9]:
    p = base(); p.lam = lam
    eq = solve_baseline(p)
    record(f"lambda={lam:.2f}", eq['gammaH'], eq['yL'])


# -------------------------------------------------------------------------
# Check 3: Sweep phi (Taylor-rule output weight) in [0.25, 4]
# -------------------------------------------------------------------------
print("\n[3] phi sweep: larger phi -> faster convergence in H (larger gammaH smaller)")
for phi in [0.25, 0.5, 1.0, 2.0, 4.0]:
    p = base(); p.phi = phi
    eq = solve_baseline(p)
    record(f"phi={phi:.2f}", eq['gammaH'], eq['yL'])


# -------------------------------------------------------------------------
# Check 4: Shock size sensitivity: yLStar in [0.9, 0.99]
# -------------------------------------------------------------------------
print("\n[4] yLStar sweep: shallower shock -> smaller deviation of yL from yH*")
for yLS in [0.90, 0.92, 0.94, 0.95, 0.97, 0.99]:
    p = base(); p.yLStar = yLS
    eq = solve_baseline(p)
    record(f"yLStar={yLS:.2f}", eq['gammaH'], eq['yL'],
           f"yL-yLStar={eq['yL'] - yLS:+.4f}")


# -------------------------------------------------------------------------
# Check 5: Ordering baseline < NKPC < first-best for yL
# -------------------------------------------------------------------------
print("\n[5] Model ordering at baseline parameters")
eqB = solve_baseline(p0)
eqN = solve_nkpc(p0)
eqI = solve_inertia(p0)
print(f"  Baseline  yL         = {eqB['yL']:.6f}")
print(f"  NKPC      yL         = {eqN['yL']:.6f}  (pi_L = {eqN['piL']:+.6f})")
print(f"  Inertia   yLBar      = {eqI['yLBar']:.6f} (piLBar = {eqI['piLBar']:+.6f})")
print(f"  First-best yL*       = {p0.yLStar:.6f}")
qual_nkpc = eqB['yL'] > eqN['yL'] > p0.yLStar
qual_inertia = eqI['yLBar'] < eqB['yL']
print(f"  Ordering yL*<yL_nkpc<yL_base? {'YES' if qual_nkpc else 'NO'}")
print(f"  Inertia steady-state below baseline yL? {'YES' if qual_inertia else 'NO'}")


# -------------------------------------------------------------------------
# Check 6: Simulated y at t=0 vs t=3 (the 'plateau'): should be ~ yL
# -------------------------------------------------------------------------
print("\n[6] Trajectory plateau: y(t=0..3) ~ yL in baseline sim")
sim = simulate_baseline(p0, eqBase)
y_plateau = sim['yTimeArr'][1:5]
print(f"  y(t=0..3) = {y_plateau}")
print(f"  |y - yL|  = {np.max(np.abs(y_plateau - eqBase['yL'])):.2e}")


# -------------------------------------------------------------------------
# Check 7: NKPC trajectory sanity (at t>=tTrans, y -> 1 monotone)
# -------------------------------------------------------------------------
print("\n[7] NKPC trajectory: after transition y rises monotonically toward yH*")
simN = simulate_nkpc(p0, solve_nkpc(p0))
post = simN['yTimeArr'][p0.tTrans + 1:]
mono_rise = all(post[i] <= post[i + 1] + 1e-12 for i in range(len(post) - 1))
print(f"  post = {post}, monotonic rising? {mono_rise}")


# -------------------------------------------------------------------------
# Check 8: Grid refinement (halve step size on yArr) stability
# -------------------------------------------------------------------------
print("\n[8] Grid refinement: step 0.001 -> 0.0005, |yL diff| should be tiny")
p_fine = base()
p_fine.yArr = np.arange(0.0, p_fine.yHStar + 1e-12, 0.0005)
eq_fine = solve_baseline(p_fine)
diff = abs(eq_fine['yL'] - eqBase['yL'])
print(f"  |yL_fine - yL_coarse| = {diff:.2e}  (should be 0 since closed form)")


# -------------------------------------------------------------------------
# Check 9: ZLB model maintains assumption 1 over eta grid
# -------------------------------------------------------------------------
print("\n[9] ZLB Assumption 1 across eta")
for eta in [0.1, 0.3, 0.5, 0.7, 0.8, 0.9]:
    p = base(); p.eta = eta
    cond = p.rho - eta / (1 - eta) * (p.yHStar - p.yLStar) < 0
    if cond:
        eq = solve_zlb(p, eta=eta)
        print(f"  eta={eta}: assumption holds, yLBar={eq['yLBar']:.4f}, yL={eq['yL']:.4f}")
    else:
        print(f"  eta={eta}: Assumption 1 FAILS (rho - eta/(1-eta)*(yH-yL) >= 0)")


# -------------------------------------------------------------------------
# Check 10: Inertia model — vary b (past-inflation weight)
# -------------------------------------------------------------------------
print("\n[10] Inertia: vary b in {0.5, 0.7, 0.9}")
for b in [0.5, 0.7, 0.9]:
    p = base(); p.b = b
    try:
        eq = solve_inertia(p)
        print(f"  b={b}: gammaH={eq['gammaH']:.4f}, gammaL={eq['gammaL']:.4f}, "
              f"yLBar={eq['yLBar']:.4f}, piLBar={eq['piLBar']:+.4f}")
    except Exception as e:
        print(f"  b={b}: FAIL {e}")


# -------------------------------------------------------------------------
# Check 11: NKPC — vary kappa
# -------------------------------------------------------------------------
print("\n[11] NKPC: vary kappa")
for kappa in [0.1, 0.3, 0.5, 1.0]:
    p = base(); p.kappa = kappa
    try:
        eq = solve_nkpc(p)
        print(f"  kappa={kappa}: gammaH={eq['gammaH']:.4f}, yL={eq['yL']:.4f}, "
              f"piL={eq['piL']:+.4f}")
    except AssertionError:
        print(f"  kappa={kappa}: polynomial stable-root assertion FAILS")


# -------------------------------------------------------------------------
# Check 12: Transition timing — sweep tTrans in {2..6}
# -------------------------------------------------------------------------
print("\n[12] tTrans sweep: longer low-supply spell -> more periods of plateau")
for tT in [2, 3, 4, 5, 6]:
    p = base(); p.tTrans = tT
    sim = simulate_baseline(p, solve_baseline(p))
    below = int(np.sum(sim['yTimeArr'] < p0.yLStar + 0.03))
    print(f"  tTrans={tT}: y-series peak={sim['yTimeArr'].max():.4f}, "
          f"<0.98 count={below}")

print("\nRobustness complete.")
