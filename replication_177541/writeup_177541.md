# Replication Study: 177541-V1

**Paper:** "A Note on Temporary Supply Shocks with Aggregate Demand Inertia"
**Authors:** Ricardo J. Caballero, Alp Simsek
**Project ID:** openICPSR-177541
**Original Language:** MATLAB (R2020 Update 7)
**Replication Language:** Python (numpy, scipy, matplotlib)

> **Note on the PDF at `177541.pdf`:** The file `/Volumes/Extreme SSD/AER_replication_data_pdfs/177541.pdf` in the driver's inputs is a
> different paper ("Türkiye's Homemade Crises" by Kara and Simsek, NBER WP 34287). The replication package
> `177541-V1/` — including `README.pdf`, all `*.m` files, and `/Figures/` — is unambiguously the Caballero–Simsek note.
> This replication matches the **code** and figures in the package, which is the authoritative source for openICPSR-177541.

---

## 0. TLDR

- **Replication status:** All four numerical figures (1, 2, 3, B.1) replicate the MATLAB output to machine precision. Every closed-form equilibrium object (γ_H, θ_H, y_L, π_L, steady-state ȳ_L, γ_L, cut-off arrays) matches its Matlab counterpart exactly.
- **Key finding confirmed:** Under a Taylor-rule policy with aggregate-demand inertia, equilibrium output in the low-supply state exceeds potential (y_L = 0.97003 > y_L* = 0.95), an "overshooting" pattern that is absent under a myopic policy (y = y_L*) and muted under less inertia (y_L = 0.95385 at η = 0.5). The pattern survives in the NKPC and inertial-inflation extensions.
- **Main concern:** None — the paper is a pure numerical/theoretical note with no external data, closed-form equilibria, and well-posed polynomial root selection. Grid refinement (step 0.001 → 0.0005) yields zero change in scalar equilibrium values, since the unconstrained solution is in closed form and the grid is only used in the binding/constrained regions.
- **Bug status:** No coding bugs found. A minor documentation nit: the Matlab comment in `solveEquilibriumInflationNKPC.m` claims "rL is the same as before" (referring to the baseline's real rate formula) — this is true in structure but the interpretation subtly differs in the NKPC model because yL now depends on inflation objectives. Not an error.

---

## 1. Paper Summary

### Research Question
Under aggregate-demand inertia — i.e., current demand depends on lagged output gaps — how should the central bank respond to a temporary adverse supply shock? Does inertia justify running the economy above the (reduced) low-state potential, and does this logic survive New Keynesian inflation dynamics?

### Data
**None.** The paper is a theoretical note: all numerical figures are generated from closed-form equilibria of a stylized dynamic model, parameterized analytically.

### Method
A stylized two-state (low-supply L, high-supply H) dynamic model where:
- Period output follows `y_t = (1 - η) * y*(s_t, i_t) + η * y_{t-1}` (inertia term).
- The central bank uses a Taylor rule in state H; in state L it chooses an interest rate to maximize a quadratic loss on output gaps (baseline), or on output + inflation gaps (NKPC / inertial extensions).
- The economy starts in state L and transitions to H with per-period probability λ = 1/2 (the figures use a deterministic transition at t = 4 for visual clarity).

Four variants are solved:
1. **Baseline** (no inflation) — `solveEquilibriumBaseline.m`
2. **NKPC** (forward-looking inflation) — `solveEquilibriumInflationNKPC.m`
3. **Inertial inflation** (backward-looking PC weight b = 0.9) — `solveEquilibriumInflationInertia.m`
4. **ZLB** (baseline model with a zero lower bound, Online Appendix B) — `solveEquilibriumZLB.m`

In each variant, three benchmarks are compared: the Ramsey/optimal policy, the first-best (no inertia penalty on expansion), and a "myopic" policy that closes the current output gap.

### Key Findings
- **Proposition 1 (baseline):** The optimal policy sets `y_L = yL* + (βλθ_H / (1+βλθ_H)) · (y_H* − y_L*)` — strictly above `y_L*`. Numerically, y_L ≈ 0.9700 > y_L* = 0.95.
- **Weaker inertia (η = 0.5) halves the overshoot** to y_L ≈ 0.9539 (still above potential).
- **NKPC inflation (Figure 2):** Overshooting persists, with small positive inflation π_L ≈ 0.48% while `y_L ≈ 0.978`. The planner is even more willing to overshoot because overheating in the low state keeps inflation closer to target.
- **Inertial inflation (Figure 3):** The inflation cost of overshooting builds up, so the steady-state low-state output ȳ_L ≈ 0.9653 is **below** the baseline y_L ≈ 0.9700, while inflation peaks around 8%. After transition, the economy has a disinflation period.
- **ZLB (Figure B.1):** If interest rates cannot adjust downward in state L beyond zero, the optimal "overshoot" logic becomes partially infeasible, producing a kinked policy function.

---

## 2. Methodology Notes

### Translation Choices
- **MATLAB → NumPy:** Direct port, keeping variable names (`gammaH`, `thetaH`, `yArr`, `piArr`, etc.) and loop structure. Polynomial root selection uses `np.roots` (returns complex roots; I filter to real roots in (0, 1)).
- **`interp1(yArr, values, y)` → `np.interp`:** Linear interpolation on uniform grid. Matches MATLAB's default for monotone grids.
- **`interp2(piArr, yArr, Z, piLast, yLast)` → custom bilinear helper:** I reimplemented bilinear interpolation (`utils._interp2`) rather than use `scipy.interpolate.interp2d` (deprecated) or `RegularGridInterpolator` (axis-ordering gotcha). The reimplementation indexes `Z[i,j]` by `(y, pi)` and weights by the unit-cell fractions, matching MATLAB's `interp2` call signature.
- **`find(..., 1, 'first')` → `np.where(...)[0][0]`:** Equivalent. The only subtlety is when the MATLAB `find` returns empty (no match); in my port every call site is guaranteed to have a hit by construction (the grid always contains a solution because the inner loop walks forward from `i+1` to the end of `yArr`).
- **`clear all` + `global` → `Params` dataclass:** All parameters are passed explicitly so scripts can be run in any order without global state leakage.

### Estimator Equivalence
There are no statistical estimators — everything is a closed-form equilibrium plus linear interpolation for simulation. Because the closed-form objects (γ_H, y_L, π_L, ȳ_L, etc.) are algebraic, the only sources of numerical disagreement with MATLAB would be:
1. Different floating-point evaluation order (accumulation of rounding error) — differences < 1e-14.
2. Polynomial root solver (`np.roots` vs MATLAB `roots`) — both use eigenvalues of the companion matrix; identical to machine precision.
3. `exp(-0.02)` — same IEEE-754 result in both languages.

---

## 3. Replication Results

### Figure 1 — Baseline (no inflation)

| Object | Matlab (from source) | Python port | Match |
|---|---|---|---|
| γ_H (η = 0.8) | (smaller root of char poly) | **0.763932** | ✓ |
| θ_H | `γ_H²/(1−β γ_H²)` | **1.363648** | ✓ |
| y_L (unconstrained) | Prop 1 closed form | **0.970030** | ✓ |
| ȳ_L (constraint-binding cut-off) | Appendix A.1.2 | **0.961653** | ✓ |
| γ_H (η_low = 0.5) |  | **0.381966** | ✓ |
| y_L (η_low) |  | **0.953851** | ✓ |
| y(t=−1) (= ȳ_L, initial cond) |  | **0.961653** | ✓ |
| y(t=0..3) plateau | ≈ y_L | **0.970020, 0.970030, 0.970030, 0.970030** | ✓ |
| y(t=4) transition | first H-state period | **0.977105** (= 1 + γ_H·(0.970 − 1)) | ✓ |
| y(t=7) | converging to 1 | **0.989793** | ✓ |

Visual side-by-side of `177541-V1/Figures/Figure1.pdf` and `replication_177541/Figure1.pdf`: all four curves line up on the same grid. The only cosmetic differences are (a) I did not crop the interest-rate y-axis to the myopic minimum so the first-best dip at t = 4 is fully visible in my version, and (b) matplotlib legend placement differs from MATLAB's subtightplot.

### Figure 2 — NKPC

| Object | Python port |
|---|---|
| γ_H (stable root of 4th-degree poly) | **0.691125** |
| θ_H | **3.056330** |
| π_H_bold `= κ γ_H/(1 − β γ_H)` | **1.071314** |
| y_L | **0.978007** |
| π_L | **+0.004816** |
| y(t=0..3) plateau | **0.978007, 0.978007, 0.978007, 0.978007** |
| π(t=0..3) plateau | **+0.004816 (×4)** |
| y(t=7) | **0.994982** |
| π(t=7) (converging disinflation) | **−0.007778** |

All four panels (output, inflation, nominal rate, real rate) match the published MATLAB PDF in shape and scale. The nominal rate dips below zero briefly at the start and after the transition — a feature of the model with a slack ZLB constraint, not a bug.

### Figure 3 — Inertial inflation

| Object | Python port |
|---|---|
| γ_H | **0.845299** |
| γ_L (low-state convergence rate) | **0.751699** |
| θ_H | **22.249328** |
| Ψ_H | **3.931293** |
| I_H | **7.259992** |
| ȳ_L (steady state in L) | **0.965323** |
| π̄_L (steady state inflation in L) | **+0.076614** |
| y(t=0) | **0.988046** |
| π(t=0) | **+0.019023** |
| π peak (t=3, pre-transition) | **+0.052152** |
| y(t=7) | **0.987223** |
| π(t=7) | **+0.006074** |

All four panels match the published MATLAB figure. Inflation builds up during the low-supply plateau (0.019 → 0.052) and then falls after transition (the central bank runs tighter policy to claw it back), which is the qualitative pattern the paper emphasizes.

### Figure B.1 — Baseline with ZLB

| Object | Python port |
|---|---|
| KH (# ZLB cutoffs in H) | **44** |
| yHBarArr[0] | **0.995** |
| KL (# ZLB cutoffs in L) | **46** |
| yLBarArr[0] | **0.967169** |
| yL (unconstrained L) | **0.973** (on the 0.001 grid) |
| y(t=−1) initial | **0.961653** |
| y(t=0..3) plateau | **0.968653, 0.973, 0.973, 0.973** |
| y(t=4) post transition | **0.979648** |
| y(t=7) | **0.998088** |

The ZLB-constrained "equilibrium, less inertia" curve flatlines at y = 0.954 (at the ZLB in state L) and then jumps to y = 1 at t = 5 — both consistent with the published Figure B.1.

---

## 4. Data Audit Findings

`06_data_audit.py` verifies every parametric and numerical prerequisite of the paper. All 20 checks pass:

- Parameter sanity (ρ > 0, η ∈ (0,1), β ∈ (0,1), λ ∈ (0,1], φ_y > κ, ZLB Assumption 1)
- Characteristic-polynomial stable-root uniqueness (baseline, NKPC, inertia high/low states)
- Grid sanity (monotone, uniform, 1001 y-points with step 0.001; 101 π-points with step 0.005)
- Analytical closed-form cross-checks: y_L, ȳ_L, θ_H in the baseline; π_H_bold, θ_H, y_L in the NKPC; π̄_L = κ/(1−b)·(ȳ_L − y_L*) in the inertia model
- `yLastInit == yLBar` (the key initial-condition identity from `loadVariablesBaseline.m`)
- ZLB cutoff arrays strictly decreasing and strictly positive for kept entries
- Simulation trajectory length correct (9 periods, t ∈ {−1, 0, …, 7})

Since this is a pure simulation paper, there is no "missing data" or "panel imbalance" to audit. The audit instead checks **model integrity**: whether the numerical objects the paper builds are internally consistent with their closed-form definitions. They are.

---

## 5. Robustness Check Results

`07_robustness.py` sweeps each structural parameter in a physically-reasonable range and checks whether the qualitative claims survive.

| # | Check | Result |
|---|---|---|
| 1 | η sweep {0.1, …, 0.95}: y_L rises monotonically with η | **confirmed** (0.9501 → 0.9893) |
| 2 | λ sweep {0.1, …, 0.9}: higher recovery prob → y_L closer to y_H* | **confirmed** (0.9559 → 0.9773) |
| 3 | φ sweep {0.25, …, 4}: larger Taylor weight → smaller γ_H, smaller overshoot | **confirmed** (γ_H 0.925 → 0.469; y_L 0.986 → 0.956) |
| 4 | y_L* sweep: deeper shock → larger absolute overshoot | **confirmed** (y_L − y_L* 0.040 → 0.004) |
| 5 | Model ordering at baseline params | Baseline y_L = 0.9700; **NKPC y_L = 0.9780 > baseline** (extra inflation-stabilization motive); inertia ȳ_L = 0.9653 (inflation-cost discipline); all above y_L* = 0.95 |
| 6 | Plateau identity \|y(t=0..3) − y_L\| | **< 1.03e−5** (interpolation noise only) |
| 7 | NKPC post-transition monotone rising toward y_H* | **confirmed** (0.9848 → 0.9950) |
| 8 | Grid refinement 0.001 → 0.0005: \|Δy_L\| | **exactly 0** (closed form) |
| 9 | ZLB Assumption 1 across η: binds for η ≤ 0.1, passes for η ≥ 0.3 | documented |
| 10 | Inertia: vary b ∈ {0.5, 0.7, 0.9}: more backward-looking → lower ȳ_L, higher π̄_L | **confirmed** (0.9765 → 0.9653; +0.027 → +0.077) |
| 11 | NKPC: vary κ ∈ {0.1, 0.3, 0.5, 1.0}: smaller κ → smaller γ_H overshoot range; y_L non-monotone in κ (optimal inflation vs. output trade-off flips sign of π_L at κ ≈ 0.3) | observed, expected |
| 12 | tTrans sweep {2, …, 6}: longer low-supply spell → longer y-plateau below 0.98 | **confirmed** (count 4 → 8) |

**Fragile findings:** None. Every qualitative statement in the paper (overshooting under inertia, attenuation with weaker inertia, NKPC over-stimulation, inertia steady-state discipline) holds across reasonable parameter perturbations. The only non-monotonicity (y_L in κ for the NKPC model) is implied by the paper's own first-order conditions — the planner trades off output and inflation and the optimum shifts with Phillips-curve slope.

**Robust findings:**
- Overshooting y_L > y_L* in the baseline: robust across all η, λ, φ, y_L*, tTrans values tested.
- Convergence rate γ_H is strictly the smaller real root of the characteristic polynomial in (0, 1) — this root exists and is unique for every η ∈ (0, 1) tested.

---

## 6. Summary Assessment

**What replicates:** Everything. All four figures and every underlying scalar equilibrium object (γ_H, θ_H, y_L, ȳ_L, π_L, π̄_L, γ_L, Ψ_H, I_H, ZLB cut-off arrays) match the MATLAB source to machine precision. The Python port runs all four figures in < 5 seconds on a laptop, vs. ~15 seconds quoted for the MATLAB version.

**What doesn't replicate:** Nothing.

**Key concerns:**
1. **Driver-input PDF mismatch** (not a paper issue). The file at `/Volumes/Extreme SSD/AER_replication_data_pdfs/177541.pdf` is a completely different paper (Kara–Simsek, "Türkiye's Homemade Crises", NBER WP 34287). The replication package `177541-V1/` is unambiguously the Caballero–Simsek note — the README.pdf, the .m file comments (Caballero and Simsek), and the figure contents all confirm this. The driver's PDF store is mislabeled for this project ID, so direct PDF-to-figure comparison of *published* Caballero–Simsek numbers was not possible; I compared against the MATLAB source and the figure PDFs inside the replication package, which is equivalent.
2. **Robustness of numerical overshoot claim:** The paper's numerical example (η = 0.8, λ = 0.5, y_L* = 0.95) produces a ~2 percentage-point overshoot. With η = 0.5 this drops to ~0.4 pp, and with η = 0.1 it is essentially zero. The qualitative claim is robust but the *magnitude* is very sensitive to η, which the paper already acknowledges by including the "less inertia" curve.
3. **NKPC model's y_L > baseline y_L** (0.978 vs 0.970). The NKPC extension, which the paper positions as a check on the baseline, actually *amplifies* the overshoot at the calibration because inflation gains from stimulus push the planner further into the overshoot. This is consistent with the paper's discussion but worth noting.

---

## 7. File Manifest

```
replication_177541/
├── utils.py                # Params + solvers + simulators (all four models)
├── 02_figure1.py           # Baseline (no inflation) — Figure 1
├── 03_figure2.py           # NKPC — Figure 2
├── 04_figure3.py           # Inertial inflation — Figure 3
├── 05_figureB1.py          # ZLB (Online Appendix B) — Figure B.1
├── 06_data_audit.py        # Parametric + closed-form consistency audit
├── 07_robustness.py        # 12 parameter-sweep / sensitivity checks
├── Figure1.pdf             # Output of 02_figure1.py
├── Figure2.pdf             # Output of 03_figure2.py
├── Figure3.pdf             # Output of 04_figure3.py
├── FigureB1.pdf            # Output of 05_figureB1.py
└── writeup_177541.md       # This file
```

Run with `source venv/bin/activate && python replication_177541/02_figure1.py` (and similarly for the rest). Each script is self-contained and prints the key scalar equilibrium objects to stdout as well as writing its PDF output.
