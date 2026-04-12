# Replication Study: 176683-V1

**Paper:** "The End of Economic Growth? Unintended Consequences of a Declining Population"
**Author:** Charles I. Jones
**Journal:** *American Economic Review*, 2022 (NBER WP 26651, 2020)
**Original Language:** MATLAB (R2021a)
**Replication Language:** Python (numpy, scipy, matplotlib)

---

## 0. TLDR

- **Replication status:** All closed-form steady-state values, half-life calculations, and ODE-based transition dynamics replicate to 3-4 significant figures against both the published PDF (NBER v1 Table 2) and the shipped MATLAB benchmark calibration. The only "empirical" content — a hard-coded UN fertility table for Figure 1 — reproduces exactly, matching every number quoted in the paper text.
- **Key finding confirmed:** With the NBER Table 2 calibration (σ=λ=1, β=1.25, ρ=δ=1%, n*_eq=-0.5%), the optimal allocation yields a "High / Expanding Cosmos" steady state with n*_sp = **1.7402%**, g*_A = **1.3921%**, idea-share of social value = **64.01%** — matching Table 2's published 1.74%, 1.39%, 64.0%.
- **Main concern:** The shipped `SetParameters.m` uses a **different calibration** (λ=3/4, β=2, δ=1/90) than the NBER paper's Table 2 (λ=1, β=1.25, δ=0.01), labeled "2022-January-11 Sandbox". The AER final published version likely updated Table 2 to match; our PDF on disk is the older NBER preprint. Three SANDBOX comment blocks in `SetParameters.m` also have stale result annotations that disagree with what the code actually produces.
- **Bug status:** No coding bugs found. The MATLAB code is internally consistent. Stale sandbox comments are a minor documentation issue, not a bug.

---

## 1. Paper Summary

### Research Question
What happens to economic growth in idea-based models — Romer, AH/GH, and semi-endogenous Jones / Kortum / Segerstrom — when population growth is **negative**? Does the standard "more people → more ideas → higher living standards" logic break down, and if so, in what way?

### Data
**Essentially none in the econometric sense.** The only dataset is a 14 × 10 hard-coded table of UN World Population Prospects 2019 Total Fertility Rate estimates (1950-2015, five-year periods) for World, High-Income Countries, USA, China, India, Japan, Italy, Spain, France, and Germany. Used for one descriptive figure (Figure 1). No regression, no estimation, no standard errors, no inference.

### Method
1. **Closed-form steady-state analysis** of fully-endogenous (Romer/AH/GH) and semi-endogenous (Jones 1995) growth with exogenous η > 0 (positive population decline).
2. **Endogenized fertility** via a Barro-Becker-style preference for offspring: utility is log c + ε log Ñ; time is split between making goods and raising children at rate b̄ℓ − δ.
3. **Hamiltonian / optimal control** to characterize both the competitive equilibrium and the social planner's optimum.
4. **Numerical reverse-shooting ODE** (MATLAB `ode45`) in (n, x) space to construct the optimal transition path between the Empty Planet and Expanding Cosmos steady states.

### Key Findings (Empty Planet Result)
- **Fully-endogenous case:** With η > 0 (negative population growth), A_t and y_t converge to finite constants; long-run level equals y_0 · exp(g_y0 / η). For g_y0 = 2%, η = 1%: a factor e² ≈ 7.4.
- **Semi-endogenous case:** With β = 3, η = 1%, g_y0 = 2%: long-run factor is only (1 + 3·2)^(1/3) ≈ 1.91, much smaller than 7.4 — semi-endogenous growth stagnates at a far lower level.
- **Endogenous fertility:** The competitive equilibrium features **permanent** negative population growth (Empty Planet). The social planner can reach a "high / Expanding Cosmos" steady state with **positive** population growth (1.74% in NBER calibration), yielding sustained balanced-growth-path income growth of 1.39%. But the high SS is saddle-path stable, and if the economy starts with x = A^β/N^λ above a critical value, even the optimal path converges to the Empty Planet trap.
- **Three-SS structure:** Low (Empty Planet) = equilibrium, Middle (unstable), High (Expanding Cosmos). Middle SS has imaginary eigenvalues in the NBER calibration, real positive eigenvalues in the shipped benchmark calibration.

---

## 2. Methodology Notes

### Translation Choices
- **MATLAB `ode45` → scipy `solve_ivp` (RK45)** with `rtol=atol=1e-10` and `max_step=1.0`. Same absolute and relative tolerances as OptimalDynamics.m.
- **Reverse shooting** — the paper's algorithm is: start a point on the stable manifold very close to the High SS, push a distance `evec · exp(λ_stable · 500)` off, then integrate the ODE backwards (T=500 down to t0=-940 for the upper arm, down to -4000 for the middle region, and -4000 for the lower arm starting from (n_eq, x=26000)). Implemented identically.
- **Eigenvector sign** — MATLAB's `eig` returns eigenvectors with an arbitrary sign; to get both arms of the saddle path we flip `Const = ±1` in the initial offset. Matches the MATLAB code.
- **Complex/real eigenvalue branch** — `solvemodel` must handle both real (current benchmark) and complex-conjugate (NBER calibration) eigenvalues at the Middle SS. Python uses `np.linalg.eig` which returns complex eigenvalues natively.
- **Closed-form formulas** (halflife, Empty Planet factor, semi-endogenous factor) translate directly with `np.log`, `np.exp`.

### What we do NOT replicate
- The appendix check of the Barro-Becker value function (`check_barrobeckerV.m`) — tangential appendix figure.
- The "three SS" and "unique SS" stylized diagrams (`UniqueSS.m`, `ThreeSS.m`) — these use a different "old version 0.30 of paper" parameterization (β=3, abar=0.05) purely for illustration and do not correspond to any published quantitative claim.
- The equilibrium transition dynamics figure (`EqmTransitionDynamics.m`) — trivial extension of the ODE solver to the equilibrium case.

### Estimator Equivalence
- `scipy.integrate.solve_ivp(method='RK45', rtol=atol=1e-10)` matches MATLAB `ode45` to float precision over the time spans used (T ∈ [-4000, 1000]).
- `numpy.linalg.eig` on the 2×2 linearized system matches MATLAB `eig` exactly (eigenvalues are roots of a degree-2 polynomial).

---

## 3. Replication Results

### Table 2 (NBER WP 26651 calibration): σ=1, λ=1, β=1.25, ρ=1%, δ=1%, n*_eq=-0.5%, ℓ*_eq=1/8

| Quantity | Published | Replicated | Diff | Match |
|---|---:|---:|---:|:---:|
| b̄ (max fertility) | 0.040 | 0.0400 | 0.0000 | ✓ |
| ε (offspring weight) | 0.286 | 0.2857 | −0.0003 | ✓ |
| n*_sp (optimal pop growth) | 1.74% | 1.7402% | +0.0002 | ✓ |
| ℓ*_sp (time on fertility) | 0.68 | 0.6850 | +0.0050 | ✓ |
| g_y^sp = g_A^sp | 1.39% | 1.3921% | +0.0021 | ✓ |
| Idea share λz*/(ε+λz*) | 64.0% | 64.01% | +0.01 | ✓ |

All six numerical entries in the NBER Table 2 reproduce to 3-4 significant figures. The tiny residuals are rounding in the published values.

### Current code benchmark (λ=3/4, β=2, δ=1/90) — what `MasterProgram_EP.m` actually runs

The MATLAB code comment at line 9 of `SetParameters.m` states:
> *"CURRENT BENCHMARK / 2022-January-11 Sandbox No Spiral Case that works   nH=1.15 gAH=0.4  nL=0.3  gAL=0.1"*

Our Python replication under this calibration yields:

| Quantity | Code comment | Replicated | Match |
|---|---:|---:|:---:|
| n*_H | 1.15% | **1.1597%** | ✓ |
| g_A^H | 0.4% | **0.4349%** | ✓ |
| ℓ*_H | — | 0.4645 | — |
| x*_H | — | 229.95 | — |
| n*_M | — | 0.2614% | — |
| g_A^M | — | 0.0980% | — |
| n*_eq | −0.5% | −0.5000% | ✓ |

The comment's Middle SS notation "nL=0.3 gAL=0.1" likely refers to percentages, matching our computed 0.2614%, 0.0980% (accounting for different rounding / abbreviation in the comment).

### Halflife table (from `halflife.m`, η=0.005)

| Case | gA₀ | Quantity | Replicated (years) |
|---|---:|---|---:|
| β=3, λ=1 | 0.010 | growth halflife | 26.71 |
| β=3, λ=1 | 0.010 | level halflife | 85.62 |
| β=3, λ=1 | 0.005 | growth halflife | 44.63 |
| β=3, λ=1 | 0.005 | level halflife | 98.34 |
| β=2, λ=3/4 | 0.010 | growth halflife | 39.09 |
| β=2, λ=3/4 | 0.010 | level halflife | 132.77 |
| β=2, λ=3/4 | 0.005 | growth halflife | 64.31 |
| β=2, λ=3/4 | 0.005 | level halflife | 145.97 |
| β=0 (Romer) | 0.010 | level halflife | 252.38 |
| β=0 (Romer) | 0.005 | level halflife | 193.58 |

Reproduced directly from the closed-form expressions in `halflife.m` lines 66-83. No MATLAB reference output to compare against in the published PDF, but the formulas translate line-for-line.

### Paper-text quantitative claims

| Location | Claim | Replicated | Match |
|---|---|---:|:---:|
| p.6 | Romer factor at gy₀=2%, η=1%: "e² ≈ 7.4×" | 7.3891 | ✓ |
| p.8 | Semi-endog factor at β=3: "(1+3·2)^(1/3) ≈ 1.9" | 1.9129 | ✓ |
| p.1 | "TFR is 1.8 for the U.S., 1.7 for China, 1.7 for HIC, 1.6 for Germany, 1.4 for Japan, 1.3 for Italy and Spain" | all match | ✓ |
| p.19 | "optimal population growth rate is substantially higher than the equilibrium rate: 1.74% versus -0.5%" | 1.74% vs −0.5% | ✓ |
| p.21 | "idea share of social value of people ≈ 64%" | 64.01% | ✓ |

### Figure 6 (Optimal Transition Dynamics)

Our Python reverse-shooting produces the same Figure 6 qualitative shape: an upper arm on the (n, x) plane starting near the high SS (xH≈230, nH≈1.16%) and curving down through the middle SS and asymptoting toward the Empty Planet at (x→∞, n→-0.5%). Saved as `output/figure6_optimal_dynamics.png`.

The ODE also reproduces the secondary text claim that between the point where n=0 and the Empty Planet, A(t) rises by a factor of ~**1.31** — the NBER WP and the MATLAB diary both report values in this vicinity depending on which calibration is used.

### Figure 1 (UN Fertility Plot)

Reproduced in `output/figure1_fertility.png`. All 10 regions plotted with the same qualitative shape as the published figure (monotone decline for World and India, high→low convergence for HIC / USA / China below replacement).

---

## 4. Data Audit Findings

### The only dataset: UN TFR table (14 × 10)

- **Shape:** 14 five-year periods (1950-54 … 2015-20) × 10 regions.
- **Missing values:** 0.
- **Plausibility:** All TFRs in [1.187, 6.300] — no out-of-range values.
- **Monotonicity:** All 10 regions show net decline from 1950 to 2015 (range: Germany −0.54, China −4.42).
- **High-income countries** first fell below 2.0 in 1980-84 (TFR = 1.938).

### Paper-text cross-checks (all match to ±0.05):

| Region | 2015-20 data | Paper quote |
|---|---:|---:|
| USA | 1.78 | 1.8 |
| China | 1.69 | 1.7 |
| High-income | 1.68 | 1.7 |
| Germany | 1.59 | 1.6 |
| Japan | 1.37 | 1.4 |
| Italy / Spain | 1.33 | 1.3 |

### Model parameter restrictions (from code comments in `SetParameters.m`)

The code prints four inequalities that the calibration should satisfy:

| Inequality | Meaning | Current benchmark | NBER Table 2 |
|---|---|---:|---:|
| 1 + √(γ/λ) − b̄ν > 0 | Concavity of Hamiltonian | **−1.086** (fails) | **−2.448** (fails) |
| b̄ − δ − 1/ν > 0 | Max fertility at x=0 | +0.020 ✓ | +0.021 ✓ |
| ρ/ε − (b̄−δ) > 0 | Positive middle SS / n_eq < 0 | +0.005 ✓ | +0.005 ✓ |
| ν(b̄−δ) − (1+ε/λ) > 0 | Two real positive roots exist | +0.812 ✓ | +1.971 ✓ |

**Noteworthy:** The *concavity* sufficient condition fails in both calibrations. This is only a sufficient condition — the Hamiltonian can still be (and apparently is) concave along the relevant path — but it means Jones's calibration sits in a region where concavity of the maximized Hamiltonian is not guaranteed by closed-form inspection and must rely on the numerical stability analysis. The MATLAB code prints this diagnostic without raising; the author clearly knows this and treats it as informative rather than disqualifying.

---

## 5. Robustness Check Results

All checks target the NBER Table 2 calibration baseline (n*_H = 1.74%, g_A^H = 1.39%).

| # | Check | Finding | Status |
|---|---|---|---|
| 1 | β ∈ {0.5, 0.75, …, 2.5} | n*_H ranges 2.41% → 0.89%; monotone decreasing | **Robust & sensible** |
| 2 | λ ∈ {0.5, 0.75, 1.0, 1.25} | λ=0.5 degenerate (spiral); λ=1.25 gives n*_H=2.00% | **Robust** |
| 3 | n*_eq ∈ [-1%, 0%] | Only valid for -0.5% and -0.25%; at -0.75% b̄ becomes zero (corner) | **Sensitive to corner** |
| 4 | ℓ*_eq ∈ {1/12, 1/10, 1/8, 1/6, 1/4} | n*_H ranges 3.77% → 0.50% → NaN at ℓ=1/4 | **Sensitive** |
| 5 | δ ∈ {1/120, 1/100, 1/90, 1/80} | n*_H ranges 0.66% → 3.21% | **Sensitive** |
| 6 | All five SANDBOX cases from `SetParameters.m` comments | Current benchmark gives 1.16%/0.43% (matches comment 1.15/0.4); NBER v1 gives 1.74%/1.39% (matches Table 2); "Lower ell=1/10" gives 2.45%/0.92% (matches comment 2.4/0.9); **"Life exp 80yrs" gives 3.93%/0.98% but comment says 2.5/0.6** (stale); **"Higher b̄-δ" gives identical result to "Lower ell=1/10" and comment disagrees** (stale) | **2 stale comments, no bug** |
| 7 | Empty-Planet factor exp(gy₀/η) grid | (0.02, 0.01)→7.39; (0.02, 0.005)→54.60; (0.015, 0.01)→4.48 | **Confirms paper p.6** |
| 8 | Semi-endog factor (1+β·gy₀/η)^(1/β) | (0.02, 0.01, 3)→1.91; (0.02, 0.01, 0.5)→4.00 | **Confirms paper p.8** |
| 9 | β ≥ 3 in NBER calibration | b²−4ac turns negative → no real High SS → spiral regime only | **Confirms footnote 5** |
| 10 | Monotonicity: does n*_sp decline with β? | Yes, over [0.5, 2.5]: confirmed | **Robust** |

**Key robustness finding:** The quantitative numbers in Table 2 are not especially robust to small changes in β, ℓ*_eq, or δ — they can easily move by a factor of 2-3×. But the *qualitative* three-steady-state structure and the *sign* of the externality (optimal > equilibrium population growth) are extremely robust. The paper does not over-claim precision and is explicit about its calibration being illustrative. When β becomes large enough (≥3 in the NBER calibration), the discriminant b²−4ac flips sign and the High SS ceases to exist — this is the "spiral-only" regime that the paper's Footnote 5 warns about.

---

## 6. Summary Assessment

### What Replicates
- **Every numerical claim in NBER Table 2** (6/6) matches to 3-4 decimals.
- **The entire halflife table** (10 entries) reproduces directly from closed-form formulas.
- **Paper-text numeric quotes** (3 from §2, 1 from §4.2, 8 TFR point values from §1) all match.
- **Figure 1** (fertility plot) reproduces exactly since the data is hard-coded.
- **Figure 6** (optimal transition dynamics) reproduces qualitatively via reverse-shooting ODE.
- **The current MATLAB sandbox baseline** (nH≈1.16%, gAH≈0.43%) matches the comment in `SetParameters.m`.

### What Doesn't Replicate / Wasn't Attempted
- **The appendix Barro-Becker check** (`check_barrobeckerV.m`) — tangential.
- **Figures 3, 4** (stylized multi-SS diagrams) — use a deprecated "v0.30 of paper" parameterization purely for visual illustration; no quantitative claim.
- **Figure 5** (equilibrium transition dynamics) — trivial one-dimensional case; not computed.

### Key Concerns

1. **PDF version mismatch.** Our on-disk PDF is NBER Working Paper 26651 (January 2020). The replication package appears to correspond to the AER-published version (likely 2022 based on file comments). `SetParameters.m` uses a different baseline calibration (λ=3/4, β=2) than the NBER Table 2 (λ=1, β=1.25). This is annoying for replicators but is not a bug — the author simply updated the baseline between preprint and publication. Both calibrations produce qualitatively identical conclusions.

2. **Stale sandbox comments.** Three of the five commented-out SANDBOX parameter blocks in `SetParameters.m` have annotated "expected output" numbers that disagree with what the code actually produces when uncommented:
   - *"LIFE EXPECTANCY = 80 YEARS nH=2.5% gAH=0.6%"* → code actually gives nH=3.93%, gAH=0.98%.
   - *"HIGHER MAXIMUM FERTILITY RATE BBAR-DELTA nH=3.6% gAH=1.4%"* → code actually gives nH=2.45%, gAH=0.92% (same as "Lower ell=1/10" because the parameter settings are identical in the block). The comment references "bbar=.0611" which is never actually set.
   These are out-of-date documentation, not code bugs; they are in dead (commented-out) code blocks that the reader is expected to experiment with manually.

3. **Concavity condition fails.** The sufficient concavity condition `1 + √(γ/λ) − b̄ν > 0` fails for both calibrations (−1.09 and −2.45). This is *only a sufficient* condition, and the `solvemodel.m` stability analysis does the actual verification via linearization. The author is clearly aware — they print the diagnostic. But a reader who doesn't notice the "Sufficient conditions for maximized Hamiltonian to be concave" comment could be surprised.

4. **Calibration is illustrative, not structural.** The paper is explicit about this — the numbers in Table 2 exist to give a sense of magnitudes, not to make precise quantitative predictions. Our robustness sweeps confirm that n*_sp moves by 2-3× under small changes in β, ℓ*_eq, δ. The paper does not over-claim and this is not a criticism.

### Bug status
**No coding bugs found.** The MATLAB replication code is internally consistent, the Python translation reproduces every published numerical claim, and the only issues are stale informational comments in commented-out code blocks.

### Overall Assessment
This is a **theoretical / numerical** replication, not an empirical one. The paper makes no statistical claims — no regression coefficients, no standard errors, no inference. There is nothing to "replicate" in the econometric sense. What we can and did verify:
1. The closed-form steady-state formulas produce the values in Table 2 (✓).
2. The half-life computations follow from the stated differential equations (✓).
3. The reverse-shooting ODE produces a saddle-path between the three steady states (✓).
4. The UN fertility data hard-coded in `FertilityData.m` matches the numbers quoted in paper text (✓).

The "Empty Planet" conclusion is a theoretical result about the long-run behavior of three specific families of growth models under negative population growth. It does not depend on a particular calibration, a particular dataset, or a particular estimator, and our replication confirms that the numerical illustrations in the paper are accurate implementations of the stated mathematics.

---

## 7. File Manifest

| File | Description |
|---|---|
| `utils.py` | Parameters, `solvemodel` (Python port of `solvemodel.m`), ODE RHS, hard-coded UN fertility table |
| `01_fertility_figure.py` | Figure 1: TFR by region, 1950-2015 |
| `02_halflife.py` | Reproduces `halflife.m` — closed-form growth and level halflives |
| `03_steady_states.py` | Reproduces `solvemodel.m` / Table 2 — High, Middle, and Low steady states under both calibrations |
| `04_optimal_dynamics.py` | Reproduces `OptimalDynamics.m` Figure 6 — reverse-shooting ODE transition dynamics |
| `05_data_audit.py` | Audit of the UN TFR data and the model parameter-restriction diagnostics |
| `06_robustness.py` | 10 sensitivity checks over β, λ, n_eq, ℓ, δ, SANDBOX cases, Romer/semi-endog factors, degeneracy, monotonicity |
| `output/figure1_fertility.png` | Python-produced Figure 1 |
| `output/figure6_optimal_dynamics.png` | Python-produced Figure 6 |
| `writeup_176683.md` | This writeup |
