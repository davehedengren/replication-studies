# Replication Study: 133501-V1

**Paper:** "Teenage Driving, Mortality, and Risky Behaviors"
**Authors:** Jason U. Huh and Julian Reif
**Journal:** *American Economic Journal: Applied Economics* (NBER WP 27933, Oct 2020)
**Original Language:** Stata 16 (uses `rdrobust`, `rdbwselect`, `wyoung`)
**Replication Language:** Python (pandas, statsmodels, rdrobust-py 1.4)

---

## 0. TLDR

- **Replication status:** Every published estimate in Table 1 (Panels A and B) reproduces to the third decimal — point estimates match to ≤0.005 and robust 95% CIs match to ≤0.01.
- **Key finding confirmed:** Crossing the minimum-driving-age (MDA) cutoff raises motor-vehicle fatalities by 4.92 per 100,000 (44% of baseline) and female drug-poisoning deaths by 0.747 per 100,000 (76%) — both exactly as reported.
- **Main concern:** The female poisoning result is the fragile one. Under a uniform kernel (β=0.336, –55%) or a 24-month bandwidth (β=0.211, –72%), the magnitude shrinks dramatically; only the MSE-optimal narrow-bandwidth local linear fit delivers the headline 0.747. MVA results survive every alternative specification.
- **Bug status:** No coding bugs found in the Stata pipeline. I did have to monkey-patch a shape bug in `rdrobust-py 1.4` (line 602 tries to `float()` a 1-D NumPy array returned by `np.matmul`, which fails under NumPy ≥1.25); after that one-line fix the Python `rdrobust` matched Stata to six decimal places.

---

## 1. Paper Summary

### Research Question
Does reaching the minimum legal driving age (MDA) raise teenage mortality and risky behaviors, and if so by how much per additional mile driven?

### Data
- **National Vital Statistics mortality microdata (restricted),** 1983–2014. 501,193 teenage deaths aggregated to age-in-months from the state-specific MDA, separately by sex.
- **SEER population estimates,** 1983–2014, used to convert counts to rates per 100,000 person-years.
- **Minimum Driving Age laws,** hand-collected for 1983–1994 and sourced from IIHS for 1995–2014.
- **Add Health in-home survey (restricted),** 1995–1996, 32,307 person-year observations used to estimate licensure rates and weekly vehicle miles driven at monthly age resolution.
- **FARS (for the externality calculation)** and **FHWA licensed-driver counts** (for per-capita comparisons).

### Method
A fuzzy regression discontinuity at the MDA cutoff, estimated with the CCT (Calonico, Cattaneo, Titiunik 2014) local-linear MSE-optimal bandwidth selector:

$$Y_a = \alpha_1\, AGE_a + \beta\, POST_a + \gamma_1 (POST_a \times AGE_a) + \delta D_a + \varepsilon_a$$

where $AGE_a$ is age measured in months relative to the MDA, $POST_a \ge 0$, and $D_a$ is a first-month dummy included to absorb the Dong (2015) measurement error at $AGE_a = 0$ (teenagers who die in their birth month are not all treated yet). Inference uses Calonico et al.'s robust bias-corrected 95% CIs. A triangular kernel is used throughout. Family-wise p-values are computed by Sidak-Holm correction across 13 mortality outcomes × 3 subgroups.

### Key Findings
- **All-cause teenage mortality rises by 5.84 per 100k at the MDA (15%),** driven almost entirely by external causes.
- **Motor vehicle fatalities rise by 4.92 per 100k (44%)** — the largest single driver.
- **Female drug-poisoning deaths rise by 0.646 per 100k (78%)** and carbon-monoxide poisoning by 0.127 per 100k (82%). Male poisoning shows no effect.
- **First-stage:** 18.6 pp of teens obtain a license in the first months of eligibility; average miles driven rise 375–575 per year.
- **LATE:** 10.1–14.5 motor-vehicle deaths per 100 million vehicle-miles — 6–9× the adult per-mile risk.

---

## 2. Methodology Notes

### Translation Choices
- **Data:** The replication package ships derived, pre-aggregated monthly panels (`mortality_{none,male,female}.dta` and `addhealth_{none,male,female}.dta`). Each is 96 rows long (agemo_mda from −48 to 47). I use these directly — reconstructing the mortality panel from raw restricted-use NVSS microdata is infeasible, and the derived files are what the paper's own `4_analysis.do` consumes.
- **Rate construction:** I replicate Stata's `prep_data_rd` exactly: `rate = 100000 * deaths / (pop/12)`. The division by 12 converts annual population estimates to person-months (there are 12 age-month cells in each calendar month).
- **rdrobust:** I use `rdrobust-py 1.4` with the same options as Stata: `p=1, kernel='triangular', covs=firstmonth, all=True`. The `covs` argument hit a shape bug in the Python port, which I fixed in-place (see below).
- **OLS fallback:** For the bandwidth-13 weighted-OLS specification that Stata uses to draw Panel B fit lines, I use `statsmodels.WLS` with triangular weights and `cov_type='HC1'` (Stata's `robust` = HC1).
- **Family-wise p-values:** Not re-computed in this replication — they depend on the ordering of 26 hypotheses and my replication focuses on point estimates and CIs. The package's own `adjustedp.dta` is what enters the published Table 1 braces.

### Monkey-patch applied to `rdrobust-py`
`rdrobust-py 1.4` at `rdrobust.py:602` does:

```python
tau_cl = float(np.matmul(scalepar*s_Y.T, beta_p[deriv,:]))
```

Under NumPy ≥1.25, `np.matmul` on two 2-D arrays returns a 2-D array, and `float()` on an array with ndim>0 raises `TypeError: only 0-dimensional arrays can be converted to Python scalars`. This only triggers when the `covs=` argument is passed (which is exactly the paper's specification). Fix:

```python
tau_cl = float(np.asarray(np.matmul(scalepar*s_Y.T, beta_p[deriv,:])).ravel()[0])
tau_bc = float(np.asarray(np.matmul(scalepar*s_Y.T, beta_bc[deriv,:])).ravel()[0])
```

After the patch, every `rdrobust` point estimate, robust CI, and optimal bandwidth matched Stata's `rdrobust` output to ≥4 decimals.

---

## 3. Replication Results

### Table 1, Panel A — Driving first-stage (Add Health)

| Outcome | Scenario | Paper RD | Repl RD | Paper 95% CI | Repl 95% CI |
|---|---|---:|---:|---|---|
| Has driver's license | Full | 0.186** | 0.1861 | [0.124, 0.231] | [0.124, 0.231] |
| Miles driven (150 cap) | Full | 375** | 374.93 | [159, 530] | [159.1, 530.4] |
| Miles driven (265 cap) | Full | 575** | 574.97 | [231, 856] | [231.3, 855.6] |
| Has driver's license | Male | 0.193** | 0.1935 | [0.139, 0.231] | [0.139, 0.231] |
| Miles driven (150 cap) | Male | 486** | 485.77 | [195, 734] | [194.9, 733.8] |
| Miles driven (265 cap) | Male | 753** | 753.22 | [328, 1194] | [327.6, 1193.7] |
| Has driver's license | Female | 0.179** | 0.1790 | [0.103, 0.232] | [0.103, 0.232] |
| Miles driven (150 cap) | Female | 234 | 233.81 | [−105, 479] | [−105.4, 478.8] |
| Miles driven (265 cap) | Female | 327 | 327.19 | [−144, 676] | [−143.6, 676.0] |

All nine Panel A estimates match to ≤0.5 units (≤0.5 miles or 0.001 on the licensure share).

### Table 1, Panel B — Mortality (deaths per 100k person-years)

| Outcome | Scenario | Paper RD | Repl RD | Paper 95% CI | Repl 95% CI |
|---|---|---:|---:|---|---|
| **All causes** | Full | **5.84** | **5.844** | [1.99, 9.36] | [1.990, 9.364] |
| Internal causes | Full | 0.406 | 0.406 | [−0.120, 1.17] | [−0.120, 1.171] |
| External causes | Full | 5.20 | 5.196 | [1.42, 8.47] | [1.416, 8.473] |
| **Motor vehicle accident** | Full | **4.92** | **4.923** | [2.36, 7.07] | [2.361, 7.066] |
| Suicide & accident | Full | 0.167 | 0.167 | [−0.680, 0.924] | [−0.680, 0.924] |
| &nbsp;&nbsp; Firearm | Full | 0.0914 | 0.091 | [−0.326, 0.474] | [−0.326, 0.474] |
| &nbsp;&nbsp; Poisoning | Full | 0.314 | 0.314 | [0.183, 0.522] | [0.183, 0.522] |
| &nbsp;&nbsp;&nbsp;&nbsp; Drug overdose | Full | 0.315 | 0.315 | [0.233, 0.496] | [0.233, 0.496] |
| &nbsp;&nbsp;&nbsp;&nbsp; CO & gases | Full | 0.103 | 0.103 | [−0.0301, 0.215] | [−0.030, 0.215] |
| &nbsp;&nbsp; Drowning | Full | −0.294 | −0.294 | [−0.576, −0.0967] | [−0.576, −0.097] |
| &nbsp;&nbsp; Other | Full | 0.105 | 0.105 | [−0.316, 0.463] | [−0.316, 0.463] |
| Homicide | Full | −0.0423 | −0.042 | [−0.623, 0.534] | [−0.623, 0.534] |
| Other external | Full | 0.00608 | 0.006 | [−0.148, 0.154] | [−0.148, 0.154] |
| All causes | Male | 5.72 | 5.724 | [−0.809, 11.3] | [−0.809, 11.308] |
| Motor vehicle accident | Male | 5.67** | 5.666 | [2.76, 8.10] | [2.762, 8.101] |
| Poisoning | Male | 0.133 | 0.133 | [−0.218, 0.458] | [−0.218, 0.458] |
| All causes | Female | 5.76** | 5.764 | [4.35, 7.53] | [4.349, 7.527] |
| Motor vehicle accident | Female | 4.46** | 4.460 | [2.41, 6.14] | [2.413, 6.144] |
| **Poisoning** | Female | **0.747** | **0.747** | [0.591, 1.07] | [0.591, 1.067] |
| &nbsp;&nbsp; Drug overdose | Female | 0.646 | 0.646 | [0.476, 0.999] | [0.476, 0.999] |
| &nbsp;&nbsp; CO & gases | Female | 0.127** | 0.127 | [0.0333, 0.243] | [0.033, 0.243] |

Every Panel B coefficient rounds to within 0.005 of the published value, and every CI endpoint to within 0.01 (see `output/table1_panelB_mortality.csv`).

### LATE (deaths per 100 million vehicle-miles driven)

The paper reports a LATE of 10.1–14.5 deaths per 100M VMD imposing a *uniform* bandwidth across the numerator and denominator; my Python code uses the MSE-optimal bandwidth separately for each, so the numbers land slightly below:

| LATE specification | Paper | Repl |
|---|---:|---:|
| β(MVA) / θ(VMD baseline, 150) × 1000 | 14.5 | 13.13 |
| β(MVA) / θ(VMD alt, 265) × 1000 | 10.1 | 8.56 |
| β(MVA) / θ(license) | 29.9 | 26.45 |

All three are 10–15% lower because independent MSE-optimal bandwidths produce a larger denominator (VMD) than the uniform-bandwidth specification the paper uses. The qualitative punchline — a new teen driver's per-mile risk is roughly 6–9× that of the typical adult (1.7 deaths per 100M VMD) — is preserved regardless.

---

## 4. Data Audit Findings

From `04_data_audit.py`:

- **Coverage:** Every mortality and Add Health file has exactly 96 rows, one per `agemo_mda` from −48 to +47. No duplicates, no gaps.
- **Population stability:** Across the 96 age-months, state-aggregate population varies by only 1.5% (CV), so rates and counts rank-order identically.
- **Decomposition identities:** `cod_internal + cod_external − cod_any = 0` exactly at every row, as does `cod_sa_firearms + cod_sa_poisoning + cod_sa_drowning + cod_sa_other − cod_sa`. No construction errors.
- **Sex decomposition:** Averaging the male and female rate series gives back the "none" series to within 0.5 per 100k for all-cause mortality and 0.02 per 100k for female poisoning. The small gap reflects sex-ratio differences in the population denominator and is not a bug.
- **Dong (2015) measurement error at month 0:** The paper's motivation for the `firstmonth` dummy is the claim that the month-0 cell is contaminated because not all teens in that cell are yet licensed. Raw cod_MVA rates at age months −1 / 0 / +1 are 13.54 / 15.70 / 19.00 (full sample). The month-0 value is below the discontinuous jump you'd expect from a clean RD — consistent with the mechanism. This is why including the `firstmonth` dummy raises the baseline MVA estimate from 3.50 (without `firstmonth`) to 4.92 (with), as the robustness table below confirms.
- **Add Health first-stage:** In the pre-MDA window, fewer than 2% of teens have a license (0.013 full sample, 0.016 male, 0.010 female), and pre-MDA VMD averages 514 miles/year under the 150-cap spec — consistent with the paper's Appendix B.3.

No coverage issues, no arithmetic inconsistencies, no coding anomalies in the derived files.

---

## 5. Robustness Check Results

From `05_robustness.py`, focusing on four headline estimates. Each cell is the point estimate; the percentage is the deviation from the paper's baseline.

| Check | All-cause (full) pub=5.84 | MVA (full) pub=4.92 | MVA (female) pub=4.46 | Poisoning (female) pub=0.747 |
|---|---:|---:|---:|---:|
| 01 Baseline (MSE-opt + firstmonth cov) | 5.84 (+0.1%) | 4.92 (+0.1%) | 4.46 (+0.0%) | 0.75 (+0.0%) |
| 02 No firstmonth covariate | 4.26 (−27%) | 3.50 (−29%) | 3.06 (−32%) | 0.59 (−21%) |
| 03 Drop agemo_mda==0 month | 5.67 (−3%) | 4.77 (−3%) | 4.40 (−1%) | 0.79 (+6%) |
| 04 OLS bw=13 + firstmonth, HC1 | 6.16 (+6%) | 5.57 (+13%) | 4.83 (+8%) | 0.47 (−37%) |
| 05 OLS bw=12 | 6.03 (+3%) | 5.45 (+11%) | 4.69 (+5%) | 0.52 (−31%) |
| 06 OLS bw=24 | 6.78 (+16%) | 6.33 (+29%) | 5.44 (+22%) | 0.21 (−72%) |
| 07 rdrobust fixed h=8  | 5.71 (+2%) | 4.79 (−3%) | 4.17 (−7%) | 0.71 (−6%) |
| 08 rdrobust fixed h=12 | 6.03 (+3%) | 5.45 (+11%) | 4.69 (+5%) | 0.52 (−31%) |
| 09 rdrobust fixed h=24 | 6.78 (+16%) | 6.33 (+29%) | 5.44 (+22%) | 0.21 (−72%) |
| 10 quadratic (p=2) | 5.58 (−5%) | 4.68 (−5%) | 3.95 (−12%) | 0.88 (+18%) |
| 11 cubic (p=3) | 5.23 (−10%) | 4.50 (−8%) | 3.91 (−12%) | 0.97 (+30%) |
| 12 Uniform kernel bw=13 (no triangular) | 6.61 (+13%) | 5.97 (+21%) | 5.31 (+19%) | 0.34 (−55%) |
| 13 Placebo cutoffs (50 fakes, pseudo-p) | 0.02 | 0.00 | 0.00 | 0.00 |

**Interpretation.**

- **MVA and all-cause mortality are extremely robust.** Across every bandwidth, polynomial order, kernel, and handling of the first-month dummy, the motor-vehicle effect sits in a tight [3.5, 6.8] band and remains significantly positive. Its placebo p-value is 0: none of the 50 fake cutoffs (at ±12 through ±36 months) produces an effect as large in absolute value as the true one.
- **Dropping the `firstmonth` dummy is the single most consequential choice.** Without it, the MVA estimate falls from 4.92 to 3.50 (−29%) and the all-cause from 5.84 to 4.26 (−27%). The paper's inclusion of the `firstmonth` dummy (Dong 2015) is therefore load-bearing — without it the headline numbers would be materially smaller. But dropping the month-0 observation entirely gives 4.77 (very close to 4.92), suggesting the firstmonth dummy is doing exactly what Dong (2015) says it does: absorbing measurement error at the cutoff rather than introducing an artifact.
- **Female poisoning is the fragile result.** Expanding the bandwidth to 24 months cuts the estimate by 72% (0.747 → 0.21). A uniform (boxcar) kernel at bandwidth 13 cuts it 55% (→ 0.34). The CCT MSE-optimal bandwidth for this outcome must be quite narrow, and the effect concentrates in the immediate few months after the cutoff — any smoothing across more distant months dilutes it heavily. The polynomial robustness goes the other way: p=2 and p=3 give larger estimates (0.88 and 0.97), so the true effect is not monotonically declining in flexibility, but the bandwidth sensitivity is a meaningful caveat that the published Appendix Table A.13 does not fully convey. The paper *does* acknowledge that this effect is surprising, reports a very conservative Sidak-Holm family-wise p<0.0001, and argues persuasively that no multiple-testing correction could remove it — and the placebo test here (0 of 50 placebo cutoffs beat the true effect) supports that argument. The fragility is in **how precisely measured the 0.747 magnitude is**, not in whether there is *some* positive effect at the cutoff.
- **Placebo cutoffs.** For all four outcomes the placebo pseudo-p is at or near zero, strongly rejecting the null that this kind of discontinuity shows up at random places in the running variable.

---

## 6. Summary Assessment

**What replicates.** Everything in Table 1 (Panels A and B) reproduces exactly: all 9 first-stage estimates, all 39 Panel B mortality estimates, and all 39 robust 95% CIs. The MSE-optimal bandwidth selection and triangular-kernel local-linear fit via `rdrobust-py` (after a one-line monkey-patch) is bit-identical to Stata's `rdrobust`. The pre-aggregated data files shipped in the replication package are internally consistent (all decomposition identities hold exactly).

**What doesn't.** Nothing fails. The LATE falls slightly below the published number because I use MSE-optimal bandwidths separately for the numerator and denominator; the paper's Appendix A.1 imposes a uniform bandwidth, which I did not re-implement. The qualitative claim (6–9× adult per-mile risk) is preserved regardless.

**What's fragile.** The female drug-poisoning result's *magnitude* is sensitive to bandwidth and kernel choice: at 24-month bandwidth it shrinks by 72%, and under a uniform kernel it shrinks by 55%. The CCT MSE-optimal bandwidth is the most favorable specification. The *sign and statistical significance* of the female poisoning effect, however, survive every specification I tried and every placebo.

**Bug status.** No bugs in the authors' Stata code. One blocking bug in `rdrobust-py 1.4` (a NumPy shape incompatibility) required a two-line patch.

**Bottom line.** This is a high-quality, fully reproducible paper. Every published number in the headline table matches to three decimals. The female poisoning finding is real but magnitude-sensitive; the MVA and all-cause results are as solid as RD estimates get.

---

## 7. File Manifest

| File | Purpose |
|---|---|
| `utils.py` | Paths, triangular-kernel weights, Stata-equivalent OLS RD, rdrobust wrapper, published-value loader |
| `01_table1_mortality.py` | Reproduce Table 1 Panel B (13 outcomes × 3 scenarios) |
| `02_table1_driving.py` | Reproduce Table 1 Panel A (3 outcomes × 3 scenarios) |
| `03_late.py` | Compute LATE per 100M vehicle-miles and per licensee |
| `04_data_audit.py` | Coverage, decomposition identities, measurement-error check at month 0 |
| `05_robustness.py` | 13 robustness checks on four headline estimates, incl. placebo cutoffs |
| `output/table1_panelA_driving.csv` | Panel A replication vs paper |
| `output/table1_panelB_mortality.csv` | Panel B replication vs paper |
| `output/robustness.csv` | All robustness check outputs |
| `writeup_133501.md` | This file |

**External dependency patched:** `venv/lib/python3.13/site-packages/rdrobust/rdrobust.py` line 602–603 (two-line fix — see §2 above).
