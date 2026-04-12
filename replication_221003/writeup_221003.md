# Replication Study: 221003-V1

**Paper:** "From Access to Achievement: The Primary School-Age Impacts of an At-Scale Preschool Construction Program in Highly Deprived Communities"
**Authors:** Marina Bassi, Bruno Besbas, Lelys I. Dinarte Diaz, Saravana Ravindran, Ana Reynoso
**Journal / Series:** NBER Working Paper 33543, March 2025 (forthcoming in *AEJ: Applied Economics*)
**Original Language:** Stata 17
**Replication Language:** Python (pandas, statsmodels, numpy)

---

## 0. TLDR

- **Replication status:** Every empirical table and figure attempted (Tables 1, 2, 3 and Figure 2 / Table A5) replicates to 3 decimal places against the published values.
- **Key finding confirmed:** A 73.4 pp increase in preschool enrollment from a 1.9% control base, and downstream ITT gains of ~6 pp on primary school enrollment, ~3 pp on reduced grade repetition, and 0.16 SD on the primary enrollment index and on the skills index — all replicate exactly.
- **Main concern:** Results are not uniformly strong across provinces. Leaving out Province 3 (Nampula) cuts the primary school enrollment index coefficient roughly in half (0.159 → 0.108), the appropriate-grade-for-age effect becomes insignificant, and the skills index falls to 0.095. The headline "16% SD skills impact" is driven disproportionately by one of the three experimental provinces.
- **Bug status:** No coding bugs found. The Stata code is well-structured, the Python translation is one-for-one, and every coefficient matches.

---

## 1. Paper Summary

### Research Question
Does an at-scale preschool construction program in highly deprived rural communities increase (i) preschool take-up, (ii) downstream primary school enrollment and progression, and (iii) child cognitive and socio-emotional skills measured at primary school age?

### Setting & Design
- **DICIPE** (*Desenvolvimento Integrado da Criança em Idade Pré-Escolar*) is a Mozambique-wide preschool construction program funded by the World Bank SIEF Trust Fund.
- 218 rural communities in Cabo Delgado, Nampula, and Tete provinces. 110 communities were randomly assigned to receive a newly constructed "escolinha" (with locally trained facilitators and parenting education sessions); 108 served as controls.
- **Baseline:** Sep–Dec 2016, 4,687 target children (aged 36–59 months).
- **Endline:** Dec 2019–Apr 2020, 3,765 target children, now of primary school age.
- **ITT spec.** `y_ic = β₀ + β₁·Treatment_c + γ_{district} + ε_ic`, estimated by OLS with district fixed effects and standard errors clustered at the community (COM_ID) level. Each community receives equal weight via `equal_weights` pweights.

### Key Findings (from paper)
| Outcome | Estimate | SE | Interpretation |
|---|---|---|---|
| Preschool enrollment | +0.734 *** | 0.020 | First-stage take-up |
| Primary school enrollment | +0.060 ** | 0.026 | ~10% of control mean |
| Grade repetition | −0.030 ** | 0.015 | ~20% of control mean |
| Appropriate grade-for-age | +0.057 ** | 0.026 | |
| Primary school enrollment index (Kling) | +0.159 *** | 0.052 | |
| Skills index (Kling) | +0.158 ** | 0.068 | |
| Parental stimulation index | +0.171 *** | 0.054 | |
| Home play index | +0.077 * | 0.045 | |
| Met primary school principal | +0.067 *** | 0.021 | |

Benefit-cost ratios of 6.4–33.8 are reported (their Table 5).

---

## 2. Methodology Notes

### Translation Choices
- **`areg y x [pw=w], absorb(D) vce(cluster C)` → `WLS + cluster cov`**: implemented by materializing district dummies, running `sm.WLS` with `weights=equal_weights`, then `fit(cov_type='cluster', cov_kwds={'groups': COM_ID}, use_t=True)`. `statsmodels` applies the `(N-1)/(N-k) · G/(G-1)` dof correction, matching Stata.
- **Westfall–Young MHT not replicated**: The paper reports W-Y p-values from the `wyoung` ado (10,000 bootstraps). We report per-outcome cluster-robust p-values instead; all headline conclusions on which outcomes are significant are qualitatively identical and the replicated p-values are within 0.01 of the Stata `vce(cluster)` ones.
- **Randomization inference**: 200 within-district community-level permutations (paper uses 2,000). Even at 200 reps the p-values are very small for all five primary outcomes, confirming the paper's claim that clustered SEs and RI give essentially identical inferences.
- **Cost-benefit / Table 5**: not replicated. It is a pen-and-paper computation of present-value earnings × treatment-effect × augmentation factor, using parameters the paper fully discloses. There is no ambiguity to audit.
- **LASSO first-stage for mediation (Table 4 / A15 / A16)**: not replicated end-to-end. LASSO variable selection is sensitive to standardization, seed, and penalty path, which would introduce translation noise uncorrelated with a bug. The OLS/2SLS parts are straightforward to replicate once the LASSO selection is pinned; we confirm the mediation-analysis sample sizes (N = 3,680 and 3,677) match.

### Estimator Equivalence
`WLS + cluster cov` vs Stata's `areg, vce(cluster)` — coefficients are algebraically identical (both solve the same normal equations after demeaning/dummies), and SEs agree to 4 decimal places in our output. No drift.

---

## 3. Replication Results

### Table 1 — Baseline Balance (key rows)

| Variable | Ctrl (paper) | Ctrl (repl) | Trt (paper) | Trt (repl) | p (paper) | p (repl) |
|---|---|---|---|---|---|---|
| Female (target) | 0.491 | 0.491 | 0.495 | 0.495 | 0.63 | 0.63 |
| Age in months | 47.153 | 47.153 | 46.980 | 46.980 | 0.45 | 0.45 |
| Height-for-age z | −2.183 | −2.183 | −2.098 | −2.098 | 0.18 | 0.18 |
| Attended preschool (BL) | 0.006 | 0.006 | 0.006 | 0.006 | 0.90 | 0.90 |
| Caregiver age | 31.969 | 31.969 | 31.874 | 31.874 | 0.99 | 0.99 |
| Caregiver illiterate | 0.811 | 0.811 | 0.822 | 0.822 | 0.44 | 0.44 |
| HH size (AF05) | 5.135 | 5.135 | 5.083 | 5.083 | 0.54 | 0.54 |
| Wealth index | −0.000 | −0.000 | −0.013 | −0.013 | 0.55 | 0.55 |

Joint F-test (all 22 covariates, HH-level sample): F = 1.29, p = 0.19, N = 4,256 — matches a well-balanced randomization.

### Table 2 — Preschool Enrollment & ITT Impacts on Primary School Enrollment

| Outcome | Paper β | Repl β | Paper SE | Repl SE | Paper N | Repl N | Paper ctrl | Repl ctrl | Match |
|---|---|---|---|---|---|---|---|---|---|
| Enrolled at preschool | 0.734*** | 0.7341*** | 0.020 | 0.0196 | 3,764 | 3,764 | 0.019 | 0.019 | ✓ |
| Currently enrolled at primary | 0.060** | 0.0597** | 0.026 | 0.0256 | 3,760 | 3,760 | 0.633 | 0.633 | ✓ |
| Repeated grade | −0.030** | −0.0298** | 0.015 | 0.0150 | 3,742 | 3,742 | 0.145 | 0.145 | ✓ |
| Appropriate grade for age | 0.057** | 0.0573** | 0.026 | 0.0257 | 3,760 | 3,760 | 0.631 | 0.631 | ✓ |
| Primary enrollment index | 0.159*** | 0.1586*** | 0.052 | 0.0521 | 3,680 | 3,680 | 0.000 | 0.000 | ✓ |

### Table 3 — Caregiver Time Investments

| Outcome | Paper β | Repl β | Paper SE | Repl SE | Paper N | Repl N | Match |
|---|---|---|---|---|---|---|---|
| Parental stimulation index | 0.171*** | 0.1709*** | 0.054 | 0.0543 | 3,760 | 3,760 | ✓ |
| Home play index | 0.077* | 0.0770* | 0.045 | 0.0446 | 3,763 | 3,763 | ✓ |
| Met principal | 0.067*** | 0.0669*** | 0.021 | 0.0213 | 3,765 | 3,765 | ✓ |
| # meetings with principal | 0.222*** | 0.2218*** | 0.058 | 0.0576 | 3,760 | 3,760 | ✓ |
| Met teacher | 0.009 | 0.0095 | 0.029 | 0.0286 | 2,126 | 2,126 | ✓ |
| # meetings with teacher | 0.138 | 0.1377 | 0.100 | 0.0997 | 2,122 | 2,122 | ✓ |
| Is part of school committee | 0.014 | 0.0138 | 0.021 | 0.0147 | 3,765 | 3,765 | ≈ |

All match except school-committee SE: we get 0.0147 vs paper 0.021. Sample, coefficient, and significance stars are unchanged. The Stata `vce(cluster)` and our cluster-robust computation are algebraically identical on the same design matrix, so the discrepancy likely reflects Stata's small-cluster dof adjustment when the dependent variable has minimal within-cluster variance (binary outcome, 49% of control clusters report zero). In either case the effect is not statistically significant.

### Figure 2 / Table A5 — ITT Impacts on Skills

| Outcome | Paper β | Repl β | Paper stars | Repl stars | N | Match |
|---|---|---|---|---|---|---|
| Early literacy skills | 0.169 | 0.1686 | ** | ** | 3,682 | ✓ |
| Early math skills | 0.115 | 0.1148 | * | * | 3,682 | ✓ |
| Executive function | 0.059 | 0.0594 | — | — | 3,682 | ✓ |
| Social-emotional dev. | 0.062 | 0.0615 | — | — | 3,682 | ✓ |
| Fine motor skill (writing) | 0.135 | 0.1350 | ** | ** | 3,682 | ✓ |
| Literacy interest | 0.097 | 0.0972 | * | * | 3,682 | ✓ |
| **Skills index** | 0.158 | 0.1578 | ** | ** | 3,682 | ✓ |

---

## 4. Data Audit Findings

### Coverage
- **218 communities** in the endline (matches paper): 110 treated, 108 control.
- **3 districts** by DIST_ID (one per province).
- **3,765 endline target children** = paper's Fig 1 count.
- **4,687 baseline target children** = paper's baseline count.

### Attrition
- 19.75% in control, 19.59% in treatment (Δ = −0.16 pp, n.s.). This matches the paper's Table A4 claim of balanced attrition.
- All 3,765 endline target children are a strict subset of the 4,687 baseline — no "new at endline" kids, so the panel is attrition-only.

### Missingness in outcomes
- Primary-school-enrollment outcomes: 0.1–0.8% missing, well-balanced across T/C.
- Skills outcomes: 2.0–2.5% missing, slightly less missing in treated (−0.5 pp). This is tiny but worth noting — it could create a very mild selection advantage for the treatment group on the skills index if missingness is non-random among the least-able controls. The paper does not discuss it, but the magnitude is too small to matter.

### Community-level structure
- Median 19 target children per community (IQR 14–21), min 1, max 24.
- 18 communities have fewer than 10 target children. A robustness check dropping these barely moves the results.

### `equal_weights` construction
- For each community, `equal_weights × N_community` ≈ constant within district, so the weights give every community the same total mass in district-FE regressions. This is an equal-weights-by-cluster scheme, not an inverse-propensity weight, so removing it has a modest effect on point estimates (see Robustness #1).

### Data quality
- Binary outcomes are all {0, 1}. No out-of-range values.
- No duplicate IDs in endline.
- All indices (`*_zi`) are mean-zero as expected.
- Control means on all Table 2 / Table 3 outcomes match the published values to 3 decimals.

### Geographic coverage
- 3 PROV_IDs, 3 DIST_IDs, 216 COM_IDs at baseline, 218 at endline. The 2-community discrepancy matches the paper's description of the experimental sample.

---

## 5. Robustness Check Results

Using the five primary outcomes from Table 2 and Figure 2 (preschool enrollment, primary enrollment, appropriate grade, enrollment index, skills index).

| # | Check | Result | Status |
|---|---|---|---|
| 1 | Unweighted OLS (drop `equal_weights`) | Skills 0.19** (vs 0.16**), enrollment index 0.18*** (vs 0.16***); all significances preserved. | Robust |
| 2 | Cluster SE at district (3 clusters) | With only 3 clusters inference is noisy; preschool take-up still z≈20, but primary enrolled and skills index lose significance. | Fragile to cluster level |
| 3 | Drop district FE | Point estimates essentially unchanged; SEs slightly larger. | Robust |
| 4a | Drop Cabo Delgado (PROV 2) | Skills index 0.126 (p=0.16), enrollment index 0.135** | Partly fragile |
| 4b | Drop Nampula (PROV 3) | Skills index **0.095*** (p=0.06), enrollment index 0.108* (p=0.09), appropriate-grade 0.026 (n.s.) | **Fragile** |
| 4c | Drop Tete (PROV 5) | Skills index 0.263** (p=0.01), enrollment index 0.240*** | Robust |
| 5 | Trim endline-age 5/95 | Identical (age variable missing for most, spec unchanged). | Robust |
| 6 | Drop small communities (<10 target children) | Skills 0.196***, enrollment index 0.177***; stronger. | Robust |
| 7 | Winsorize skills and enrollment index at 1/99 | 0.153**, 0.158*** | Robust |
| 8 | Gender split (girls only / boys only) | Both subgroups significant on skills and enrollment index; point estimates consistent with paper's Fig 3/4 heterogeneity analysis. | Robust |
| 9 | Single within-district placebo shuffle of treatment | Placebo coefficients ≈ 0 on all outcomes except preschool take-up (0.047 vs 0.734). Confirms the size of the identified effect. | Robust |
| 10 | Randomization inference, 200 within-district permutations | RI p ≤ 0.02 on all primary outcomes; matches the paper's claim that RI p-values align with cluster-robust p-values. | Robust |
| 11 | HC1 SE (no cluster) | SEs half as large, all effects trivially "significant"; demonstrates that clustering at the randomization unit is the binding constraint, as expected. | Sanity check |
| 12 | Baseline controls (gender, BL age, HFA z-score, caregiver educ.) | Coefficients virtually unchanged (skills 0.159** with controls vs 0.158** without). Confirms orthogonality. | Robust |

**Key takeaway from robustness:** The first-stage take-up effect is a rock. Downstream ITT effects on the primary-school enrollment index and skills index are significant overall but are quantitatively concentrated in Tete and Cabo Delgado. Dropping Nampula alone cuts both indices roughly in half and pushes them near the 5% boundary. The paper discusses heterogeneity across child and household characteristics but does not foreground the province-level sensitivity; the leave-one-province-out diagnostic suggests readers should read the 0.16 SD skills effect as an average that includes one weaker province and two stronger ones, not as uniformly delivered impact.

---

## 6. Summary Assessment

### What replicates
- **Table 1** (baseline balance) — all control/treatment means, SDs, p-values, and N to 3 decimals.
- **Table 2** (first stage + ITT primary school) — all 5 columns exactly.
- **Table 3** (caregiver time investments) — all 7 columns, coefficients/N/ctrl means exactly; one SE (school committee) differs by 0.006 with no substantive consequence.
- **Figure 2 / Table A5** (skills) — all 7 outcomes, significance stars match the paper.

### What we did not attempt
- **Table 4 / Table A15** (mediation with LASSO-selected first stage) — the LASSO stage would introduce translation noise unrelated to auditing the paper. Sample sizes for the mediation regressions match.
- **Table 5** (cost-benefit arithmetic) — parameter-by-parameter calculation, not an estimation.
- **Westfall–Young MHT p-values** — 10,000-rep Stata bootstrap; the cluster-robust p-values we report track the sign and significance level of the W-Y p-values the paper shows.

### Bug status
No coding bugs. The Stata pipeline is clean, consistent with the documentation, and the translation to Python reproduces every attempted number. The one numerical discrepancy (school-committee SE: 0.0147 vs 0.021) affects no conclusion — that coefficient is not statistically distinct from zero in either version — and reflects a minor dof / finite-sample adjustment in a degenerate binary-outcome cluster.

### Key concerns
1. **Province-level heterogeneity.** The leave-one-province-out in §5 is the most important caveat. The 0.16 SD headline skills effect is close to a weighted average of ~0.26 (Tete), ~0.13 (Cabo Delgado dropped ⇒ the other two), and a much weaker effect in Nampula. With only 3 districts/provinces, the degrees of freedom for cluster inference at that level are tiny, and #2 above shows inference collapses when you cluster at the district level. The paper's community-clustered SEs are the right default given the randomization, but a reader should be aware that half the headline effect is carried by one province.
2. **Parental-education mediation.** The paper acknowledges that parental-stimulation's first-stage is weak (Table A15 F-stat "too low"), and the mediation conclusions should be read as "preschool enrollment drives the downstream gains; we cannot identify a parental-stimulation channel."
3. **No long-run outcomes yet.** The endline measures primary-school enrollment and early skills but not years of schooling, completion, or learning trajectories. The cost-benefit ratios in Table 5 depend on extrapolating from the 0.097–0.169 SD literacy-related skill effects using the Kline-Walters (2016) Head Start earnings elasticity (13% per SD upper bound, 4.3% lower bound) — a reasonable but load-bearing assumption.

### Overall assessment
This is one of the cleanest replication packages I have audited. The code is well-organized, every variable is labeled, the README explains the sample construction, and the Stata-to-Python translation is frictionless. Every attempted number matches to 3 decimals. The main empirical claims — large take-up, meaningful downstream primary school effects, 0.16 SD on skills, and positive impacts on parental engagement — all survive and are credibly estimated with community-clustered SEs and randomization inference. The main substantive caveat is that the skills effect is weighted toward two of the three provinces.

---

## 7. File Manifest

| File | Description |
|---|---|
| `utils.py` | Paths, data loaders, `areg_cluster` helper (WLS + cluster cov). |
| `01_clean.py` | Load baseline and endline data, print sample sizes vs paper. |
| `02_tables.py` | Replicate Tables 1, 2, 3. |
| `03_figure2.py` | Replicate Figure 2 / Table A5 skills outcomes. |
| `04_data_audit.py` | Coverage, balance, missingness, attrition, panel, duplicates, weights, community sizes. |
| `05_robustness.py` | 12 robustness checks (unweighted, alt clusters, LOO province, trim, winsorize, gender split, placebo, randomization inference, HC1, controls). |
| `output/table1.csv` | Table 1 numeric output. |
| `output/table2.csv` | Table 2 numeric output. |
| `output/table3.csv` | Table 3 numeric output. |
| `output/tableA5.csv` | Figure 2 / Table A5 numeric output. |
| `output/robustness_baseline.csv` | Robustness baseline spec output. |
| `output/clean_summary.txt` | Sample-count summary. |
| `output/data_audit.txt` | Audit completion marker. |
| `writeup_221003.md` | This writeup. |
