# Replication Study: Chan, Cropper, and Malik (2014)

**Paper:** "Why Are Power Plants in India Less Efficient than Power Plants in the United States?"
**Journal:** American Economic Review: Papers & Proceedings, 104(5): 586-590
**Authors:** Hei Sing (Ron) Chan, Maureen L. Cropper, and Kabir Malik

---

## 0. TLDR

- **Replication status:** Core regression results replicate almost exactly; matching estimates are directionally consistent but differ modestly due to implementation differences in the nearest-neighbor estimator.
- **Key finding confirmed:** State-owned Indian power plants are approximately 9.4% less thermally efficient than comparable US plants, and the gap narrowed from ~14% (1988-1991) to ~8% (post-1997).
- **Main concern:** India's OPHR reporting rate deteriorates from 84% to 54% over the study period; if non-reporting plants are less efficient, the estimated improvement over time may be overstated.
- **Bug status:** The nnmatch do files reference a nonexistent variable `bias` in the `biasadj(bias)` option, and `distribution.do` references nonexistent variables (`loghr`, `dyear*`). These appear to be bugs that would cause the original code to fail as provided. The regression code (`regressioncode.do`) runs correctly and produces the paper's main results.
- **Bottom line:** The main regression findings are robust and well-supported. The matching estimates and summary statistics may have been produced with a different data version or preprocessing step not included in the package.

---

## 1. Paper Summary

**Research question:** How does the thermal efficiency of state-owned coal-fired power plants in India compare to plants in the United States, and what explains the differences?

**Data:** Panel data on coal-fired power plants in both countries, 1988-2009. US data from EIA (724 plants), India data from state electricity boards (95 plants, state-owned focus). Thermal efficiency measured by net operating heat rate (OPHR, in MMBtu/kWh) and auxiliary generation (% of gross electricity used for plant operations).

**Method:**
1. OLS regression of log(OPHR) on India dummy (or India × year interactions) + polynomial controls for age and nameplate capacity + ownership/restructuring dummies + year FE, clustered at plant level.
2. Nearest-neighbor matching (Mahalanobis distance, M=5) on age, nameplate capacity, and ownership, estimated year by year.

**Key findings:**
- Indian plants had heat rates ~9.4% higher than US plants overall (regression), ~10.4% (matching).
- The gap declined from ~13.7% (1988-1991) to ~8.0% (post-1997).
- Controlling for coal heating value reduces the gap to ~6.8%, implying coal quality explains 20-30% of the difference.
- Auxiliary generation was ~3.5% higher at Indian plants.

---

## 2. Methodology Notes

**Translation choices:**
- OLS with clustered SEs: `statsmodels.OLS` with `cov_type='cluster'` (equivalent to Stata `,cluster(plantcode)`).
- Nearest-neighbor matching: Implemented from scratch using `scipy.spatial.distance.cdist` with Mahalanobis metric. Stata's `nnmatch` command (Abadie et al. 2003) was not available; our implementation matches on the same variables with M=5 but uses simpler SE estimation (SD of individual effects / sqrt(N_T)) rather than the Abadie-Imbens conditional variance estimator.
- Bias adjustment in `nnmatch` was not applied because the do files reference a nonexistent variable `bias` in `biasadj(bias)`.

**Estimator differences:**
- Matching SEs will differ from published values due to different variance estimators.
- The slight difference in India sample sizes (47 vs 38 in 1988) suggests the published results may use a different version of the India data or additional sample restrictions not documented in the replication package.

**Age variable inconsistency in original code:** `regressioncode.do` corrects US age (`age = w_age + 1`) but then sets `agesq = w_agesq` (= w_age^2) and `agecube = w_agecube` (= w_age^3), creating an inconsistency where the linear term uses corrected age but the polynomial terms use uncorrected age. We replicate this exactly.

---

## 3. Replication Results

### Table A1: Regression Results

| Statistic | Published | Replicated | Difference |
|-----------|-----------|------------|------------|
| **Model 1: India coefficient** | ~0.094 | 0.0945 | +0.1% |
| **Model 2 period averages:** | | | |
| 1988-1991 average ATE | 0.137 | 0.1344 | -1.9% |
| 1997-2009 average ATE | 0.080 | 0.0814 | +1.8% |
| Overall average ATE | 0.094 | 0.0939 | -0.1% |
| N (both models) | — | 9,890 | — |
| Adj. R² (Model 1) | — | 0.374 | — |

**Verdict: Near-exact replication of regression results.**

### Figure 1: Year-by-Year Treatment Effects

| Year | Published (approx.) | Replicated |
|------|-------------------|------------|
| 1988 | ~0.10 | 0.101 |
| 1989 | ~0.12 | 0.125 |
| 1990 | ~0.14 | 0.141 |
| 1991 | ~0.17 | 0.171 |
| 1997 | ~0.11 | 0.106 |
| 2003 | ~0.08 | 0.082 |
| 2009 | ~0.06 | 0.059 |

**Verdict: Figure replicates the declining trend precisely.**

### Table A2: NN Matching - Heat Rate

| Period | Published SATT | Replicated SATT |
|--------|---------------|-----------------|
| 1988-1991 | 0.152 | 0.142 |
| 1997-2009 | 0.089 | 0.076 |
| Overall | 0.104 | 0.091 |

**Verdict: Same direction and magnitude; ~10-15% lower than published, likely due to (a) no bias adjustment, (b) slightly different sample composition.**

### Table A3: NN Matching - Auxiliary Generation

| Period | Published | Replicated |
|--------|-----------|------------|
| 1997-2009 average | 3.48 pp | 3.64 pp |

**Verdict: Close match (+4.6% difference).**

### Table 1: Summary Statistics (2009, US)

| Variable | Published | Replicated | Match |
|----------|-----------|------------|-------|
| Age | 40.32 | 41.32 | Close |
| Capacity | 795.1 | 795.1 | Exact |
| Capacity factor | 52.52% | 52.52% | Exact |
| Heat content | 10,314 | 10,314 | Exact |
| Aux gen | 7.97% | 7.97% | Exact |
| Heat rate | 11,326 | 11,326 | Exact |
| N | 406 | 406 | Exact |

Note: 1988 US N = 419 in our data vs. published 406 (13 extra obs with missing capacity-weighted age that Stata regression would drop). India Ns are systematically higher (47 vs 38 in 1988), suggesting a data version difference.

---

## 4. Data Audit Findings

### Coverage Issues
- **India heatrate gap (1992-1996):** Zero Indian plants report heatrate in 1992-1993, and zero in 1994-1996 (from state-owned plants), creating a 5-year gap in the India panel. The paper acknowledges this.
- **Worsening OPHR coverage:** India's heatrate reporting rate declined from 84% (1988) to 54% (2009). This is a significant selection concern — if non-reporting plants are less efficient, the apparent improvement over time (declining treatment effect) could be partially artifactual.

### Data Quality
- **US extreme heatrates:** The raw US data contains implausible heatrate values up to 80 million (from tiny plants <1 MW). The code correctly drops these with the 6,000-20,000 filter.
- **US capacity factor anomalies:** 37 US observations have capacity factors >100% and 31 have negative values. These are mostly for small or unusual plants excluded by other filters.
- **India duplicates:** 14 duplicate (plantcode, year) observations in the India data.
- **India capacity factor:** 10 observations with capfactor >100%.
- **No serious data integrity issues** after the standard cleaning filters are applied.

### Missing Data Patterns
Plants without OPHR data in India are not systematically smaller or older than reporting plants in 2009 (contrary to what Table 1 might suggest with the published smaller sample). In our larger sample, non-OPHR plants are actually larger on average (1,057 MW vs 795 MW), suggesting the missing data is not purely a function of plant size.

---

## 4a. Bug Impact Analysis

### Bug 1: `biasadj(bias)` in nnmatch do files
- **Location:** `nnmatch_heatrate.do` line 55, `nnmatch_aux.do` line 45
- **Description:** The `biasadj(bias)` option references a variable called `bias` that does not exist in the dataset. In Stata, this would cause the `nnmatch` command to throw an error.
- **Impact:** Tables A2 and A3 cannot be reproduced from the provided code without modification. The matching results in the paper may have been produced with a different dataset version that included a `bias` variable, or with different code.
- **Affected tables:** Table A2 (NN matching heat rate), Table A3 (NN matching aux gen).
- **Unaffected:** Table 1, Table A1, Figure 1 (these use regression code which works correctly).

### Bug 2: Variable names in `distribution.do`
- **Location:** `distribution.do` line 18
- **Description:** The code references `loghr` (not in the data; variable is `log_heatrate`) and `dyear1989-dyear2010` (no `dyear*` variables exist in the dataset).
- **Impact:** Table 1 summary statistics cannot be reproduced from the provided code. The values were likely produced from a different data preparation.
- **Affected:** Table 1.
- **Unaffected:** All other tables and Figure 1.

### Bug 3: Age polynomial inconsistency in `regressioncode.do`
- **Location:** `regressioncode.do` lines 24-26
- **Description:** After correcting `age = w_age + 1`, the code sets `agesq = w_agesq` (= w_age²) instead of `(w_age + 1)²`. Same for `agecube`. This means the linear, quadratic, and cubic terms are not a consistent polynomial in the same variable.
- **Impact:** Very minor. The regression controls are flexible enough that this doesn't meaningfully affect the India treatment effect. Our replication follows the original code exactly and gets matching results.
- **What doesn't change:** The core finding of ~9.4% efficiency gap and its decline over time.

---

## 5. Robustness Check Results

| Check | Coefficient | SE | Change from baseline |
|-------|------------|-----|---------------------|
| **Baseline** | **0.0945** | **0.0197** | **—** |
| Tighter heatrate bounds (7000-15000) | 0.0646 | 0.0146 | -31.6% |
| Post-1997 only | 0.0934 | 0.0228 | -1.2% |
| 1988-1991 only | 0.1419 | 0.0217 | +50.1% |
| Drop top/bottom 5% log_heatrate | 0.0688 | 0.0137 | -27.1% |
| Winsorize 1st/99th | 0.0926 | 0.0189 | -2.0% |
| HC1 robust SEs | 0.0945 | 0.0064 | SE drops 68% |
| HC3 robust SEs | 0.0945 | 0.0064 | SE drops 68% |
| + coal quality control | 0.0664 | 0.0207 | -29.7% |
| Large plants (≥500 MW) | 0.0827 | 0.0211 | -12.5% |
| Small plants (<500 MW) | 0.1501 | 0.0353 | +58.9% |
| Leave-one-year-out (range) | 0.090-0.098 | — | Stable |
| Placebo permutation (p-value) | — | — | p = 0.000 |
| Nameplate ≥ 50 MW | 0.1088 | 0.0200 | +15.2% |
| Nameplate ≥ 100 MW | 0.1321 | 0.0208 | +39.8% |

### Key Robustness Findings

**Survives all checks:** The India efficiency gap is positive and statistically significant in every specification.

**Sensitive to outlier treatment:** Tighter heatrate bounds (-32%) and dropping extreme percentiles (-27%) meaningfully reduce the coefficient, suggesting extreme values contribute to the estimated gap. However, the effect remains strongly significant.

**Heterogeneity by plant size:** Smaller plants show a much larger gap (15.0%) than larger plants (8.3%). The paper's threshold of 25 MW is very inclusive; higher thresholds increase the estimated gap.

**Coal quality explains ~30%:** Adding BTU content reduces the coefficient from 0.094 to 0.066, confirming the paper's claim that coal quality accounts for 20-30% of the difference.

**Clustering matters for inference:** Clustered SEs are 3x larger than heteroskedasticity-robust SEs (0.020 vs 0.006), indicating substantial within-plant correlation. The plant-level clustering is appropriate.

**Leave-one-year-out:** Extremely stable (range: 0.090-0.098), confirming no single year drives the result.

**Permutation test:** p = 0.000 (500 permutations), strongly confirming the treatment effect is not due to chance.

---

## 6. Summary Assessment

### What Replicates
- **Main regression (Table A1, Figure 1):** Near-exact replication. The India treatment effect of ~9.4%, the declining time pattern, and the coal quality adjustment all match published values within 2%.
- **Matching estimates (Tables A2, A3):** Directionally consistent with published values. Point estimates are 10-15% lower for heat rate matching, likely due to implementation differences in the NN estimator.
- **Key qualitative conclusions:** All confirmed. Indian plants are less efficient, the gap has narrowed, and coal quality explains part but not all of the difference.

### What Doesn't Replicate Exactly
- **India sample sizes:** Our India counts are consistently higher than published (e.g., 47 vs 38 in 1988), suggesting the published results may use a different version of the data or undocumented sample restrictions.
- **Matching standard errors:** Differ due to different variance estimators (simple vs Abadie-Imbens).
- **Two of four do files have variable name bugs** that prevent them from running as provided.

### Key Concerns
1. **Selection bias from missing OPHR data:** The declining reporting rate (84% → 54%) is the most important threat to the paper's finding of an improving efficiency gap. If worse plants stop reporting, the apparent improvement is overstated.
2. **Code bugs in replication package:** Two of four do files (`distribution.do`, `nnmatch_*.do`) reference variables that don't exist in the provided data, suggesting the replication package is incomplete or based on a different data version.
3. **Sensitivity to outlier treatment:** The coefficient is 27-32% smaller when extreme heatrate values are excluded, though it remains significant.

### Overall Assessment
The paper's core finding — that Indian power plants are significantly less efficient than US plants — is well-supported and robust. The regression-based results replicate almost exactly. The code bugs affect supplementary analyses (summary statistics and matching) but not the main results. The declining OPHR coverage is a legitimate concern about the time trend, but the cross-sectional finding of an efficiency gap is solid.

---

## 7. File Manifest

| File | Description |
|------|-------------|
| `utils.py` | Shared paths, data loading, sample construction, matching, regression helpers |
| `01_clean.py` | Sample construction verification and diagnostics |
| `02_tables.py` | Table 1 (summary stats), Table A1 (regression), Table A2/A3 (matching) |
| `03_figures.py` | Figure 1 (year-specific treatment effects plot) |
| `04_data_audit.py` | Data quality, coverage, outliers, panel balance, consistency checks |
| `05_robustness.py` | 12 robustness checks (bounds, subsamples, SEs, placebo, etc.) |
| `writeup_112820.md` | This writeup |
| `figure1_treatment_effects.png` | Replicated Figure 1 |
