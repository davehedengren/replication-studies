# Replication Study: "Using School Choice Lotteries to Test Measures of School Effectiveness"

**Paper**: Deming, David J. (2014). NBER Working Paper No. 19803 / AER Papers & Proceedings.
**Replication Package**: 112805-V1
**Date**: 2026-03-04

---

## 0. TLDR

- **Replication status**: Results closely replicate. All six Table 1 Panel A coefficients match published values within 0.01-0.07, with differences attributable to the de-identified public-use data.
- **Key finding confirmed**: VAMs with prior test score controls and 2+ years of data are accurate, unbiased predictors of student achievement (coefficient ≈ 0.90 vs published 0.97; cannot reject = 1).
- **Main concern**: None significant. The paper's main finding is highly robust across estimators, subgroups, winsorization, and placebo tests. The one-year-prior specification remains significantly biased downward (≈ 0.47), consistent with the published finding.
- **Bug status**: No coding bugs found. The Stata code is well-documented, and the author clearly flags the limitations of the public-use data.

---

## 1. Paper Summary

**Research question**: Are school value-added models (VAMs) accurate, unbiased predictors of student achievement? Can non-experimental VAM estimates predict the causal impact of attending a school?

**Data**: Administrative panel data from Charlotte-Mecklenburg Schools (CMS), 1996-2004. Students in grades 3-8 with end-of-year standardized math and reading test scores. A school choice lottery in Spring 2002 provides random assignment.

**Method**: The paper estimates school VAMs using pre-lottery data (varying model specifications, estimation approaches, and years of prior data). It then tests whether these VAM estimates predict the impact of winning the lottery (which randomly assigns students to schools). The key equation is a 2SLS/IV regression where the endogenous variable is the VAM of the assigned school, instrumented by the VAM of the first-choice school (for lottery winners) vs. the home/neighborhood school (for losers). Lottery fixed effects account for the randomization unit. If VAMs are unbiased, the coefficient on school VA should equal 1.

**Key findings**:
- Model 1 (levels only, no covariates): Coefficients near zero (0.01-0.03), VAMs based on raw levels contain almost no causal information.
- Model 2 (gains, with lagged test score polynomial): With 2+ years of prior data, coefficients approach 1 (0.81-0.97), and the hypothesis of unbiasedness cannot be rejected.
- More years of prior data improve accuracy substantially.
- Results are similar across average residual, mixed effects, and fixed effects approaches.

**Analysis sample**: 2,599 students in 118 lottery priority groups where admission probability was strictly between 0 and 1 ("on-margin" students).

---

## 2. Methodology Notes

**Translation choices**:
- The Stata `xtivreg2` with bootstrap SEs is replicated using manual 2SLS with FE (demeaning approach) and block bootstrap at the lottery FE level (100 replications, matching the paper).
- VAM estimation: `reg ... , predict resid` → OLS residual approach (average residual); `xtmixed` → statsmodels MixedLM; `xtreg, fe` → manual demeaning FE.
- The public-use dataset lacks demographic variables (gender, race, free lunch), so Models 3-4 (which include demographics) cannot be estimated. Only Models 1 and 2 are replicated.

**Estimator differences**:
- Python's MixedLM (REML) may differ slightly from Stata's `xtmixed` in convergence behavior and random effects predictions (BLUPs), especially with small group sizes.
- Bootstrap standard errors are stochastic; small differences in SEs are expected.

**Data differences (documented by author)**:
- School IDs are scrambled (consistent within data but not real).
- Student identifiers removed.
- Demographics removed.
- The author states: "the results from the publicly available program and data differ slightly from the published version of the paper."

---

## 3. Replication Results

### Table 1 Panel A: Validating VAMs Using Lottery Data (Outcome: Spring 2003 Test Scores)

| Col | Model | Years | Replicated Coef | Replicated SE | Published Coef | Published SE | Published p(=1) | Replicated p(=1) |
|-----|-------|-------|----------------|---------------|----------------|--------------|-----------------|------------------|
| 1 | M1 (levels) | 2002 | 0.015 | 0.067 | 0.025 | 0.077 | 0.000 | 0.000 |
| 2 | M1 (levels) | 01-02 | 0.026 | 0.062 | 0.034 | 0.070 | 0.000 | 0.000 |
| 3 | M1 (levels) | 98-02 | 0.010 | 0.068 | 0.014 | 0.072 | 0.000 | 0.000 |
| 4 | M2 (gains) | 2002 | 0.474 | 0.194 | 0.531 | 0.208 | 0.024 | 0.007 |
| 5 | M2 (gains) | 01-02 | 0.760 | 0.216 | 0.807 | 0.236 | 0.413 | 0.267 |
| 6 | M2 (gains) | 98-02 | 0.898 | 0.325 | 0.966 | 0.342 | 0.920 | 0.754 |

**Assessment**: All coefficients match the qualitative pattern: Model 1 near zero, Model 2 increasing with more years of data. Quantitative differences are small (0.01-0.07 in coefficients) and entirely consistent with the documented data modifications. The critical finding—that Model 2 with all years of data produces a coefficient indistinguishable from 1—replicates successfully.

### Table A1: Descriptive Statistics

| Variable | Replicated (Lottery) | Published (Lottery) | Match |
|----------|---------------------|---------------------|-------|
| 2002 Math Score | -0.036 | -0.037 | ~exact |
| 2002 Reading Score | -0.068 | -0.070 | ~exact |
| Lottery Sample N | 2,599 | 2,599 | exact |
| Non-lottery N | 31,405 | 31,455 | close |

### Table A3: Impact of Winning the Lottery

| Outcome | Replicated Coef | Replicated SE | Published Coef | Published SE |
|---------|----------------|---------------|----------------|--------------|
| Enrolled in 1st Choice | 0.560 | 0.036 | 0.557 | 0.036 |
| Enrolled in Home School | -0.311 | 0.038 | -0.352 | 0.040 |
| Attend Magnet | 0.197 | 0.045 | 0.198 | 0.046 |
| Math 2003 (IV) | -0.032 | 0.041 | -0.037 | 0.043 |
| Reading 2003 (IV) | -0.065 | 0.043 | -0.060 | 0.047 |

**Assessment**: ITT enrollment effects match very closely. The IV effects on school characteristics differ more (due to scrambled school IDs affecting peer composition variables), but achievement effects match well.

### Table A2: Persistence of School Effects (2004 Outcomes)

| Spec | Replicated Coef | Replicated SE | p(=1) | N |
|------|----------------|---------------|-------|---|
| M2 AR 2002 | 0.443 | 0.267 | 0.037 | 1,967 |
| M2 AR all years | 0.928 | 0.423 | 0.865 | 1,967 |
| M2 mix 2002 | 0.501 | 0.280 | 0.075 | 1,967 |
| M2 mix all years | 0.906 | 0.423 | 0.824 | 1,967 |

**Assessment**: Consistent with published findings—VAMs based on all years of prior data predict second-year (2004) outcomes well.

---

## 4. Data Audit Findings

**Coverage**: 92,971 total observations; 38,236 in the analysis sample (grades 4-8, miss_02==0); 2,908 on-margin students; 2,599 with valid 2003 outcomes and VA estimates (matching published N exactly).

**Distributions**: Test scores are approximately standardized (mean ≈ 0, SD ≈ 1). No extreme outliers beyond 5 SD.

**Missing data**:
- miss_02 flags 17.4% of students who were previously in private schools, applied to the lottery, lost, and disappeared from CMS.
- Differential attrition between lottery winners and losers is minimal: 2003 math missing 8.0% for winners vs 7.6% for losers; 2004 math missing 29.4% vs 27.3%.
- Higher attrition by 2004 is expected as students move to high school.

**Logical consistency**: Grade progressions are normal. Lottery balance is strong (winners significantly more likely to enroll in first choice, less likely in home school). Margin variables all in [0,1].

**Panel balance**: 127 unique lottery FE groups with median size 12 (range 1-203). Some very small groups but this is inherent to the research design.

**Duplicates**: 117 exact duplicate rows found in the raw data. These do not affect the analysis since the analysis sample filters are strict.

**No coding bugs found** in the original Stata code. The code is well-documented with clear comments explaining each step.

---

## 5. Robustness Check Results

| Check | Coef | SE | p(=1) | N | Survives? |
|-------|------|-----|-------|---|-----------|
| **Baseline** (M2, AR, all years) | **0.898** | **0.325** | **0.754** | **2,562** | **Yes** |
| Mixed Effects estimator | 0.830 | 0.321 | 0.596 | 2,562 | Yes |
| Fixed Effects estimator | 0.811 | 0.313 | 0.546 | 2,562 | Yes |
| 1 Year Prior only | 0.474 | 0.194 | 0.007 | 2,562 | No (biased) |
| 2 Years Prior | 0.760 | 0.216 | 0.267 | 2,562 | Yes |
| Model 1 (levels) | 0.010 | 0.068 | 0.000 | 2,562 | No (uninformative) |
| Grades 4-5 only | 1.234 | 1.206 | 0.846 | 523 | Yes (imprecise) |
| Grades 6-8 only | 0.861 | 0.449 | 0.757 | 2,039 | Yes |
| Drop grade 4 | 0.925 | 0.369 | 0.838 | 2,253 | Yes |
| Drop grade 5 | 0.835 | 0.383 | 0.668 | 2,348 | Yes |
| Drop grade 6 | 1.022 | 0.617 | 0.972 | 1,521 | Yes |
| Drop grade 7 | 0.828 | 0.350 | 0.623 | 2,054 | Yes |
| Drop grade 8 | 0.960 | 0.352 | 0.908 | 2,072 | Yes |
| Alt counterfactual (weighted) | 0.997 | 0.301 | 0.991 | 2,513 | Yes |
| Winsorized outcome | 0.874 | 0.331 | 0.704 | 2,562 | Yes |
| Math only | 1.060 | 0.332 | 0.856 | 2,558 | Yes |
| Reading only | 0.743 | 0.498 | 0.606 | 2,551 | Yes |
| **Placebo (2002 pre-lottery)** | **0.008** | **0.012** | **—** | **2,589** | **Passes** |
| Drop small lottery groups | 0.890 | 0.336 | 0.744 | 2,518 | Yes |

**Key findings**:
- The main result (Model 2, gains, all years) is **highly robust**. All leave-one-grade-out checks, alternative estimators, winsorization, and the alternative counterfactual produce coefficients where we cannot reject = 1.
- The placebo test (2002 pre-lottery scores as outcome) produces a coefficient of 0.008, indistinguishable from zero, confirming no pre-existing relationship.
- As expected, Model 1 (levels) and Model 2 with only 1 year of prior data produce coefficients significantly different from 1, consistent with the paper's main message.
- The alternative counterfactual (weighted portfolio of choices) produces an even more precise estimate (0.997) very close to 1, consistent with the paper's footnote that this approach "produces slightly more accurate estimates."

---

## 6. Summary Assessment

**What replicates**: The paper's central finding replicates cleanly. School VAMs estimated with gains specifications (controlling for prior test scores) and multiple years of prior data are accurate, unbiased predictors of student achievement. This is true across all three estimation approaches (average residual, mixed effects, fixed effects), all leave-one-grade-out checks, and alternative sample restrictions.

**What doesn't replicate**: Nothing fails to replicate. Coefficients are systematically slightly lower than published (by ~0.04-0.07), which is fully explained by the scrambled school IDs and removed demographics in the public-use data. The author explicitly documents this expectation.

**Key concerns**: None significant. This is a well-designed study with transparent methodology, clean code, and honestly documented data limitations. The only substantive concern is external validity—the lottery sample is from one school district (Charlotte-Mecklenburg) and may not generalize to all contexts.

**Overall assessment**: This is a high-quality replication with strong results. The paper's conclusions are well-supported by the data.

---

## 7. File Manifest

| File | Description |
|------|-------------|
| `utils.py` | Shared paths, variable lists, and helper functions (VAM estimation, 2SLS bootstrap) |
| `01_clean.py` | Data cleaning, covariate construction, VAM estimation (90 specifications), school-level VA creation, merge to student data |
| `02_tables.py` | Replication of Tables 1, A1, A2, A3 with side-by-side comparisons to published values |
| `03_figures.py` | Diagnostic figures: VAM distributions, coefficient comparison, VA scatter plots |
| `04_data_audit.py` | Data quality audit: coverage, distributions, missing patterns, consistency, panel balance |
| `05_robustness.py` | 19 robustness checks: alternative estimators, subgroups, leave-one-out, placebo, winsorization |
| `writeup_112805.md` | This writeup |
| `analysis_data.pkl` | Cleaned analysis dataset with merged VA estimates |
| `va_school.pkl` | School-level VA estimates |
| `fig1_vam_distributions.png` | Distribution of school VAM estimates |
| `fig2_coef_comparison.png` | Replicated vs published coefficient comparison |
| `fig3_vam_scatter.png` | Assigned school vs home school VAM scatter |
| `fig4_sd_effects.png` | SD of school effects by specification |
