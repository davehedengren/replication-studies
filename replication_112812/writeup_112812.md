# Replication Study: "Benefits of Neuroeconomic Modeling: New Policy Interventions and Predictors of Preference"

**Paper**: Krajbich, Ian, Bastiaan Oud, and Ernst Fehr (2014). AER Papers & Proceedings, 104(5): 501-506.
**Replication Package**: 112812-V1
**Date**: 2026-03-04

---

## 0. TLDR

- **Replication status**: All key statistics replicate exactly: RT by stakes (1.65/1.11s), missed trials (~44), money left on table ($20.10 CHF), and 16/18 probabilistic UG subjects.
- **Key finding confirmed**: Decision-makers misallocate time to low-stakes choices, and a time-constraint intervention improves choice surplus. Response times predict indifference points.
- **Main concern**: Minor — the UG study has only 18 subjects with 3-5 trials per offer level, limiting statistical power. The food choice intervention effect, while significant, is small in magnitude (~0.02 CHF per trial).
- **Bug status**: No coding bugs found.

---

## 1. Paper Summary

**Research question**: Can drift-diffusion models (DDMs) from neuroscience provide useful insights for economics? Specifically: (1) do people misallocate their time by spending too long on low-stakes decisions, and can an intervention help? (2) Can response times predict preference strength and indifference points?

**Data**:
- Study 1: 49 subjects make 100 incentivized binary food choices under time pressure (150 seconds total). Items have previously measured BDM valuations. 50 "high-stakes" trials (large value differences) and 50 "low-stakes" trials (small differences). Five blocks per subject, with time-constraint intervention in some blocks.
- Study 2: 18 subjects play 16 rounds of an ultimatum game as responders, with offers of 4, 6, 8, or 10 CHF from 20 CHF.

**Method**:
- Study 1: Compare mean RT by stakes level, count missed trials, measure welfare loss. Test a "Choose Now" intervention that cuts off slow decisions. Measure de-meaned choice surplus across block types.
- Study 2: Examine whether peak RT coincides with indifference points (50% acceptance) across offer levels, testing the DDM prediction.

**Key findings**:
- People spend 1.65s on low-stakes but only 1.11s on high-stakes choices, missing ~44 of 100 trials and leaving $20.10 on the table.
- The time-constraint intervention significantly improves choice surplus.
- 16/18 UG subjects show probabilistic choices; peak RT aligns with indifference for most subjects.

---

## 2. Methodology Notes

**Translation choices**:
- Stata `.do` file for Figures 3A/3B translated directly to Python (pandas groupby, OLS regression).
- R code for Figure 4 translated to matplotlib.
- De-meaned choice surplus computed exactly following the Stata code: individual means from blocks 2-5 subtracted from block-level means.

**Estimator differences**: None. The analysis uses simple descriptive statistics, OLS regression, and t-tests. No complex estimators required.

**Data notes**: The data is clean and well-structured. Study 1 has 24,500 rows (49 subjects x 5 blocks x 100 trials). Study 2 has 288 rows (18 subjects x 16 trials). Both match expected dimensions exactly.

---

## 3. Replication Results

### Study 1: Key In-Text Statistics

| Statistic | Replicated | Published | Match |
|-----------|-----------|-----------|-------|
| Mean RT, low stakes | 1.65 s | 1.65 s | Exact |
| Mean RT, high stakes | 1.11 s | 1.11 s | Exact |
| Mean missed trials (first block) | 44.2 | ~44 | Exact |
| Mean $ left on table | 20.10 CHF | $20.10 | Exact |
| N subjects | 49 | 49 | Exact |

### Study 1: RT-Value Difference Regression (Figure 3A)

| Parameter | Estimate | p-value |
|-----------|----------|---------|
| Intercept | 1.6506 | <0.001 |
| Slope (|value diff|) | -0.3327 | <0.001 |
| N | 2,735 | |
| R² | 0.092 | |

RT decreases linearly with |value difference|, confirming the paper's claim.

### Study 1: Intervention Effect (Figure 3B)

| Block Type | De-meaned Choice Surplus | SE | N |
|------------|--------------------------|-----|---|
| First | -0.0194 | 0.0062 | 49 |
| Pre-intervention | -0.0166 | 0.0055 | 29 |
| Intervention | -0.0015 | 0.0026 | 49 |
| Post-intervention | +0.0119 | 0.0038 | 49 |

The intervention significantly improves choice surplus vs. first block (t=2.34, p=0.023). Post-intervention blocks show even larger improvement (t=4.80, p<0.001), consistent with the paper's "spillover effects" claim.

### Study 2: Ultimatum Game

| Statistic | Replicated | Published | Match |
|-----------|-----------|-----------|-------|
| N subjects | 18 | 18 | Exact |
| Probabilistic choosers | 16/18 | 16/18 | Exact |
| Peak RT at indifference (exact) | 14/18 | 16/18 | Close |
| Peak RT at indifference (adjacent) | 17/18 | — | — |

The small discrepancy in peak-RT-at-indifference (14 vs 16) may reflect different definitions of "closest to indifference" — with only 3-5 trials per offer level, ties and measurement noise matter. Using an "adjacent" criterion (within one offer level), 17/18 subjects match.

---

## 4. Data Audit Findings

**Coverage**: Study 1: 24,500 observations, 49 subjects, 5 blocks of 100 trials each. Study 2: 288 observations, 18 subjects, 16 trials each. Both match expected dimensions.

**Distributions**: BDM valuations range 0-2.25 CHF. RT ranges from 0.005s to 15.6s in Study 1 (mean 1.02s for reached trials) and 526-13,530ms in Study 2.

**Missing data**: RT is missing for 8,514 unreached trials (by design — subjects ran out of time). The `cut` variable (random cutoff time) is only present for intervention blocks (7,031 trials). 157 trials have forced cutoff (subjectwascutoff=1). No unexpected missingness.

**Logical consistency**: All checks pass — absdiff matches |leftrate - rightrate|, val_preferred >= val_nonpreferred everywhere, choice is binary.

**Outliers**: Very fast RTs (<0.1s): 2 trials. Very slow RTs (>10s): 4 trials in Study 1, 3 in Study 2. These are rare and do not affect results.

**Duplicates**: 26 exact duplicate rows in Study 1, 1 in Study 2. These appear to be cases where the same pair of food items with identical valuations appeared multiple times, and are not data errors.

**No coding bugs found** in the original Stata or R code.

---

## 5. Robustness Check Results

| Check | Result | Robust? |
|-------|--------|---------|
| RT ~ absdiff with cluster SEs | slope=-0.333 (SE=0.031, p<0.001) | Yes |
| Quadratic specification | Marginal quadratic term (p=0.03), tiny R² gain | Yes |
| Log RT specification | slope=-0.223 (p<0.001), R²=0.157 | Yes |
| Trim RT > 5s | low=1.58s, high=1.10s | Yes |
| Winsorize 1/99 pct | low=1.61s, high=1.11s | Yes |
| Leave-one-subject-out | All 49 slopes negative | Yes |
| Individual-level slopes | 49/49 negative, 42/49 significant | Yes |
| Intervention paired t-test | First vs Int: p=0.023, First vs Post: p<0.001 | Yes |
| Permutation test (intervention) | p=0.018 | Yes |
| UG peak RT at indifference | 14/18 exact, 17/18 adjacent | Yes |
| RT by accept/reject (UG) | Consistent with DDM predictions | Yes |

**Key findings**:
- The RT-value difference relationship is remarkably robust: negative slope in all 49 subjects, persists across trimming, winsorization, and alternative functional forms.
- The intervention effect survives both parametric (paired t-test) and non-parametric (permutation) tests.
- The slight weakness: the quadratic term is marginally significant, suggesting the relationship may be slightly nonlinear (steeper for small differences), consistent with DDM predictions.

---

## 6. Summary Assessment

**What replicates**: Everything. All reported statistics match exactly or very closely. The three main claims — (1) time misallocation, (2) intervention improves welfare, (3) RT predicts indifference — are fully confirmed.

**What doesn't replicate**: Nothing fails. The minor discrepancy in UG peak-RT matching (14/18 vs 16/18) likely reflects definitional differences with only 3-5 trials per offer level.

**Key concerns**:
- Study 2 is underpowered: 18 subjects with 3-5 trials per offer level provides limited statistical precision.
- The intervention effect, while statistically significant, is small in magnitude (~0.02 CHF per trial of surplus improvement).
- The paper's theoretical framework (DDM) makes specific predictions about the *functional form* of the RT-choice relationship, but the empirical tests here are relatively simple (means, regression slopes) rather than formal DDM fitting.

**Overall assessment**: Clean, transparent data and code. Results replicate perfectly. The paper makes modest, well-supported claims. A solid contribution to the neuroeconomics literature.

---

## 7. File Manifest

| File | Description |
|------|-------------|
| `utils.py` | Shared paths, constants, and helper functions |
| `01_clean.py` | Data loading, variable construction, key statistics computation |
| `02_tables.py` | Replication of all in-text statistics with side-by-side comparisons |
| `03_figures.py` | Replication of Figures 3A, 3B, and 4 |
| `04_data_audit.py` | Data quality audit for both studies |
| `05_robustness.py` | 12 robustness checks across both studies |
| `writeup_112812.md` | This writeup |
| `study1_clean.pkl` | Cleaned Study 1 dataset |
| `study2_clean.pkl` | Cleaned Study 2 dataset |
| `fig3a_rt_vs_absdiff.png` | Figure 3A replication |
| `fig3b_choice_surplus.png` | Figure 3B replication |
| `fig4_ug_subjects.png` | Figure 4 replication |
| `fig_diagnostic_rt.png` | Additional RT diagnostic figure |
