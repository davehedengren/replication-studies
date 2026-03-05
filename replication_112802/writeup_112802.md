# Replication Study: Altonji, Kahn, and Speer (2014)

**Paper:** "Trends in Earnings Differentials Across College Majors and the Changing Task Composition of Jobs"
**Published in:** AEA Papers and Proceedings, May 2014
**Replication package:** 112802-V1

---

## 0. TLDR

- **Replication status:** Full replication. All coefficients in Tables 1 and 2 match published values to within rounding precision (max difference: 0.0005).
- **Key finding confirmed:** Earnings differentials across college majors widened ~24% from 1993-2003 and ~13% from 1993-2011; task intensity measures account for about two-thirds of the increase.
- **Main concern:** The 1993-2011 widening (delta_0911=0.018) is not statistically significant under a permutation test (p=0.17), while the 1993-2003 widening is highly significant (p=0.00). The 1993-2011 trend is also driven almost entirely by older workers (age 41-55).
- **Bug status:** No coding bugs found.

---

## 1. Paper Summary

**Research question:** How have earnings differentials across college majors changed over time, and can changing demand for labor market skills (abstract, routine, manual tasks) account for the trends?

**Data:** Pooled cross-sections from the National Survey of College Graduates 1993 (N=85,957) and 2003 (N=47,920), plus the American Community Survey 2009-2011 (N=763,798). Sample restricted to employed, full-time college graduates aged 25-55 with non-missing earnings. Total N=897,675.

**Method:** Weighted least squares regressions of log earnings on standardized major returns (beta^m) interacted with time dummies. Major-specific task intensity measures (abstract, routine, manual from DOT/Autor-Dorn) are used to decompose the variance of major returns over time. Standard errors clustered at the major-by-year level (240 clusters).

**Key findings:**
1. The earnings advantage of a high-paying major (1 SD above mean) increased by 23.5% from 1993-2003, partially reversing to +13.2% by 2009-2011.
2. Changing returns to task measures account for 73% (1993-2003) and 69% (1993-2011) of the increase in inequality across majors.
3. With graduate education controls, rising returns to abstract tasks drive the 1993-2003 increase, while routine tasks dominate the 1993-2011 change.

---

## 2. Methodology Notes

**Translation choices:**
- Python (numpy, pandas, statsmodels) replication of Stata code
- WLS via `statsmodels.regression.linear_model.WLS` with `cov_type='cluster'`
- Stata's probability weights (`[pw=]`) are equivalent to statsmodels WLS weights with robust/cluster SEs
- Weight normalization correction applied for RMSE/variance computation: Stata normalizes weights to sum=N for RMSE calculation; we correct by multiplying `scale` by N/sum(weights)

**Estimator differences:**
- No estimator substitution needed — all regressions are WLS, which is directly available in statsmodels
- Potential experience is demeaned (weighted mean ≈ 17.78) before creating polynomial terms and interactions, matching the Stata code in `paper_results.do`
- First-stage regression of beta^m on A, R, M is run at the individual level with survey-adjusted weights (`newweight`), matching Stata

**Design matrix construction:**
- Survey dummies: surveydum1 (1993), surveydum2 (2003); surveydum3 (2009) omitted as base
- Year dummies: year_2009, year_2010 included; year_2003 dropped (collinear with surveydum2), year_2011 dropped (collinear)
- Graduate education controls: HGC dummies (18, 19, 20; base=16) plus HGC × survey interactions for HGC levels 16, 18, 19 (HGC 20 × survey interactions dropped for collinearity)

---

## 3. Replication Results

### Table 1: Log(Earnings) and Major Characteristics

| Variable | Col 1 (Python) | Col 1 (Pub) | Col 2 (Python) | Col 2 (Pub) | Col 3 (Python) | Col 3 (Pub) | Col 4 (Python) | Col 4 (Pub) |
|---|---|---|---|---|---|---|---|---|
| beta^m / v^m | 0.1359 | 0.136 | 0.1386 | 0.139 | 0.1305 | 0.131 | 0.1395 | 0.140 |
| beta^m/v^m × D03 | 0.0322 | 0.032 | 0.0350 | 0.035 | 0.0297 | 0.030 | 0.0375 | 0.038 |
| beta^m/v^m × D0911 | 0.0177 | 0.018 | 0.0247 | 0.025 | 0.0184 | 0.018 | 0.0308 | 0.031 |
| A^m | — | — | — | — | 0.0540 | 0.054 | 0.0599 | 0.060 |
| A^m × D03 | — | — | — | — | 0.0207 | 0.021 | 0.0284 | 0.028 |
| A^m × D0911 | — | — | — | — | -0.0013 | -0.001 | 0.0153 | 0.015 |
| R^m | — | — | — | — | 0.0682 | 0.068 | 0.0618 | 0.062 |
| R^m × D03 | — | — | — | — | 0.0064 | 0.006 | 0.0002 | 0.000 |
| R^m × D0911 | — | — | — | — | 0.0130 | 0.013 | 0.0047 | 0.005 |
| M^m | — | — | — | — | -0.0273 | -0.027 | -0.0188 | -0.019 |
| M^m × D03 | — | — | — | — | -0.0127 | -0.013 | -0.0089 | -0.009 |
| M^m × D0911 | — | — | — | — | -0.0126 | -0.013 | 0.0017 | 0.002 |
| Grad Controls | X | X | — | — | X | X | — | — |
| R² | 0.2185 | 0.218 | 0.1599 | 0.160 | 0.2188 | 0.219 | 0.1602 | 0.160 |
| N | 897,675 | 897,675 | 897,675 | 897,675 | 897,675 | 897,675 | 897,675 | 897,675 |

**Assessment:** All coefficients match to within 0.001 (consistent with 3-decimal rounding in the paper). R-squared values match to 3 decimal places. Maximum coefficient discrepancy: 0.0005.

### Table 2: Variance Decomposition

| | Grad Controls | | No Grad Controls | |
|---|---|---|---|---|
| | 1993-2003 | 1993-2011 | 1993-2003 | 1993-2011 |
| **Var(β^m_93)** | 0.0186 (pub: 0.019) | | 0.0193 (pub: 0.019) | |
| **% Change** | 54.89% (54.89%) | 27.63% (27.63%) | 60.38% (60.38%) | 38.81% (38.81%) |
| **Abstract** | 46.05% (46.05%) | -5.29% (-5.29%) | 56.78% (56.78%) | 44.67% (44.67%) |
| **Routine** | 12.37% (12.37%) | 52.00% (52.00%) | 0.30% (0.30%) | 11.95% (11.95%) |
| **Manual** | 11.69% (11.69%) | 23.09% (23.09%) | 6.04% (6.04%) | -1.53% (-1.53%) |
| **Total** | 73.36% (73.36%) | 68.51% (68.51%) | 64.82% (64.82%) | 55.93% (55.93%) |
| **Residual** | 26.64% (26.6%) | 31.49% (31.5%) | 35.18% (35.2%) | 44.07% (44.1%) |

**Assessment:** All variance decomposition percentages match exactly. The base variance (0.0186 vs published 0.019) differs by rounding — the paper rounds to 3 decimal places.

---

## 4. Data Audit Findings

### Coverage
- Full sample of 897,675 observations across 51 majors, 5 year-groups
- 7 of 51 majors have zero observations in at least one survey (minor gaps)
- ACS (2009-2011) comprises 85% of observations; weight adjustments upweight the smaller NSCG surveys substantially (NSCG 2003 gets 19.7x weight amplification per observation)

### Distributions
- Log earnings: mean=10.975, SD=0.650, range [6.22, 12.90]
- 0.57% of observations at the $400,000 topcode
- 2,873 extreme low outliers (below Q1 - 3×IQR) in log earnings
- All standardized major-specific variables have weighted mean ≈ 0 and SD ≈ 1 (by construction)

### Logical Consistency
- All sample restrictions (age 25-55, employed, full-time, non-missing) properly applied
- Demographic interaction variables correctly computed (zero mismatches)
- Major-level variables are perfectly constant within major (zero within-major variation)
- Cluster variable (240 groups) matches major × year grouping exactly

### Weight Structure
- The `newweight` variable reweights observations so that each of the three surveys (NSCG93, NSCG03, ACS09-11) gets roughly equal total weight
- This substantially inflates the weight of NSCG observations (12.9x for 1993, 19.7x for 2003) relative to ACS observations (1.1x)
- The extreme weight ratios are intentional (to give each time period equal importance) but mean individual coefficient estimates are sensitive to a few heavily-weighted NSCG observations

### Data Quality Assessment
- No anomalies or quality issues detected
- Data is clean, well-constructed, and suitable for analysis
- The main limitation is the cross-sectional design using different survey instruments across periods, partially addressed by survey fixed effects

---

## 5. Robustness Check Results

### Which findings survive:

1. **Drop outlier majors:** sigma_93=0.1346, delta_03=0.0347, delta_0911=0.0214. Very stable; top/bottom 3 majors do not drive results.

2. **Winsorize earnings (1st/99th percentile):** sigma_93=0.1366, delta_03=0.0290, delta_0911=0.0155. Results moderately robust; winsorization slightly attenuates time trends.

3. **No grad controls:** sigma_93=0.1386, delta_03=0.0350, delta_0911=0.0247. Larger delta_0911 (0.025 vs 0.018), as documented in the paper. The role of graduate education in mediating major returns matters for the 1993-2011 comparison.

4. **Males vs. Females:** Both genders show similar base-year returns (males: 0.145, females: 0.133). Time trends are slightly weaker for females. No evidence of gender-specific divergence driving results.

5. **Levels vs. logs:** sigma_93 corresponds to $9,004 (12.6% of mean earnings); delta_03 is $3,042 (4.3%); delta_0911 is $3,497 (4.9%). Results qualitatively consistent.

6. **Alternative clustering (major level only):** SEs generally similar to major × year clustering. Clustering at the major level produces smaller SEs for sigma_93 and delta_03 but larger SE for delta_0911 (0.0072 vs 0.0059), weakening its significance.

### Which findings are fragile:

7. **Placebo test (permutation):** The 1993-2003 widening is highly significant under permutation (p=0.00 from 100 permutations). However, **the 1993-2011 widening is NOT significant** (p=0.17). This suggests the 1993-2011 result could arise from chance variation in major assignments.

8. **Age subgroups:** Strong heterogeneity. Older workers (41-55) drive the time trends: delta_03=0.039, delta_0911=0.030. Younger workers (25-40) show weak trends: delta_03=0.025, delta_0911=0.006. The 1993-2011 widening essentially disappears for younger workers.

9. **Leave-one-survey-out:** Dropping the 1993 survey substantially changes the baseline return estimate (from 0.136 to 0.105). Dropping 2003 or 2009 has less impact on remaining estimates, suggesting the 1993 survey is particularly influential.

### SE Comparisons:
| | HC1 | HC3 | Cluster (major×year) |
|---|---|---|---|
| sigma_93 | 0.0024 | 0.0024 | 0.0051 |
| delta_03 | 0.0044 | 0.0044 | 0.0077 |
| delta_0911 | 0.0022 | 0.0022 | 0.0059 |

Clustering roughly doubles SEs relative to robust (HC1/HC3) SEs, reflecting that the variation in beta^m is at the major level.

---

## 6. Summary Assessment

### What replicates:
- **All published results replicate exactly.** Every coefficient in Tables 1 and 2 matches to within rounding precision. The code is well-documented and the working data is clean.
- The core finding — that earnings differentials across majors widened from 1993 to 2003 — is robust to outlier majors, winsorization, gender subsamples, and permutation testing.
- The variance decomposition showing that task measures account for ~70% of the variance change is mechanically accurate and replicates precisely.

### What deserves caution:
- **The 1993-2011 trend is less robust than 1993-2003.** It fails the permutation test (p=0.17 vs p=0.00 for 1993-2003) and is driven almost entirely by older workers. For younger workers, the 1993-2011 widening is essentially zero.
- **The weight structure gives outsized influence to NSCG observations.** The NSCG 2003 observations receive ~20x weight amplification relative to ACS observations. While this equal-weighting approach is defensible, it means a few heavily-weighted observations could be influential.
- **The 1993 survey is particularly influential.** Dropping it changes the baseline return estimate by 23%, substantially more than dropping either other survey.
- **The variance decomposition is somewhat mechanical.** The task measures (A, R, M) have a high R² (0.685) in predicting beta^m, so decomposing changes in beta^m into changes in task prices is partly tautological. The decomposition tells us about the covariance structure, not necessarily about causal mechanisms.
- **The sensitivity to graduate education controls** (noted in the paper) is confirmed: whether abstract or routine tasks dominate depends on this specification choice.

### Key concerns:
1. The paper's claim about 1993-2011 widening is more fragile than the 1993-2003 finding
2. Age heterogeneity suggests cohort effects may confound the time trends
3. The reliance on cross-sectional variation across three different surveys with different instruments introduces potential comparability concerns

---

## 7. File Manifest

| File | Description |
|---|---|
| `utils.py` | Shared utilities: data loading, design matrix construction, WLS regression |
| `01_clean.py` | Data verification and summary statistics |
| `02_tables.py` | Replication of Table 1 (regression results) and Table 2 (variance decomposition) |
| `04_data_audit.py` | Data quality audit: coverage, distributions, consistency, missing data |
| `05_robustness.py` | 10 robustness checks: leave-one-out, subgroups, placebo, winsorization, etc. |
| `writeup_112802.md` | This writeup |
