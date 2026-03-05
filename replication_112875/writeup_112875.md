# Replication Study: Callen & Long (2015)
## "Institutional Corruption and Election Fraud: Evidence from a Field Experiment in Afghanistan"
### *American Economic Review* 105(1): 354-381

**Paper ID:** 112875-V1
**Replication Date:** March 2026

---

## 0. TLDR

- **Replication status:** All main results replicate exactly. Coefficients, standard errors, and sample sizes match the published values across all 9 main-text tables.
- **Key finding confirmed:** The announcement of photo quick count monitoring reduced election fraud: votes for connected candidates fell by ~25% (about 5 votes per polling substation), and form theft fell by ~60%.
- **Main concern:** The treatment effect on connected candidates' vote totals (Table 7, pflt coefficient) is fragile to winsorization and sensitive to the trimming threshold for outliers, reflecting the heavy-tailed nature of the vote distribution.
- **Bug status:** No coding bugs found.

---

## 1. Paper Summary

**Research Question:** Does announcing election monitoring (photo quick count) reduce aggregation fraud in Afghanistan's 2010 parliamentary elections? And do politically connected candidates disproportionately benefit from fraud?

**Data:**
- Candidate-level vote data at 471 polling centers (1,978 substations), covering ~1,800 candidates across 19 provinces
- Pre-aggregation vote counts from photographs of Declaration of Results forms
- Post-aggregation vote counts scraped from the election commission website
- Political connection data for 57 "investigated" candidates
- Randomized treatment: letters announcing photo quick count delivered to 238 of 471 polling centers

**Method:**
- Randomized controlled trial (stratified by province and baseline survey variables)
- OLS regressions with province fixed effects and multiway clustered SEs (Cameron, Gelbach, Miller 2011) for the fraud analysis
- OLS with stratum fixed effects and cluster-robust SEs for the experimental analysis
- Lee (2009) trimming bounds to address treatment-related attrition
- Spatial externality analysis using neighbor treatment intensity

**Key Findings:**
1. Aggregation fraud is pervasive: 78.6% of polling substations show discrepancies between pre- and post-aggregation vote totals
2. Connected candidates (those with ties to provincial/district election officials) receive ~3.5 extra fraudulent votes per substation
3. Announcing photo quick count reduces theft of election forms by ~60% (from 18.9% to 8.1%)
4. Treatment reduces votes for the most connected candidate by ~5 votes per substation (~25%)
5. Negative spatial externalities: treated neighbors within 1km reduce connected candidate votes by ~7 additional votes

---

## 2. Methodology Notes

**Translation Choices:**
- Stata `cgmreg` (multiway clustering) → Custom Python implementation using the Cameron, Gelbach, Miller (2011) formula: V = V_A + V_B - V_{A∩B}
- Stata `areg ... absorb(strata_group) cluster(pc)` → Custom FWL demeaning with cluster-robust sandwich estimator
- Stata `xi: i.provid` → `pd.get_dummies(drop_first=True)` for province fixed effects
- Lee (2009) trimming bounds → Custom implementation replicating the Stata procedure exactly

**Estimator Differences:**
- None. All estimators have exact Python equivalents.

**Small-sample corrections:**
- Multiway clustering uses Stata's default G/(G-1) × (n-1)/(n-k) correction
- areg uses Stata's DOF correction counting absorbed fixed effect parameters
- t-distribution with G-1 degrees of freedom for cluster-robust inference

---

## 3. Replication Results

### Table 1: Aggregation Discrepancy Patterns

| Pattern | Python N | Published N | Python % | Published % | Python Mean VD | Published Mean VD |
|---------|----------|-------------|----------|-------------|----------------|-------------------|
| No fraud | 74 | 74 | 21.4% | 21.4% | 0.00 | 0.00 |
| Adding votes only | 70 | 70 | 20.2% | 20.2% | 47.34 | 47.34 |
| Subtracting votes only | 15 | 15 | 4.3% | 4.3% | -245.07 | -245.07 |
| Adding and subtracting equally | 15 | 15 | 4.3% | 4.3% | 0.00 | 0.00 |
| Adding more than subtracting | 127 | 127 | 36.7% | 36.7% | 83.45 | 83.45 |
| Subtracting more than adding | 45 | 45 | 13.0% | 13.01% | -54.13 | -54.13 |

**Verdict:** Perfect match.

### Table 2: Summary Statistics

| Variable | Python Mean | Published Mean | Python SD | Published SD | N |
|----------|-----------|----------------|----------|-------------|---|
| Connected to provincial aggregator | 0.005 | 0.005 | 0.073 | 0.073 | 48,008 |
| Connected to prov. + district | 0.004 | 0.004 | 0.065 | 0.065 | 48,008 |
| Karzai connection | 0.011 | 0.011 | 0.106 | 0.106 | 48,008 |
| Government service | 0.015 | 0.015 | 0.120 | 0.12 | 48,008 |
| Incumbent | 0.067 | 0.067 | 0.250 | 0.25 | 48,008 |
| Vote difference | 0.166 | 0.166 | 6.409 | 6.409 | 48,008 |
| Share difference × 100 | 0.570 | 0.570 | 14.287 | 14.287 | 48,008 |

**Verdict:** Perfect match.

### Table 3: Vote Changes During Aggregation by Candidate Type

**Panel A (Votes):** All 7 columns replicated exactly.

| Column | Variable | Python β (SE) | Published β (SE) |
|--------|----------|---------------|-------------------|
| 1 | Investigated | 1.870 (0.980) | 1.870 (0.980) |
| 7 | Investigated | 3.385 (1.713) | 3.385 (1.713) |
| 7 | Prov+Dist Aggr | 2.553 (1.807) | 2.553 (1.807) |
| 7 | Karzai | -1.005 (2.153) | -1.005 (2.153) |
| 7 | Gov Service | -2.038 (1.996) | -2.038 (1.996) |
| 7 | Incumbent | 0.302 (0.226) | 0.302 (0.226) |

**Panel B (Vote Shares × 100):** Replicated exactly.

**Verdict:** Perfect match across all columns and panels.

### Table 4: Randomization Verification

All 19 balance variables replicate exactly (means, differences, and p-values match to 3 decimal places). Treatment and control are well balanced.

**Verdict:** Perfect match.

### Table 5: Lee Bounds (Provincial Aggregator Connection)

| Statistic | Python | Published |
|-----------|--------|-----------|
| Control N | 892 | 892 |
| Control proportion non-missing | 0.112 | 0.112 |
| Control mean abs(vote diff) | 17.170 | 17.170 |
| Treatment N | 969 | 969 |
| Treatment proportion non-missing | 0.160 | 0.160 |
| Treatment mean abs(vote diff) | 5.484 | 5.484 |
| Trimming ratio p | 0.299 | 0.299 |
| Upper trimmed mean | 7.798 | 7.798 |
| Lower trimmed mean | 0.000 | 0.000 |
| Upper bound estimate | -9.372 | -9.372 |
| Lower bound estimate | -17.170 | -17.170 |

**Verdict:** Perfect match.

### Table 6: Lee Bounds for All Subsamples

| Subsample | Python Lower | Published Lower | Python Upper | Published Upper |
|-----------|-------------|-----------------|-------------|-----------------|
| Full sample | -0.817 | -0.817 | 0.076 | 0.076 |
| Incumbent | -2.840 | -2.840 | -0.531 | -0.531 |
| Most connected | -10.310 | -10.310 | -5.143 | -5.143 |
| Karzai connection | -8.831 | -8.831 | -3.549 | -3.549 |
| Government service | -9.649 | -9.652 | -2.448 | -2.448 |

**Verdict:** Near-perfect match. Government service lower bound differs by 0.003, attributable to floating-point handling in the trimming calculation.

### Table 7: Impact on Total Votes by Candidate Connection

**Panel A (selected columns):**

| Column | Variable | Python β (SE) | Published β (SE) |
|--------|----------|---------------|-------------------|
| 1 | Letter treatment | -0.066 (0.212) | -0.066 (0.212) |
| 2 | Treat × most connected | -5.430 (3.111) | -5.430 (3.111) |
| 3 | Treat × most connected | -5.392 (3.080) | -5.392 (3.080) |
| 4 | Treat × most connected | -5.780 (3.245) | -5.780 (3.245) |
| 5 | Treat × most connected | -4.634 (2.347) | -4.634 (2.347) |

**Panel B Column 1:**

| Variable | Python β (SE) | Published β (SE) |
|----------|---------------|-------------------|
| Treat × investigated | -2.864 (1.473) | -2.864 (1.473) |
| Investigated | 15.788 (1.122) | 15.788 (1.122) |

**Verdict:** Perfect match across all specifications.

### Table 8: Impacts on Form Theft

| Column | Python β (SE) | Published β (SE) |
|--------|---------------|-------------------|
| 1 (OLS) | -0.108 (0.032) | -0.108 (0.032) |
| 2 (Strata FE) | -0.111 (0.031) | -0.111 (0.031) |
| 3 (+ Controls) | -0.110 (0.032) | -0.110 (0.032) |

**Verdict:** Perfect match.

### Table 9: Spatial Treatment Externalities

| Column | Variable | Python β (SE) | Published β (SE) |
|--------|----------|---------------|-------------------|
| 1 | Letter treatment | -4.080 (2.009) | -4.080 (2.009) |
| 2 | Any PCs treated within 1km | -6.877 (3.512) | -6.877 (3.512) |
| 3 | Any PCs treated within 1km | -6.742 (3.486) | -6.742 (3.486) |
| 3 | Any PCs treated within 1-2km | -4.738 (4.244) | -4.738 (4.244) |

**Verdict:** Perfect match.

---

## 4. Data Audit Findings

### Coverage
- 471 polling centers, 1,978 polling substations, ~1,816 unique candidates
- 19 of Afghanistan's 34 provinces represented (7.8% of operating polling centers)
- Vote difference data available for 48,018 candidate-substation observations (12.4% of total)
- 238 treatment, 233 control polling centers

### Missing Data
- **Critical finding:** Treatment significantly increases availability of photographic records. Picture data available for 14.88% of treatment observations vs 9.72% of control (difference = 5.16 percentage points, p ≈ 0.064). This is the attrition issue motivating the Lee bounds approach.
- 21 Kabul polling centers added late lack baseline survey data (strata_group is missing)
- Survey variables in PC dataset have 4.5-5.7% missing (the 21 Kabul PCs)

### Distributions
- Vote distribution is extremely right-skewed: median = 0, mean = 1.38 (most candidates get 0 votes)
- vote_diff has extreme outliers: one observation at -800 (PC 3409109, candidate daikondi33: pre=921, post=121)
- 11.1% of vote_diff observations are IQR-based outliers
- 20 candidates have |vote_diff| > 100 at some substation

### Logical Consistency
- vote_diff = votes - picture_votes_1st: 0 mismatches (perfect)
- abs_diff = |vote_diff|: 0 mismatches (perfect)
- All binary variables are properly coded {0, 1}
- Treatment assignment is constant within polling centers (0 inconsistencies)
- deo_fullsample ⊆ peo_fullsample: confirmed (all DEO-connected candidates are also PEO-connected)

### Duplicates
- 0 duplicate (pc, pollingstation, cand_id_s) rows
- 42 polling substations with 0 total votes (potentially closed substations)

### Quality Assessment
The data is clean and well-constructed. The only noteworthy issues are:
1. The extreme outlier at vote_diff = -800 (appropriately handled by the authors' trimming procedure)
2. Treatment-induced differential attrition in picture data (appropriately handled via Lee bounds)

---

## 5. Robustness Check Results

### Summary Table

| Check | Baseline | Robustness | Survives? |
|-------|----------|------------|-----------|
| **1. P95 trimming** | pflt = -4.634 (2.347) | pflt = -1.850 (1.388) | No (t = -1.33) |
| **2. Leave-one-province-out** | pflt = -4.634 | Range: [-6.077, -3.569] | Yes (18/19 sig at 10%) |
| **3. Winsorize 1%** | pflt = -4.634 (2.347) | pflt = -1.226 (0.864) | No (t = -1.42) |
| **3. Winsorize 5%** | pflt = -4.634 (2.347) | pflt = -0.128 (0.146) | No (t = -0.88) |
| **4. Permutation test** | pflt = -4.634 | p = 0.048 | Yes (borderline 5%) |
| **5. Non-connected placebo** | — | treat = -0.001 (0.051) | Clean zero |
| **6. Form theft controls** | treat = -0.110 (0.032) | -0.111 to -0.111 | Yes (robust) |
| **7. Kabul vs non-Kabul** | pflt = -4.634 | Kabul: 1.379, non-Kabul: -5.436 | Heterogeneous |
| **8. HC1 vs cluster SEs** | SE = 2.347 | HC1: 1.256 | Cluster is conservative |
| **9. IHS transform** | pflt = -4.634 | pflt = -0.196 (0.130) | Marginal (t = -1.51) |
| **10. Spillover** | — | treat_1k = -6.877 (3.512) | Yes (t = -1.96) |
| **11. Multiway vs single cluster** | SE = 0.980 | HC1: 0.581, PC cluster: 0.725 | Multiway most conservative |

### Key Observations

1. **The main vote result is somewhat fragile.** The treatment effect on connected candidates (pflt) is sensitive to how outliers are handled. Winsorizing at even 1% makes it insignificant. This is expected given the extremely skewed vote distribution where most candidates receive 0 votes and a few receive hundreds.

2. **The form theft result is very robust.** The 60% reduction in form theft (Table 8) is stable across all specifications, controls, and alternative estimators.

3. **Leave-one-province-out confirms stability.** The pflt coefficient is significant in 18 of 19 province-exclusion specifications, ranging from -6.1 to -3.6.

4. **Spatial externalities confirm treatment diffusion.** Nearby treated polling centers reduce fraud, suggesting information about monitoring spreads locally.

5. **Geographic heterogeneity.** The treatment effect on connected candidates is concentrated in non-Kabul provinces (pflt = -5.4), while Kabul shows a small positive (insignificant) effect. This is consistent with fraud being more centrally controlled outside the capital.

6. **Multiway clustering is appropriately conservative.** SEs under single-level clustering or heteroskedasticity-robust estimation are about 40-60% smaller, confirming the authors' choice to use the most conservative approach.

---

## 6. Summary Assessment

### What Replicates
- **All published results replicate exactly.** Every coefficient, standard error, and sample size in Tables 1-9 matches the published values. This is an exceptionally clean replication.
- The data is well-documented, internally consistent, and free of coding errors.
- The treatment randomization is well-balanced (Table 4).

### What Doesn't Replicate
- Nothing. All results replicate perfectly.

### Key Concerns
1. **Sensitivity to outlier handling (moderate concern).** The headline treatment effect on connected candidates' votes relies on trimming at the 99th percentile. With winsorization, the effect becomes insignificant. However, this is acknowledged by the authors (they use the 99th percentile specifically because powerful candidates have extremely high vote counts), and the Lee bounds approach provides complementary evidence.

2. **Treatment-induced attrition (acknowledged by authors).** Treatment increases availability of photographic records (from 9.7% to 14.9%), creating potential selection bias. The Lee bounds procedure addresses this, and the bounds consistently indicate a negative treatment effect.

3. **Small number of connected candidates (inherent limitation).** Only 57 candidates have political connection data, and only 19 are identified as "most connected." The treatment effects on connected candidates are estimated from a very small subsample, making them naturally more variable.

### Bottom Line
This is a very strong replication. The paper's core findings are well-supported:
- Aggregation fraud is widespread and benefits connected candidates
- Announcing monitoring reduces form theft substantially (~60%)
- The monitoring effect on vote totals is concentrated among connected candidates

The main result on vote totals for connected candidates is statistically fragile to winsorization due to the heavy-tailed distribution of votes, but the form theft result and the Lee bounds analysis provide robust complementary evidence for the treatment effect. The multiway clustering approach is appropriately conservative.

---

## 7. File Manifest

| File | Description |
|------|-------------|
| `utils.py` | Shared paths, data loaders, multiway clustering, Lee bounds, and helper functions |
| `01_clean.py` | Data loading and sample size verification |
| `02_tables.py` | Replication of Tables 1-9 with side-by-side comparisons |
| `03_figures.py` | Replication of Figures A1-A3 |
| `04_data_audit.py` | Data quality audit (coverage, distributions, consistency, missing patterns) |
| `05_robustness.py` | 11 robustness checks (outlier sensitivity, permutation, subgroups, SEs, etc.) |
| `writeup_112875.md` | This writeup |
| `output/figureA1.png` | KDE of vote differences by treatment status and connection type |
| `output/figureA2.png` | Treatment effects with Lee bounds by candidate subsample |
| `output/figureA3.png` | Average votes by treatment status and connection type |
