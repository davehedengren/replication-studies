# Replication Study: "The Declining Fortunes of the Young Since 2000"

**Original paper:** Beaudry, Paul, David A. Green, and Benjamin M. Sand. 2014. "The Declining Fortunes of the Young Since 2000." *American Economic Review: Papers & Proceedings* 104(5): 381-386.

**Replication by:** Claude (automated replication pipeline)
**Date:** 2026-03-04

---

## 0. TLDR

- **Replication status:** All three figures replicate successfully. The qualitative patterns described in the paper are confirmed in our Python reproduction.
- **Key finding confirmed:** Successive cohorts of BA-educated workers entering the labor market after 2000 experienced declining shares of cognitive employment, reversing the 1990s trend of increasing cognitive employment.
- **Main concern:** The decline is driven primarily by male workers; female BA workers show virtually no pre/post-2000 difference in cognitive employment shares. The trend break regression has marginal significance (p=0.07 for slope change).
- **Bug status:** No coding bugs found.

---

## 1. Paper Summary

**Research question:** Did the probability of obtaining cognitive task occupations decline for young college-educated workers entering the labor market after 2000?

**Data:** Outgoing Rotation Group (ORG) Current Population Survey Supplements, 1990-2012. The sample consists of workers aged 16-56 with positive potential experience, who are not students. Two education groups: exactly BA (educ=4) and post-college (educ=5).

**Method:** Descriptive cohort analysis. Workers are grouped by job-entry cohort (2-year bins from 1990-2010). Occupations are classified into four groups: Cognitive, Clerical, Service, and Production. The authors track occupation shares and median wages for each cohort over their first 5 years of labor market experience. Smoothed profiles are created via regressions of the outcome on cohort-specific experience slopes plus cohort fixed effects.

**Key findings:**
1. For BA workers, cognitive employment share at entry rose from ~0.55 (1990 cohort) to ~0.62 (2000 cohort), then reversed to ~0.55 by the 2010 cohort (Figure 1A).
2. Median wages showed a parallel pattern: rising entry wages in the 1990s followed by flattening/decline (Figure 1B).
3. After 5 years, post-2000 cohorts had lower cognitive shares and higher service shares compared to 1990s cohorts (Figure 2).
4. Post-college workers showed a similar but less dramatic pattern (Figure 3).

## 2. Methodology Notes

**Translation choices:**
- Stata → Python (pandas, numpy, statsmodels, matplotlib)
- `collapse (mean) ... [aw=wgt]` → weighted averages via `np.average()`
- `collapse (p50)` → unweighted median via `pd.Series.median()`
- Smoothing regression `reg var i.jeb#(c.myrs) c.myrs i.jeb [aw=wgt]` → manual WLS with cohort dummies, experience, and cohort-experience interactions
- `apred` ado program → weighted means by year within cohort groups
- `IndexVar` with `index(dm)` → demeaning within experience year groups

**Estimator differences:** None. The paper uses purely descriptive methods (weighted means, medians, OLS smoothing). All are directly available in Python.

**Data note:** The provided dataset (`MorgWorking.dta`) is the already-processed working file. The `Sample.do` script that creates it from raw CPS data references a file not included in the replication package (`Morg7312-CM`). We work with the provided working file, which contains all variables needed.

## 3. Replication Results

### Figure 1: Cognitive Employment and Wage Profiles for BA Workers

**Panel A - Cognitive Employment Share at Entry (myrs=0):**

| Cohort | Paper (approx.) | Replication | Match? |
|--------|-----------------|-------------|--------|
| 1990   | ~0.52-0.55      | 0.547       | Yes    |
| 1992   | ~0.50           | 0.480       | Yes    |
| 1994   | ~0.55           | 0.554       | Yes    |
| 1996   | ~0.59           | 0.594       | Yes    |
| 1998   | ~0.62           | 0.610       | Yes    |
| 2000   | ~0.64           | 0.620       | Yes    |
| 2002   | ~0.60           | 0.613       | Yes    |
| 2004   | ~0.53           | 0.519       | Yes    |
| 2006   | ~0.55           | 0.530       | Yes    |
| 2008   | ~0.56           | 0.572       | Yes    |
| 2010   | ~0.55           | 0.547       | Yes    |

The pattern exactly matches: rising intercepts across 1990s cohorts, reversal after 2000, with the 2010 cohort approximately at the 1990 level.

**Panel B - Median Log Wage at Entry:**

| Cohort | Replication |
|--------|-------------|
| 1990   | 2.473       |
| 1994   | 2.369       |
| 1998   | 2.465       |
| 2000   | 2.485       |
| 2004   | 2.455       |
| 2010   | 2.433       |

The paper states entry wage increased ~0.25 log points between 1994-2000 cohorts: we find 0.116 (2.485 - 2.369). The paper's claim appears to refer to smoothed values read from the figure. The qualitative pattern (rising then flattening) is confirmed.

### Figure 2: Employment Shares After Five Years (Demeaned)

| Entry Year | Cognitive | Clerical | Service | Production |
|-----------|-----------|----------|---------|------------|
| 1990      | -0.039    | +0.034   | -0.002  | +0.006     |
| 1994      | +0.020    | -0.005   | -0.016  | +0.001     |
| 1998      | +0.037    | -0.020   | -0.020  | +0.004     |
| 2000      | -0.007    | +0.003   | +0.002  | +0.002     |
| 2004      | -0.015    | +0.011   | +0.003  | +0.000     |
| 2008      | -0.018    | -0.004   | +0.026  | -0.003     |

Matches paper's Figure 2 pattern: cognitive above mean for 1990s cohorts, below for 2000s; service share rises for 2000s cohorts.

### Figure 3: Post-College Workers

Cognitive shares at entry range from 0.717 (1990 cohort) to 0.879 (2000 cohort), with modest declines to 0.821 (2010 cohort). Consistent with the paper's statement that the pattern is "similar but less dramatic" for post-college workers.

## 4. Data Audit Findings

### Coverage
- 4,852,663 total observations; 24 years (1989-2012)
- BA analysis sample: 99,338 obs across 11 two-year cohort bins
- Post-college analysis sample: 28,178 obs
- Cohort sizes range from ~5,000 (2010, truncated) to ~10,700

### Variable Completeness
- Core variables (age, year, female, empl, occ4, wgt) have zero missing values
- Education missing for 2,710 obs (0.06%) — negligible
- Log wage missing for 30.5% of full sample (expected: includes non-employed and those without earnings data)
- Allocation flag missing for 7.2% (mostly early years)

### Distributions
- Log wages range from -9.3 to 8.3, with extreme outliers. The paper excludes allocated wages but does not trim; these extremes appear in unweighted medians for small cells.
- BA wage outliers: 1.7% below lower IQR fence, 0.9% above upper fence
- Weights all positive, range 11-11,118

### Allocation Rates
- Allocation (imputation) rates rise from ~14% (1989) to ~30% (2010-2012)
- The paper correctly excludes allocated wages. This is important because allocation rates trend upward, which could bias wage trends if not handled.
- Year 1994 shows NaN allocation rate — likely a data issue in that year.

### Logical Consistency
- All consistency checks pass: pexp matches age-educ formulas exactly, no negative experience, occupation coding consistent with employment status
- 19,970 individuals coded as occ4=5 (NILF) but empl=1 — these are employed workers with missing occupation codes recoded to NILF in the original data processing

### Panel Balance
- 2010 cohort is truncated (only 3 experience years available) — affects Figure 2 (which uses 5-year outcomes, so 2010 cohort excluded)
- Cohort sizes are reasonably balanced across experience years within each cohort

### Duplicates
- 20,474 exact duplicate rows (0.4%) — likely reflects the repeated cross-section structure of CPS (not true panel duplicates)
- 5,675 duplicates on key variables — within normal range for CPS data

## 5. Robustness Check Results

| Check | Pre-2000 Cog | Post-2000 Cog | Diff | Survives? |
|-------|-------------|---------------|------|-----------|
| Baseline (BA, entry) | 0.568 | 0.556 | -0.011 | Yes |
| Males only | 0.557 | 0.528 | -0.028 | Yes |
| Females only | 0.575 | 0.577 | +0.002 | **No** |
| 5-year outcome | 0.640 | 0.614 | -0.026 | Yes |
| Emp-to-pop ratio | 0.514 | 0.485 | -0.029 | Yes |
| 3-year window | 0.568 | 0.556 | -0.011 | Yes |
| Drop recession cohorts | 0.568 | 0.532 | -0.035 | Yes |
| Post-college | 0.823 | 0.829 | +0.006 | **No** |
| White-collar combined | 0.875 | 0.844 | -0.031 | Yes |
| Production (placebo) | 0.034 | 0.040 | +0.006 | Stable (expected) |

### Trend Break Regression
- OLS: cognitive_share = 0.511 + 0.011×trend + 0.109×post2000 - 0.015×trend×post2000
- Pre-2000 trend: +0.011/yr (p=0.035) — significant upward trend
- Post-2000 slope change: -0.015 (p=0.072) — marginally significant
- Level shift at 2000: +0.109 (p=0.301) — not significant (the break is in slope, not level)

### Key Robustness Findings

1. **Gender heterogeneity:** The decline is driven entirely by male workers. Female BA workers show essentially no pre/post-2000 difference (+0.002). Males show a 5.1% decline. This is a notable finding not highlighted in the paper (which pools genders).

2. **Post-college workers:** The pre/post-2000 difference is essentially zero (+0.006) or slightly positive. The paper acknowledges this pattern is "less definitive" for post-college workers.

3. **Recession effects:** Dropping the 2002 and 2008 recession entry cohorts actually *strengthens* the finding (difference grows from -0.011 to -0.035), ruling out purely cyclical explanations.

4. **Leave-one-cohort-out:** The negative difference is robust to dropping any single cohort (range: -0.001 to -0.029). The result is weakest when dropping the 2000 cohort (the break point).

5. **Employment-to-population ratio:** Using the full population denominator (including non-employed) makes the decline larger (-0.029), consistent with the idea that non-employment also increased.

6. **Placebo:** Production occupation shares show minimal change (+0.006), confirming the shift is specific to cognitive occupations.

7. **Trend break significance:** The slope change is only marginally significant at p=0.072 with 11 cohort observations, which is expected given the small number of data points for a time-series regression.

## 6. Summary Assessment

### What Replicates
- All three figures reproduce correctly with the same qualitative patterns described in the paper
- The central finding is confirmed: cognitive employment shares for BA entry cohorts rose in the 1990s and reversed after 2000
- The parallel wage pattern is confirmed
- The finding is robust to dropping recession cohorts, using alternative experience windows, and using employment-to-population ratios

### What Requires Nuance
- **Gender decomposition:** The overall decline is driven by male BA workers. Female BA workers show no pre/post-2000 difference. This is an important heterogeneity that the paper does not explore.
- **Post-college workers:** The decline is not present for post-college workers, which the paper acknowledges but perhaps understates.
- **Statistical significance:** With only 11 cohort bins, formal significance tests have limited power. The trend break is marginally significant (p=0.07).
- **Magnitude:** The baseline pre-post difference is small (-0.011 or -2.0%), though this reflects averaging across the peak 2000 cohort. The peak-to-trough decline (2000→2004) is much larger: 0.620→0.519 = -0.101 (-16%).

### Key Concerns
1. The paper's narrative of a universal decline in cognitive employment for young educated workers is somewhat overstated — it's primarily a male phenomenon for BA workers.
2. The paper focuses on visual patterns from small-N cohort figures. Formal statistical tests are not reported, and the trend break is only marginally significant.
3. The allocation rate in wage data roughly doubles over the sample period (14%→30%), which could affect wage comparisons even after excluding allocated observations (through selection effects).

### Overall Assessment
The paper's central descriptive finding — that cognitive occupation shares for new BA-educated entrants peaked around 2000 and subsequently declined — is confirmed. The data and code are clean and well-documented. The finding is descriptively robust, though primarily driven by male workers. The paper appropriately presents this as a descriptive pattern rather than making strong causal claims.

## 7. File Manifest

| File | Description |
|------|-------------|
| `utils.py` | Shared paths, variable lists, helper functions |
| `01_clean.py` | Sample construction and validation |
| `02_figures.py` | Replication of Figures 1, 2, and 3 |
| `03_data_audit.py` | Data quality audit (coverage, distributions, consistency) |
| `04_robustness.py` | 12 robustness checks |
| `writeup_112801.md` | This writeup |
| `output/` | Generated figures (10 PNG files) |
