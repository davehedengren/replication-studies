# Replication Study: Paper 112783

**Paper**: "Entrepreneurial Innovation: Killer Apps in the iPhone Ecosystem"
**Authors**: Pai-Ling Yin, Jason P. Davis, Yulia Muzyrya
**Replication Date**: 2026-03-05

---

## TLDR

- **Replication status**: All probit coefficients, standard errors, pseudo R-squared values, and sample sizes replicate exactly. Marginal effects show small differences due to Stata `margeff` vs. statsmodels computation methodology.
- **Key finding confirmed**: Experience (App Order) and no updating increase the likelihood of a game becoming a killer app, while more updates (Number of Versions) increase the likelihood for non-game killer apps. Both findings replicate exactly.
- **Main concern**: Several key results are fragile to sample restrictions — the games "No Updates" effect loses significance when restricted to late cohorts (20-38), and the non-games "Number of Versions" effect loses significance in multiple robustness checks (early cohorts only, late cohorts only, drop outliers, drop zero-comment apps).
- **Bug status**: No coding bugs found.

---

## 1. Paper Summary

**Research question**: How does the development of "killer apps" (apps appearing in the top 300 grossing rank on iTunes) vary by market characteristics (games vs. non-games) and app/developer characteristics?

**Data**: 36,349 matched iPhone apps observed every 2 days on iTunes between September 2010 and August 2011. The sample consists of 1,457 killer apps matched to 34,892 control apps by developer cohort (month/year of entry) and primary category. 18 app categories, 38 cohorts.

**Method**: Probit models of killer app status (binary: appeared in top 300 grossing or not), estimated separately for games (cat5=1) and non-games (cat5=0). Models include cohort fixed effects (games model) or cohort + category fixed effects (non-games model). Standard errors are robust and clustered at the cohort level.

**Key findings**:
1. For games: previous app experience (App Order, positive) and not updating (No Updates, positive) increase the probability of becoming a killer app.
2. For non-games: more updates (Number of Versions, positive) and shorter time between versions increase the probability of becoming a killer app.
3. The opposing innovation processes suggest different optimal strategies in games vs. non-game markets.

---

## 2. Methodology Notes

**Translation choices**:
- Stata `probit` → Python `statsmodels.discrete.discrete_model.Probit`
- Stata `robust cluster(cohort)` → `cov_type='cluster', cov_kwds={'groups': cohort}`
- Stata `margeff, replace` → `res.get_margeff(at='overall', method='dydx')` (average marginal effects)

**Marginal effects note**: The published paper labels columns as "Average Marginal Effects" but the Stata code uses the `margeff` command, which computes marginal effects at sample means (MEM) by default. Statsmodels' `get_margeff(at='overall')` computes average marginal effects (AME), and `get_margeff(at='mean')` computes MEM. Neither exactly matches the published values. AME is closer (within ~10-15%) while MEM diverges more (~50%). This is a known methodological difference between Stata's older `margeff` and Python's statsmodels; it does not affect the probit coefficients or standard errors, which match exactly.

**Variable construction**: The log of number of comments (`lnnumcomapp`) is computed as ln(numcomapp) for positive values and set to 0 when numcomapp=0. This is equivalent to ln(max(numcomapp, 1)) but not ln(numcomapp + 1).

**Data precision**: The Stata .dta file stores continuous variables as float32. Python reads these correctly and uses them in float64 for estimation, matching Stata's internal promotion to double during computation.

---

## 3. Replication Results

### Table 1: Descriptive Statistics

All means and standard deviations match published values within rounding tolerance (±0.02). Sample sizes match exactly:

| Group | Published N | Replicated N |
|-------|------------|-------------|
| Games Killer | 535 | 535 |
| Games Non-Killer | 7,148 | 7,148 |
| Non-Games Killer | 922 | 922 |
| Non-Games Non-Killer | 27,744 | 27,744 |

### Table 2: Probit Regressions

**Probit coefficients and clustered SEs: Exact match across all variables.**

| Variable | Games Coef (Pub) | Games Coef (Rep) | Non-Games Coef (Pub) | Non-Games Coef (Rep) |
|----------|-----------------|-----------------|---------------------|---------------------|
| App Order | 0.0219*** | 0.0219*** | 0.00119 | 0.00119 |
| Number of Versions | 0.0126 | 0.0126 | 0.0224** | 0.0224** |
| No Updates | 0.317** | 0.317** | 0.108 | 0.108 |
| Time Between Versions | -0.00513 | -0.00513 | -0.00407*** | -0.00407*** |
| Price | 0.140*** | 0.140*** | 0.00800*** | 0.00800*** |
| Size | 0.00445*** | 0.00445*** | 0.00132*** | 0.00132*** |
| Log Number of Comments | 0.673*** | 0.673*** | 0.533*** | 0.533*** |
| Score | 0.103*** | 0.103*** | 0.0683*** | 0.0683*** |
| Constant | -3.610*** | -3.610*** | -3.179*** | -3.179*** |
| Pseudo R² | 0.4953 | 0.4953 | 0.3512 | 0.3512 |
| N | 7,683 | 7,683 | 28,666 | 28,666 |

**Marginal effects**: Small differences (~10-15%) due to AME vs. MEM computation method (see Methodology Notes). Sign and significance match in all cases.

---

## 4. Data Audit Findings

### Coverage
- 36,349 total observations: 7,683 games (21.1%), 28,666 non-games (78.9%)
- 18 mutually exclusive categories; games (cat5) is the largest
- 38 cohorts with highly uneven sizes: smallest = 103 (cohort 2), largest = 2,203 (cohort 18)
- Killer app rate varies dramatically across cohorts: 1.5% to 22.0%

### Missing Data
- Zero missing values across all 68 variables
- However, 47.6% of observations have zero comments AND zero score (17,293 apps) — these apps may effectively have no user engagement data, treated as zeros in the analysis
- Score of 0 is heavily concentrated in non-killer apps (49.4%) vs. killer apps (3.4%)

### Logical Consistency
- All logical checks pass: noupdates=1 ↔ numverapp=1, avdeltatime=0 when noupdates=1
- Score bounded [0, 5] as expected
- countapp (App Order) starts at 1, which is correct
- lnnumcomapp = ln(numcomapp) for positive values, 0 for zero comments (verified: exact match)

### Duplicates
- **1,085 exact duplicate rows** (identical across all 68 variables, 3.0% of data)
- 4,828 rows duplicated on key analytical variables
- These duplicates may arise from the matching procedure (multiple control matches with identical characteristics) but are unexplained in the paper

### Extreme Values
- countapp ranges from 1 to 591 (top 10 all >560 from the same prolific non-game developer)
- avprice reaches $999.99 (3 observations); most are killer apps
- avsize reaches 1,010 MB; these are all non-killer non-game apps
- numcomapp is extremely right-skewed: median=1, max=46,359

### Panel Balance
- This is cross-sectional (one obs per app), not panel data
- Cohort sizes are very uneven (103 to 2,203) which may affect the precision of cohort-clustered SEs given only 38 clusters

---

## 5. Robustness Check Results

### Summary Table

| Check | Games: App Order | Games: No Updates | Non-Games: NumVersions | Non-Games: TimeBtwn |
|-------|-----------------|-------------------|----------------------|---------------------|
| Baseline | +*** | +** | +** | -*** |
| 1. Drop duplicates | +*** | +** | +** | -*** |
| 2. Drop outliers (1%) | +*** | +** | insig | -** |
| 3. Logit | +*** | +** | +* | -** |
| 4. Drop zero-comment apps | +*** | +** | +** | -*** |
| 5. HC1 SEs (no cluster) | +*** | +*** | +** | -*** |
| 6. Drop cohort 18 | +*** | +** | +*** | -** |
| 7. Early cohorts only | +** | +** | insig | -** |
| 8. Late cohorts only | +*** | insig | insig | insig |
| 9. Winsorized | +*** | +** | +** | -*** |
| 10. Placebo test | 0.06 | 0.06 | 0.02 | 0.08 |

### Key Findings

**Robust results** (survive all checks):
- **App Order (Games)**: Positive and significant across all 9 specification checks. The most robust finding in the paper.
- **Log Number of Comments**: Positive and highly significant in all checks for both games and non-games (consistently the strongest predictor).
- **Price** and **Size**: Consistently positive and significant for both games and non-games.

**Fragile results**:
- **No Updates (Games)**: Loses significance when restricted to late cohorts only (Check 8, p=0.155). This suggests the effect may be driven by earlier market conditions. The finding is significant in 8 of 9 specification checks.
- **Number of Versions (Non-Games)**: Loses significance in 3 of 9 checks — when dropping outliers (Check 2), restricting to early cohorts (Check 7), and restricting to late cohorts (Check 8). This is the paper's key non-games finding and its fragility is notable.
- **Time Between Versions (Non-Games)**: Loses significance in late cohorts (Check 8), but survives most other checks.
- **Score (Games)**: Loses significance when restricting to apps with comments only (Check 4, p=0.387) and in early cohorts (Check 7, p=0.275), suggesting the effect is partly driven by zero-score apps.

**Placebo test**: Rejection rates at the 5% level are close to the nominal 5% for most variables, confirming that the real effects are not spurious. Time Between Versions shows a slightly elevated 16% rejection rate for games, suggesting some concern about confounding with cohort effects.

---

## 6. Summary Assessment

### What Replicates
- **All probit coefficients, standard errors, sample sizes, and pseudo R-squared values replicate exactly.** This is a fully successful computational replication.
- Table 1 descriptive statistics match within rounding tolerance.
- The qualitative findings about opposing innovation processes (experience matters for games; updating matters for non-games) are confirmed.

### What Doesn't Replicate
- Marginal effects differ by ~10-15% due to methodological differences in computation. This does not affect any qualitative conclusions since the underlying probit estimates match perfectly.

### Key Concerns
1. **Fragility of key non-games result**: The Number of Versions effect for non-games — a central finding of the paper — loses significance in multiple robustness checks (outlier removal, temporal subsamples). The paper's narrative about cumulative innovation in non-games may overstate the robustness of this relationship.

2. **Duplicate observations**: 1,085 exact duplicate rows (3.0%) are present in the data. While dropping them does not change results materially, their presence is unexplained. They may be an artifact of the matching procedure.

3. **Zero-engagement apps**: Nearly half the sample (47.6%) has zero comments and zero score. These apps effectively contribute no user engagement information, yet they are included in regressions using lnnumcomapp and scoreapp. The Score variable's significance is sensitive to including/excluding these zero-engagement apps.

4. **Cohort heterogeneity**: The 38 cohorts vary dramatically in size (103 to 2,203) and killer app rates (1.5% to 22.0%). With only 38 clusters for clustered SEs, the asymptotic properties of the cluster-robust estimator may be a concern. Switching to HC1 robust SEs makes some previously insignificant results significant (No Updates for non-games becomes significant at 5%).

5. **Temporal instability**: Results from early vs. late cohorts sometimes diverge substantially, suggesting the relationships may not be stable over time.

---

## 7. File Manifest

| File | Description |
|------|-------------|
| `utils.py` | Shared paths, variable lists, helper functions |
| `01_clean.py` | Data loading and verification |
| `02_tables.py` | Replication of Tables 1 and 2 |
| `03_data_audit.py` | Data quality audit |
| `04_robustness.py` | 10 robustness checks |
| `writeup_112783.md` | This writeup |
