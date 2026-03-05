# Replication Study: Charness & Levin (2005)

**Paper**: "When Optimal Choices Feel Wrong: A Laboratory Study of Bayesian Updating, Complexity, and Affect"
**Authors**: Gary Charness and Dan Levin
**Paper ID**: 112319
**Replication Date**: 2026-03-04

---

## 0. TLDR

- **Replication status**: Core results largely replicate with small quantitative discrepancies (typically 1-8 error counts per cell in Tables 1-3), likely due to ambiguous color-to-variable mapping in the data; starting errors and regression patterns match closely.
- **Key finding confirmed**: When reinforcement and Bayesian updating conflict, ~50% of switching decisions after Left draws violate BEU predictions, while Right-draw errors are below 5% -- this central result is confirmed.
- **Main concern**: Small but systematic discrepancies in switching-error counts across Tables 1-3 suggest either an undocumented data coding convention or minor computational differences; no single explanation accounts for all deviations.
- **Bug status**: No coding bugs found -- the paper contains no analysis code, so the replication checks the data against published statistics directly.
- **Bottom line**: The paper's qualitative conclusions are fully supported by the data. Quantitative differences are minor and do not affect any claims.

---

## 1. Paper Summary

**Research Question**: How does the propensity to use Bayes' rule depend on whether reinforcement (emotional affect from outcomes) aligns with or conflicts with Bayesian updating?

**Data**: Laboratory experiment at UCSB with 165 participants across 3 treatments:
- Treatment 1 (59 participants, 60 periods): Baseline with two urns (Left: mixed, Right: all-or-nothing)
- Treatment 2 (54 participants, 60 periods): Simplified Bayesian inference (Left Down urn changed from 3B/3W to 2B/4W)
- Treatment 3 (52 participants, 80 periods): Affect removed from first draw (no payoff, no success/failure labeling)

**Method**: Descriptive statistics (error rates), nonparametric tests (Wilcoxon, binomial), Spearman correlations, OLS and Random Effects regressions.

**Key Findings**:
1. When reinforcement and BEU agree (Right draws), error rates are very low (~4-5%)
2. When they clash (Left draws), error rates approach 50%
3. Simplifying the Bayesian calculation (T2) does not reduce errors
4. Removing affect (T3) substantially reduces errors after favorable draws
5. There is an inverse relationship between error cost and frequency
6. Female participants have higher error rates
7. Individual heterogeneity is substantial, with identifiable "reinforcement types"

---

## 2. Methodology Notes

**Translation choices**: No original analysis code was provided. The replication package contains only the data file (`new_big_data.xls`) and the paper PDF. All analysis was reconstructed from the paper's descriptions.

**Data mapping**: The data uses "Cyan" as the ball color variable rather than "Black/White" as in the paper. For T1/T2, `Cyan_Pays=1` throughout, so `Cyan1=1` maps to a "Black" (valuable) ball. For T3, `Cyan_Pays` varies randomly, and `first_draw_favorable = (Cyan1 == Cyan_Pays)` maps to the paper's favorable/unfavorable classification.

**Treatment identification**: Individuals 1-59 = T1, 60-113 = T2, 114-165 = T3. This was validated by checking Left-Down urn draw probabilities (T1 ~0.50, T2 ~0.33).

**BEU predictions**: Computed from Bayesian posterior probabilities and expected utility calculations as described in the paper and appendices.

**Regression approach**: The paper uses Random Effects for specs (5)-(6); we approximate with OLS clustered by individual, which should yield similar inference.

---

## 3. Replication Results

### Table 1: Treatment 1 Switching-Error Rates

**After Right draw:**

| Draw | Phase I | Phase II | Phase III | Aggregated |
|------|---------|----------|-----------|------------|
| Black | Pub: 12/286 (4.2%), Mine: 12/288 (4.2%) | Pub: 20/596 (3.4%), Mine: 20/596 (3.4%) | Pub: 20/161 (12.4%), Mine: 22/161 (13.7%) | Pub: 52/1043 (5.0%), Mine: 54/1045 (5.2%) |
| White | Pub: 15/304 (4.9%), Mine: 16/302 (5.3%) | Pub: 23/683 (3.4%), Mine: 26/683 (3.8%) | Pub: 7/160 (4.4%), Mine: 7/160 (4.4%) | Pub: 45/1147 (3.9%), Mine: 49/1145 (4.3%) |

**After Left draw (Phases I-II):**

| Draw | Phase I | Phase II | Agg I+II | Phase III |
|------|---------|----------|----------|-----------|
| Black | Pub: 179/332 (53.6%), Mine: 171/332 (51.5%) | Pub: 180/272 (66.2%), Mine: 180/272 (66.2%) | Pub: 359/604 (59.4%), Mine: 351/604 (58.1%) | Pub: 31/157 (19.7%), Mine: 31/157 (19.7%) |
| White | Pub: 98/258 (38.0%), Mine: 98/258 (38.0%) | Pub: 63/219 (28.8%), Mine: 64/219 (29.2%) | Pub: 161/477 (33.8%), Mine: 162/477 (34.0%) | Pub: 17/112 (15.2%), Mine: 17/112 (15.2%) |

**Assessment**: Most cells match exactly or within 1-2 counts. Largest discrepancy: T1 Left Black Phase I (179 vs 171, diff=8). Denominators mostly match; the key qualitative pattern (high Left errors, low Right errors) is clearly replicated.

### Table 4: Starting-Error Rates -- PERFECT MATCH

| Treatment | Phase | BEU-start | Published | Replicated |
|-----------|-------|-----------|-----------|------------|
| 1 | II | R | 491/1770 (27.7%) | 491/1770 (27.7%) |
| 2 | II | R | 351/1620 (21.7%) | 351/1620 (21.7%) |
| 1 | III | L | 321/590 (54.4%) | 321/590 (54.4%) |
| 2 | III | R | 201/540 (37.2%) | 201/540 (37.2%) |

### Table 7: Switching-Error Rates over Time (T3) -- PERFECT MATCH

| Draw | Periods 1-20 | Periods 21-50 | Periods 51-70 |
|------|-------------|---------------|---------------|
| L'B' | Pub: 13.4%, Mine: 13.4% | Pub: 13.9%, Mine: 13.9% | Pub: 13.1%, Mine: 13.1% |
| L'W' | Pub: 46.2%, Mine: 46.2% | Pub: 41.0%, Mine: 41.0% | Pub: 41.5%, Mine: 41.5% |

### Table 8: Regressions

| Variable | Published (1) OLS | Replicated (1) OLS | Published (5) RE | Replicated (5) Clustered |
|----------|------------------|-------------------|-----------------|------------------------|
| Constant | .469*** (.042) | .495 (.041) | .100** (.051) | .120 (.060) |
| Cost | -.928*** (.153) | -.962 (.134) | -.483*** (.090) | -.514 (.078) |
| Left | -- | -- | .226*** (.033) | .205 (.037) |
| Affect | -- | -- | .118*** (.031) | .123 (.036) |
| Female | -- | -- | .079*** (.025) | .076 (.021) |
| Adj R² | .609 | .651 | .252 | .259 |

**Assessment**: All coefficient signs match. Magnitudes are close. Slight differences likely due to: (a) different number of error type categories (28 vs 24 in population-level specs), (b) OLS with clustering vs RE in individual-level specs.

---

## 4. Data Audit Findings

**Coverage**: Complete balanced panel. 10,940 observations = 59×60 + 54×60 + 52×80. No missing data in any raw variable.

**Randomization quality**: State (Up) rates are close to 50% across all treatments (T1: 48.8%, T2: 48.4%, T3: 50.1%). Lag-1 autocorrelation in states is approximately zero.

**Logical consistency**: All checks pass:
- Right urn perfectly reveals state (100% Cyan when Up, 0% when Down)
- Left urn draw rates match theoretical compositions within 95% CIs
- Phase I restrictions enforced correctly (odd=Left, even=Right for T1/T2; always Left for T3)

**Position variable**: Ranges from 1-12 (not 1-6 as might be expected for 6-ball urns). T1 and T3 use only positions 1-6; T2 uses positions 1-12. This likely reflects different web interface layouts between treatments.

**Gender**: 56-60% female across all treatments. Gender is constant within individuals.

**T3 Cyan_Pays**: Approximately 50/50 split (2090 vs 2070), marginally dependent on state (chi2=3.12, p=0.077).

**Anomalies**: One T2 individual always switches urns (100% switch rate). No other extreme patterns.

**Quality assessment**: Data quality is excellent. Clean, complete, well-structured. No coding anomalies or implausible values detected.

---

## 5. Robustness Check Results

### Check 1: Drop Worst Offenders
Removing 7 worst offenders per treatment (~12% of subjects) eliminates ~72% of Right-draw errors (rate drops from ~5% to ~1.5%), consistent with paper's claim of ~75%. Left-draw error rates decline modestly (47.5% to 43.1% for T1).

### Check 2: Bootstrap CIs
Cluster-bootstrapped 95% CIs confirm the wide gap between Left and Right error rates:
- T1 Right: [0.019, 0.077] vs T1 Left: [0.412, 0.524]
- T3 Favorable: [0.094, 0.178] vs T3 Unfavorable: [0.342, 0.506]

### Check 3: Permutation Test for H1
Permutation test (10,000 iterations) yields p=0.0000 for both T1 and T2, strongly confirming Left > Right switching errors.

### Check 4: Gender Subgroups
Female error rates are higher in all treatments. Statistically significant in T2 (MWU p=0.007) and marginally significant in T3 (p=0.084), consistent with the paper's findings.

### Check 5: Early vs Late Learning
No significant difference between early (rounds 21-35) and late (rounds 36-50) Left-draw error rates in Phase II for either T1 (p=0.37) or T2 (p=0.71), confirming the paper's finding that errors do not diminish over time.

### Check 6: Probit Alternative
Probit marginal effects are consistent with OLS: Cost (-0.790***), Left (+0.132***), Affect (+0.097***), Female (+0.081***). Same signs and significance as OLS.

### Check 7: CRRA Risk Aversion
BEU prediction to switch to Right after Left-Black holds for all reasonable CRRA coefficients (rho < 0.9). Only at extreme risk aversion (rho ≥ 0.9) does the prediction flip, confirming the paper's argument.

### Check 8: Leave-One-Out
LOO range for T1 Left errors: [0.465, 0.487] (full: 0.475). No single individual drives the result.

### Check 9: Alternative SEs
HC0 and HC3 robust standard errors are similar to or smaller than OLS SEs, confirming significance.

### Check 10: Right-Draw Placebo
Right-draw errors are significantly > 0 in all phases but remain very low (3.6-9.0%), concentrated among worst offenders. This is expected -- even simple decisions have some noise.

### Check 11: Voluntary vs Forced Starts
Voluntary Left-Black error rates are 12-15 percentage points higher than forced Left-Black error rates (T1: 66.2% vs 51.5%; T2: 48.4% vs 36.8%), supporting the affect channel interpretation.

### Check 12: Binomial Test
103/113 individuals have higher Left than Right error rates (Z=8.75, p<0.001). Paper reports 111/113 (Z=10.44) -- our count is slightly lower (103 vs 111) due to small discrepancies in error classification, but the overwhelming significance is confirmed.

**Summary**: All 12 robustness checks confirm the paper's main findings. No fragile results identified.

---

## 6. Summary Assessment

**What replicates**:
- The central finding that switching errors are dramatically higher after Left draws (~48%) than Right draws (~5%) is fully confirmed
- Starting-error rates replicate perfectly (Table 4)
- Treatment 3 time-segment error rates replicate perfectly (Table 7)
- Regression coefficients have correct signs and similar magnitudes (Table 8)
- All qualitative patterns and hypothesis evaluations are confirmed
- The gender effect and absence of learning over time are robust

**What doesn't replicate exactly**:
- Small discrepancies in switching-error counts in Tables 1-3 (typically 1-8 counts off per cell)
- Spearman correlations (Table 5/6) differ somewhat, likely due to differences in how the cross-period reinforcement measures are computed
- Binomial test count (103/113 vs 111/113) slightly lower

**Key concerns**:
- The absence of original analysis code means the exact computational pipeline cannot be verified
- Small systematic discrepancies in Tables 1-3 suggest either an undocumented data coding convention (perhaps related to the Position variable or a different mapping between Cyan and Black/White) or minor data-processing differences
- None of these concerns affect qualitative conclusions

**Overall assessment**: This is a strong replication. The paper's central claims about the clash between reinforcement and Bayesian updating, the role of affect, and the absence of learning are all well-supported by the data. The experimental design is clean and the data quality is excellent.

---

## 7. File Manifest

| File | Description |
|------|-------------|
| `utils.py` | Shared paths, treatment definitions, BEU predictions, data loading |
| `01_clean.py` | Data loading, validation, and cleaning |
| `02_tables.py` | Reproduction of Tables 1-8 with side-by-side comparisons |
| `03_figures.py` | Reproduction of Figures 1-6 (histograms and scatter plot) |
| `04_data_audit.py` | Data quality audit (coverage, consistency, randomization) |
| `05_robustness.py` | 12 robustness checks |
| `writeup_112319.md` | This replication report |
| `output/` | Generated figures and cleaned data |
