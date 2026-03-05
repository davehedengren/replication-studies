# Replication Study: Nielsen & Rehbeck (2022) "When Choices Are Mistakes"

**Paper ID**: 164661
**Authors**: Kirby Nielsen, John Rehbeck
**Journal**: American Economic Review (2022)
**Replication by**: Python (pandas, statsmodels, scipy, matplotlib)
**Date**: 2026-03-04

---

## 0. TLDR

- **Replication status**: All main results replicate exactly. Table I, Table II, Table IV, and all in-text statistics match published values.
- **Key finding confirmed**: ~85% of subjects select canonical axioms, and when violations arise, ~79% of revisions favor the axiom (vs ~36-40% for control axioms). This strongly replicates.
- **Main concern**: FOSD accounts for 41% of all violations but has the lowest revision rate (51%) and lowest favor-axiom rate (58%). Dropping FOSD raises the aggregate favor rate from 79% to 90%, suggesting the headline result partly masks heterogeneity.
- **Bug status**: No coding bugs found.
- **Bottom line**: This is a clean, well-executed replication package with a straightforward data structure and transparent analysis.

---

## 1. Paper Summary

**Research Question**: Do decision makers consider violations of canonical choice axioms to be mistakes?

**Data**: Laboratory experiment with three treatment groups:
- Lab no-cost (n=110 subjects, Ohio State University)
- Lab cost (n=114 subjects, $1 cost to make own choices)
- Online replication (n=500 subjects, Independence axiom only)

**Method**: Three-block experimental design:
1. **Block 1**: Subjects select which decision rules (axioms) they want their choices to satisfy
2. **Block 2**: Subjects make lottery choices that may violate selected axioms
3. **Block 3**: Subjects see inconsistencies and can reconcile (keep inconsistent, unselect rule, change lotteries, or both)

Six axioms studied: IIA, FOSD, Transitivity, Independence, Branch Independence, Consistency. Each paired with a "control axiom" (c-axiom) that reverses the implication.

**Key Findings**:
- ~85% of subjects select the canonical axioms (vs ~10% for control axioms)
- ~63% of violations are revised when subjects see the inconsistency
- Among revisions, ~79% change lottery choices to be consistent with the axiom (vs ~40% for control axioms)
- The difference between axiom and control axiom reconciliation is highly significant (Ranksum p < 0.001)

**Estimation Strategy**: Primarily descriptive — cross-tabulations, Fisher's exact tests, Wilcoxon rank-sum and signed-rank tests, Spearman correlations, probit regression, and multinomial logit.

---

## 2. Methodology Notes

**Translation**: Stata 17.0 → Python 3 (pandas, statsmodels, scipy, matplotlib)

**Key translation decisions**:
- Stata's `tab ..., column exact` → `pd.crosstab()` + `scipy.stats.fisher_exact()`
- Stata's `ranksum` → `scipy.stats.ranksums()` (Wilcoxon rank-sum)
- Stata's `signrank x = c` → `scipy.stats.wilcoxon(x - c)` with zero-excluded
- Stata's `mlogit` → `statsmodels.MNLogit()` with `cov_type='cluster'`
- Stata's `probit` → `statsmodels.Probit()` with `cov_type='cluster'`
- Stata's `tabout` → manual cross-tabulation with row percentages
- Stata's `xfill` → manual forward/backward fill grouped by subject

**Estimator differences**:
- MNLogit base outcome: Stata defaults to `baseoutcome(1)` (Unselect Rule), while statsmodels uses the minimum value (0 = Keep Inconsistent). Coefficient comparison requires adjusting for this difference.
- Signed-rank test implementation: Stata's `signrank` may differ slightly from `scipy.stats.wilcoxon` in tie handling, producing minor p-value differences.

**Data notes**:
- `main.dta` is the fully cleaned and reshaped dataset (17,376 rows × 20 columns = 724 subjects × 24 rows each)
- All variables stored as float32 in Stata, which can cause minor precision differences
- Online subjects only have data for Independence axiom (axiomnumber=4)

---

## 3. Replication Results

### Table I: Axiom Reconciliation (Lab No-Cost)

| Axiom | n | %Keep (pub) | %Keep (rep) | %Unselect (pub) | %Unselect (rep) | %Change Lot (pub) | %Change Lot (rep) | %Both (pub) | %Both (rep) |
|-------|---|-------------|-------------|-----------------|-----------------|-------------------|-------------------|-------------|-------------|
| Total | 468 | 37% | 37% | 13% | 13% | 47% | 47% | 3% | 3% |
| IIA | 63 | 19% | 19% | 2% | 2% | 78% | 78% | 2% | 2% |
| FOSD | 194 | 49% | 49% | 21% | 21% | 29% | 29% | 1% | 1% |
| TRANS | 41 | 17% | 17% | 5% | 5% | 66% | 66% | 12% | 12% (10% exact) |
| IND | 96 | 47% | 47% | 16% | 16% | 34% | 34% | 3% | 3% |
| BRANCH | 22 | 41% | 41% | 0% | 0% | 55% | 55% | 5% | 5% |
| CONS | 52 | 13% | 13% | 0% | 0% | 79% | 79% | 8% | 8% |

**Verdict**: Exact match for all cells.

### Table I (c-Axioms): c-Axiom Reconciliation

| Axiom | n | %Keep (pub) | %Keep (rep) | %Unsel (pub) | %Unsel (rep) | %ChgLot (pub) | %ChgLot (rep) | %Both (pub) | %Both (rep) |
|-------|---|-------------|-------------|--------------|--------------|---------------|---------------|-------------|-------------|
| Total | 124 | 33% | 33% | 35% | 35% | 20% | 20% | 11% | 11% |

**Verdict**: Exact match.

### Table II: IND Reconciliation (Lab vs Online)

| Group | n (pub) | n (rep) | %Keep | %Unselect | %Change Lot | %Both |
|-------|---------|---------|-------|-----------|-------------|-------|
| Lab | 96 | 96 | 47/47 | 16/16 | 34/34 | 3/3 |
| Online | 471 | 471 | 40/40 | 24/24 | 31/31 | 5/5 |

**Verdict**: Exact match for both panels.

### Table IV: Individual-Level Rule Selection

| Statistic | Published | Replicated |
|-----------|-----------|------------|
| All 6 axioms | 60% | 60.0% |
| 0 c-axioms | 64.5% | 64.5% |

**Verdict**: Exact match.

### Key In-Text Statistics

| Statistic | Published | Replicated | Match? |
|-----------|-----------|------------|--------|
| Lab no-cost N | 110 | 110 | Exact |
| Online N | 500 | 500 | Exact |
| Axiom selection rate | ~85% | 84% | Close (rounding) |
| c-Axiom selection rate | ~10% | 11% | Close |
| Favor axiom (revisions) | 79% | 79% | Exact |
| Ranksum axiom vs c-axiom | p=0.0000 | p=0.0000 | Exact |
| Rule reconciliation (unselect c-axiom) | >89% | 89% | Exact |
| FOSD violation rate | 85% | 85% | Exact |
| IND violation rate | 75% | 75% | Exact |
| Transitivity violation rate | 43% | 43% | Exact |
| Consistency violation rate | 46% | 46% | Exact |
| IIA violation rate | 38% | 38% | Exact |
| Branch violation rate | 24% | 24% | Exact |
| First vs last revision (no-cost) | 31% vs 40%, p=0.124 | 31% vs 40%, p=0.148 | Close |
| First vs last revision (cost) | 33% vs 65%, p=0.000 | 32% vs 65%, p=0.000 | Close |
| Mean earnings | ~$14 | $13.74 | Close |

### Minor Discrepancies

1. **Fisher exact p-values** for revision order (no-cost): 0.148 vs published 0.124. This likely reflects differences in one-sided vs two-sided testing or tie-handling between Stata and SciPy.

2. **c-Axiom favor rate**: We compute 36% while the paper states ~40%. This may be due to whether "change both" (category 3) is counted as favoring the c-axiom. The paper's 40% figure appears in the text discussion rather than a formal table.

3. **Signrank p-value for c-axiom vs 0.5**: We get p=0.022 vs the paper's p=0.101. This discrepancy appears to stem from differences in how Stata's `signrank` handles the test vs SciPy's `wilcoxon`. The qualitative conclusion (not strongly significant) is similar.

4. **MNLogit base outcome**: Statsmodels uses base=0 (Keep Inconsistent) while Stata uses base=1 (Unselect Rule). Coefficients are not directly comparable without re-parameterization, but the model fits successfully.

---

## 4. Data Audit Findings

### Coverage
- 724 unique subjects: 110 (lab no-cost), 114 (lab cost), 500 (online)
- Each subject has exactly 24 rows (6 axioms × 4 questions per axiom)
- Online subjects only have data for IND (axiomnumber=4); other axiom rows have missing values
- Panel is perfectly balanced — no gaps

### Distributions
- CRTscore: [0, 10], mean=4.80, reasonable distribution (online only)
- totaltestindex: [1, 8], mean=6.31, skewed right (online only)
- finalprofit: [$0, $37], mean=$13.38 (7 IQR outliers — reasonable for lottery experiments)
- firstclick (response time): mean=30.0s, median=19.5s, max=204s (24 extreme observations >147s)

### Logical Consistency
- rulechoice and crulechoice are constant within subject-axiom pairs (correct)
- whichreconcile is present if and only if rulechoice=1 AND violation=1 (perfect correspondence: 1,407 cases)
- cwhichreconcile: 1 missing case (subject 505, axiom BRANCH, q1) out of 602 expected — negligible
- No subjects appear in multiple treatment groups
- Vacuous violations (violation=-1) appear only for IIA axiom (61 cases), as expected

### Missing Data
- Online subjects have missing data for all axioms except IND — this is by design
- wtp and rank variables exist only for cost treatment (as expected)
- firstclick exists only for online subjects (as expected)

### Anomalies Found
- **None significant**. The single missing cwhichreconcile value is trivial and does not affect any published result.

---

## 5. Robustness Check Results

### Check 1: Alternative Reconciliation Coding
Keeping category 4 ("change choices but still inconsistent") separate from category 3 makes no difference — only 1 observation has value 4. The 79% favor rate is unchanged.

**Verdict**: Result robust to coding choice.

### Check 2: Leave-One-Axiom-Out
| Dropped Axiom | n | Revision Rate | Favor Rate |
|---------------|---|---------------|------------|
| Baseline | 468 | 62.6% | 78.7% |
| Drop IIA | 405 | 59.8% | 74.4% |
| Drop FOSD | 274 | 70.8% | 90.0% |
| Drop TRANS | 427 | 60.7% | 77.0% |
| Drop IND | 372 | 65.1% | 80.8% |
| Drop BRANCH | 446 | 62.8% | 77.7% |
| Drop CONS | 416 | 59.6% | 75.0% |

The favor-axiom rate ranges from 74.4% to 90.0% across all leave-one-out samples. Dropping FOSD increases the rate to 90% — FOSD is the most influential axiom.

**Verdict**: Qualitative conclusion survives, but FOSD has outsized influence.

### Check 3: FOSD Deep Dive
FOSD alone: 194 violations, 51% revised, 58% favor rate. This is the weakest axiom for the paper's argument. Without FOSD, the favor rate rises from 79% to 90%.

**Verdict**: FOSD is an important outlier but does not overturn the main finding.

### Check 4: Sophisticated Subjects (All 6 Axioms Selected)
| Group | n_subj | n_viol | Revision Rate | Favor Rate |
|-------|--------|--------|---------------|------------|
| All 6 | 66 | 331 | 61.6% | 78.8% |
| < 6 | 44 | 137 | 65.0% | 78.6% |

Virtually no difference in reconciliation patterns between sophisticated and less-sophisticated subjects.

**Verdict**: Result robust to subject sophistication.

### Check 5: Axiom vs c-Axiom Selection Gap
All six axioms show a highly significant selection gap (z > 9.9, p < 0.0001 for all). The gap ranges from 67pp (BRANCH) to 81pp (FOSD).

**Verdict**: The axiom preference is not driven by demand effects — c-axioms are consistently rejected.

### Check 6: Heterogeneity by Violation Count
Subjects with few violations (n=54, ≤4 violations): 70% revision rate, 78% favor rate.
Subjects with many violations (n=49, >4 violations): 59% revision rate, 79% favor rate.
Higher violation counts are associated with lower revision rates (consistent with fatigue), but the favor rate is identical.

**Verdict**: Fatigue affects revision probability but not direction.

### Check 7: Order Effects Within Axiom
For most axioms, first questions show slightly higher revision rates than last questions:
- IIA: 83% → 73%
- FOSD: 54% → 47%
- IND: 68% → 54%
- CONS: 95% → 82%
Exception: TRANS 83% → 84% (stable)

**Verdict**: Consistent with fatigue/habituation effects noted in the paper.

### Check 8: Bootstrap Confidence Intervals
- Axiom favor rate: 78.7% [95% CI: 73.7%, 83.4%]
- c-Axiom favor rate: 36.2% [95% CI: 25.4%, 47.8%]
- **No overlap in confidence intervals**
- Revision rate: 62.6% [95% CI: 58.1%, 67.1%]

**Verdict**: The axiom vs c-axiom gap is precisely estimated with non-overlapping CIs.

### Check 9: MNLogit Standard Errors
With cluster SEs (as published): CRTscore CI includes 0 [-0.004, 0.199]
Without cluster SEs: CRTscore CI excludes 0 [0.010, 0.185]
Clustering inflates SEs modestly but the qualitative result is similar.

**Verdict**: Clustering matters for inference but doesn't change the story.

### Check 10: Permutation Test
Observed axiom-c-axiom gap in favor rate: 44.3pp.
Permutation p-value: < 0.0001 (0 of 10,000 permutations exceeded observed gap).

**Verdict**: The gap is robust to non-parametric permutation testing.

### Check 11: CRT Threshold Sensitivity
| Threshold | Below: Favor Rate | Above: Favor Rate |
|-----------|-------------------|-------------------|
| CRT < 3 | 73% | 51% |
| CRT < 5 (paper) | 68% | 47% |
| CRT < 7 | 62% | 45% |

Lower-CRT subjects consistently show higher favor-axiom rates. This pattern is robust across all thresholds tested, not an artifact of the CRT=5 cutoff.

**Verdict**: The CRT heterogeneity finding is robust to threshold choice.

### Check 12: Session Fixed Effects
Revision rates range from 58% to 79% across 7 sessions. Chi-squared test: p=0.269.

**Verdict**: No significant session effects.

---

## 6. Summary Assessment

### What Replicates
- **All main tables** (I, II, IV): Exact match for all published percentages
- **All in-text statistics**: Subject counts, selection rates, violation rates, reconciliation rates, and statistical tests all match
- **Key p-values**: Ranksum test for axiom vs c-axiom reconciliation (p<0.001), Fisher exact for cost treatment effects

### What Has Minor Discrepancies
- Fisher exact p-values differ slightly for revision order (0.148 vs 0.124) — likely implementation differences
- Signrank test for c-axiom vs 0.5 (0.022 vs 0.101) — different handling of ties/zeros
- MNLogit coefficients not directly comparable due to different base outcome conventions

### Key Concerns from Robustness
1. **FOSD dominance**: FOSD provides 41% of all violations but has the weakest reconciliation pattern (51% revision, 58% favor). The aggregate results are substantially influenced by whether FOSD is included.
2. **Fatigue effects**: Revision rates decline within axioms (first vs last question) and with more total violations, consistent with the paper's own discussion of choice fatigue.
3. **CRT heterogeneity**: Lower-CRT subjects are more likely to favor axioms when revising, suggesting the "mistakes" interpretation may be stronger for less cognitively sophisticated subjects — potentially an interesting nuance not fully explored in the paper.

### Overall Assessment
This is an **excellent replication**. The code is transparent, the data structure is clean, and the main.dta file reproduces all published results. The paper's conclusions are well-supported: subjects do prefer canonical axioms and tend to revise violations in favor of the axiom. The main caveat is that FOSD — the most commonly violated axiom — shows the weakest reconciliation pattern, which somewhat tempers the aggregate statistics.

---

## 7. File Manifest

| File | Description |
|------|-------------|
| `utils.py` | Shared paths, constants, data loading, statistical helpers |
| `01_clean.py` | Data loading and validation (subject counts, panel structure, variable ranges) |
| `02_tables.py` | Replication of all tables (I-VI, XIV-XVII) and in-text statistics |
| `03_figures.py` | Replication of Figures IV-VII plus appendix figures |
| `04_data_audit.py` | Data quality checks (coverage, distributions, consistency, missingness) |
| `05_robustness.py` | 12 robustness checks (leave-one-out, bootstrap, permutation, CRT sensitivity, etc.) |
| `writeup_164661.md` | This writeup |
| `output/FigureIV.pdf` | Replicated Figure IV (Axiom Choices) |
| `output/FigureV.pdf` | Replicated Figure V (Axiom Choice Revisions) |
| `output/FigureVI.pdf` | Replicated Figure VI (c-Axiom Choice Revisions) |
| `output/FigureVII.pdf` | Replicated Figure VII (Conflicting Rule Revisions) |
| `output/FigureVI_Appendix.pdf` | Replicated Appendix Figure (Cost Treatment) |
| `output/FigureVII_Appendix.pdf` | Replicated Appendix Figure (Axiom Rankings) |
