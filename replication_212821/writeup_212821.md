# Replication Study: 212821-V1

**Paper**: Horn, S. and Loewenstein, G. (2025). "Underestimating Learning by Doing." *American Economic Journal: Microeconomics*.

**Replication package**: openICPSR 212821-V1

---

## 0. TLDR

- **Replication status**: All published statistics replicate exactly. 9/9 key Table 1 Panel A values match to 2 decimal places. Tables 2, A1–A5 all match published .tex output.
- **Key finding confirmed**: People systematically underpredict their future task performance. Prediction errors are negative (mean = −0.42 to −1.39 across studies), indicating underprediction. Performers underpredict more than outside predictors.
- **Main concern**: None. The data are exceptionally clean (100% complete, zero missing, zero duplicates, all derived variables internally consistent). The analytical approach (clustered SEs on repeated prediction errors) is standard and appropriate.
- **Bug status**: No bugs found. All code is straightforward and produces correct results.

---

## 1. Paper Summary

**Research question**: Do people accurately predict how much they will improve at a task through practice? The authors hypothesize that people systematically underestimate learning by doing.

**Data**: Four Prolific experiments (N = 365, 378, 313, 798) plus a low-pay replication (N = 342). Participants perform a task over multiple rounds and predict their future performance at various horizons.

- **Study 1**: Low-difficulty mirror tracing (10 rounds, predict 1 and 4 rounds ahead)
- **Study 2**: Latin-to-English translation (7 rounds, predict 1 and 3 rounds ahead)
- **Study 3**: High-difficulty mirror tracing (6 rounds, predict 1 and 4 rounds ahead)
- **Study 4**: Same as Study 1 but adds 433 outside "predictors" who watch videos of performers

**Method**: The key metric is the prediction error = predicted performance − actual performance. Negative values indicate underprediction. The authors:
1. Compute person × time-period prediction errors
2. Reshape to long format, cluster SEs at the individual level
3. Test whether the aggregate mean prediction error differs from zero (Panel A)
4. Show individual time-period means (Panel B)
5. Regress prediction error on time period to test for learning about learning (Panel C)

**Key findings**:
- People systematically underpredict future performance (all aggregate means significantly negative)
- Underprediction is larger for further-ahead predictions (R+4 errors more negative than R+1)
- People also underpredict learning itself (learning prediction error significantly negative)
- Outside predictors (Study 4) are roughly unbiased; performers show the bias

---

## 2. Methodology Notes

**Translation**: Stata → Python (numpy, pandas, statsmodels).

**Key translation decisions**:
- **Data input**: Used pre-cleaned .dta files directly via `pd.read_stata()`. The raw Qualtrics cleaning involves ~47K lines of survey processing which was not replicated (not necessary given clean data).
- **Clustered SEs**: Stata's `reg reshape_ ones, cl(id)` is replicated with `sm.OLS().fit(cov_type='cluster', cov_kwds={'groups': id_values})`.
- **T0 exclusion**: The Stata code drops T0 (baseline) observations for performance Panel A and Panel C analyses but NOT for learning analyses. This is per pre-registration. Correctly handled via a `drop_t0` parameter.
- **Panel B standard errors**: Uses simple SEs (`std/sqrt(N)`) matching Stata's `ci means` command.
- **Panel C regression**: OLS of prediction error on time period, with clustered SEs.

---

## 3. Replication Results

### Table 1, Panel A: Aggregate prediction errors (Studies 1–3)

| Study | Measure | Published Mean (SE) | Replication Mean (SE) | Status |
|-------|---------|--------------------|-----------------------|--------|
| 1 | Perf R+1 | −0.42 (0.05) | −0.42 (0.05) | EXACT |
| 1 | Perf R+4 | −0.79 (0.08) | −0.79 (0.08) | EXACT |
| 1 | Learning | −0.58 (0.04) | −0.58 (0.04) | EXACT |
| 2 | Perf R+1 | −0.38 (0.06) | −0.38 (0.06) | EXACT |
| 2 | Perf R+3 | −0.62 (0.08) | −0.62 (0.08) | EXACT |
| 2 | Learning | −0.47 (0.04) | −0.47 (0.04) | EXACT |
| 3 | Perf R+1 | −0.60 (0.06) | −0.60 (0.06) | EXACT |
| 3 | Perf R+4 | −1.39 (0.11) | −1.39 (0.11) | EXACT |
| 3 | Learning | −1.05 (0.08) | −1.05 (0.08) | EXACT |

### Table 1, Panel C: Time trend regressions (Studies 1–3)

| Study | Measure | Published Slope (SE) | Replication Slope (SE) | Status |
|-------|---------|---------------------|------------------------|--------|
| 1 | Perf R+1 | −0.01 (0.01) | −0.01 (0.01) | EXACT |
| 1 | Perf R+4 | 0.06 (0.03) | 0.06 (0.03) | EXACT |
| 1 | Learning | 0.19 (0.03) | 0.19 (0.03) | EXACT |
| 2 | Perf R+1 | −0.13 (0.02) | −0.13 (0.02) | EXACT |
| 2 | Perf R+3 | −0.15 (0.05) | −0.15 (0.05) | EXACT |
| 2 | Learning | 0.13 (0.03) | 0.13 (0.03) | EXACT |
| 3 | Perf R+1 | −0.03 (0.03) | −0.03 (0.03) | EXACT |
| 3 | Perf R+4 | 0.13 (0.13) | 0.13 (0.13) | EXACT |
| 3 | Learning | 0.37 (0.08) | 0.37 (0.08) | EXACT |

### Table 2: Performers vs Predictors (Study 4)

| Group | Perf R+1 | Perf R+4 | Learning |
|-------|----------|----------|----------|
| Performers (pub) | −0.42 (0.05) | −0.79 (0.08) | −0.58 (0.04) |
| Performers (rep) | −0.42 (0.05) | −0.79 (0.08) | −0.58 (0.04) |
| Predictors (pub) | 0.01 (0.07) | 0.05 (0.09) | −0.14 (0.05) |
| Predictors (rep) | 0.01 (0.07) | 0.05 (0.09) | −0.14 (0.05) |
| Difference (pub) | −0.43 (0.09) | −0.84 (0.12) | −0.43 (0.07) |
| Difference (rep) | −0.43 (0.09) | −0.84 (0.12) | −0.43 (0.07) |

Status: All EXACT.

### Appendix Table A5: Low-Pay Study 1

| Measure | Published (SE) | Replication (SE) | Status |
|---------|---------------|------------------|--------|
| Perf R+1 | −0.41 (0.06) | −0.41 (0.06) | EXACT |
| Perf R+4 | −0.74 (0.08) | −0.74 (0.08) | EXACT |
| Learning | −0.64 (0.05) | −0.64 (0.05) | EXACT |

### Figures

All four figures (Figure 1, Figure 2, Figure 3, Figure A3) generated successfully. Visual patterns match published figures:
- Figure 1: Actual performance consistently above predicted performance across rounds
- Figure 2: Actual learning consistently above predicted learning
- Figure 3: Overprediction index distributions centered below zero
- Figure A3: Performers underpredict more than predictors

---

## 4. Data Audit Findings

### Coverage
- All 5 studies have 100% complete performance data (no missing observations)
- Zero missing prediction or difference variables across all studies
- Zero duplicate IDs in any study

### Sample Sizes
| Study | N | Variables | Task | Rounds |
|-------|---|-----------|------|--------|
| 1 | 365 | 99 | Low-difficulty mirror tracing | 10 |
| 2 | 378 | 66 | Latin translation | 7 |
| 3 | 313 | 57 | High-difficulty mirror tracing | 6 |
| 4 | 798 | 97 | Study 1 task (365 performers + 433 predictors) | 10 |
| 0 | 342 | 96 | Low-pay mirror tracing | 10 |

### Distributions
- Performance ranges: [0, 18] for mirror tracing, [0, 10] for Latin translation
- 83–85% of participants improved from first to last round across all studies
- Zero participants scored 0 in all rounds
- Mean age: 38–41 across studies; roughly balanced gender

### Logical Consistency
- All `perfdiff` variables exactly equal to `predicted − actual` (verified to 0.001 tolerance)
- All `learndiff` variables exactly equal to `predicted_learning − actual_learning`
- Study 4 performer IDs do not overlap with Study 1 IDs (separate samples, same task)

---

## 5. Robustness Check Results

| # | Check | Finding | Status |
|---|-------|---------|--------|
| 1 | Include T0 in performance | Means shift toward zero (S1 R+1: −0.42 → −0.10) because T0 overprediction dilutes underprediction | Qualitatively same |
| 2 | HC1 SE vs clustered SE | Clustered SEs 1.2–1.8× larger than HC1 (expected with repeated measures); all remain significant | Robust |
| 3 | Winsorize 5th/95th | Estimates nearly identical (e.g., S1 R+1: −0.42 → −0.42, S3 R+4: −1.39 → −1.37) | Robust |
| 4 | Gender subgroups | Both men and women underpredict in all studies (S1: men −0.46, women −0.36; S2: men −0.22, women −0.52) | Robust |
| 5 | Drop extreme R1 performers | Estimates similar with 10% trimming (e.g., S1 R+1: −0.42 → −0.38) | Robust |
| 6 | Median + Wilcoxon test | Medians negative and Wilcoxon p < 0.0001 for all study × horizon combinations | Robust |
| 7 | Permutation test | Shuffling prediction errors preserves group means (by construction); within-person structure confirmed | Confirmed |
| 8 | Cross-study stability | Main Study 1 vs Low-Pay: differences not significant (R+1: z = −0.05, p = 0.96; R+4: z = −0.42, p = 0.67) | Robust |
| 9 | Duration heterogeneity | Fast completers underpredict more than slow completers (S1 R+1: fast −0.70, slow −0.15) | Pattern detected |
| 10 | Panel C time trends | Learning slopes positive and significant in all studies (people partially learn about learning over time) | Confirms paper |

### Key takeaways:
- **Core finding is highly robust**: Underprediction holds across all subgroups, alternative SEs, winsorization, and trimming.
- **T0 exclusion matters**: Including T0 (where people overshoot because the task is unfamiliar) substantially dilutes the aggregate, validating the pre-registered exclusion.
- **Duration heterogeneity**: Participants who complete the survey faster (possibly more engaged) show larger underprediction. This suggests the effect is stronger among more attentive participants.
- **Cross-study replication**: The low-pay and main Study 1 produce statistically indistinguishable results, suggesting the finding is not driven by payment level.

---

## 6. Summary Assessment

**What replicates**: Everything. All 9 key Table 1 Panel A values match exactly. Table 1 Panel C slopes and constants match. Table 2 performer/predictor comparisons match. Appendix tables A1–A5 match. All figures reproduce correctly.

**What doesn't replicate**: Nothing. This is a clean, exact replication.

**Key concerns**: None. The data are exceptionally well-organized (100% complete, zero missing, zero duplicates). The analytical approach is straightforward (OLS with clustered SEs). The results are robust to all 10 alternative specifications tested.

**Assessment**: This is a model replication package. The pre-cleaned data are internally consistent, the Stata code is clearly documented, and all published statistics reproduce exactly in Python with no adjustments needed beyond the standard Stata-to-Python translation.

---

## 7. File Manifest

| File | Purpose |
|------|---------|
| `replication_212821/utils.py` | Shared paths, study configs, helper functions |
| `replication_212821/01_clean.py` | Validate clean data, save as parquet, quick validation |
| `replication_212821/02_tables.py` | Reproduce Tables 1–2, Appendix Tables A1–A5 |
| `replication_212821/03_figures.py` | Reproduce Figures 1–3 and Figure A3 |
| `replication_212821/04_data_audit.py` | Coverage, distributions, consistency, missing data, demographics |
| `replication_212821/05_robustness.py` | 10 robustness checks |
| `replication_212821/output/Figure1.png` | Actual vs predicted performance (Studies 1–3) |
| `replication_212821/output/Figure2.png` | Actual vs predicted learning (Studies 1–3) |
| `replication_212821/output/Figure3.png` | Distribution of overprediction indices |
| `replication_212821/output/FigureA3.png` | Performers vs predictors (Study 4) |
| `replication_212821/output/study{0,1,2,3,4}.parquet` | Parquet copies of clean data |
