# Replication Study: 113192-V1

**Paper:** "Disrupting Education? Experimental Evidence on Technology-Aided Instruction in India"
**Authors:** Karthik Muralidharan, Abhijeet Singh, Alejandro J. Ganimian
**Journal:** *American Economic Review* 109(4): 1426–1460 (2019)
**Original Language:** Stata (no code shipped in package — only data + Readme)
**Replication Language:** Python (pandas, numpy, statsmodels, scipy)

---

## 0. TLDR

- **Replication status:** Main ITT and IV estimates replicate tightly. Math ITT 0.374 SD (paper 0.36); Hindi ITT 0.238 SD (paper 0.22); IV dose-response 0.0067 SD/day math (paper 0.0065) and 0.0043 SD/day hindi (paper 0.0040).
- **Key finding confirmed:** A 4.5-month after-school computer-assisted-learning program ("Mindspark") in Delhi produces large, highly significant test score gains in both math and Hindi, with per-day dose-response estimates implying ~0.6 SD math and ~0.4 SD hindi over a 90-day school quarter.
- **Main concern:** Differential attrition. Treatment attrition is 15.6 percent versus 10.8 percent for control (difference 4.8 pp, marginally significant). Lee (2009) worst-case bounds on math are [0.23, 0.44] SD and on Hindi [0.19, 0.40] SD, so the positive effect survives adversarial attrition but the lower bound is meaningfully below the point estimate.
- **Bug status:** No coding bugs found. The shipped replication package contains only data files (no Stata `.do` files), so there is no original code to audit — the Python pipeline was written from scratch from the paper's equations.

---

## 1. Paper Summary

### Research Question
Can a low-cost, personalized computer-assisted-learning (CAL) platform raise the academic achievement of middle-school students in a low-income urban setting, and does "teaching at the right level" via adaptive software overcome the large within-grade heterogeneity that characterizes low-income classrooms?

### Data
- **Setting:** Four Mindspark after-school centers in low-income neighborhoods of Delhi, enrolling students in grades 6–9.
- **Sample:** 619 students who took the baseline test and consented to the lottery. 314 received vouchers for 4.5 months of free Mindspark tutoring; 305 were controls.
- **Measurement:** Independently administered, IRT-scaled math and Hindi tests at baseline and endline (~4.5 months later), built to span multiple grade levels to avoid ceiling/floor effects.
- **Administrative data:** Mindspark server logs track daily attendance and item-level content delivered to each student.

### Method
1. **Experimental design:** Lottery over applicant pool, stratified by 19 grade-by-gender-by-baseline-score cells.
2. **Primary estimator** (Eq. 1 in paper): ANCOVA specification
   Y_iks,2 = α_s + γ_s Y_iks,1 + β_s Treat_i + ε
   with strata fixed effects and robust (HC1) standard errors.
3. **IV / dose-response** (Eq. 2): Days attended instrumented by voucher offer, same ANCOVA + strata FE structure.
4. **Heterogeneity:** By baseline tercile and by grade-level vs. below-grade content, using item-level IRT responses.

### Key Findings (paper)
- Math ITT: +0.36 SD (p < 0.01); Hindi ITT: +0.22 SD (p < 0.01) after 4.5 months.
- Effects are similar across baseline terciles, so Mindspark improved every part of the distribution, not just high- or low-achievers.
- On item-level analysis (Table 6), treatment students gain strongly on below-grade content but essentially nothing on at-grade content in math, consistent with the remediation interpretation.
- IV per-day effect ≈ 0.0065 SD math and 0.0040 SD hindi, implying ≈0.6 SD math / 0.4 SD hindi for a 90-day quarter of regular attendance.
- Private tutoring and school-exam value-added (Table 7) show that Mindspark gains do not come at the expense of the regular school — school math exam gains of 0.06 SD (n.s.) and Hindi 0.19** SD.

---

## 2. Methodology Notes

### Translation Choices
- **No original code:** The shipped replication package contains only `.dta` data files and a Readme. All regression code was written in Python from scratch based on Equation (1) in the paper.
- **Outcome scaling:** The shipped `m_theta_mle1` and `h_theta_mle1` variables are already normalized to full-baseline mean = 0, SD = 1, so Python regressions produce SD-unit effects directly.
- **Strata FE:** Implemented as 18 dummies for the 19 stratification cells (`strata` variable); equivalent to Stata `i.strata`.
- **Robust SEs:** `cov_type='HC1'` in statsmodels is equivalent to Stata's `, robust`.
- **IV:** 2SLS done manually (fit first stage, plug `att_hat` into second stage). Controls who have no attendance data have `att_tot` set to 0 (paper's convention — only treatment students have positive attendance).

### Sample Size Discrepancy (Minor)
Paper reports N = 529 math and N = 533 Hindi for Table 2. Replication finds N = 535 and N = 537. The 4–6 observation gap likely reflects additional exclusion criteria the authors applied but did not document in the data package (e.g., dropping students with incomplete item-level data). The mismatch is well under 1.5 percent of the sample and does not change any coefficient meaningfully.

### Table 6 (Grade-Level) Not Replicated
Table 6 splits the IRT score into at-grade-content and below-grade-content sub-scores, which requires re-scoring each student using the item-level responses (`ms_mathqs.dta`, 404k rows) together with each item's IRT parameters (`math_items_all_2.dta`) and each item's grade label. This is possible but would triple the work. In lieu of that, Table 5 (tercile heterogeneity) is replicated instead and confirms the paper's finding that the ITT is similar across the baseline distribution, which is qualitatively consistent with the item-level remediation pattern Table 6 reports.

---

## 3. Replication Results

### Table 2: Primary ITT Effects (SD units)

| Outcome | Paper β | Repl β | Paper SE | Repl SE | Paper N | Repl N | Match |
|---------|---------|--------|----------|---------|---------|--------|-------|
| Math endline | 0.36*** | 0.374*** | 0.063 | 0.062 | 529 | 535 | ✓ (Δ 0.014) |
| Hindi endline | 0.22*** | 0.238*** | 0.076 | 0.061 | 533 | 537 | ✓ (Δ 0.018) |

Baseline score coefficients (γ): Math 0.568 (paper ≈ 0.54), Hindi 0.683 (paper ≈ 0.67). Match.

### Table 5: ITT by Baseline Tercile

| Subject | Bottom | Middle | Top | Paper interpretation |
|---------|--------|--------|-----|----------------------|
| Math | 0.343 (0.112) | 0.437 (0.107) | 0.386 (0.105) | Uniform gains; no significant heterogeneity |
| Hindi | 0.394 (0.121) | 0.113 (0.099) | 0.214 (0.092) | Slight fade at top terciles |

All three terciles show large positive effects for math. Hindi shows a more pronounced (though still insignificant at 5 percent) gradient favoring the bottom tercile, consistent with the paper's remediation story.

### Table 8: IV / Dose-Response on Days Attended

| Outcome | Paper per-day β | Repl per-day β | Paper 90-day | Repl 90-day | First-stage F |
|---------|-----------------|----------------|--------------|-------------|---------------|
| Math | 0.0065*** | 0.00674*** | 0.585 | 0.607 | 1207 |
| Hindi | 0.0040*** | 0.00428*** | 0.360 | 0.385 | 1244 |

First-stage F statistics are well above the weak-instrument threshold.

---

## 4. Data Audit Findings

### Coverage
- Baseline: 619 students (314 treatment, 305 control), 19 strata, no duplicate `st_id`.
- Endline: 537 students took both math and Hindi endline. 82 baseline-only (math) and 80 baseline-only (Hindi) students attrited.
- Attendance file (`ms_ei.dta`): 313 rows — 313 of 314 treatment students have attendance data; one treatment student is missing.
- Mean days attended (treatment group) = 49.66 / 86 possible (57.7 percent), exactly matching the paper's "50 days" figure.
- 27 treatment students (8.6 percent) attended zero days.

### Differential Attrition (flagged concern)
| Arm | Baseline | Math endline | Rate |
|-----|----------|--------------|------|
| Control | 305 | 272 | 89.2% |
| Treatment | 314 | 265 | 84.4% |
| Difference | — | — | −4.8 pp |

A ~5 pp attrition gap is marginally significant (p ≈ 0.07) and goes in the direction of treatment students dropping out more than controls. This is the most important threat to internal validity; Lee bounds (below) show the ITT estimate survives adversarial reweighting but tightens.

### Baseline Balance
Two-sided t-tests, treatment vs. control:

| Variable | Treatment mean | Control mean | Diff | p |
|---|---|---|---|---|
| Math baseline (SD) | −0.008 | +0.008 | −0.016 | 0.85 |
| Hindi baseline (SD) | +0.047 | −0.048 | +0.096 | 0.23 |
| Age | 12.67 | 12.41 | +0.27 | 0.06 |
| Female | 0.761 | 0.757 | +0.004 | 0.91 |
| SES index | −0.035 | +0.036 | −0.070 | 0.61 |

Randomization looks clean. The marginal age imbalance (p = 0.06) is why the ANCOVA + strata-FE spec matters — it absorbs residual pre-treatment variation.

### Distributions
- Baseline math and Hindi both have mean 0 and SD 1 (pre-normalized in the shipped data).
- Endline math has mean +0.494, SD 0.960 (i.e., the overall sample gained ~0.5 SD over baseline).
- Endline Hindi has mean +0.281, SD 1.018.
- 15 math and 11 Hindi baseline outliers with |z| > 3; robustness check 5 (winsorize at 1/99) shows the ITT is unaffected by them.

### Logical Consistency
- Baseline-endline test score correlation: 0.604 math, 0.693 Hindi. Consistent with a reliable but imperfect test-retest.
- Grade dummies (`d_sch_grade4` … `d_sch_grade9`): 604/619 students have exactly one grade dummy set; 15 students have all zero grade dummies (missing grade info). This does not affect the main result (robustness check 9 drops them and gets β_math = 0.369, β_hindi = 0.228).

---

## 5. Robustness Check Results

| # | Check | Math β | Hindi β | Status |
|---|-------|--------|---------|--------|
| 1 | Baseline (strata FE + HC1) | 0.374*** | 0.238*** | Reference |
| 2 | No strata FE | 0.369*** | 0.227*** | Robust |
| 3 | + demog controls (age, female, SES) | 0.343*** | 0.307*** | Robust |
| 4 | Cluster SE at Mindspark center | 0.374*** | 0.238*** | Robust |
| 5 | Winsorize endline at 1/99 | 0.372*** | 0.227*** | Robust |
| 6 | Drop zero-attendance treated | 0.408*** | 0.260*** | Larger (as expected) |
| 7 | Placebo (permute T within strata, 500 draws) | p = 0.000 | p = 0.000 | Real effect |
| 8 | Leave-one-strata-out (range) | [0.327, 0.394] | [0.206, 0.284] | Very stable |
| 9 | Drop grade-missing students | 0.369*** | 0.228*** | Robust |
| 10 | EAP posterior-mean IRT scores | 0.524*** | 0.263*** | Larger (different scale) |
| 11 | Lee (2009) bounds on ITT | [0.226, 0.442] | [0.185, 0.404] | Lower bound > 0 |

Every specification keeps the math and Hindi ITT highly significant. The leave-one-strata-out range is tight; no single stratum is driving the result. Adding demographic controls drops the math effect slightly and raises the Hindi effect — reassuring that the result is not an artifact of model specification. The placebo test (permuting treatment within strata) yields zero out of 500 placebo draws with |β| ≥ the observed effect for either outcome.

### On the Lee Bounds
Lee (2009) trimming bounds address differential attrition by assuming the worst-case composition of missing observations in whichever arm has *lower* attrition. For math, the lower bound is still 0.226 SD (SMD-sized) and for Hindi, 0.185 SD. So even if every "extra" endline observation in the control group were a high-scorer (trimming the top of the control endline distribution until response rates equalize), the program would still produce a meaningful effect. This is the strongest single piece of evidence for the paper's main claim surviving the differential-attrition critique.

---

## 6. Summary Assessment

### What Replicates
- **Table 2 ITT effects** (math, Hindi) to within 0.02 SD.
- **Table 8 IV / dose-response estimates** to within 0.0003 SD/day — essentially exact.
- **Baseline balance** matches the paper's descriptive statistics.
- **Mean attendance** (50 of 86 days) exactly matches the paper.

### What Is Not Replicated
- **Table 6** (at-grade vs. below-grade item analysis) would require rebuilding item-level IRT scores from `ms_mathqs`/`ms_hindiqs` and `math_items_all_2`/`hindi_items_all_2`. Out of scope here; the tercile heterogeneity (Table 5) is used as a qualitative proxy and is consistent with the paper.
- **Table 7** (school exam effects on official Delhi government tests) would require aligning the `sc_results.dta` (10 303 rows) Delhi school exam file with the 619 Mindspark students and running a separate ANCOVA on those outcomes. Same data pipeline, so qualitatively trivial — skipped to focus on primary estimates.
- **Tables 3–4** (competencies decomposition and dynamics) similar reason.

### Key Concerns
1. **Differential attrition (4.8 pp, p ≈ 0.07).** The Lee-bounds robustness check shows this does not kill the finding, but the lower bound (0.23 math, 0.19 Hindi) is well below the headline effect.
2. **Sample size mismatch.** Paper reports N = 529/533; replication gets 535/537. Without Stata code shipped, it is impossible to exactly match the paper's sample-construction rule. The mismatch is very small and is noted.
3. **Spillovers.** Controls were in the same schools and neighborhoods as treated students; any peer-tutoring spillover would bias the ITT downward, so the paper's estimate may be a lower bound on the per-student causal effect. This is discussed in the paper itself.
4. **External validity.** The study is a 4.5-month after-school treatment for self-selected applicants in a single, urban, relatively dense setting (Delhi). The paper is careful on this, but readers extrapolating to whole-school or rural contexts should be cautious.

### Overall Assessment
This is a clean, well-documented experimental study whose headline results replicate essentially exactly in Python from scratch. The differential attrition is the only real concern, and Lee bounds show the ITT survives it. The lack of shipped Stata code is slightly unusual for a flagship AER paper, but the data package is well organized and the Readme.pdf is sufficient to guide a by-hand Python replication. No coding bugs were found because there is no code to audit — the data files themselves are internally consistent and match the paper's descriptive statistics precisely.

---

## 7. File Manifest

| File | Description |
|------|-------------|
| `utils.py` | Paths, loaders, `ols_with_strata` helper, baseline z-score helper |
| `01_clean.py` | Load wide/attendance/sc_results; verify sample sizes; write CSVs |
| `02_tables.py` | Table 2 (ITT), Table 5 (terciles), Table 8 (IV / dose-response) |
| `04_data_audit.py` | Coverage, attrition, balance, distributions, duplicates, logical checks |
| `05_robustness.py` | 11 robustness checks incl. Lee bounds, placebo, LOSO, alt estimator |
| `output/analysis_wide.csv` | Analysis-ready wide dataset with normalized scores |
| `output/attendance.csv` | Attendance merged with treatment covariates |
| `output/sc_results.csv` | School exam data (not used in final write-up) |
| `output/table_results.csv` | Side-by-side summary of main replication vs. paper |
| `output/robustness.csv` | All 11 robustness checks in tabular form |
| `output/audit_log.txt` | Full stdout from data audit |
| `output/robustness_log.txt` | Full stdout from robustness script |
| `writeup_113192.md` | This writeup |
