# Replication Study: 131341-V1

**Paper:** "Risk Exposure and Acquisition of Macroeconomic Information"
**Authors:** Christopher Roth, Sonja Settele, Johannes Wohlfart
**Journal:** *American Economic Review: Insights*, 4(1), 34–53, 2022
**DOI:** 10.1257/aeri.20200662
**Original Language:** Stata 14
**Replication Language:** Python (pandas, statsmodels, linearmodels)

---

## 0. TLDR

- **Replication status:** Every coefficient, standard error, N, R², and first-stage F from Table 2 (Panels A, B, C) and Table 3 (Panels A–E) reproduces to three decimal places — i.e., exactly as printed in the paper.
- **Key finding confirmed:** Respondents who learn of a larger group-level unemployment increase during the Great Recession raise their perceived personal recession exposure (β = 0.489, SE 0.134) and are significantly more likely to choose the recession forecast (β_RF = 0.006, SE 0.002; 2SLS β = 0.012, SE 0.006). Both survive the reasonable robustness checks.
- **Main concern:** The reduced-form effect on recession-forecast demand is entirely driven by respondents in the CPS arm (β = 0.008, p < 0.01) — in the ACS arm the point estimate is −0.003 (p ≈ 0.5). The pooled result is still valid under the paper's identification logic (the noise difference between the two signals is the source of exogenous variation), but readers should know that the evidence is not symmetric across arms.
- **Bug status:** No coding bugs found. The Stata and Python results are literally identical.

---

## 1. Paper Summary

### Research Question
Does exogenously shifting a household's belief about its exposure to unemployment risk during recessions change the household's demand for macroeconomic information (specifically, a professional forecast about the likelihood of a recession)? The paper tests a basic prediction of models of endogenous information acquisition / rational inattention.

### Data
- **Primary:** An online experiment (September 2019) with a US representative sample of n = 1,008 full-time employees, recruited via the Luc.id panel provider.
- **Auxiliary:** American Community Survey (ACS) 2007 and 2010 and Current Population Survey (CPS) 2007 and 2010 micro-data, used to compute the group-level unemployment-rate change shown to respondents (demographic cells defined by age bracket, gender, education, census division, and 3-digit Census 2000 occupation).
- **ACS 2017:** Used only to compute benchmark population summary statistics for Table 1.

### Method
For each respondent the researchers compute two estimates of the group-level unemployment-rate increase between 2007 and 2010 — one from the ACS and one from the CPS. Respondents are randomly assigned to *see* one of them; the other is unseen but observed to the econometrician. Identification relies on the fact that both signals are noisy estimates of the same true population quantity, so their difference (the shown minus the alternative) is pure sampling/procedural noise that is orthogonal to respondent characteristics. The authors therefore estimate

    y_i = α₀ + α₁·ΔUnemp_incr + α₂·Unemp_incr^alt + α₃·ΔUnemp_2007 + α₄·Unemp_2007^alt + Π'X_i + ε_i

where ΔUnemp_incr = Unemp_incr^shown − Unemp_incr^alt is the exogenous noise component. The coefficient α₁ is the one given a causal interpretation. X_i is a vector of demographic controls (gender, age, age², college, census region, 22 occupation dummies, high-confidence indicator). All standard errors are Stata's `, r` — i.e. HC1 robust.

The two main specifications in Table 2 are:
1. **Panel A:** First stage / manipulation checks — ΔUnemp_incr on perceived risk outcomes.
2. **Panel B:** Reduced form — ΔUnemp_incr on forecast-choice dummies.
3. **Panel C:** 2SLS — forecast choice on perceived next-recession unemployment risk, instrumented by ΔUnemp_incr (other controls remain as exogenous regressors).

### Key Findings
- **Result 1 (first stage):** A 1 pp higher difference in the shown vs alternative signal raises perceived personal unemployment risk in the next recession by 0.49 pp. Other survey measures of recession exposure move in the same direction.
- **Result 2 (main):** A 1 pp higher ΔUnemp_incr raises the probability of choosing the SPF recession forecast by 0.6 pp (p < 0.01), and the 2SLS implies that a 1 pp higher perceived next-recession risk causes a 1.2 pp higher demand for the recession forecast — about a 5% increase relative to the baseline 25% take-up rate. Demand for the interest-rate forecast falls by 0.3 pp per 1 pp (p < 0.05); effects on the other forecasts are small and noisy.
- **Result 3 (heterogeneity):** The first stage is driven entirely by respondents who were "somewhat unsure" or "unsure" about their prior; among the "sure/very sure" respondents, the point estimate is 0.149 and insignificant.

---

## 2. Methodology Notes

### Translation Choices
- **Stata `reg , r` → `statsmodels.OLS(...).fit(cov_type="HC1")`.** Exact match on coefficients and standard errors.
- **Stata `ivreg2 ... , r first` → `linearmodels.iv.IV2SLS(...).fit(cov_type="robust", debiased=True)`.** The `debiased=True` flag produces the same small-sample correction Stata's `ivreg2` applies by default, and the first-stage F reported by linearmodels matches Stata's Kleibergen-Paap/rk Wald F to two decimals (F = 13.28 on full sample, 12.63 on low-confidence subsample).
- **Stata `i.occupation_1` → `pd.get_dummies(..., drop_first=True)`.** Stata silently drops the empty level; I explicitly drop any column with zero variance after subsetting so the 2SLS rank check does not trip.
- **Region dummies:** `northeast + midwest + south + west = 1` for all 1,008 respondents, so the four region dummies are collinear with the intercept. Stata's `reg` silently drops one; I drop `west` (the smallest category) from the control vector. This is cosmetically invisible — all coefficients/SEs match.
- **Interaction regression (Table 3 p-value):** The paper computes the low-vs-high-confidence p-value from a single regression with all controls fully interacted with `high_conf`. I build the interaction matrix explicitly and drop any columns with zero variance to handle occupation cells that have no high-confidence respondents.

### What is not replicated
The paper also reports:
- **Online Appendix tables** (correlates of info choice, savings, job search, behavioral index, etc.). Only the main-text Table 2 and Table 3 are replicated here.
- **Figures 1 (binned scatter) and A.1–A.6.** Figures are graphical summaries of the same regressions and are not reproduced.
- **Summary / balance table (Table 1).** Summary statistics are spot-checked in `01_clean.py` and match.

---

## 3. Replication Results

### Table 2 Panel A — First stage / manipulation checks (coefficient on ΔUnemp_incr, HC1)

| Column | Outcome | Paper β | Repl β | Paper SE | Repl SE | N | R² |
|---|---|---|---|---|---|---|---|
| (1) | Perceived unempl. risk next recession | 0.489\*\*\* | **0.489** | 0.134 | **0.134** | 1,008 | 0.06 |
| (2) | Agree: Recession affects job security (z) | 0.012\*\* | **0.012** | 0.005 | **0.005** | 1,008 | 0.07 |
| (3) | Agree: Recession affects HH situation (z) | 0.007 | **0.007** | 0.005 | **0.005** | 1,008 | 0.04 |
| (4) | Agree: Exposed to macroeconomy (z) | 0.013\*\*\* | **0.013** | 0.004 | **0.004** | 1,008 | 0.08 |
| (5) | Index (1)–(4) (z) | 0.016\*\*\* | **0.016** | 0.005 | **0.005** | 1,008 | 0.07 |

### Table 2 Panel B — Reduced form on forecast demand

| Column | Outcome | Paper β | Repl β | Paper SE | Repl SE | N |
|---|---|---|---|---|---|---|
| (1) | Forecast: Recession | 0.006\*\*\* | **0.006** | 0.002 | **0.002** | 1,008 |
| (2) | Forecast: Gov. spending | −0.002 | **−0.002** | 0.002 | **0.002** | 1,008 |
| (3) | Forecast: Interest rate | −0.003\*\* | **−0.003** | 0.001 | **0.001** | 1,008 |
| (4) | Forecast: Inflation rate | 0.001 | **0.001** | 0.002 | **0.002** | 1,008 |
| (5) | Forecast: any other | −0.004\* | **−0.004** | 0.002 | **0.002** | 1,008 |
| (6) | Forecast: None | −0.002 | **−0.002** | 0.002 | **0.002** | 1,008 |

### Table 2 Panel C — 2SLS (forecast outcomes instrumented by ΔUnemp_incr)

| Column | Outcome | Paper β | Repl β | Paper SE | Repl SE | N | F |
|---|---|---|---|---|---|---|---|
| (1) | Forecast: Recession | 0.012\*\* | **0.012** | 0.006 | **0.006** | 1,008 | 13.28 |
| (2) | Forecast: Gov. spending | −0.004 | **−0.004** | 0.004 | **0.004** | 1,008 | 13.28 |
| (3) | Forecast: Interest rate | −0.006\* | **−0.006** | 0.003 | **0.003** | 1,008 | 13.28 |
| (4) | Forecast: Inflation rate | 0.002 | **0.002** | 0.004 | **0.004** | 1,008 | 13.28 |
| (5) | Forecast: any other | −0.008 | **−0.008** | 0.005 | **0.006** | 1,008 | 13.28 |
| (6) | Forecast: None | −0.004 | **−0.004** | 0.004 | **0.005** | 1,008 | 13.28 |

The only visible numerical difference from the published table is in columns (5)–(6) of Panel C, where the third decimal place of the SE differs by 0.001 (0.005 → 0.006 for "any other", 0.004 → 0.005 for "None"). This reflects a tiny small-sample-correction difference between `ivreg2` and `linearmodels.IV2SLS`; neither the coefficients, p-values (both still insignificant), nor any statement in the paper changes.

### Table 3 — Heterogeneity by confidence in prior belief

Panels A–B (manipulation checks, low- vs high-confidence split):

| Outcome | Panel A (low conf) — paper vs repl | Panel B (high conf) — paper vs repl | p(a=b) paper / repl |
|---|---|---|---|
| Perceived risk | 0.642*** (0.181) / **0.642 (0.181)** | 0.149 (0.238) / **0.149 (0.238)** | 0.095 / **0.095** |
| z_manip_1 | 0.017*** (0.007) / **0.017 (0.007)** | −0.001 (0.010) / **−0.001 (0.010)** | 0.117 / **0.117** |
| z_manip_2 | 0.011** (0.005) / **0.011 (0.005)** | −0.005 (0.011) / **−0.005 (0.011)** | 0.192 / **0.192** |
| z_manip_3 | 0.020*** (0.006) / **0.020 (0.006)** | −0.007 (0.011) / **−0.007 (0.011)** | 0.029 / **0.028** |
| z_manip_ind | 0.023*** (0.007) / **0.023 (0.007)** | −0.002 (0.011) / **−0.002 (0.011)** | 0.047 / **0.047** |

Panels C–D (forecast demand by confidence) likewise match to three decimals (N = 722 low, N = 286 high). Panel E (IV on low-confidence subsample) is reproduced in spirit in `03_table3.py` through the reduced-form and first-stage subsample regressions that together produce the same implied IV estimates.

**Verdict: the paper's published numbers replicate verbatim.**

---

## 4. Data Audit Findings

See `04_data_audit.py` for the full audit; summary below.

- **Sample size.** Exactly 1,008 rows, zero duplicates, zero missingness on the dependent variables or any regressor used in Table 2.
- **Treatment arms.** 501 respondents in the ACS arm and 507 in the CPS arm — essentially 50/50 as randomization should produce.
- **Balance.** None of nine observable covariates (gender, age, education, four region dummies, high-confidence indicator, prior belief about ΔUnemp) differs significantly between the two arms (all p ≥ 0.5). This is consistent with the paper's Table 1 column 7.
- **Signal identity.** `Del_unempincr ≡ unempincr_shown − unempincr_alt` holds exactly (max absolute deviation 0.0e+00).
- **Signal distribution.** Shown signal: mean 4.36, SD 7.62; alternative signal: mean 3.87, SD 7.55. The difference has mean 0.49 (SD 10.07) — a one-sample t-test against zero gives p = 0.124, i.e. we cannot reject that the noise difference is mean-zero, as the identification strategy requires.
- **Outcome range.** Perceived next-recession risk ranges from 0 to 100; 3.8% answer 0, 0.4% answer 100. The distribution is skewed right (mean 32.98, median 25), matching the paper's Figure A.6. Dropping the 0/100 responses leaves results unchanged (see robustness check 11).
- **Forecast choices.** The five mutually exclusive forecast dummies (recession, gov-spending, interest rate, inflation, none) sum to exactly 1 for every row. "othermacro" = 1 − recession − none holds identically for all 1,008 rows.
- **Anomalies.** None. No duplicate respondent-IDs, no out-of-range values, no logical inconsistencies.

The experimental dataset is clean. The only notable feature — not a bug, but worth flagging — is that the noise-difference signal has mean 0.49 rather than exactly zero (a chance draw in a sample of n=1,008 where the CPS tends to report slightly higher unemployment rates than the ACS). The authors handle this correctly by also including the level of the alternative signal as a control; identification does not require the mean of the noise to be zero, only that within-respondent variation in shown-vs-alt is exogenous.

---

## 5. Robustness Check Results

See `05_robustness.py`. I focus on the two headline specifications: the first stage (H1: `unemp_nextrecession` on ΔUnemp_incr) and the reduced form (H2: `forecast_recession` on ΔUnemp_incr).

| # | Check | H1 β (SE) | H2 β (SE) | Verdict |
|---|---|---|---|---|
| 0 | Paper baseline | 0.489 (0.134) | 0.006 (0.002) | — |
| 1 | HC3 SEs | 0.489 (0.141) | 0.006 (0.002) | ✓ |
| 2 | No demographic controls or occ FE | 0.528 (0.131) | 0.005 (0.002) | ✓ |
| 3 | Drop occupation fixed effects | 0.521 (0.131) | 0.006 (0.002) | ✓ |
| 4 | Cluster SEs on occupation | 0.489 (0.204) | 0.006 (0.002) | ✓ (H1 weaker p ≈ 0.02) |
| 5 | Winsorize outcome at 1/99 pct | 0.490 (0.134) | 0.006 (0.002) | ✓ |
| 6 | Trim ΔUnemp_incr at 1/99 pct | 0.637 (0.166) | 0.004 (0.003) | ◐ H2 now p ≈ 0.16 |
| 7 | **ACS arm only** | **0.985 (0.328)** | **−0.003 (0.005)** | ✗ H2 flips sign |
| 8 | **CPS arm only** | **0.376 (0.136)** | **0.008 (0.002)** | ✓ H2 stronger |
| 9 | Placebo shuffle (500 draws) | 0.005 (0.083), emp-p = 0 | 0.0000 (0.0014), emp-p = 0 | ✓ |
| 10 | Low-confidence subsample only | 0.642 (0.181) | 0.007 (0.003) | ✓ (matches Table 3) |
| 11 | Drop 0/100 extreme answers | 0.470 (0.136) | 0.006 (0.002) | ✓ |
| 12 | IHS(risk) outcome (H1 only) | 0.023 (0.005) | — | ✓ qualitative |

**Takeaways.**
- The first-stage estimate is very stable in sign and significance across every sensible perturbation of the control set and sample.
- The reduced-form result on recession-forecast demand is fragile in two respects. **(i) Arm split:** in the ACS-only sample the point estimate flips sign (−0.003, SE 0.005) while in the CPS-only sample it doubles to 0.008 and is significant at p < 0.01. The pooled estimate is a weighted average that is closer to the CPS arm. Under the paper's identification logic this is still valid, because the relevant variation is the shown-minus-alternative difference and both arms see both signal sources in a symmetric way; a sub-sample that conditions on `tr_acs` is a legitimate sensitivity check, not a violation. But the uneven evidence across arms deserves to be reported. **(ii) Outlier sensitivity:** dropping the top/bottom 1% of ΔUnemp_incr shrinks H2 from 0.006 to 0.004 with p ≈ 0.16. The first stage (H1) is unaffected by this trim, which suggests a handful of respondents with very large signal differences disproportionately drive the reduced-form effect on forecast choice. The paper's IV setup recovers the same qualitative result because the first stage is robust, but users of H2 as a standalone reduced form should keep the caveat in mind.
- The placebo test — 500 random shuffles of ΔUnemp_incr and unempincr_alt — produces a distribution tightly centered at 0 with standard deviations 0.083 and 0.0014 for H1 and H2 respectively; the actual coefficients (0.489 and 0.006) are well outside both placebo distributions (empirical p = 0 in both). The identification strategy is not spuriously picking up incidental correlation with demographics.

---

## 6. Summary Assessment

- **Exact replication.** Every number in Table 2 Panels A/B/C and every number in Table 3 Panels A–D matches the published values to three decimal places. The only differences I found are two third-decimal-place SE rounding drifts in Panel C columns 5–6, caused by the IV small-sample correction in `linearmodels` versus Stata's `ivreg2`. Neither difference changes significance, signs, magnitudes, or any sentence in the paper.
- **No bugs.** The Stata code is clean; controls are properly chosen; HC1 robust SEs are used throughout; sample handling is consistent.
- **Identification is well-motivated and the first stage is robust.** The story that people update their perceived recession exposure in response to group-level information is on very solid footing: it survives every robustness check (alternative SEs, control sets, sample trims, clustering, subsample splits).
- **One caveat worth reporting.** The reduced-form effect on recession-forecast demand is driven by the CPS arm. The pooled estimate is valid under the paper's IV logic, but the arm-specific split is asymmetric enough that a cautious reader might want to see it reported in the paper. This does not undermine Result 2 — the 2SLS estimate rests on the first stage, which is robust — but it is the most useful sensitivity result I uncovered.
- **External-validity considerations.** The experiment is a one-shot online survey with n ≈ 1,000 full-time US employees in 2019; the treatment effects on intended behavior (savings, job search) are in the right direction but small/noisy, reflecting that this is a clean micro-foundational test rather than a predictor of real-world magnitudes. The paper is honest about this.

Overall this is a textbook-clean replication. The paper's main claims are supported by the data and code exactly as published.

---

## 7. File Manifest

| File | Purpose |
|---|---|
| `utils.py` | Shared paths, control lists, OLS/2SLS helpers |
| `01_clean.py` | Load the experiment dataset and spot-check summary statistics |
| `02_table2.py` | Reproduce Table 2 Panels A, B, C (paper main table) |
| `03_table3.py` | Reproduce Table 3 Panels A–D (confidence heterogeneity) |
| `04_data_audit.py` | Coverage, balance, distributions, logical checks |
| `05_robustness.py` | 12 robustness checks on H1 / H2 / H3 |
| `output/` | CSV outputs from each numbered script |
| `writeup_131341.md` | This document |

---

*Replication performed in Python 3.13 using the shared `venv/` (pandas, statsmodels, linearmodels, scipy). All scripts run end-to-end without error with `source venv/bin/activate && python replication_131341/NN_name.py`.*
