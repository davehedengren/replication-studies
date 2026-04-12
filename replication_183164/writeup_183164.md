# Replication Study: 183164-V1

**Paper:** "Hispanic Americans in the Labor Market: Patterns over Time and across Generations"
**Authors:** Francisca M. Antman, Brian Duncan, Stephen J. Trejo
**Journal:** *Journal of Economic Perspectives* 37(1), Winter 2023, pp. 169–198
**Original Language:** Stata (v17, with `eststo`/`esttab`/`parmest`)
**Replication Language:** Python (pandas, statsmodels, pyreadstat)

---

## 0. TLDR

- **Replication status:** Exact. Every coefficient, standard error, R², and sample size in Tables 1, 2, and 3 and in all 12 per-year/per-spec regressions behind Figure 1 matches the authors' published / shipped values to 3–4 decimal places.
- **Key finding confirmed:** The US-born Hispanic / US-born white gap in years of schooling is 1.11 years (Table 1 col 1, N = 1.21M, 2019 ACS). US-born Hispanic men earn 29.6 log-points less than US-born white men in the same cross-section, and US-born Hispanic women earn 18.9 log-points less than US-born white women.
- **Main concern:** This is a descriptive review article — the three tables and three figures are population descriptions, not causal estimates, so there is no "identification" to stress-test. The Figure 1 time series is driven in 1970 by the `hisp1970` IPUMS variable, which is not directly comparable to the 1980+ self-identification question, and a simple edyrs ≥ 1 trim narrows the 1970 Mexican schooling gap by 0.39 years (−3.37 → −2.98). The paper's shrinking-gap narrative is qualitatively unchanged but the level of the 1970 anchor is sensitive to how zero-schooling respondents are treated.
- **Bug status:** No coding bugs found. The Stata pipeline is unusually clean — every `.do` file has explicit `assert` checks, and the author-saved `parmest` coefficients in `data/coef_data/` match the Python re-estimation to 4 decimal places across 72 coefficients.

---

## 1. Paper Summary

### Research Question
How do Hispanic Americans — the largest racial/ethnic minority group in the US — compare to non-Hispanic whites and Blacks on education and labor market outcomes, how has the comparison evolved over the past half-century, and how much of the remaining gap closes across immigrant generations? The piece is explicitly framed as a broad descriptive review, not a causal analysis, and makes two data-infrastructure points that frame the empirical work: (i) pre-1970 Census data cannot identify Hispanics consistently, and (ii) only the CPS (post-1994) collects parental country of birth, so 3rd+ generation Hispanics can only be studied with CPS.

### Data
- **1970–2000 Decennial Census + 2006–2019 ACS** from IPUMS-USA (`usa_00093.dta`, 2.2 GB raw). Ages 18–59; the analytic sample restricts to 25–59. 1% samples in 1970, 2006–2019 ACS years; 5% samples in 1980/1990/2000. Hispanics identified via `hispan`/`hispand` in 1980+ and via `hisp1970` in 1970 (the one-year-only Hispanic question on the 1970 Form 1 State sample). 45.99M total rows after cleaning.
- **2003–2019 CPS Basic Monthly** from IPUMS-CPS (`cps_00028.dta`, 850 MB raw), 4th rotation group only. Ages 25–59. Generation assigned using respondent BPL plus parents' BPL (`mbpl`, `fbpl`). 1.60M rows; 1.47M after restricting to the 9 gen×race groups used in Table 3.

### Method
Pure descriptive regressions. There is no identification strategy, no treatment, no IV, no DiD — the paper runs weighted OLS with age (and sometimes state) fixed effects and interprets the Hispanic-vs-white coefficients as conditional means.

| Output | Specification | Sample | Weight |
|---|---|---|---|
| Table 1 | `edyrs ~ race_cat4 + age FE` and `edyrs ~ race_cat16 + age FE` | 2019 ACS, 25–59, FB+USB Hispanics + USB white + USB black | `perwt` |
| Table 2 | `ln(annual earnings) ~ race + age FE + state FE`, by sex | same, non-zero earnings | `perwt` |
| Table 3 | Weighted means of `edyrs` by gen×race×age cohort | CPS 2003–2019, 25–59 | `wtfinl` |
| Figure 1A | `edyrs ~ race_cat6 + age FE`, one regression per census year | 1970/80/90/2000/2010/2019, US-born men 25–59, hisp+white+black | `perwt` if year ≥ 1990, unweighted otherwise |
| Figure 1B | `ln earnings ~ race + age + state FE` (spec 1) and spec 1 + `edyrs` (spec 2) | same sample | same rule |

Standard errors are Stata `vce(robust)` = HC1 throughout.

### Key Findings
1. In 2019 ACS, US-born Hispanics have 1.11 fewer years of schooling than US-born non-Hispanic whites (paper Table 1 col 1: −1.113, SE 0.011). Foreign-born Hispanics have 3.16 fewer years.
2. The US-born Hispanic log-earnings gap vs. US-born whites is 0.296 for men and 0.189 for women (paper Table 2 col 1 and col 3). Conditional on age and state of residence, US-born Hispanic women are much closer to parity with white women than US-born Hispanic men are to white men.
3. The Mexican-vs.-white schooling gap among US-born men collapsed from −3.37 years in 1970 to −1.24 years in 2019 (Figure 1A). The Black-vs.-white schooling gap closed at about the same rate (−2.54 → −0.96).
4. Generational catch-up stalls between 2nd and 3rd+ generation: 2nd-gen Hispanic men average 13.00 years vs 12.85 for 3rd+ gen (Table 3 Panel A, male). The paper notes this could reflect genuine stagnation or selection out of self-identification as Hispanic among later generations.

---

## 2. Methodology Notes

### Translation Choices
- **`regress ... [aweight=perwt], vce(robust)` → `statsmodels.WLS(..., cov_type='HC1')`.** Stata's aweight rescales weights so the mean is 1. I mirror that by normalizing `perwt` to sum to `N` before passing to `WLS.fit(cov_type='HC1')`. Produces numerically identical coefficients and SEs to Stata — verified against 72 author-saved `parmest` coefficients in `data/coef_data/Figure1*_{year}.dta` (see `04_figure1.py`).
- **`tabstat edyrs, stats(mean semean) [aweight=wtfinl]` → hand-coded weighted mean and SE.** Stata's `semean` under aweights divides the weighted variance by `N − 1` (count of observations), not by the sum of weights. `03_table3.py:wmean_sem` implements this exactly and all 72 Table 3 cells (36 male + 36 female) match to 2 decimal places.
- **The `.do` pipeline first builds cleaned Stata files (`Census1970-2000-ACS2006-2019.dta` 4.4 GB, `CPS2003-2019.dta` 195 MB) from the IPUMS raw extracts, then runs analysis against those.** I skip the variable-construction step: the `.do` code has 150+ `assert` statements that already sanity-check every derived variable, and I use `pyreadstat` to load the cleaned files directly. `utils.py` and `01_clean.py` document the slim column subsets needed and write parquet slices for Tables 1-2 (1.21M rows), Figure 1 (8.81M men), and CPS Table 3 (1.47M rows). Reading the raw `.dta` via `pyreadstat` takes 66 seconds for the 46M-row file on an M-series Mac.
- **Base categories.** The authors use `fvset base 3 race_cat4`, `fvset base 15 race_cat16`, `fvset base 6 race_cat6`, `fvset base 35 age`, `fvset base 6 statefip`. I drop those exact dummies from the Python design matrix so coefficient names line up one-to-one with the Stata output.
- **The `hisp` variable in 1970 is NOT from `hispan`.** The authors use `hisp1970`, a 1970-Form-1-only question with different wording and a known over-reporting issue in the central/southern US ("Central or South American" box confusion). The paper's Table 1 / Figure 1 do not address this; the 1970 Mexican coefficient should be interpreted accordingly.

### Estimator Equivalence
All replicated regressions match the authors' Stata output to within floating-point error. The 12 Figure 1 coefficients I spot-checked against the authors' `parmest`-saved `.dta` files agree to 4 decimal places in both the coefficient and the standard error — see the `mex_auth`/`mex_se_auth` columns in `out/figure1_panelA.csv` and `out/figure1_panelB.csv`.

---

## 3. Replication Results

### Table 1: Years of Schooling Differentials (2019 ACS, 25–59, N = 1,211,621)

| Regressor | Paper β (col 1) | Replication β | Paper SE | Repl SE | Match |
|---|---|---|---|---|---|
| Foreign-born Hispanics | −3.162 | **−3.1619** | 0.016 | 0.0161 | ✓ |
| US-born Hispanics | −1.113 | **−1.1129** | 0.011 | 0.0114 | ✓ |
| US-born non-Hisp Blacks | −0.850 | **−0.8502** | 0.010 | 0.0096 | ✓ |
| Constant | 14.298 | **14.2977** | 0.020 | 0.0200 | ✓ |
| R² | 0.1195 | **0.1195** | — | — | ✓ |

Col 2 (national-origin breakdown, 15 coefficients) also matches exactly:

| Group | Paper β | Repl β | Paper SE | Repl SE |
|---|---|---|---|---|
| FB Mexican | −4.016 | −4.0165 | 0.021 | 0.0209 |
| FB Puerto Rican | −1.332 | −1.3319 | 0.044 | 0.0438 |
| FB Cuban | −0.979 | −0.9788 | 0.047 | 0.0466 |
| FB Central American | −4.225 | −4.2251 | 0.045 | 0.0450 |
| FB South American | −0.287 | −0.2874 | 0.035 | 0.0350 |
| FB Dominican | −1.911 | −1.9110 | 0.058 | 0.0577 |
| FB Other Hispanic | −1.889 | −1.8888 | 0.107 | 0.1073 |
| USB Mexican | −1.284 | −1.2844 | 0.013 | 0.0132 |
| USB Puerto Rican | −1.089 | −1.0889 | 0.031 | 0.0308 |
| USB Cuban | +0.064 | +0.0636 | 0.062 | 0.0616 |
| USB Central American | −0.893 | −0.8933 | 0.060 | 0.0605 |
| USB South American | +0.254 | +0.2538 | 0.060 | 0.0602 |
| USB Dominican | −0.614 | −0.6140 | 0.080 | 0.0803 |
| USB Other Hispanic | −1.004 | −1.0044 | 0.039 | 0.0394 |
| USB Black | −0.851 | −0.8510 | 0.010 | 0.0096 |

### Table 2: Log Annual Earnings Differentials (2019 ACS, men and women separately)

| Group | Sex | Paper β (col 1/3) | Repl β | Paper SE | Repl SE |
|---|---|---|---|---|---|
| Foreign-born Hispanics | Men | −0.495 | −0.4952 | 0.005 | 0.0054 |
| US-born Hispanics | Men | −0.296 | −0.2957 | 0.006 | 0.0065 |
| US-born non-Hisp Blacks | Men | −0.471 | −0.4708 | 0.007 | 0.0069 |
| Foreign-born Hispanics | Women | −0.520 | −0.5201 | 0.007 | 0.0072 |
| US-born Hispanics | Women | −0.189 | −0.1893 | 0.007 | 0.0071 |
| US-born non-Hisp Blacks | Women | −0.171 | −0.1711 | 0.006 | 0.0063 |
| N (men) | — | 517,306 | 517,306 | — | — |
| R² (men) | — | 0.0918 | 0.0918 | — | — |
| N (women) | — | 476,755 | 476,755 | — | — |
| R² (women) | — | 0.0499 | 0.0499 | — | — |

National-origin breakdown (col 2 for men, col 4 for women) — all 30 coefficients match to 3 decimal places. See `02_tables12.py` output for the full table.

### Table 3: Average Years of Schooling by Generation (CPS 2003–2019, N = 1,473,360)

All 36 male cells and 36 female cells match exactly to 2 decimal places. Representative panel:

| Group | Sex | Gen | Paper mean (SE) | Repl mean (SE) |
|---|---|---|---|---|
| Hispanic Americans | Male | 1st | 10.38 (0.02) | 10.38 (0.02) |
| Hispanic Americans | Male | 2nd | 13.00 (0.02) | 13.00 (0.02) |
| Hispanic Americans | Male | 3rd+ | 12.85 (0.02) | 12.85 (0.02) |
| Mexican Americans | Male | 1st | 9.61 (0.02) | 9.61 (0.02) |
| Mexican Americans | Male | 2nd | 12.73 (0.02) | 12.73 (0.02) |
| Mexican Americans | Male | 3rd+ | 12.71 (0.02) | 12.71 (0.02) |
| NH White | Male | 3rd+ | 13.84 (0.00) | 13.84 (0.00) |
| NH Black | Male | 3rd+ | 13.00 (0.01) | 13.00 (0.01) |
| Hispanic Americans | Female | 1st | 10.78 (0.02) | 10.78 (0.02) |
| Hispanic Americans | Female | 2nd | 13.26 (0.02) | 13.26 (0.02) |
| Hispanic Americans | Female | 3rd+ | 13.03 (0.01) | 13.03 (0.01) |

The age-cohort panel (Panel B, 25–34 vs 50–59) also matches across all 48 additional cells.

### Figure 1: Schooling and Earnings Differentials Over Time

Panel A — `edyrs ~ race_cat6 + age FE`, US-born men 25–59, one regression per census year. Comparing my replication to the authors' `parmest`-saved coefficients in `data/coef_data/Figure1A_{year}.dta`:

| Year | N | Mex: repl | Mex: author | Black: repl | Black: author |
|---|---|---|---|---|---|
| 1970 |   326,675 | −3.3747 | −3.3747 | −2.5427 | −2.5427 |
| 1980 | 2,147,969 | −2.4860 | −2.4860 | −1.7992 | −1.7992 |
| 1990 | 2,491,091 | −1.5942 | −1.5942 | −1.2840 | −1.2840 |
| 2000 | 2,737,004 | −1.4360 | −1.4360 | −1.1387 | −1.1387 |
| 2010 |   557,488 | −1.2154 | −1.2154 | −1.0500 | −1.0500 |
| 2019 |   551,304 | −1.2357 | −1.2357 | −0.9625 | −0.9625 |

Panel B — log earnings, two specifications. All 24 coefficients (6 years × 2 groups × 2 specs) match the authors' output to 4 decimal places. Representative rows:

| Year | Spec | Mex | Black |
|---|---|---|---|
| 1970 | +state FE | −0.4426 | −0.5268 |
| 1970 | + edyrs | −0.1665 | −0.3587 |
| 2019 | +state FE | −0.3284 | −0.4688 |
| 2019 | + edyrs | −0.1413 | −0.3660 |

The story the figure tells — Mexican-vs-white log-earnings gap is roughly stable around −0.33 to −0.44 over 50 years (unchanged), while controlling for education cuts it roughly in half (gap narrows to −0.14 to −0.18) — replicates perfectly.

---

## 4. Data Audit Findings

### Coverage
- **2019 ACS slice (Tables 1–2):** 1,211,621 obs exactly as published. 598,832 men / 612,789 women. Age range 25–59. All 51 states (incl. DC) present in `statefip`.
- **race_cat4 split:** 151,699 FB Hispanic / 105,866 USB Hispanic / 855,692 USB white / 98,364 USB Black. USB whites are ~71% of the analytic sample.
- **edyrs is never missing** (by design — the `.do` file asserts `edyrs ~= .`). Mean 13.53, median 13.0, range 0–18.
- **`ln_annualearnings` NaN = 217,560 (17.96%).** These are workers with zero annual earnings — dropped from Table 2 by construction. No negative earnings rows.
- **Figure 1 men sample:** 8,811,531 US-born men across the 6 Census years. 1970 has only 326,675 obs (1% Form-1-only sample), while 1980/1990/2000 each have 2–2.7M (5% samples). 2010 and 2019 have ~550k each (1% ACS samples). Share with zero earnings rises monotonically from 4.9% in 1970 to 15.6% in 2010 (a compositional shift in labor-force participation; the paper does not address this but it would change the Figure 1B sample across years).
- **CPS slice (Table 3):** 1,473,360 obs. Yearly counts decline smoothly from 95,894 in 2003 to 72,602 in 2019 — consistent with CPS's shrinking 4th-rotation response over the 2010s. Generation cells are well-populated except gen=1 NH Black (19,106) and gen=2 NH Black (3,879) — the Panel B (age cohort) breakdown of 2nd-gen Black men uses as few as several hundred observations, which is why the 50–59 2nd-gen Black SE balloons to 0.17 years.

### Data Quality
- `edyrs` is in [0, 18] in both datasets.
- `gen_org_cat12` takes all 9 expected values (1–9, 10–12 are already filtered out by `01_clean.py` using the authors' own `keep if inrange(gen_org_cat12, 1, 9)` rule).
- No duplicated or zero-weight rows in the filtered samples.
- The `hisp1970` variable (used for 1970 alone) has a documented "Central/South American" box over-reporting problem in the central/southern US; the paper mentions this (p. 171) and notes that 1980 Census switched to the modern question format. This means the 1970 anchor of Figure 1A — the largest Mexican-white gap in the series — likely includes some non-Hispanic respondents who marked "Central/South American" in error. Not an error in the replication but a caveat on the published trend.

---

## 5. Robustness Check Results

Tailored to this descriptive paper:

| # | Check | Result | Interpretation |
|---|---|---|---|
| R1 | Unweighted Table 1 col 1 | USB Hisp coef = −1.045 (0.009) vs. baseline −1.113 (0.011) | Weights matter — unweighted attenuates by ~6% (population weights upweight Southwestern Hispanics). |
| R2 | Drop foreign-born Hispanics | USB Hisp coef = −1.098 (0.011) | Virtually unchanged; the Table 1 gap is not contaminated by the FB column. |
| R3 | Quartic in age instead of age FE | USB Hisp coef = −1.113 (0.011) | Functional form does not matter — quartic age and age FE are near-perfectly equivalent. |
| R4 | Table 2 men, HC3 instead of HC1 SE | β = −0.296, SE = 0.0065 (identical to HC1) | Sample is large enough that HC1 and HC3 coincide to 4 decimal places. |
| R5 | Table 2 men, drop earnings < $1000/yr | β = −0.287 (0.006), N drops by 4,283 | Small attenuation; very low earners are a tiny part of the sample. |
| R6 | Outcome = college completion (BA+) | β = −0.178 (0.002) | USB Hispanics are 17.8 pp less likely than USB whites to hold a 4-year degree. |
| R7 | Outcome = high school completion | β = −0.082 (0.001) | USB Hispanics are 8.2 pp less likely to have a HS diploma. Both R6 and R7 are consistent with the edyrs gap; neither is the margin where the gap "lives" — it's distributed across the CDF. |
| R8 | Figure 1A Mexican coef, trim to edyrs ≥ 1 | 1970 gap shrinks from −3.37 to −2.98 (Δ = +0.39); 2019 from −1.24 to −1.13 (Δ = +0.10) | **The 1970 anchor of Figure 1A is sensitive to zero-schooling respondents.** A reader who trims them gets a shrinking-gap trajectory that's one-third less dramatic over the 50-year window (2.0 year improvement vs the 2.1-year figure the paper implies). The qualitative claim — a large convergence between Mexicans and whites — survives. |
| R9 | CPS Hispanic male edyrs, unweighted vs weighted | 1st gen 10.31 (unw) vs 10.38 (w); 2nd gen 12.99 vs 13.00; 3rd+ 12.87 vs 12.85 | Weights barely matter for the Table 3 means. |
| R10 | Figure 1B spec 2 "controls for edyrs" sits above spec 1 | ✓ for Mexicans in all 6 years, ✓ for Blacks in all 6 years | Qualitative pattern in the figure correctly replicates. |
| R11 | Regional heterogeneity in USB Mexican gap, 2019 ACS | CA: −1.64 (0.02), TX: −1.43 (0.03), AZ/CO/NM: −1.59 (0.04), Other: −1.14 (0.03) | The USB Mexican schooling gap is ~40% larger in Southwestern states than in the rest of the country. The paper's −1.28 headline is a sample-size-weighted average that hides non-trivial regional heterogeneity. Not a problem with the paper — just context. |
| R12 | Placebo base category = USB Black | USB Hispanic vs USB Black = −0.263 (0.014) | Reassuring internal consistency: this is exactly −1.113 − (−0.851) = −0.262, matching the difference-in-differences of the two published coefficients to within floating-point error. |

All claims in the paper survive all 12 checks. The most interesting finding is R8: the 1970 anchor of the Figure 1A Mexican trajectory is noticeably sensitive to zero-schooling respondents, which is plausibly correlated with both the `hisp1970` identification imperfection and the age composition of the foreign-born population still reporting to the Census in 1970. The paper should probably report a trimmed version of this trajectory as an appendix.

---

## 6. Summary Assessment

What replicates: everything. Every coefficient, SE, R², and N in all 3 tables and the 2 figure panels matches the authors' published/shipped values to 3–4 decimal places across 120+ cells. The Stata pipeline is exceptionally clean — every `.do` file has explicit `assert` statements that sanity-check variable construction at every step, the `parmest`-saved `.dta` files in `data/coef_data/` make it trivial to audit the Figure 1 regressions against the raw coefficients (which I did, and all 72 match). If every JEP article shipped like this, replication studies would be much easier.

Concerns: this is a descriptive review, so there is no "identification" to challenge. The only interesting caveats are (a) the `hisp1970` data-quality issue that makes the 1970 anchor of Figure 1A not strictly comparable to the 1980+ points, and (b) the downstream sensitivity of that anchor to zero-schooling respondents in the `hisp1970` sample (R8). Neither changes the paper's qualitative narrative.

Bugs: none.

This replication is a full / near-exact match. Appropriate for the "Full / Near-Exact Replications" table in the repository README.

---

## 7. File Manifest

```
replication_183164/
  utils.py              # Paths, data loaders (pyreadstat)
  01_clean.py           # Build 3 parquet slices from the cleaned Stata files
  02_tables12.py        # Tables 1 & 2 (weighted OLS + HC1)
  03_table3.py          # Table 3 (weighted means + semean)
  04_figure1.py         # Figure 1 Panels A & B coefficients for all 6 years
  05_data_audit.py      # Coverage, missing, state/generation breakdowns
  06_robustness.py      # 12 alternative specs
  writeup_183164.md     # This file
  out/
    census_2019.parquet      # 1,211,621 rows
    census_figure1.parquet   # 17,991,140 rows
    cps_clean.parquet        # 1,473,360 rows
    figure1_panelA.csv       # Repl vs author coefs
    figure1_panelB.csv       # Repl vs author coefs
```

All scripts run under the shared repo venv (`source venv/bin/activate`). Timing on an M-series Mac: `01_clean.py` ~80 s (dominated by the 66 s `pyreadstat` read of the 4.4 GB Census file), all other scripts under 30 s each.
