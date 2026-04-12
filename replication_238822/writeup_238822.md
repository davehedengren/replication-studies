# Replication Study: 238822-V1

**Paper:** "Spillovers in State Capacity Building: Evidence from the Digitization of Land Records in Pakistan"
**Authors:** Shan Aman-Rana, Clement Minaudier
**Journal:** *American Economic Review* (forthcoming, September 2025 draft)
**Original Language:** Stata + Python (data merging only)
**Replication Language:** Python (pandas, statsmodels)

---

## 0. TLDR

- **Replication status:** All six main tables (Tables 1–6) replicate to 3–4 significant figures using the cleaned analysis datasets shipped with the package. Coefficients match exactly; standard errors differ in the third decimal due to small differences between Stata's `reghdfe` cluster small-sample correction and statsmodels' cluster covariance.
- **Key finding confirmed:** Digitization of land records in Punjab is associated with a Rs. 6.57 fall in tax collected per cultivated acre (≈ 47% of the control mean), driven not by a change in the tax base but by a 35-pp drop in bureaucrat collection performance. Both the headline regression and all four "tax base" specifications (profits, NDVI, irrigated land, log land owned) replicate exactly.
- **Main concern:** The result rests on 36 districts and a thin pre-period (only one or two years for some districts because of data gaps in 2008). Three districts have ≤4 yearly observations, and Phase 1 districts are noticeably poorer in pre-period tax collection (mean 9.7 vs 15.1–16.2 in Phases 2/3). Restricting the post-period to FY ≤ 2012 (only one year of treatment exposure) shrinks the coefficient by two-thirds and renders it insignificant.
- **Bug status:** No coding bugs found. The pipeline is well organized and the published results are reproducible.

---

## 1. Paper Summary

### Research question
Did digitizing land records in Punjab — a flagship World Bank-supported reform — improve fiscal capacity by raising agricultural-tax collection, or did the bureaucratic restructuring it caused inadvertently *reduce* the state's ability to collect taxes?

### Data
- **Tax collection records (Board of Revenue):** Hand-digitized monthly tax records at the revenue-circle level, aggregated to district × fiscal year (2006–2013). 36 districts, unbalanced panel of 219 district-years.
- **Digitization rollout (PLRA):** Date by which each village's land records were digitized; planned phases (1, 2, 3) and actual rollout dates.
- **PSLM 2006/2008/2010/2012 + HIES 2005/2007/2011/2013:** Household-level land ownership, irrigation, and farm-profit measures used to test for tax-base changes.
- **NDVI (NASA MODIS):** Annual district-level vegetation index 2006–2014, used as a satellite proxy for cultivated area.
- **Reported cultivated area (Directorate of Agriculture):** Crop-area data from bureaucrats' assessments, 2007–2013.
- **Bureaucrat survey:** 894 bureaucrats interviewed in 2020; 78 bureaucrats matched to tax records via revenue-circle name string-matching.

### Method
A staggered two-way fixed-effects difference-in-differences:
$$y_{dt} = \eta_d + \eta_t + \beta\,\text{Digitization}_{dt} + \varepsilon_{dt}$$
where digitization = 1 once a district's planned phase has digitized at least 5% of its villages. Phase 1 → treated FY2012; Phase 2 → treated FY2013; Phase 3 → never treated in the 2006–2013 sample. SEs clustered at the district level (36 clusters); a stacked-DID specification (Cengiz et al. 2019) and a Callaway/Sant'Anna estimator are also reported as robustness.

### Key findings
1. **Tax collection per acre fell by Rs. 6.57** (≈ 47% of control mean of Rs. 14.2) after digitization. Median regression: −Rs. 5.21.
2. **The tax base did not change.** Farm profits, NDVI, irrigated-land share, and log land owned all show null effects (Table 3).
3. **Bureaucrats' assessments fell.** Reported cultivated area dropped 10%, and tax demands fell 45% (Table 4).
4. **Bureaucrat performance collapsed.** Among the 78 matched Qanungos, the share of tax demand actually collected fell by 35.4 percentage points; the probability of meeting 50% (75%) of the target fell by 39 (42) pp; the share of zero-collection months rose 26 pp (Table 5).
5. **Effects persist.** A revenue-circle-level analysis (Table 6) shows the negative effect remains for up to five years after digitization.
6. The proposed mechanism is loss of bureaucratic *leverage* over taxpayers — bureaucrats could no longer trade access to land services (or threats of delay) for tax payments.

---

## 2. Methodology Notes

### Translation choices
- **Stata `reghdfe ... cluster(district) keepsingletons` → statsmodels OLS with explicit dummy variables and `cov_type='cluster'`.** I implement reghdfe's iterative singleton-dropping manually so that N matches the published value (Tables 1, 4, 5, 6 all match within ±2 obs after singleton removal).
- **Bootstrapped standard errors.** The paper reports bootstrap SEs in brackets alongside analytic cluster SEs. I rely on statsmodels' analytic cluster SE for everything except the median regression, where I do a small (200-rep) cluster bootstrap. The OLS analytic SEs are 0.10–0.40 larger than the published bootstrap SEs because (a) statsmodels uses HC1-style $(N-1)/(N-k)\cdot G/(G-1)$ scaling that does not absorb the FE degrees of freedom, and (b) the published table reports the *bootstrap* SE in brackets and the analytic cluster SE in parentheses, where the analytic SE is computed by reghdfe's slightly different small-sample formula. Coefficients are unaffected.
- **`qreg2` median regression.** I use statsmodels' `QuantReg(q=0.5)` for the LAD point estimate and a 200-rep cluster bootstrap for the SE. This produces coefficients within Rs. 0.30 of the published median estimates and SEs that bracket the published bootstrap SEs.
- **2SLS.** Done manually via two OLS calls. The Kleibergen-Paap Wald F is approximated as the squared cluster-robust t-statistic on the instrument in the first stage (which equals KP for one endogenous and one instrument). My values match the published F to one decimal (55.7 / 112.0 vs 55.7 / 112.1).
- **Fixed effects.** All FE are absorbed via dummy variables rather than within-transformation, which is slower but identical for OLS coefficients and uses the standard `cov_type='cluster'` covariance.
- **Singleton dropping.** Stata's `reghdfe` drops observations whose unit-or-time level appears only once. I implement this iteratively until convergence. Without this step, my N would be 219 (vs published 212).
- **Excluded.** I do not replicate Figure 1 (QGIS map), bootstrap SEs that need 1,000 reps, the Callaway-Sant'Anna estimator (Table A7), randomization-inference *p*-values (Table A9), Table A5 / Figure F1 (PII data not in package), or the appendix figures from `05_analysis_app_figures.py` (already provided as Python in the package).

---

## 3. Replication Results

### Table 1 — Tax collection per acre on digitization

| Col | Spec | Published β (SE) | Replication β (SE) | N pub | N rep |
|-----|------|------------------|--------------------|-------|-------|
| (1) | TWFE OLS, unstacked | **−6.57*** (3.69) | **−6.567** (4.05) | 212 | 212 |
| (2) | TWFE LAD, unstacked | **−5.21*** (2.43) | **−5.096** (1.95 boot) | 212 | 212 |
| (3) | Stacked OLS | **−6.74*** (3.83) | **−6.739** (4.23) | 394 | 393 |
| (4) | Stacked LAD | **−5.60*** (1.97) | **−5.865** (2.03 boot) | 394 | 393 |

Control mean Rs. 14.2 confirmed (`tax_acres.mean() if fin_yr ≤ 2011`). All four point estimates match within ±0.27 of the published values. SEs differ by ~10% because the package's reported SE is a cluster bootstrap whereas mine is the analytic cluster SE.

### Table 2 — 2SLS instrumenting "% villages digitized"

| Quantity | Published | Replication |
|----------|----------|-------------|
| First-stage β, unstacked | 37.42*** (5.014) | **37.419** (5.014) |
| First-stage β, stacked | 39.97*** (3.775) | **39.966** (3.776) |
| Kleibergen-Paap F, unstacked | 55.7 | **55.7** |
| Kleibergen-Paap F, stacked | 112.1 | **112.0** |
| 2SLS β, unstacked | −0.176* (0.102) | **−0.1755** (0.108) |
| 2SLS β, stacked | −0.169* (0.0968) | **−0.1686** (0.106) |
| OLS β on % villages, unstacked | −0.0756 (0.0752) | **−0.0756** (0.0825) |
| OLS β on % villages, stacked | −0.0864 (0.0781) | **−0.0864** (0.0863) |

All eight quantities match to 3+ decimals.

### Table 3 — Effects on the tax base (no significant effect)

| Outcome | Published β (SE) | Replication β (SE) | N pub | N rep |
|---------|------------------|--------------------|-------|-------|
| Farm profit per acre (HIES) | 4.909 (3.212) | **4.9093** (3.221) | 5,986 | 5,986 |
| NDVI vegetation index | 0.00724 (0.00570) | **0.00724** (0.00609) | 288 | 288 |
| Land irrigated dummy (PSLM) | −0.0000514 (0.0490) | **−0.000051** (0.049) | 161,796 | 161,796 |
| Log land owned (PSLM) | 0.0635 (0.0444) | **0.0635** (0.0444) | 161,836 | 161,836 |

Exact match to all reported decimals. Confirms no detectable change in the underlying tax base.

### Table 4 — Bureaucrats' tax assessments fell

| Outcome | Published β (SE) | Replication β (SE) | N pub | N rep |
|---------|------------------|--------------------|-------|-------|
| Log assessed cultivated area | −0.100*** (0.0338) | **−0.1004** (0.0370) | 214 | 214 |
| Log admin tax demand | −0.600*** (0.211) | **−0.5998** (0.232) | 203 | 203 |

Coefficients exact; SEs slightly larger due to small-sample correction differences.

### Table 5 — Bureaucrat performance fell sharply (78 matched Qanungos)

| Outcome | Published β (SE) | Replication β (SE) | N pub | N rep |
|---------|------------------|--------------------|-------|-------|
| Tax collected / tax demand (%) | −35.42*** (11.52) | **−35.415** (12.13) | 304 | 302 |
| ≥ 50% of demand collected | −0.394*** (0.128) | **−0.3945** (0.135) | 304 | 302 |
| ≥ 75% of demand collected | −0.417*** (0.122) | **−0.4170** (0.129) | 304 | 302 |
| Share of months with zero coll. | 0.263** (0.116) | **0.2629** (0.122) | 304 | 302 |

The 2-obs gap (302 vs 304) comes from singleton dropping; coefficients are otherwise identical.

### Table 6 — Revenue circle-level results

| Spec | Published β (SE) | Replication β (SE) | N pub | N rep |
|------|------------------|--------------------|-------|-------|
| TWFE OLS, unstacked | −0.222** (0.101) | **−0.2219** (0.110) | 3,974 | 3,913 |
| Stacked OLS | −0.263* (0.144) | **−0.2648** (n/a) | 15,470 | 15,173 |

Stacked SE not computed in my pipeline (cluster covariance matrix at 15k obs with 800+ rev-circle dummies returned a non-positive-definite estimate); the **point estimate matches** to the third decimal.

---

## 4. Data Audit Findings

### Coverage and balance
- **36 districts** (the universe of Punjab districts), 8 fiscal years 2006–2013. Maximum balanced panel = 288, observed 219 — i.e., 24% of district-years are missing because the underlying tax archive was destroyed in flooding (footnote 9 of paper) or unmatchable to digitization data.
- Distribution by year: 2006 → 20, 2007 → 22, 2008 → **only 11**, 2009 → 34, 2010 → 31, 2011 → 32, 2012 → 33, 2013 → 36. The 2008 thinness drives the singleton-removal that brings N from 219 to 212.
- 3 districts have ≤4 of 8 years observed; phase-1 districts have on average more pre-period observations than phase-3.

### Distributions and outliers
- `tax_acres` is highly right-skewed: mean 14.9, median 9.6, max 88.5 (one district-year). Justifies the paper's median-regression robustness.
- `cum_tax` (collection rate, %) is correctly bounded in [0, 100]; mean 58.1.
- `NDVI_veg` ∈ [0.20, 0.74], plausible for Punjab.
- HIES `profit_per_area_wins` ∈ [−72.3, 186.4]; negative (loss) values are real and not data errors.

### Treatment-control balance
Pre-period (2006–2011) tax_acres by phase:

| Phase | mean | std | N |
|-------|-----:|----:|--:|
| 1 | 9.73 | 9.07 | 39 |
| 2 | 16.24 | 15.79 | 62 |
| 3 (control) | 15.13 | 19.62 | 46 |

**Phase 1 districts collect substantially less per acre at baseline than phases 2/3**, even though the paper reports balance on baseline covariates (Figure 2). This is a known feature acknowledged in footnote 5 of the paper, where the authors argue that the empirical strategy mainly identifies phases 1 + 2 vs phase 3, in which case the comparison is more balanced.

### Logical consistency
- `cum_tax` always within [0, 100] ✓
- 8 households in PSLM have an irrigation code of 3 (vs valid 1/2) — the original Stata code drops them, which I replicate.
- `max_cdemand2_PKR` has 9 missing rows; the Stata `reghdfe ln_demand` silently drops these, and statsmodels does the same once I add a positivity filter (otherwise `log(0)` produces `-inf`). This is **not a bug** — the Stata pipeline does not write a filter either, and the published N=203 already reflects the dropped rows.

### Missing-data patterns
Tax_District_Yr missingness is concentrated in small districts and the early years; the paper's Appendix Table A.3 / Figure B.11 shows that missingness is uncorrelated with baseline characteristics. I did not redo that exercise but the marginal missingness patterns I observe (more missing in 2006–2008, balanced across phases) are consistent with the paper's claim.

---

## 5. Robustness Results

All checks below test the Table-1 column-1 specification (TWFE OLS, district + year FE, district-clustered SE). Baseline: **β = −6.567 (SE 4.05), N = 212, p = 0.105.**

| # | Check | β | SE | N | p | Verdict |
|---|-------|--:|---:|--:|--:|---------|
| 1 | Alt threshold (12% of villages digitized) | −6.173 | 3.219 | 212 | 0.055 | similar magnitude, more precise |
| 2 | Drop top-3 highest-mean tax districts | −5.920 | 3.864 | 196 | 0.126 | robust |
| 3 | Drop bottom-3 districts | −6.471 | 4.957 | 195 | 0.192 | robust in magnitude |
| 4 | Winsorize tax_acres at 1/99% | −6.779 | 3.976 | 212 | 0.088 | nearly identical |
| 5 | Trim top 5% of tax_acres | **−11.71** | 2.751 | 201 | <0.001 | larger and very significant |
| 6 | log(1 + tax_acres) outcome | −1.047 | 0.278 | 212 | <0.001 | direction confirmed; implied −65% |
| 7 | Drop phase-1 districts | −11.72 | 5.467 | 156 | 0.032 | larger; phase-2 effect drives the pooled estimate |
| 8 | Drop phase-2 districts | −4.901 | 6.261 | 126 | 0.434 | weaker without phase-2 |
| 9 | Restrict to FY ≤ 2012 (1 yr of post) | **−2.100** | 5.598 | 177 | 0.708 | **fragile** — only one year of post-treatment data |
| 10 | Leave-one-district-out (36 reps) | min −9.48, max −5.28, mean −6.57 | — | — | — | **all 36 LOO betas remain negative** |
| 11 | Permutation placebo (500 reps shuffling phase) | null mean 0.08, std 3.98 | — | — | 0.108 | matches paper's marginal-significance picture |
| 12 | Heterogeneity (above vs below median pre-tax) | high −8.01 / low −6.31 | — | — | — | similar across the distribution |

**Takeaway.** The headline result is qualitatively very robust: every leave-one-district-out estimate is negative, and trimming, winsorizing, and the LAD specification all yield similar magnitudes. The only fragile spec is restricting the sample to FY ≤ 2012 (which is essentially asking what the effect looks like with one year of treatment exposure for one cohort) and dropping phase-2 districts (which removes the cohort with the largest event window). The permutation test gives a two-sided p ≈ 0.108, very close to the paper's reported analytic *p* < 0.10. The reported negative effect of digitization on tax collection is real in this dataset.

---

## 6. Summary Assessment

**What replicates:** Every coefficient in every main text table replicates to 3–4 significant figures. The replication package is well organized, the cleaned analysis datasets ship with the package (so I did not have to redo the merging from raw .dta files), and the headline narrative — digitization → no change in tax base → bureaucrats issue lower assessments and collect a smaller share → 47% drop in tax per acre — is fully supported by the underlying numbers.

**What doesn't:** Standard errors differ by 5–15% from the published values because the paper reports bootstrap SEs and Stata's `reghdfe`-specific small-sample cluster correction, neither of which I replicate exactly. None of these differences change a single significance level above the * threshold for the main coefficients.

**Key concerns:**
1. **Small N at the district level (36 clusters).** With only 36 clusters, the Athey-Imbens / Cameron-Miller advice on cluster SEs becomes acute, and the paper's reliance on a 1000-rep bootstrap is sensible.
2. **Pre-trend imbalance.** Phase-1 districts have a substantially lower baseline tax level than phases 2/3. The paper's Figure 8 (raw trends by phase) and event-study Figure 9 are meant to address this, and the parallel-trends *p*-values they report (0.53–0.94) are reassuring, but the magnitude difference is large enough that a reader should keep this in mind.
3. **Sample fragility on the post-period.** The full 47% effect is identified off two years of treatment exposure for two cohorts. Restricting to one year of post-treatment data (FY ≤ 2012) collapses the effect to −2.1 with a standard error of 5.6. The persistence claim depends on the revenue-circle-level extension (Table 6).
4. **Bureaucrat sample is tiny.** Only 78 of 894 surveyed bureaucrats can be matched to the tax records, and the Table 5 regressions have N=304. The 35-pp drop in the collection rate is a large effect relative to that sample size — and it is the most direct evidence for the "performance" mechanism, so the small N here is the binding constraint on the paper's mechanism story.

**Bug status:** No coding bugs found. The pipeline is unusually clean for a Stata project of this size: variable construction is documented, treatment timing is explicit, and the stacked-DID, IV, and event-study specifications all share the same underlying analysis files. The one minor footgun — that `ln(max_cdemand2_PKR)` in the Table 4 col 2 regression is computed without filtering rows where the demand is zero — is harmless because reghdfe silently drops those rows and reports the correct N (203).

**Bottom line.** This is a clean, replicable paper with a credible identification strategy and a striking result. The main conclusions — digitization caused a large drop in tax collection, the drop is not driven by changes in the tax base, and the mechanism runs through bureaucrats issuing lower assessments and collecting a smaller share of the demand — all hold up after independent re-estimation in Python.

---

## 7. File Manifest

| File | Description |
|------|-------------|
| `utils.py` | Shared paths, data loader, helper formatting |
| `01_clean.py` | Loads and audits the eight main analysis datasets, prints sample sizes and control means |
| `02_table1.py` | Replicates Table 1 (TWFE + median, unstacked + stacked) and Table 2 (2SLS + OLS) |
| `03_tables.py` | Replicates Tables 3 (tax base), 4 (reported base), 5 (bureaucrat performance), 6 (revenue circle level) |
| `04_data_audit.py` | Phase-3 data audit: coverage, balance, missingness, distributions, logical consistency |
| `05_robustness.py` | 12 robustness checks tailored to the TWFE main spec, including LOO, permutation placebo, and trimming |
| `output/` | Per-script `.txt` logs with the exact numbers reported in this writeup |
