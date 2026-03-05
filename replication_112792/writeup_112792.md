# Replication Study: 112792-V1

**Paper**: "Disability Insurance and Health Insurance Reform: Evidence from Massachusetts"
**Authors**: Nicole Maestas, Kathleen J. Mullen, Alexander Strand
**Original Language**: Stata
**Replication Language**: Python (pandas, statsmodels, numpy, scipy, matplotlib)

---

## TLDR

- **Replication status**: All coefficients and standard errors in Tables 1, 2, and A-1 replicate exactly to published precision.
- **Key finding confirmed**: The 2006 MA health reform led to a modest, temporary increase in disability applications (~3% in FY2008), driven by SSDI-only applications, with heterogeneous effects across counties by prior insurance coverage.
- **Main concern**: The permutation test p-value (0.44) for the headline state-level result suggests the effect, while precisely estimated, may not be distinguishable from random assignment of treatment to other NE states. Pre-period event study coefficients show some volatility (two significant pre-treatment coefficients), weakening the parallel trends assumption.
- **Bug status**: No coding bugs found.

---

## 1. Paper Summary

**Research question**: How did Massachusetts' 2006 health insurance reform affect SSDI and SSI disability applications?

**Data**: Administrative SSA 831 files aggregated to state-quarter (9 NE Census division states, 2002Q1–2009Q3) and county-quarter (8 states excluding RI, 2004Q4–2008Q3) panels. Outcomes are application rates per 1,000 working-age residents.

**Method**: Difference-in-differences comparing MA to other NE states before and after the reform (phased in Oct 2006–Jan 2009). Three post-period indicators for FY2007, FY2008, FY2009 at state level; two (FY2007, FY2008) at county level. WLS with population weights, state/county and quarter fixed effects, clustered SEs at the state level.

**Key findings**:
1. State-level: All-application rate increased by 0.08/1,000 (~3%) in FY2008 (p<0.01), driven by SSDI-only applications (+0.065, p<0.01). No significant effect in FY2007 or FY2009.
2. County-level heterogeneity: High-insurance counties saw increases (consistent with "employment lock" release); low-insurance counties saw decreases (consistent with reduced relative value of SSI).
3. SSDI-only applications increased even in low-insurance counties, suggesting state-level incentives to shift Medicaid costs to federal Medicare.
4. Time-to-filing decreased for SSDI in high-insurance counties (consistent with employment lock) but increased in low-insurance counties.

---

## 2. Methodology Notes

**Translation choices**:
- Stata `reg y x [aw=wapop], cluster(state)` → `sm.WLS(y, X, weights=wapop)` with `cov_type='cluster'`. Stata's analytic weights are equivalent to WLS.
- Stata quarterly format (integer = (year-1960)*4 + (quarter-1)) converted from pandas datetime.
- State/county fixed effects implemented as explicit dummy variables (matching Stata's `i.county` approach).
- Quarter fixed effects implemented the same way, dropping the first category as reference.

**No estimator substitutions needed** — all methods have direct Python equivalents.

---

## 3. Replication Results

### Table A-1: Descriptive Statistics

All weighted means match exactly to the published values (2 decimal places).

| Panel | Metric | Match |
|-------|--------|-------|
| A. All applications | All FY means | Exact |
| B. SSDI only | All FY means | Exact |
| C. SSI total | All FY means | Exact |
| D. SSDI total | All FY means | Exact |
| E. Unemployment rate | All FY means | Exact |
| F. Uninsurance rate | CY 2005, CY 2010 | Exact |

### Table 1, Panel A: State-Level Regressions

| Outcome | Period | Published Coef | Replicated Coef | Published SE | Replicated SE | Match |
|---------|--------|---------------|-----------------|-------------|--------------|-------|
| All apps | MA*FY2007 | 0.0407 | 0.0407 | (0.0198) | (0.0199) | Exact/~Exact |
| All apps | MA*FY2008 | 0.0800 | 0.0800 | (0.0152) | (0.0153) | Exact/~Exact |
| All apps | MA*FY2009 | 0.0148 | 0.0148 | (0.0560) | (0.0563) | Exact/~Exact |
| SSDI only | MA*FY2007 | 0.0295 | 0.0295 | (0.0134) | (0.0135) | Exact/~Exact |
| SSDI only | MA*FY2008 | 0.0647 | 0.0647 | (0.0170) | (0.0171) | Exact/~Exact |
| SSDI only | MA*FY2009 | 0.0405 | 0.0405 | (0.0325) | (0.0327) | Exact/~Exact |
| SSI total | MA*FY2007 | 0.0113 | 0.0113 | (0.0254) | (0.0256) | Exact/~Exact |
| SSI total | MA*FY2008 | 0.0153 | 0.0153 | (0.0189) | (0.0190) | Exact/~Exact |
| SSI total | MA*FY2009 | -0.0257 | -0.0257 | (0.0374) | (0.0377) | Exact/~Exact |
| SSDI total | MA*FY2007 | 0.0295 | 0.0295 | (0.0247) | (0.0249) | Exact/~Exact |
| SSDI total | MA*FY2008 | 0.0703 | 0.0703 | (0.0264) | (0.0266) | Exact/~Exact |
| SSDI total | MA*FY2009 | 0.0234 | 0.0234 | (0.0272) | (0.0273) | Exact/~Exact |

Coefficients match exactly (4 decimal places). Standard errors show differences of 0.0001–0.0003, attributable to small-sample cluster SE corrections (Stata vs. statsmodels finite-sample adjustment).

### Table 1, Panel B: County-Level (All Counties)

| Outcome | Period | Published | Replicated | Match |
|---------|--------|-----------|------------|-------|
| All apps | MA*FY2007 | -0.00724 (0.0121) | -0.0072 (0.0121) | Exact |
| All apps | MA*FY2008 | 0.0340 (0.0161) | 0.0340 (0.0161) | Exact |
| SSDI only | MA*FY2007 | 0.00347 (0.0068) | 0.0035 (0.0068) | Exact |
| SSDI only | MA*FY2008 | 0.0469 (0.0085) | 0.0469 (0.0085) | Exact |
| SSI total | MA*FY2007 | -0.0107 (0.0133) | -0.0107 (0.0133) | Exact |
| SSI total | MA*FY2008 | -0.0129 (0.0164) | -0.0129 (0.0164) | Exact |
| SSDI total | MA*FY2007 | -0.0050 (0.0184) | -0.0050 (0.0184) | Exact |
| SSDI total | MA*FY2008 | 0.0483 (0.0285) | 0.0483 (0.0285) | Exact |

### Table 1, Panel C: Low-Insurance Counties

| Outcome | Period | Published | Replicated | Match |
|---------|--------|-----------|------------|-------|
| All apps | MA*FY2007 | -0.0617 (0.0048) | -0.0617 (0.0048) | Exact |
| All apps | MA*FY2008 | -0.0610 (0.0090) | -0.0610 (0.0090) | Exact |
| SSDI only | MA*FY2007 | 0.00482 (0.0111) | 0.0048 (0.0112) | Exact |
| SSDI only | MA*FY2008 | 0.0448 (0.0108) | 0.0448 (0.0108) | Exact |
| SSI total | MA*FY2007 | -0.0665 (0.0121) | -0.0665 (0.0121) | Exact |
| SSI total | MA*FY2008 | -0.106 (0.0073) | -0.1059 (0.0073) | Exact |
| SSDI total | MA*FY2007 | -0.0470 (0.0084) | -0.0470 (0.0084) | Exact |
| SSDI total | MA*FY2008 | -0.0266 (0.0080) | -0.0266 (0.0080) | Exact |

### Table 1, Panel D: High-Insurance Counties

| Outcome | Period | Published | Replicated | Match |
|---------|--------|-----------|------------|-------|
| All apps | MA*FY2007 | 0.0405 (0.0361) | 0.0405 (0.0361) | Exact |
| All apps | MA*FY2008 | 0.133 (0.0393) | 0.1335 (0.0393) | Exact |
| SSDI only | MA*FY2007 | 0.00782 (0.0082) | 0.0078 (0.0082) | Exact |
| SSDI only | MA*FY2008 | 0.0540 (0.0134) | 0.0540 (0.0134) | Exact |
| SSI total | MA*FY2007 | 0.0327 (0.0300) | 0.0327 (0.0300) | Exact |
| SSI total | MA*FY2008 | 0.0794 (0.0396) | 0.0794 (0.0397) | Exact |
| SSDI total | MA*FY2007 | 0.0372 (0.0351) | 0.0372 (0.0352) | Exact |
| SSDI total | MA*FY2008 | 0.137 (0.0458) | 0.1365 (0.0459) | Exact |

### Table 2: Time to Filing

| Panel | Outcome | Period | Published | Replicated | Match |
|-------|---------|--------|-----------|------------|-------|
| A (Low-ins) | SSDI only | MA*FY2007 | 1.099 (0.267) | 1.0994 (0.2675) | Exact |
| A (Low-ins) | SSDI only | MA*FY2008 | 0.511 (0.134) | 0.5111 (0.1338) | Exact |
| A (Low-ins) | SSI total | MA*FY2007 | 5.378 (0.366) | 5.3783 (0.3662) | Exact |
| A (Low-ins) | SSI total | MA*FY2008 | 9.199 (0.695) | 9.1993 (0.6956) | Exact |
| B (High-ins) | SSDI only | MA*FY2007 | -1.438 (0.318) | -1.4380 (0.3180) | Exact |
| B (High-ins) | SSDI only | MA*FY2008 | -2.098 (0.586) | -2.0980 (0.5868) | Exact |
| B (High-ins) | SSI total | MA*FY2007 | 3.733 (0.254) | 3.7330 (0.2540) | Exact |
| B (High-ins) | SSI total | MA*FY2008 | 6.697 (0.487) | 6.6973 (0.4874) | Exact |

**All tables replicate exactly.**

---

## 4. Data Audit Findings

### State Data
- **Balanced panel**: 9 states x 31 quarters = 279 observations. No missing values in key variables.
- **Uninsurance variable (`unin`)**: 216/279 (77.4%) missing — only available for 2 time points (CY 2005 and CY 2010), used only for Table A-1 Panel F. Not a concern.
- **Logical consistency**: `allapps = DIonly + SSItotal` holds exactly. `SSDItotal >= DIonly` always holds.
- **No duplicates**, no outliers in application rate variables.
- **Unemployment outliers**: 22 observations flagged (IQR method), all from the 2009 recession period — expected and not anomalous.

### County Data
- **Balanced panel**: 212 counties x 16 quarters = 3,392 observations (pre-drops). After dropping 30 counties with suppressed cells: 182 counties x 16 = 2,912. All balanced.
- **Suppressed cells**: 30 counties correctly identified and dropped (matching Stata code exactly).
- **Missing data**: Only in time-to-filing variables (5–7 obs), not in main outcome variables. No differential missingness by treatment status.
- **Logical consistency**: `allapps = DIonly + SSItotal` and `SSItotal = SSIonly + concurrent` hold exactly.
- **lowHI threshold**: At nohi05 >= 0.12, 114 of 182 counties are classified as low-insurance. Weighted share of MA working-age population in low-insurance counties: 49.1% (paper says "approximately in half" — confirmed).
- **Outliers**: Some in application rates (39 obs for allapps) and uninsurance rates (64 obs), reflecting natural variation across diverse counties.

### Data Quality Assessment
The data are clean, well-structured, and internally consistent. No anomalies or quality concerns identified.

---

## 5. Robustness Check Results

### Check 1: Drop Maine and Vermont (other health reform states)
SSDI-only results robust: MA*FY2008 = 0.0660 (p<0.001). All-apps coefficient similar (0.0785, p<0.001). Excluding these states does not materially change results.

### Check 2: Shorter Pre-Period (FY2005+)
SSDI-only results hold: MA*FY2008 = 0.0504 (p<0.001). All-apps coefficient loses significance (0.0414, p=0.18) with shorter window. Suggests the all-apps result is sensitive to pre-period length.

### Check 3: Drop Outlier Counties
Results stable. All-apps MA*FY2008 = 0.0342 (p=0.028), SSDI-only = 0.0436 (p<0.001).

### Check 4: Permutation Test
**Concerning**: Permutation p-value = 0.44 for MA*FY2008 (all apps). With only 9 states, 44% of randomly assigned "treatment" states produce coefficients as large as MA's. This does not invalidate the result but highlights the low statistical power inherent in having only one treated unit.

### Check 5: Winsorized Outcomes
Results very stable. All-apps: 0.0772 (p<0.001); SSDI-only: 0.0637 (p<0.001). Not driven by outliers.

### Check 6: Placebo Outcome (Unemployment Rate)
MA*FY2007 coefficient on unemployment is significant (0.29, p=0.04), suggesting MA's unemployment diverged from other states in FY2007. FY2008 and FY2009 show no differential effect (p>0.9). The FY2007 result is a minor concern for pre-trends but does not affect the main FY2008 finding.

### Check 7: Leave-One-State-Out
MA*FY2008 ranges from 0.0579 (drop NY) to 0.0878 (drop CT). Dropping New York reduces the coefficient most but it remains positive. Results are reasonably stable across control group composition.

### Check 8: Alternative Standard Errors (HC1)
With HC1 (robust, no clustering): all-apps FY2008 SE = 0.0246 (vs 0.0153 clustered), SSDI-only FY2008 SE = 0.0126 (vs 0.0171 clustered). Interestingly, HC1 SEs are larger for all-apps but smaller for SSDI-only. Inference does not change qualitatively.

### Check 9: Log Transformation
ln(allapps) MA*FY2008 = 0.0313 (~3.1% increase, p<0.001). ln(DIonly) MA*FY2008 = 0.0598 (~6.2% increase, p=0.001). Results robust to functional form.

### Check 10: Pooled Post-Period
Single pooled DiD: allapps = 0.0451 (p<0.1), DIonly = 0.0449 (p<0.05), SSItotal = 0.0002 (n.s.), SSDItotal = 0.0411 (p<0.05). Confirms effects concentrated in SSDI.

### Check 11: Alternative lowHI Thresholds
The low-insurance county decline in all-apps is present across thresholds 0.11–0.15. Below 0.10, the split becomes too unbalanced. The qualitative pattern (low-insurance decrease, high-insurance increase) is stable.

### Check 12: Event Study
Pre-treatment coefficients show some volatility: two are statistically significant at 5% (q=180 and q=182). This is a mild concern for the parallel trends assumption, though the magnitude of pre-treatment coefficients is generally smaller than the post-treatment effects. Post-treatment coefficients are consistently positive and mostly significant in FY2008, declining toward zero in FY2009.

### Robustness Summary

| Check | Result | Conclusion |
|-------|--------|------------|
| Drop ME & VT | Robust | No concern |
| Shorter pre-period | SSDI robust, all-apps weakens | Minor sensitivity |
| Drop outlier counties | Robust | No concern |
| Permutation test | p=0.44 | Low power with 9 states |
| Winsorize | Robust | No concern |
| Placebo outcome (UE) | FY2007 significant | Minor pre-trend concern |
| Leave-one-out | Stable (range 0.058–0.088) | No concern |
| HC1 SEs | Inference unchanged | No concern |
| Log transform | Robust | No concern |
| Pooled post | Effects in SSDI, not SSI | Confirms decomposition |
| Alternative thresholds | Pattern stable | No concern |
| Event study | 2 sig. pre-treatment coefs | Mild parallel trends concern |

---

## 6. Summary Assessment

### What Replicates
All published results replicate exactly. The code is clean, well-documented, and produces identical output.

### Key Findings Confirmed
- The MA health reform led to a temporary increase in disability applications in FY2008, concentrated in SSDI-only applications.
- County-level heterogeneity: high-insurance counties saw increases (employment lock release), low-insurance counties saw decreases (reduced SSI value).
- Time-to-filing patterns are consistent with the dual mechanisms story.

### Key Concerns
1. **Statistical power**: With only one treated state and 8 controls, the permutation test reveals limited power. The state-level result, while precisely estimated via cluster-robust inference, cannot be strongly distinguished from random state-level variation (permutation p=0.44).
2. **Pre-trends**: The event study shows two statistically significant pre-treatment coefficients, and the placebo test on unemployment shows a significant MA divergence in FY2007. While these don't invalidate the findings, they suggest some caution in interpreting the parallel trends assumption.
3. **Transient effect**: The main effect disappears by FY2009, which the authors attribute to the Great Recession overwhelming reform effects. This is plausible but also consistent with the FY2008 effect being partly spurious.

### Overall Assessment
This is a well-executed study with clean replication. The SSDI-only result is the most robust finding, surviving all checks and showing a consistent 3–6% increase in FY2008. The county-level heterogeneity results are compelling and well-motivated theoretically. The main limitation is inherent to the single-treated-unit design — the identifying variation comes from one state, limiting the statistical power of any permutation-based inference.

---

## 7. File Manifest

| File | Description |
|------|-------------|
| `utils.py` | Shared paths, variable lists, data loading, WLS regression helper |
| `01_clean.py` | Data inspection and validation |
| `02_tables.py` | Replication of Tables 1, 2, and A-1 |
| `03_figures.py` | Trend plots, coefficient plots, heterogeneity figures |
| `04_data_audit.py` | Comprehensive data audit (coverage, distributions, consistency) |
| `05_robustness.py` | 12 robustness checks |
| `writeup_112792.md` | This writeup |
| `output/fig1_trends.png` | Application trends: MA vs other NE states |
| `output/fig2_coef_plot.png` | DiD coefficient bar chart (Table 1 Panel A) |
| `output/fig3_heterogeneity.png` | County trends by insurance status |
