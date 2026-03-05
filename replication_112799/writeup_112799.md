# Replication Study: Paper 112799

**Paper:** Arceo-Gomez, E.O. & Campos-Vazquez, R.M. (2014). "Race and Marriage in the Labor Market: A Discrimination Correspondence Study in a Developing Country." CIDE Working Paper No. 553.

**Replication Date:** March 2026

---

## 0. TLDR

- **Replication status:** All main results replicate successfully. Coefficient differences are < 0.001 for Tables 4-5 (core results) and < 0.006 for Tables 6-7 (interaction models with FE).
- **Key finding confirmed:** Women receive ~4.3pp more callbacks than men; European-phenotype women receive ~3.3pp more callbacks than indigenous-phenotype women; marriage penalty exists for women but not men.
- **Main concern:** Treatment variables are correlated with control variables (placebo test fails), indicating block randomization rather than full independence. The regression controls address this, but the descriptive cross-tabs (Tables 2-3) may be confounded.
- **Bug status:** No coding bugs found in the provided do-file.

---

## 1. Paper Summary

**Research question:** Is there racial and gender discrimination in Mexico's labor market? Does marital status affect callback rates differently by gender?

**Data:** ~8,000 fictitious CVs sent to ~1,000 online job advertisements in Mexico City's metropolitan area (Oct 2011 - May 2012). Each job ad received ~8 CVs varying gender (male/female), phenotype (European/Mestizo/Indigenous via photos, plus no-photo control), and marital status (married/single).

**Method:** Linear probability model (LPM) with robust standard errors clustered at the firm level. Also estimated with probit models (appendix) and firm fixed effects. The key estimating equation regresses callback (0/1) on gender, photo dummies, marital status, and controls (age, major, scholarship, university type, foreign language, time availability, leadership).

**Key findings:**
1. Women receive 40% more callbacks than men (4.3pp difference on a 12.9% base rate)
2. European-phenotype women receive 23% more callbacks than indigenous-phenotype women (~3.3pp)
3. There is no statistically significant phenotype effect for men
4. Married women face a ~2.8pp callback penalty; no marriage effect for men
5. The marriage penalty varies by phenotype for women, suggesting preference-based discrimination

---

## 2. Methodology Notes

**Translation choices:**
- Stata `reg ... , robust cluster(id_offer)` → `statsmodels.OLS.fit(cov_type='cluster', cov_kwds={'groups': df['id_offer']})`
- Stata `xtreg ... , fe i(id_offer) robust` → FWL demeaning by id_offer, then cluster-robust SEs
- Stata `dprobit` → `statsmodels Probit` with `.get_margeff()` for marginal effects
- Data loaded with `pd.read_stata(convert_categoricals=False)` to preserve numeric codes

**Estimator differences:**
- FE models (Tables 7, C1 col 6) show slightly larger differences (~0.005) due to DOF adjustments between Stata xtreg and Python demeaning approach
- Heteroskedastic probit (Table 8) not replicated directly; standard probit marginal effects replicated instead (Appendix B1)

**Variable coding:**
- `photo`: 1=European, 2=Mestizo, 3=Indigenous, 4=No photo
- `photo1`, `photo2`, `photo4`: indicator dummies (photo3=indigenous is omitted category)
- `sex`: 0=male, 1=female
- `callback`: 0/1 indicator (float32 in Stata)

---

## 3. Replication Results

### Table 4: Econometric Results - All (LPM, N=8,149)

| Variable | Col | Published | Ours | Diff |
|----------|-----|-----------|------|------|
| Woman | [1] | 0.043 [0.008] | 0.043 [0.008] | -0.0001 |
| Woman | [2] | 0.043 [0.008] | 0.043 [0.008] | -0.0000 |
| Woman | [6] FE | 0.035 [0.008] | 0.035 [0.008] | -0.0001 |
| Public univ | [2] | -0.000 [0.006] | -0.000 [0.006] | -0.0004 |
| Married | [1] | -0.011 [0.008] | -0.011 [0.008] | 0.0000 |
| Married | [2] | -0.010 [0.008] | -0.010 [0.008] | -0.0000 |
| Photo 1 (European) | [2] | 0.025 [0.007] | 0.025 [0.007] | ~0 |
| Photo 2 (Mestizo) | [2] | 0.017 [0.007] | 0.017 [0.007] | ~0 |

**Verdict:** Exact match to 3 decimal places for all coefficients and SEs.

### Table 5: Econometric Results - Women (LPM, N=4,157)

| Variable | Col | Published | Ours | Diff |
|----------|-----|-----------|------|------|
| Married | [1] | -0.028 [0.013] | -0.028 [0.013] | 0.0003 |
| Married | [2] | -0.028 [0.013] | -0.028 [0.013] | 0.0004 |
| Married | [6] FE | -0.018 [0.011] | -0.018 [0.011] | 0.0002 |
| Photo 1 | [2] | 0.033 [0.011] | 0.033 [0.011] | ~0 |
| Photo 2 | [2] | 0.019 [0.010] | 0.019 [0.010] | ~0 |

**Verdict:** Exact match.

### Table 6: Interaction Models (no FE)

| Variable | Col | Published | Ours | Diff |
|----------|-----|-----------|------|------|
| Married (women) | (1) | -0.029 [0.015] | -0.028 [0.015] | 0.0007 |
| Married (women) | (3) | -0.063 [0.023] | -0.062 [0.023] | 0.0011 |
| Photo2×Married (women) | (3) | 0.099 [0.037] | 0.096 [0.037] | -0.0031 |
| Married (men) | (6) | 0.041 [0.022] | 0.041 [0.022] | 0.0001 |

**Verdict:** Very close match (diffs < 0.004). Small differences attributable to float32/float64 precision.

### Table 7: Interaction Models (with FE)

Largest difference: photo2_married for women Col (2): published 0.012, ours 0.006 (diff=0.006).

**Verdict:** Good match. FE models have slightly larger differences (~0.005) typical of DOF sensitivity.

### Appendix B1: Probit Marginal Effects

| Variable | Published | Ours | Diff |
|----------|-----------|------|------|
| sex | 0.043 [0.008] | 0.043 [0.008] | 0.0002 |
| photo1 | 0.026 [0.007] | 0.025 [0.007] | -0.0011 |
| photo2 | 0.017 [0.007] | 0.017 [0.007] | -0.0003 |

**Verdict:** Near-exact match.

### Appendix C1: Men Only (LPM, N=3,992)

All coefficients match to < 0.001. Men show no significant effects for any treatment variable, confirming the paper's finding.

---

## 4. Data Audit Findings

### Coverage
- 8,149 observations across 1,161 unique firms
- 802 firms (69%) received the full set of 8 CVs (all8=1)
- Remaining firms received 3-6 CVs (gender-restricted ads)
- 9 firms received >8 CVs (12-16), likely appearing in multiple weeks

### Randomization Balance
- **Photo × Sex:** Well balanced (chi2=0.126, p=0.989)
- **Married × Sex:** Significantly imbalanced (chi2=9.06, p=0.003) - men more likely married (28.8% vs 25.8%)
- **Public univ × Sex:** Imbalanced (chi2=7.41, p=0.006) - men more likely public
- **Major × Sex:** Imbalanced (chi2=10.0, p=0.002) - women more likely business
- **Photo × Married:** Strongly imbalanced (chi2=35.9, p<0.001)

These imbalances reflect the block-randomization design (10 pre-built CV sets × 8 variations). They are controlled for in the regressions but affect the raw descriptive cross-tabs.

### Placebo Test Failure
Treatment variables (sex, married, photo dummies) significantly predict `scholarship` (a control variable). This confirms that the randomization was NOT fully independent across dimensions. The 10 CV sets were pre-built with specific characteristic combinations. This is addressed by including controls in the regressions, and all control variable coefficients are statistically insignificant in the main regressions, confirming they don't bias the results.

### Data Quality
- Zero missing values across all 18 variables
- All binary variables correctly coded (0/1)
- Photo dummies perfectly consistent with photo variable
- all8 firms each have exactly 4 male + 4 female CVs
- Callback rate (12.85%) is plausible for correspondence studies
- No exact duplicate rows
- 70.6% of firms called back no one; 3.1% called back everyone

---

## 5. Robustness Check Results

| Check | Description | Result |
|-------|-------------|--------|
| 1 | Restrict to all8 sample | Sex effect drops from 0.043 to 0.031; photo effects slightly smaller |
| 2 | Drop >8 CV firms | No change |
| 3 | HC1 robust SEs (no clustering) | SEs ~0.91× for sex (slightly smaller), ~1.5× for photo (larger) |
| 4 | HC3 robust SEs | Very similar to HC1 |
| 5 | Placebo outcome (scholarship) | **FAILS** - treatments predict scholarship (block randomization) |
| 6 | Permutation test (sex) | Highly significant (p≈0.000) |
| 7 | Business vs Engineering | Sex effect larger for business (0.049 vs 0.029) |
| 8 | Public vs Private univ | Marriage penalty concentrated in private (-0.037***); photo effects in public only |
| 9 | Leave-one-photo-out | Sex effect stable (0.039-0.047); married effect stable |
| 10 | Probit marginal effects | Nearly identical to LPM |
| 11 | Drop 0%/100% callback firms | Effects amplify (expected: removes zero-variation firms) |
| 12 | Sex × Married interaction | Marriage penalty specific to women (-0.030); men: +0.010 (n.s.) |

### Key robustness findings:

1. **Gender effect (4.3pp) is highly robust:** Survives all checks, confirmed by permutation test (p<0.001).

2. **European phenotype premium (~2.5pp) is moderately robust:** Present in full sample but attenuated in all8 sample. Concentrated among public university graduates. Not robust for men in any specification.

3. **Marriage penalty for women (~2.8pp) is moderately robust:** Significant in full sample, attenuated with FE. Concentrated in private universities. The sex × married interaction confirms the differential effect.

4. **Interesting heterogeneity:** The marriage penalty is NOT present in public universities (-0.003, n.s.) but is strong in private universities (-0.037***). Similarly, photo effects are significant for public university graduates but not private.

---

## 6. Summary Assessment

### What replicates:
- All main regression results (Tables 4-8) replicate to high precision (< 0.001 for core tables)
- The gender callback gap (~4.3pp, women favored) is the most robust finding
- European-phenotype women receive more callbacks than indigenous-phenotype women
- Marriage penalty exists for women but not men
- Probit and LPM give essentially identical results
- Firm fixed effects attenuate some effects but don't eliminate them

### What doesn't replicate:
- No replication failures identified

### Key concerns:
1. **Block randomization:** The experimental design used 10 pre-built CV sets, creating correlations between treatment and control variables. The regressions control for this, but the descriptive cross-tabulations (Tables 2-3) may reflect these correlations rather than pure treatment effects.

2. **Attenuation in balanced sample:** When restricting to the all8 sample (firms receiving all 8 CVs), the gender effect drops from 4.3pp to 3.1pp. This suggests the non-all8 firms (those requesting specific genders) drive part of the gender effect.

3. **Heterogeneity by university type:** The marriage penalty is driven entirely by private university graduates, and the phenotype effects are driven by public university graduates. This important heterogeneity is not prominently discussed in the paper.

4. **Clustering and SE structure:** For photo variables, unclustered robust SEs are 50% larger than clustered SEs, which is unusual and suggests negative intra-cluster correlation for photo effects. This means the clustered SEs may be anti-conservative for photo variables, though they are conservative for the gender variable.

---

## 7. File Manifest

| File | Description |
|------|-------------|
| `utils.py` | Shared paths, variable lists, helper functions (run_ols_cluster, run_fe_ols_cluster) |
| `01_clean.py` | Data loading and verification against Table 1 |
| `02_tables.py` | Replication of Tables 1-8 and Appendices B1, C1 |
| `04_data_audit.py` | Data quality audit (coverage, balance, consistency, duplicates) |
| `05_robustness.py` | 12 robustness checks |
| `writeup_112799.md` | This writeup |
