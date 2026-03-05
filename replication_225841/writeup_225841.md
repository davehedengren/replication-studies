# Replication Study: Paper 225841

**Paper**: "Do Credit Conditions Move House Prices?"
**Authors**: Daniel Greenwald and Adam Guren
**Source**: American Economic Review, 2025
**Replication by**: Claude (automated replication pipeline)

---

## 0. TLDR

- **Replication status**: Methodology successfully replicated using non-confidential pseudodata. Local projection IRFs match the paper's approach (reghdfe with year + CBSA FE, clustered SEs, population weights). Exact coefficient comparison not possible because the published results use confidential GG microdata.
- **Key finding replicated (qualitatively)**: Credit supply shocks (Loutskina-Strahan instrument) positively affect homeownership rates at short horizons (h=0,1), consistent with the paper's main result.
- **Main limitation**: Pseudodata precludes exact coefficient comparison with published figures.
- **Bug status**: No coding bugs found.

---

## 1. Paper Summary

**Research question**: Do credit conditions causally affect house prices and homeownership rates?

**Data**: CBSA-level panel (413 CBSAs, 1990-2017) combining Census HVS homeownership data, FHFA house price indices, and credit supply instruments.

**Method**: Local projection impulse response functions (LP-IRF). For each horizon h=0,...,5:
```
reghdfe F(h).y z controls [aw=pop2000], absorb(year cbsa) cluster(cbsa)
```
where `z` is the Loutskina-Strahan credit supply instrument.

**Key findings**: Credit supply shocks raise homeownership rates and house prices, with effects building over several years.

---

## 2. Methodology Notes

**Translation choices**:
- Stata `reghdfe ... absorb(year cbsa) cluster(cbsa)` translated using FWL demeaning by year and CBSA groups + manual cluster-robust variance
- Analytical weights (Stata `[aw=pop2000]`) implemented as WLS with normalized weights
- Local projections computed by creating forward leads `F(h).y` within each CBSA panel

**Three instruments analyzed in paper**:
- Loutskina-Strahan (LS): Credit supply shock from securitization exposure (replicated here)
- DiMaggio-Kermani (DK): Anti-predatory lending laws
- Mian-Sufi (MS): Household debt expansion

---

## 3. Replication Results

Results shown use non-confidential pseudodata. Methodology verified against Stata code.

### Figure 3: LS Instrument IRFs (HVS Common Sample)

**Homeownership Rate IRF** (qualitative match):

| Horizon | Coefficient | SE | 95% CI | N |
|---------|:-----------:|:--:|:------:|:-:|
| h=0 | 1.797 | 0.734 | [0.36, 3.24] | 943 |
| h=1 | 2.741 | 1.165 | [0.46, 5.02] | 902 |
| h=2 | 0.980 | 1.615 | [-2.19, 4.15] | 861 |
| h=3 | 1.259 | 1.456 | [-1.59, 4.11] | 820 |
| h=4 | 0.789 | 1.465 | [-2.08, 3.66] | 779 |
| h=5 | -0.682 | 1.162 | [-2.96, 1.60] | 738 |

The positive and significant effect at h=0 and h=1 is consistent with the paper's main finding that credit supply shocks increase homeownership rates.

---

## 4. Data Audit Findings

**Coverage**: 413 CBSAs, 28 years (1990-2017), balanced panel (28 obs per CBSA)

**Sample sizes**: HVS Common Sample = 943 obs (41 CBSAs), HVS All Sample = 1,426 obs (62 CBSAs), GG All Sample = 8,970 obs (390 CBSAs)

**Missing data**: Price-rent ratio and homeownership rate available only for HVS-covered CBSAs (~15% of panel). HPI available for all CBSAs.

**Instrument variation**: LS instrument mean = 0.0005, sd = 0.0012. Within-CBSA sd (0.0010) exceeds between-CBSA sd (0.0006), confirming time-series variation drives identification.

---

## 5. Robustness Check Results

### Results that survive (with pseudodata)

**Homeownership rate at h=0**: Positive and significant across HVS Common (coef=1.80, t=2.45), HVS All (coef=1.64, t=2.33), and HVS Unbalanced (coef=2.16, t=2.95) samples.

**Weighted vs unweighted**: Both show positive significant effects (weighted: 1.80, unweighted: 2.48).

### Results that are sensitive

**Employment controls**: Adding industry employment shares reduces the h=0 homeownership coefficient from 1.80 to 1.11 and it loses significance (SE=0.75).

**Price-rent ratio**: Very large SEs (49.3) make this outcome uninformative in the pseudodata, likely due to pseudodata noise.

---

## 6. Summary Assessment

### What replicates
The local projection methodology is correctly implemented and produces qualitatively consistent results with the paper's main finding: credit supply shocks (LS instrument) positively affect homeownership rates at short horizons.

### Key limitations

1. **Pseudodata**: The non-confidential dataset contains pseudodata for some variables, making exact coefficient comparison impossible. The paper's main results require confidential GG microdata.

2. **Small effective samples**: The HVS-based samples have only 41-62 CBSAs, limiting statistical power for the price-rent ratio outcome.

3. **Employment controls sensitivity**: The homeownership result weakens substantially when controlling for industry employment shares, suggesting some confounding between credit supply and local labor market conditions.

### Overall assessment
The replication successfully implements the paper's econometric approach (local projection IRFs with FWL demeaning, cluster-robust SEs, and population weights). Qualitative findings are consistent with the paper. Full verification requires access to the confidential dataset.

---

## 7. File Manifest

| File | Description |
|------|-------------|
| `utils.py` | Data loading, `run_local_projection()` LP-IRF function, `print_irf()` |
| `02_tables.py` | Replication of main LS IRFs (Figures 3a-3d equivalent) |
| `04_data_audit.py` | Data audit: coverage, distributions, panel balance, instrument variation |
| `05_robustness.py` | Robustness: sample sensitivity, employment controls, outcomes, weighting |
| `writeup_225841.md` | This writeup |
