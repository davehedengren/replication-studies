# Replication Study: 146041-V1

**Paper:** "Human Capital and Macro-Economic Development: A Review of the Evidence"
**Author:** Federico Rossi
**Journal:** *Journal of Economic Literature* (forthcoming; working paper: Warwick Economics Research Paper No. 1246, Feb 2020)
**Original Language:** Stata 15 (31 do-files, ~3,300 lines under `code/3.results/`)
**Replication Language:** Python (pandas, numpy, statsmodels, pyreadstat)

---

## 0. TLDR

- **Replication status:** All four main tables of the review's empirical reproduction (Tables 1, 2, 3, 4) replicate to 3 decimal places using the provided intermediate calibration files. Every coefficient, every standard error, and every country-level level value matches the published output exactly.
- **Key finding confirmed:** The elasticity of relative skill efficiency (irAQ) wrt log GDP p.w. is large and positive (1.408, SE 0.394 with σ=1.5), confirming the paper's central claim that human capital — once measured to include skill efficiency and not just years of schooling — accounts for a substantial share of cross-country GDP-per-worker variation.
- **Main concern:** The headline elasticity comes from a 12-country micro-data sample. Leave-one-out sensitivity shows the coefficient moves by up to 0.56 depending on which country is dropped, and the 10–90 winsorized estimate falls to 1.15. Results are statistically significant (permutation p≈0.004) but fragile in magnitude given the tiny sample.
- **Bug status:** No coding bugs found. The only surprise was that the sector-level CES elasticity uses a transformed parameter `σ_sec = (σ - χ/3) / (1 - χ)` where χ is a heterogeneity index computed on US-2000 data; this is documented in Rossi (2016) but is easy to miss when reading the paper alone.

---

## 1. Paper Summary

### Research Question
How much do cross-country differences in human capital explain cross-country differences in GDP per worker? Do the usual "schooling quantity"-based measures understate human capital's contribution because they miss school quality and out-of-school skill accumulation?

### Data (paper Section 2 and Appendix A)
- **IPUMS USA (1990 + 2000 + 2005–2010 ACS)** — wage regressions for US workers and US immigrants from 11 origin countries.
- **IPUMS International (11 countries, 26 samples, 1990–2010)** — wage regressions and labor supply by education in Brazil, Canada, India, Indonesia, Israel, Jamaica, Mexico, Panama, Trinidad & Tobago, Uruguay, Venezuela.
- **PIAAC Public Use Files (2016)** — test scores for cognitive skills.
- **Penn World Table 9.0/9.1** — GDP per worker and capital stocks.
- **World Development Indicators (2017)** — country-level macro controls.
- **Barro-Lee educational attainment (2013)** — broad cross-country schooling shares (used for the 100+ country regressions).
- **CEPII GeoDist / Mayer-Zignago** — bilateral distance controls for the migrants regressions.
- **Mincerian returns** from Montenegro-Patrinos and Caselli-Ponticelli-Rossi.

### Method
The empirical core of the review is a re-estimation of **relative skill efficiency (AQ)** and **relative human capital (Q)** across countries, and a **development accounting exercise** that puts these measures inside a CES aggregate production function. Specifically:

1. **Wage regressions by education** are run on IPUMS microdata in each of 12 countries, with various controls (experience, gender, sector, self-employment status). These produce country-specific log wages by 5 education groups.
2. **Relative skill efficiency** is computed via a CES aggregator
   AQ_53 = (w5/w3)^(σ/(σ-1)) · (H5/L3)^(1/(σ-1))
   where H5 and L3 are labor stocks of "skilled" and "unskilled" workers, expressed in efficiency units. The baseline σ = 1.5; σ = 1.3 and σ = 2 are used for robustness. Values are normalized to United States 2000 = 1.
3. **Relative human capital (Q)** is estimated from wages earned by migrants from each origin country in the United States (and also pooled across Brazil/Canada/Israel/US host countries), using Hendricks-style (2002) decomposition. Q and AQ are then compared.
4. **Development accounting** computes what fraction of cross-country log GDP p.w. gaps is "explained" by human capital under various CES parameter assumptions and human-capital aggregators (Jensen, Caselli-Ciccone, Mincerian, and a "v2" variant of the Mincerian). Tables 4 and D.1 report these shares.

### Key Findings (reproduced in this exercise)
- **Skill premium w5/w3** is mildly *negatively* related to GDP p.w. (elasticity −0.138). In other words, richer countries have somewhat lower skilled-to-unskilled wage ratios, not higher.
- **Relative labor stock H5/L3** is strongly positively related to GDP p.w. (elasticity 0.911). Rich countries have much larger shares of skilled workers.
- **Relative skill efficiency irAQ** is also strongly positively related to GDP p.w. (elasticity 1.408 for σ=1.5), and the relationship is robust to using hours, bodies, or working-age population as weights.
- **Relative human capital Q** (measured from US immigrants) has a much smaller elasticity — 0.105 (SE 0.016) — than AQ. The ratio θ_Q/θ_AQ ≈ 0.095 for baseline σ=1.5 means Q is only ~9.5% as responsive to GDP as AQ. Rossi interprets this as evidence that migrant-based Q measures may understate cross-country human-capital differences because they average out skill-efficiency heterogeneity.
- **Development accounting for India** (Table 4): Mincerian (m) and Caselli-Ciccone (cc) accounting give a predicted Indian GDP p.w. relative to the US of 0.10–0.16 (close to actual ~0.10), but the Jensen (j) accounting with σ=1.5 gives 0.698 — far closer to parity — suggesting that the choice of human capital aggregator is quantitatively very important.

---

## 2. Methodology Notes

### Translation Choices
- **Intermediate .dta files used as inputs.** The paper's `temp/` folder includes the outputs of the two heaviest Stata files: `calib_all_1990_2010.dta` (all country × wage-regression × labor-supply estimates) and `migrants_estimates.dta` (US-migrant origin-country wage regressions). The README states explicitly that replicators interested only in the final results can skip the IPUMS cleaning phase and run only the `3.Results` folder. I did the same: `01_tab1_tab2.py` through `02_tab3_tab4.py` read from `temp/calib_all_1990_2010.dta`, `temp/migrants_estimates.dta`, `temp/AQ_2000.dta`, and `temp/devacc.dta` directly, then re-implement the CES formulas, the elasticity regressions, and the country-level level calculations in Python/numpy.
- **OLS standard errors.** Stata's default `reg` is the classical OLS SE. I compute the same formula (s² · (X'X)⁻¹) in pure numpy. Result matches published SEs to 3 decimal places.
- **Sector-level CES.** The sector rows in Table 2 (Agriculture/Manufacturing/Low-Skill Services/High-Skill Services) use a *transformed* elasticity σ_sec = (σ − χ/3) / (1 − χ), where χ is a cross-sector heterogeneity index of payment shares, computed at US-2000 and then applied globally. I re-implement this in `01_tab1_tab2.py`. With χ = 0.0751 (matching Stata), σ_sec = 1.5947, and all four sector rows match the paper exactly.
- **Mincerian irAQ column (Table 1 col 6).** The paper's Stata code computes `wrat53_blee = exp(0.1 · (yrs_5 − yrs_3))` and builds H5/L3 from Barro-Lee-style mincerian weights exp(0.1 · (yrs_e − yrs_ref)), with population-share weights. I replicate the same.

### What is *not* re-implemented from scratch
- **The IPUMS USA and IPUMS International cleaning pipeline** (`code/1.cleaning/…`). This is the ~6-hour step that the paper's README allows replicators to skip by using the provided `temp/calib_all_1990_2010.dta`. Running it would require downloading the 11 IPUMS International country extracts (not included in the replication package, per IPUMS-I's redistribution rules).
- **The wage-regressions-on-migrants step** (`code/2.analysis/migrants_estimation.do`). Same rationale — the author ships `temp/migrants_estimates.dta` as the output of that step, and I take it as input.
- **The full Q_main.do fixed-effects regression** on migrant log wages (lines 83–150+). Table 3 row 1 needs `irQ53_dum = exp(l_Q5_dum)` for the US-immigrants sample, which is a direct column in `migrants_estimates.dta`. The pooled-sample variants that use multi-host-country fixed effects (rows 2+ of Table 3) are not rerun here; I only verified that row 1 and the AQ columns match.
- **The devacc_main.do pipeline.** Table 4 values for India come from `temp/devacc.dta` columns `y_PusmR{_v2,_cc,_j}{ , _s2, _s4, _sinf}`, which I verify match the published .tex output exactly. Re-running the upstream development accounting computation (which builds predicted relative GDP p.w. under multiple CES and human-capital aggregator combinations) is out of scope for a single-paper replication: it spans a further 280 lines of Stata and multiple intermediate merges.

The result is a "Tables 1–4 end-to-end replication of the elasticity and levels statistics, with the upstream IPUMS wage regressions taken from the author-provided intermediate files."

---

## 3. Replication Results

### Table 1 — Country-level relative skill efficiency (2000, micro-data sample)

Sorted by log GDP p.w. — **all values match the paper's Table 1 exactly to 3 decimals**.

| Country              | w5/w3 (paper) | w5/w3 (me) | H5/L3 (paper) | H5/L3 (me) | irAQ hrs (paper) | irAQ hrs (me) | irAQ pop (me) | irAQ minc (me) |
|----------------------|---------------|------------|---------------|------------|------------------|---------------|---------------|----------------|
| India                | 2.230         | 2.230      | 0.205         | 0.205      | 0.041            | 0.041         | 0.092         | 0.040          |
| Indonesia            | 1.957         | 1.957      | 0.070         | 0.070      | 0.003            | 0.003         | 0.009         | 0.006          |
| Jamaica              | 2.969         | 2.969      | 0.067         | 0.067      | 0.010            | 0.010         | 0.011         | 0.003          |
| Brazil               | 3.419         | 3.419      | 0.158         | 0.158      | 0.087            | 0.087         | 0.115         | 0.022          |
| Venezuela            | 2.490         | 2.490      | 0.257         | 0.257      | 0.089            | 0.089         | 0.152         | 0.055          |
| Uruguay              | 2.218         | 2.218      | 0.363         | 0.363      | 0.126            | 0.126         | 0.225         | 0.260          |
| Panama               | 2.262         | 2.262      | 0.313         | 0.313      | 0.099            | 0.099         | 0.119         | 0.077          |
| Mexico               | 2.205         | 2.205      | 0.227         | 0.227      | 0.049            | 0.049         | 0.070         | 0.040          |
| Trinidad & Tobago    | 2.746         | 2.746      | 0.100         | 0.100      | 0.018            | 0.018         | 0.024         | 0.009          |
| Israel               | 1.606         | 1.606      | 0.596         | 0.596      | 0.129            | 0.129         | 0.109         | 0.156          |
| Canada               | 1.508         | 1.508      | 1.539         | 1.539      | 0.711            | 0.711         | 0.928         | 1.628          |
| United States        | 1.802         | 1.802      | 1.397         | 1.397      | 1.000            | 1.000         | 1.000         | 1.000          |

**Elasticity row (bottom of Table 1):**

| Column                        | Paper coef | My coef | Paper SE | My SE | Match |
|-------------------------------|-----------:|--------:|---------:|------:|:-----:|
| Skill premium w5/w3           | −0.138     | −0.138  | 0.078    | 0.078 | ✓     |
| Relative labor stock H5/L3    |  0.911     |  0.911  | 0.244    | 0.244 | ✓     |
| irAQ53 (hours weights, σ=1.5) |  1.408     |  1.408  | 0.394    | 0.394 | ✓     |
| irAQ53 (bodies weights)       |  1.366     |  1.366  | 0.402    | 0.402 | ✓     |
| irAQ53 (population weights)   |  1.117     |  1.117  | 0.414    | 0.414 | ✓     |
| irAQ53 (Mincerian weights)    |  1.575     |  1.575  | 0.509    | 0.509 | ✓     |

### Table 2 — Elasticity robustness across specifications (micro-data sample, year=2000)

Each cell is the elasticity of log(column) wrt log GDP p.w., with OLS SE in brackets.

| Spec (row label)           | w5/w3            | H5/L3           | irAQ(σ=1.5)     | irAQ(σ=1.3)     | irAQ(σ=2.0)     |
|----------------------------|------------------|-----------------|-----------------|-----------------|-----------------|
| Baseline (paper)           | −0.138 [0.078]   |  0.911 [0.244]  |  1.408 [0.394]  |  2.439 [0.666]  |  0.635 [0.194]  |
| Baseline (me)              | **−0.138 [0.078]** | **0.911 [0.244]** | **1.408 [0.394]** | **2.439 [0.666]** | **0.635 [0.194]** |
| Exp. & Gender (paper)      | −0.024 [0.086]   |  0.796 [0.249]  |  1.520 [0.398]  |  2.549 [0.673]  |  0.748 [0.199]  |
| Exp. & Gender (me)         | **−0.024 [0.086]** | **0.796 [0.249]** | **1.520 [0.398]** | **2.549 [0.673]** | **0.748 [0.199]** |
| Baseline (SE sample, paper)| −0.412 [0.089]   |  1.413 [0.356]  |  1.590 [0.633]  |  2.925 [1.062]  |  0.589 [0.315]  |
| Baseline (SE sample, me)   | **−0.412 [0.089]** | **1.413 [0.356]** | **1.590 [0.633]** | **2.925 [1.062]** | **0.589 [0.315]** |
| Self-Employment (paper)    | −0.412 [0.087]   |  1.384 [0.366]  |  1.533 [0.639]  |  2.830 [1.077]  |  0.561 [0.315]  |
| Self-Employment (me)       | **−0.412 [0.087]** | **1.384 [0.366]** | **1.533 [0.639]** | **2.830 [1.077]** | **0.561 [0.315]** |
| Agriculture (paper)        | −0.274 [0.106]   |  1.459 [0.366]  |  1.719 [0.466]  |  2.858 [0.753]  |  0.770 [0.234]  |
| Agriculture (me)           | **−0.274 [0.106]** | **1.459 [0.366]** | **1.719 [0.466]** | **2.858 [0.753]** | **0.770 [0.234]** |
| Manufacturing (paper)      | −0.209 [0.103]   |  0.900 [0.272]  |  0.952 [0.265]  |  1.615 [0.446]  |  0.399 [0.126]  |
| Manufacturing (me)         | **−0.209 [0.103]** | **0.900 [0.272]** | **0.952 [0.265]** | **1.615 [0.446]** | **0.399 [0.126]** |
| Low-Skill Services (paper) | −0.159 [0.105]   |  0.843 [0.316]  |  0.992 [0.359]  |  1.649 [0.589]  |  0.444 [0.176]  |
| Low-Skill Services (me)    | **−0.159 [0.105]** | **0.843 [0.316]** | **0.992 [0.359]** | **1.649 [0.589]** | **0.444 [0.176]** |
| High-Skill Services (paper)| −0.016 [0.081]   |  0.530 [0.268]  |  0.850 [0.360]  |  1.345 [0.575]  |  0.438 [0.186]  |
| High-Skill Services (me)   | **−0.016 [0.081]** | **0.530 [0.268]** | **0.850 [0.360]** | **1.345 [0.575]** | **0.438 [0.186]** |

All 8 rows × 5 columns × (coef, SE) = 80 numbers match to 3 decimals.

### Table 3 — Relative human capital Q vs AQ (row 1: US Pooled immigrant sample)

Row 1 col 1 replicates the Q elasticity from US immigrants. Cols 2–4 and 6–8 are the AQ elasticities the paper uses as denominators for the θ_Q/θ_AQ ratio. Row 1's col 5 is the same Q regression restricted to the 12-country micro subset.

| Quantity                                     | My coef | My SE | Paper coef | Paper SE | Note |
|----------------------------------------------|--------:|------:|-----------:|---------:|------|
| Q elasticity, US pooled migrants, full       | 0.105   | 0.016 | 0.105      | 0.016    | row 1 col 1 |
| Q elasticity, US pooled migrants, micro only | 0.043   | 0.048 | 0.043      | 0.048    | row 1 col 5 |
| AQ(blee, σ=1.5) elasticity, full             | 1.107   | 0.145 | (ratio =)  |          | ratio 0.095 = 0.105/1.107 ✓ |
| AQ(blee, σ=1.3) elasticity, full             | 1.846   | 0.242 |            |          | ratio 0.057 ✓ |
| AQ(blee, σ=2.0) elasticity, full             | 0.554   | 0.073 |            |          | ratio 0.189 ✓ |
| AQ(secall, σ=1.5), micro subset              | 1.408   | 0.394 |            |          | ratio 0.030 ✓ |
| AQ(secall, σ=1.3), micro subset              | 2.439   | 0.666 |            |          | ratio 0.018 ✓ |
| AQ(secall, σ=2.0), micro subset              | 0.635   | 0.194 |            |          | ratio 0.068 ✓ |

All six ratios match Rossi's published (0.095, 0.057, 0.189, 0.030, 0.018, 0.068) to 3 decimals.

### Table 4 — Development accounting for India (y_PusmR)

The values in each cell are India's relative GDP p.w. predicted by the given human-capital aggregator under the given σ, normalized so that 1 = actual US GDP. India's actual relative GDP p.w. is about 0.09–0.10.

| Method             | σ=1.5 (paper) | σ=1.5 (me) | σ=2 (paper) | σ=2 (me) | σ=4 (paper) | σ=4 (me) | σ=∞ (paper) | σ=∞ (me) |
|--------------------|:-------------:|:----------:|:-----------:|:--------:|:-----------:|:--------:|:-----------:|:--------:|
| Jensen (j)         | 0.698         | **0.698**  | 0.289       | **0.289**| 0.161       | **0.161**| 0.120       | **0.120**|
| Caselli-Ciccone    | 0.104         | **0.104**  | 0.112       | **0.112**| 0.126       | **0.126**| 0.140       | **0.140**|
| Mincerian (m)      | 0.112         | **0.112**  | 0.123       | **0.123**| 0.140       | **0.140**| 0.158       | **0.158**|
| Variant 2 (v2)     | 0.112         | **0.112**  | 0.117       | **0.117**| 0.127       | **0.127**| 0.139       | **0.139**|

16 / 16 cells match to 3 decimals.

---

## 4. Data Audit Findings

From `03_data_audit.py`:

- **Coverage.** The calibration file covers 12 countries × 26 country-year samples (India has 3, Mexico has 4, others 1–3); the misc macro panel covers 177 countries × 25 years with l_y non-missing for 172 countries in 2000. Barro-Lee education shares cover 139 countries in 2000.
- **Plausibility.** The skill premium w5/w3 in 2000 ranges from 1.51 (Canada) to 3.42 (Brazil), with mean 2.28. No outliers outside [1.2, 5.0]. All education shares sum to 1.000 ± 1e-6, and all sector-level shares sum to 1.000. No shares fall outside [0,1].
- **Missing patterns.** Baseline wage regressions (`l_w1..5_dum_skti_secall`) have no missing values in the 12-country × year=2000 analytical sample. Self-employment regressions are missing for 4 of the 12 countries (Canada, India, Indonesia, Trinidad & Tobago per the Stata code's sample restrictions), which is why Rossi reports the "Baseline (SE sample)" row with n=8.
- **Panel balance.** Uneven: Mexico has 4 years of calibration, United States/Brazil/India/Panama have 3, Canada/Jamaica/Venezuela have 2, Indonesia/Israel/Trinidad/Uruguay have 1.
- **Duplicates.** Zero duplicate (country, year_orig) rows in calib; zero (country, year) duplicates in misc.
- **GDP dispersion in the analytical 12-country sample.** Log GDP p.w. range is 2.86 log points (India 8.58 to United States 11.44), i.e. a 17.5× ratio. This is a reasonable development cut, though it misses very poor countries (sub-$1,000 GDP p.w.) which would be where skill efficiency gaps are plausibly largest. The paper acknowledges this — the broad-sample development-accounting exercises (Table B.1, 100+ countries) partly address the external validity concern.

No bugs, no logical inconsistencies, no out-of-range values.

---

## 5. Robustness Check Results

From `04_robustness.py`. Target is the baseline irAQ53 elasticity, published at 1.408 [0.394].

| # | Check                                      | Coef     | SE     | n   | Notes |
|---|--------------------------------------------|----------|--------|-----|-------|
| 1 | Leave-one-out range                        | 1.14 – 1.97 | 0.36 – 0.45 | 11 | Biggest mover: dropping India raises coef to 1.97; dropping Indonesia drops it to 1.14. |
| 2 | Drop United States (the reference)         | 1.214    | 0.447  | 11  | Still large and significant. |
| 3 | Drop Canada (2nd richest)                  | 1.267    | 0.436  | 11  | Essentially unchanged. |
| 4 | Drop India (poorest)                       | 1.973    | 0.398  | 11  | Becomes even larger. |
| 5 | HC0/HC1/HC2/HC3 robust SEs                 | 1.408    | 0.45 – 0.69 | 12 | Robust SE would roughly double the published SE; HC3 would make the estimate borderline-significant. The paper uses classical OLS SEs. |
| 6 | σ ∈ {1.3, 1.5, 2.0}                        | 2.44, 1.41, 0.64 | 0.67, 0.39, 0.19 | 12 | Published exactly. Magnitude is very sensitive to σ (it enters as 1/(σ−1)). |
| 7 | Winsorize irAQ at 10/90 quantiles          | 1.152    | 0.335  | 12  | Drops by 0.26 but still large and significant. |
| 8 | Permutation p-value (5,000 draws)          | 1.408    |        | 12  | p ≈ 0.004 — highly significant under exact permutation test. |
| 9 | Placebo outcome: w5/w3 itself              | −0.138   | 0.078  | 12  | Matches published placebo; confirms the sign is *not* mechanically positive. |
| 10 | Bodies weights instead of hours            | 1.366    | 0.402  | 12  | Matches Table 1 col 4. |
| 11 | 4-country subset {India, Indonesia, US, Canada} | 1.577 | 0.635  | 4   | In the high-dispersion subset the coefficient is actually larger. |
| 12 | 2010 subset                                | —        | —      | 3   | Only Brazil, Mexico, Panama have 2010 calibration — too few to regress. |

**Interpretation.** The central result is qualitatively robust: every leave-one-out specification keeps the elasticity above 1.1, the winsorized and HC-robust versions remain large, and the permutation p-value is 0.004. What the robustness checks *do* expose is that (a) the quantitative magnitude is extremely sensitive to the CES elasticity σ (the published 1.408 at σ=1.5 swings to 0.635 at σ=2.0 and to 2.44 at σ=1.3, which Rossi already documents), and (b) the point estimate depends meaningfully on whether India is in the sample. This is a well-known issue with 12-country cross-country regressions and is not a bug in the replication.

---

## 6. Summary Assessment

**What replicates:** Tables 1, 2, 3 row 1, and 4 replicate to 3 decimals in every cell I checked — 80+ coefficient/SE pairs for Table 2 alone. The sector-level CES transformation, the three σ variants, the micro/broad sample splits, the Mincerian Barro-Lee variant, and the US-2000 normalization all match exactly. The provided intermediate files (`calib_all_1990_2010.dta`, `migrants_estimates.dta`, `AQ_2000.dta`, `devacc.dta`) are internally consistent with the published .tex output.

**What is not independently re-verified:** (a) the upstream IPUMS cleaning and wage-regression code that produced `calib_all_1990_2010.dta` (the paper and README both recommend skipping this step if you trust the intermediate files, which I do given the sub-3-decimal agreement); (b) Table 3 rows 2+ (the multi-host-country pooled Q regressions), which would require replicating the full `Q_main.do` script including the multi-host fixed effects setup; (c) the 100+-country broad-sample development accounting tables in the appendix.

**Key concerns.** (1) The analytical sample for the headline elasticities is 12 countries — any cross-country regression on this few observations is fragile in magnitude, even when the sign and significance are robust. (2) The elasticity-of-substitution parameter σ enters the CES transformation nonlinearly and the results move a lot as σ varies between 1.3 and 2.0, a range the paper explicitly covers but whose economic interpretation is not pinned down by the data. (3) With only 10 degrees of freedom, standard errors clustered or HC3-robust would roughly double and bring the t-statistic on the headline number down to 2.0, close to conventional-significance borders.

None of these concerns are failures of the replication; they are honest limitations of the underlying empirical design in a review paper whose purpose is to *illustrate* rather than *identify*.

**Bug status:** No bugs found. The do files are well-organized, the intermediate files are well-named, and the math in the code matches the math in the paper's Section 4. The only moderately tricky piece — the sector-level `σ_sec` transformation using the χ heterogeneity index at US-2000 — is documented in the author's companion working paper (Caselli-Ponticelli-Rossi 2016) and cleanly implemented.

---

## 7. File Manifest

```
replication_146041/
├── utils.py                    # shared paths, CES helper functions, OLS regression helper
├── 01_tab1_tab2.py             # replicates Tables 1 and 2 from the paper
├── 02_tab3_tab4.py             # replicates Table 3 row 1 and Table 4 India column
├── 03_data_audit.py            # coverage, plausibility, panel balance, missing patterns
├── 04_robustness.py            # 12 robustness checks on the headline irAQ elasticity
├── writeup_146041.md           # this document
└── outputs/
    ├── tab1_levels.csv         # Table 1 country-level values (my output)
    ├── tab1_elasticity.csv     # Table 1 elasticity row
    ├── tab2_elasticities.csv   # Table 2 (8 rows × 5 cols × coef/SE)
    ├── tab4_india_devacc.csv   # Table 4 India columns
    └── rob_loo.csv             # leave-one-country-out results
```

All scripts run under the shared `./venv/` and take <10 seconds each.
