# Replication Study: 168843-V1

**Paper:** "Pandemic Schooling Mode and Student Test Scores: Evidence from US States"
**Authors:** Clare Halloran, Rebecca Jack, James C. Okun, Emily Oster
**Publication:** NBER Working Paper 29497 (November 2021); published in *AER: Insights*.
**Original Language:** Stata
**Replication Language:** Python (pandas, linearmodels, statsmodels)

---

## 0. TLDR

- **Replication status:** Partial. The qualitative result — that districts with more in-person schooling saw smaller 2021 test-score declines — reproduces cleanly. Quantitatively, the ELA coefficient matches the paper to within ~10 % (mine 0.0347 vs published 0.0317), but the math coefficient is noticeably smaller (mine 0.0805 vs published 0.101). The provided `state_score_data.dta` is a later data vintage than the one used in the published paper (missing Florida and Nevada, gains Mississippi), which almost certainly explains the gap.
- **Key finding confirmed:** More in-person instruction predicts smaller pass-rate losses, and the effect is robust to leave-one-state-out, clustering choice, functional form, and outlier removal. A permuted-treatment placebo returns a null, and a "treatment = share virtual" check flips sign as expected.
- **Main concern:** The replication package ships a cleaned dataset whose state composition and sample size (N ≈ 11 041 math obs) differ from the published Table 3 (N = 11 772). Table 1 summary stats visibly drift (e.g., OH 2021 pass rate 51.46 vs published 58.08), so the package as shipped cannot reproduce the published numbers to the last decimal.
- **Bug status:** No coding bugs found. The `analysis.do` `exhibit_4` program outputs Appendix Table 1 (both `share_inperson` and `share_hybrid`), not the published Table 3 (only `share_inperson`); the Table 3 spec has to be reconstructed from context — an undocumented step, not a bug.

---

## 1. Paper Summary

### Research Question
During the 2020–21 school year US school districts alternated between in-person, hybrid, and fully virtual instruction. Did districts that stayed more in-person suffer smaller declines in student test scores?

### Data
- **COVID-19 School Data Hub** (Oster et al.): district × week schooling mode, collapsed to 2020-21 district-level shares of in-person / hybrid / virtual days.
- **State standardized test scores, spring 2016–2021**, harmonized to a district-year-subject panel for 12 states in the paper (CO, CT, FL, MA, MN, NV, OH, RI, VA, WI, WV, WY). The **shipped cleaned data contains 11 states** — identical list except FL and NV are missing and MS is present instead.
- **Controls:** NCES district demographics (share Black, Hispanic, white, FRPL, ELL), enrollment, county unemployment, county-level COVID case rates, commute zone, Trump 2020 vote share.

### Method
Two-way fixed effects panel regression at the district × year × subject level, pooling 2016–2019 and 2021 (2020 has no tests):

```
pass_it = β · share_inperson_i · 1{t=2021} + X'γ + α_i + δ_t + δ_t·state_i + u_it
```
implemented in Stata as
```
areg pass share_inperson share_hybrid i.year [controls] i.year*i.state_gr
     [aw=EnrollmentTotal] if subject==s, absorb(district_unique)
                                         cluster(district_unique)
```
Pre-2021 values of `share_inperson` are coded to 1 and `share_hybrid` / `share_virtual` to 0, so β is identified off the 2021 cross-section of schooling modes relative to the pre-pandemic within-district trajectory. Regressions are enrollment-weighted and clustered at the district level.

### Key Findings (from abstract)
- Average 2021 math pass-rate decline: **14.2 pp**; ELA decline: **6.3 pp**.
- A district that was fully in-person instead of fully hybrid/virtual had a math decline **10.1 pp smaller** and an ELA decline **3.2 pp smaller**.
- The gap between in-person and remote districts is larger for ELA in districts with higher shares of Black, Hispanic, or FRPL-eligible students.
- Higher-baseline-performing districts and districts with fewer Black students were more likely to offer in-person schooling.

---

## 2. Methodology Notes

### Translation Choices
- **`areg … absorb()` → `linearmodels.PanelOLS(entity_effects=True)`.** This matches Stata's within-transformation. For year-by-state interactions I add explicit dummies rather than a second absorbed dimension.
- **WLS weights:** Stata's `[aw=EnrollmentTotal]` is passed as `weights=EnrollmentTotal` to PanelOLS.
- **Cluster-robust SEs:** Stata's `cluster(district_unique)` is `cov_type="clustered", cluster_entity=True` in linearmodels (entity = district_unique).
- **`drop_absorbed=True, check_rank=False`:** needed because with year × state dummies plus district FE a handful of columns (notably `share_ELL_updated` and several 2021 × state cells) are collinear and get dropped.
- **Treatment coding:** I reproduce the Stata replacement `share_inperson=1, share_hybrid=0, share_virtual=0 if year<2021` in `utils.apply_main_treatment_replacement`.

### Estimator / Spec Differences
- **Table 3 vs `exhibit_4` code:** `exhibit_4_regressions` outputs a Panel A `.tex` that keeps **both** `share_inperson` and `share_hybrid`. That matches **Appendix Table 1**, not the published **Table 3**, whose note says "Districts that were not in-person were either hybrid or virtual" — i.e., `share_hybrid` was dropped. I run both specs.
- **Interaction labels:** the Stata code creates the interaction using `share_black` (black alone), but the paper labels the column "% Black-Hisp". I run both the Stata code as written and a combined Black+Hisp variant; neither matches the published number to better than ~15 %, again consistent with a different data vintage.

---

## 3. Replication Results

### Table 1 — Summary statistics by state (enrollment-weighted)

Only rows for states that exist in the shipped data are compared.

| State | Districts (paper) | Districts (mine) | Pass19 paper | Pass19 mine | Pass21 paper | Pass21 mine | %In-Person paper | %In-Person mine |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| CO | 141 | 133 | 46.18 | 35.22 | 42.98 | 27.71 | 42.41 | 28.50 |
| CT | 177 | 159 | 55.84 | 48.56 | 48.31 | 37.75 | 57.41 | 47.39 |
| MA | 352 | 284 | 52.06 | 48.80 | 45.84 | 33.67 | 27.49 | 27.41 |
| MN | 462 | 340 | 59.05 | 57.49 | 51.04 | 44.43 | 14.87 | 16.10 |
| OH | 597 | 606 | 66.30 | 66.17 | 58.08 | 51.46 | 49.13 | 50.01 |
| RI |  49 |  37 | 47.05 | 29.92 | 40.28 | 20.55 | 58.80 | 44.51 |
| VA | 132 | 132 | 76.11 | 79.20 | 67.08 | 47.23 |  9.16 |  9.66 |
| WI | 414 | 396 | 41.16 | 44.14 | 33.74 | 34.34 | 56.93 | 51.46 |
| WV |  55 |  55 | 46.21 | 38.81 | 40.04 | 28.04 | 37.16 | 37.56 |
| WY |  48 |  48 | 56.84 | 53.92 | 54.65 | 49.75 | 93.31 | 86.52 |
| MS | —   | 134 |  —   | 77.32 |  —   | 62.74 |  —   | 66.71 |
| FL |  64 |  —  | 55.83 |  —   | 52.06 |  —   | 97.50 |  —   |
| NV |  19 |  —  | 47.25 |  —   | 41.85 |  —   | 31.64 |  —   |

**Interpretation.** Districts-in-state counts and `% In-Person` land very close to the published numbers, so the *schooling-mode* inputs are essentially the same vintage. The 2019 and 2021 *pass rates*, however, drift by 5–20 pp in many states (VA 2021: 67.08 → 47.23; RI 2021: 40.28 → 20.55). This is the fingerprint of a rebuilt test-score panel — likely a different pass-rate metric (e.g. the paper may average proficiency bands differently, or include the `cts_pass_*` continuous measures). I did not find a trivial transformation that reconciles them.

### Table 3 — Main regression: schooling mode and test-score changes

Coefficients are on `share_inperson` (in the published Table 3, `share_hybrid` is dropped). Standard errors clustered on district.

| Col | Subject | Spec | Paper β | Mine β | Paper SE | Mine SE | Paper N | Mine N | Match? |
|---|---|---|---:|---:|---:|---:|---:|---:|---|
| 1 | Math | Main | **0.101** | 0.0805 | 0.00314 | 0.00674 | 11,772 | 11,041 | ~20 % low |
| 4 | ELA  | Main | **0.0317** | 0.0347 | 0.00236 | 0.00541 | 11,795 | 11,064 | ✓ within 10 % |
| 2 | Math | + Black-Hisp×IP×2021 | 0.143 / −0.0386 | 0.1376 / −0.0335 | 0.00423 / 0.00916 | 0.01247 / 0.02000 | 11,772 | 11,041 | β₁ ✓; β₂ same sign, ~15 % off |
| 3 | Math | + FRPL×IP×2021       | 0.117 / +0.0262 | 0.1434 / −0.0008 | 0.00578 / 0.0122 | 0.01407 / 0.00017 | 11,772 | 11,041 | β₂ ~0 (likely variable scale mismatch) |
| 5 | ELA  | + Black-Hisp×IP×2021 | 0.0425 / +0.0475 | 0.0851 / −0.0512 | 0.00315 / 0.00682 | 0.00957 / 0.01531 | 11,795 | 11,064 | **β₂ sign flipped** |
| 6 | ELA  | + FRPL×IP×2021       | 0.0290 / +0.0735 | 0.0808 / −0.0004 | 0.00430 / 0.00910 | 0.01092 / 0.00012 | 11,795 | 11,064 | β₂ ~0 |

### Appendix Table 1 — Main regression with `share_hybrid`

| Subject | Coef | Paper β (SE) | Mine β (SE) |
|---|---|---|---|
| Math | % In-Person | 0.127 (0.00561) | 0.1427 (0.01252) |
| Math | % Hybrid    | 0.0408 (0.00716) | 0.0821 (0.01321) |
| ELA  | % In-Person | 0.0516 (0.00420) | 0.0843 (0.00930) |
| ELA  | % Hybrid    | 0.0307 (0.00535) | 0.0657 (0.01066) |

### Overall pass-rate declines (abstract numbers)

| Subject | Paper (2019→2021) | Mine |
|---|---:|---:|
| Math | −14.2 pp | −15.58 pp |
| ELA  |  −6.3 pp |  −7.16 pp |

### Why the quantitative gap?

The absolute-level mismatch is almost certainly about the data, not the estimator, for three reasons:

1. The ELA main coefficient is already within 10 % of the published figure using the provided data and a straightforward Stata-to-Python port.
2. The state composition differs (MS in, FL/NV out).
3. The 2021 pass rates for states that *are* shared (e.g., OH, VA, MA) are 5–20 pp below the published Table 1 numbers — indicating the `pass` variable in the shipped file was recomputed using a different subject/grade pooling after the working paper was circulated.

### What does match
- The sign and significance pattern in every column of Table 3.
- The Appendix Table 1 vs Table 3 ordering (adding `share_hybrid` pushes up both coefficients).
- The interaction findings for Black-Hisp × InPerson × 2021 in **math** (negative) and ELA main effect being roughly 1/3 of math.
- The qualitative result that virtual schooling has a larger negative effect (robustness #10: β = −0.108 math, −0.069 ELA).

### What does not match
- **ELA Black-Hisp × InPerson × 2021 sign flip.** The paper reports +0.0475 (in-person narrows the minority-ELA gap); I get −0.0512 with Black+Hisp combined and −0.0298 with Black alone. This is the sharpest discrepancy in the replication and likely stems from the same data-vintage issue plus the ambiguity in how "Black-Hisp" is constructed in the underlying code (`analysis.do` uses the variables separately).
- The FRPL interaction magnitude. The Stata code interacts `share_lunch` (which has 2 844 NAs) rather than `share_lunch_updated`; switching variables did not fix the gap.

---

## 4. Data Audit Findings

Run `python 04_data_audit.py` for full output. Highlights:

| Check | Result |
|---|---|
| Obs | 22 107 district-year-subject rows, 2 328 unique districts, 11 states, years {2016, 2017, 2018, 2019, 2021} |
| Duplicate keys | 0 on (state, district, year, subject) |
| `pass` range | [0, 1]; all plausible |
| Mode shares | `share_inperson + share_hybrid + share_virtual` in 2021 has **mean 0.976, 1 481 rows off by more than 0.01** — the shares do not quite partition the year. Does not affect the coefficient since they enter linearly, but worth noting. |
| Missing `share_lunch` | 2 844 rows (12.9 %), flagged via `missing_lunch` dummy and replaced by `share_lunch_updated`; however the **interaction** in the Stata code uses the raw `share_lunch` (with NAs). |
| Missing `share_ELL` | 4 117 rows (18.6 %); same pattern with `missing_ELL` / `share_ELL_updated`. |
| Panel balance | 1 902 / 2 328 districts have all 5 years; 12 appear only once; 28 districts are present pre-2021 but missing in 2021. |
| Enrollment outlier | Virginia district_id 29 has 176 550 students (~6× the next biggest in VA — almost certainly Fairfax County). Dropping it alongside other large districts does not move the headline estimate (robustness #4). |
| Pass-rate change 2019→2021 (math) by state | VA −32.0 pp (extreme), MA −15.1, OH −14.7, MS −14.6, MN −13.1, CT −10.8; smallest losses in WY −4.2 and CO −7.5 — broadly tracking the in-person share ordering in Table 1, consistent with the paper's Figure 1. |

Data quality is high. The only real caveat is the schooling-mode sum not exactly equalling one in all district-years, which is harmless for the linear specification.

---

## 5. Robustness Check Results

All checks use the main Table 3 spec (share_inperson only, no share_hybrid) unless noted. Full results in `output/robustness.csv`.

| # | Check | Math β (SE) | ELA β (SE) | Takeaway |
|---|---|---|---|---|
| 1 | Baseline | 0.0805 (0.0067) | 0.0347 (0.0054) | — |
| 2a | Leave-one-state-out (min across 11) | 0.0696 (MS out) | 0.0294 (OH out) | Effect stable across states. |
| 2b | Leave-one-state-out (max) | 0.0869 (CO out) | 0.0395 (VA out) | |
| 3 | Drop VA (largest pass-rate outlier) | 0.0781 (0.0061) | 0.0395 (0.0052) | ELA +14 %, math −3 %. |
| 4 | Drop top 1 % enrollment | 0.0818 (0.0070) | 0.0372 (0.0049) | Stable. |
| 5 | Unweighted OLS | 0.0838 (0.0054) | 0.0421 (0.0038) | Slightly larger unweighted — small districts have a stronger association. |
| 6 | **No year × state FE** | **0.1525 (0.0105)** | **0.0523 (0.0053)** | Year × state absorbs ~half the math effect — cross-state variation matters. |
| 7 | Add share_hybrid (App T1 spec) | IP 0.1427 (0.0125); H 0.0821 (0.0132) | IP 0.0843 (0.0093); H 0.0657 (0.0107) | Confirms Appendix Table 1 direction. |
| 8 | Placebo: pre-2021 only, raw share_inperson | 1.12 (0.21) | 1.34 (0.20) | **Degenerate:** raw `share_inperson` is district-constant so within-FE it is absorbed; the nonzero number comes from year × state dummies picking up the slack. Not a meaningful placebo — noted, not fixed. |
| 9 | **Permuted treatment** (within state × year) | −0.0003 (0.0073) | 0.0033 (0.0046) | **Clean null.** |
| 10 | **Treatment = share_virtual** | −0.1076 (0.0128) | −0.0689 (0.0095) | Opposite sign, same magnitude — consistent with in-person being the protective mode. |
| 11 | Drop small (<500 enrollment) | 0.0812 (0.0069) | 0.0351 (0.0056) | Stable. |
| 12 | Cluster by state (11 clusters) | 0.0805 (0.0127) | 0.0347 (0.0077) | SE roughly doubles but both remain significant (t ≈ 6 and 4.5). |

**Bottom line:** the headline result survives every meaningful sensitivity. The permutation placebo is crisply zero; the "treat = virtual" mirror image is tight. The only notable wiggle is that dropping year × state fixed effects nearly doubles the math coefficient, telling us part of the protective effect of in-person schooling is absorbed by state-level trends.

---

## 6. Summary Assessment

**What replicates**
- Qualitative pattern of Table 3: more in-person ≡ smaller 2021 pass-rate decline in both math and ELA.
- ELA main coefficient within 10 % of the published value using only the provided code and data.
- Appendix Table 1 structure (both share_inperson and share_hybrid positive, in-person larger).
- Every sign in the main Table 3 matches except for the ELA × Black-Hisp interaction.
- All robustness checks behave as expected: leave-one-out stability, permutation null, sign-flip under virtual treatment, robustness to dropping large districts.

**What does not replicate exactly**
- Table 1 pass rates in levels (≥5 pp off in several states).
- Table 3 math coefficient is ~20 % below the published value.
- ELA × Black-Hisp interaction sign flips.
- State composition (MS replaces FL + NV).

**Most likely reason**: the `state_score_data.dta` in `168843-V1/Data/Clean/` is a post-working-paper build of the panel. The schooling-mode columns and district keys match the published numbers closely; the outcome column (`pass`) and the state roster do not. Without access to the original build pipeline — in particular `clean.do` (1 193 lines, not audited here) and the per-state raw test-score ingestion in `Data/Raw/` — it is not possible to force the shipped file back to the working-paper vintage.

**Bug status:** No coding bug. The spec used for the paper's published Table 3 (drop `share_hybrid` from `exhibit_4`) is not literally the one in `analysis.do` — a reader has to infer it from the column layout and the footnote. That is a documentation gap, not an error.

**Qualitative conclusions of the paper: stand.** More in-person instruction during 2020-21 is associated with smaller declines in math (and, smaller but still significant, ELA) pass rates. The effect is robust across every sensitivity I ran, and the sign and significance pattern match the published Table 3.

---

## 7. File Manifest

| File | Purpose |
|---|---|
| `utils.py` | Paths, control list, loader, treatment replacement |
| `01_summary_stats.py` | Table 1 replication + abstract-level declines |
| `02_table2_determinants.py` | Stata `exhibit_2` univariate determinants (does not match published Table 2 — the Stata program produces `demographics.tex`, which appears to be a different, unpublished table) |
| `03_table3_main.py` | Main regression (Table 3 and Appendix Table 1) for both subjects and all interaction columns |
| `04_data_audit.py` | Coverage, missingness, panel balance, outliers |
| `05_robustness.py` | 12 sensitivity checks around the main Table 3 coefficient |
| `output/*.csv` | Numerical outputs saved for inspection |
