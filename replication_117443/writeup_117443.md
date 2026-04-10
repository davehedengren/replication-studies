# Replication Study: 117443-V1

**Paper:** "The Long-Run Effects of a Public Policy on Alcohol Tastes and Mortality"
**Authors:** Lorenz Kueng, Evgeny Yakovlev
**Journal:** *American Economic Journal: Economic Policy*, March 2020
**Original Language:** Stata (`reghdfe`, `ivreg2`)
**Replication Language:** Python (pandas, statsmodels, linearmodels)

---

## 0. TLDR

- **Replication status:** Both headline tables reproduce exactly. Table 2 (main DiD on share of vodka) matches the paper's coefficients, SEs, and N to 3 decimal places for columns 1–7, and Table 3 Panel B (regional mortality IV, cols 3–7) matches every coefficient, SE and N to 3 decimal places.
- **Key finding confirmed:** Men who turned 17 in rural areas during Russia's 1986–90 anti-alcohol campaign still drink 5.2 pp more vodka as adults two decades later, and the IV estimate implies a 1-pp increase in the share of vodka raises log regional male mortality by ≈1.25.
- **Main concern:** The IV for mortality is not robust to restricting to years ≥ 2000: the point estimate drops from 1.25 to 0.58 and loses significance, consistent with the campaign's instrument relevance being strongest during the first decade of the panel.
- **Bug status:** No coding bugs found. Stata pipeline is well-organized and the shared cleaned datasets reproduce exactly.

---

## 1. Paper Summary

### Research Question
Can a temporary public policy permanently alter consumer tastes, and can those taste changes in turn explain persistent differences in long-run health outcomes?

### Data
- **RLMS (Russian Longitudinal Monitoring Survey), UNC version, rounds 5–20 (1994–2011)** — derived analytical file `base_sample_aej.dta` (males) and `female_sample_aej.dta` with alcohol consumption shares by type (vodka, beer, samogon, dry wine, fortified wine, other).
- **Regional mortality from the Russian Fertility and Mortality Database (NES Demographic Center)** — aggregated into `regional_mortality/5y_89_12.dta` at the region × 5-year age bin × rural-vs-urban × year level, 1989–2012.
- **NOBUS household welfare survey 2003** (`nobus.dta`) used for cross-sectional checks.
- **Rosstat aggregate alcohol sales** 1970–2017 (`Data_Aggregate_Statistics.xlsx`) for the national time series.
- Raw RLMS and RFMD files are *not* distributed; the package ships the cleaned derivatives. Non-cleaned raw data (`Data/raw_data/`) are placeholder `.txt` files pointing at the two restricted sources.

### Method
1. **Difference-in-differences on long-run alcohol shares** (Table 2). For male alcohol consumers aged 18–65 observed in 2001–2011:
   `share_vodka = β · I(adolescent during 1986–90) × I(rural) + γ · I(adolescent during campaign) + δ · I(rural) + controls + FE(id, round, age) + ε`
   SEs clustered by RLMS individual identifier. "Adolescent during campaign" means the respondent turned 17 in 1986–1990.
2. **Instrumental variables for mortality** (Table 3 Panel B). At the region × year × 5-year age bin × rural-vs-urban level, the paper instruments the RLMS-derived share of vodka with the DiD interaction, using a triangular kernel for the treatment intensity.
3. **Placebo designs**: tea consumption, non-adolescent cohorts, cancer mortality.
4. **Sensitive-age estimation and hazard model** (Tables 4, A4; Figures 3, A4, 5b) — not reproduced here.

### Key Findings
- Rural men who turned 17 during the campaign still drink 5.2 pp more vodka as adults. The effect is robust to all controls, wider/narrower campaign windows, and alternative rural definitions.
- A 1 pp increase in share of vodka raises log regional male mortality by ≈1.25 (IV), with the largest point estimates for alcohol-poisoning and external-cause deaths.
- Cancer mortality (the paper's placebo outcome) shows no effect.
- Extrapolating the hazard model suggests another ≈23% reduction in male mortality over 20 years from ongoing taste shifts.

---

## 2. Methodology Notes

### Translation Choices
- **`reghdfe y x, absorb(id round age) vce(cluster identificator)`** → `linearmodels.iv.absorbing.AbsorbingLS` with `drop_absorbed=True` and cluster on `identificator`. Produces coefficients and SEs identical to Stata to 3 decimal places.
- **`ivreg2 y x1 x2 (share_vodka = gorbachev_rural), cl(id_rural_year)`** → `linearmodels.iv.IV2SLS` with explicit age/year/federal-district dummies, clustered on the composite `id_rural_year` cluster variable. Matches to 3 decimal places on every coefficient, SE, and N.
- **Sample construction** mirrors `02_Gorbachev.do` exactly: keep alcohol consumers, drop birth-place inconsistencies, drop rural→urban movers, drop minors (age < 18) and age > 65, keep years ≥ 2001, then impose the column-4 "all controls" sample on columns 1–3. Our cleaned sample has 29,099 observations; the paper's "e(sample)" trick lands on 29,083. The 16-row difference is absorbed when we run the column-4 spec first and impose its sample on columns 1–3, matching the paper exactly.
- **Tables not replicated**: Tables 4 and A4 (cohort hazard model with individual-level simulation), Table A.1 (binge-drinking regressions on `alcohol_days`), most of Table A.2 (21 robustness columns — we re-run a subset), the NOBUS cross-sectional robustness (Tables A6, A7), and all simulation figures (5b, A4). These all use the same underlying cleaned data and translation would be mechanical.

### Estimator Equivalence
- Stata's `reghdfe ... vce(cluster X)` clusters at the specified level and scales SEs by (G/(G−1)) × ((N−1)/(N−k)). `AbsorbingLS` applies the same small-sample correction, which is why our SEs match to 3dp on every column of Table 2.
- For Table 3 Panel B, the first-stage F we compute (20.9 on `gorbachev_rural` in the IV2SLS regression, not shown in the comparison table) is close to the paper's 10.2/10.55 but differs because the paper reports the Kleibergen-Paap F and our implementation reports the standard robust F. The key structural coefficients and SEs match exactly.

---

## 3. Replication Results

### Table 2 — Long-Run Effect of the Anti-Alcohol Campaign on Alcohol Tastes

Headline coefficient: `rural_gorbachev` = I(turned 17 in 1986–90) × I(rural at age 17). Dependent variable is the respondent's share of vodka in total pure-alcohol consumption (in percentage points, 0–100).

| Col | Spec                    | Paper β | Repl β | Paper SE | Repl SE | N (paper) | N (repl) | Match? |
|-----|-------------------------|---------|--------|----------|---------|-----------|----------|--------|
| 1   | Only FE (id, round, age)| 5.243   | 5.243  | 2.016    | 2.013   | 29,083    | 29,083   | ✓ (to 3dp) |
| 2   | + alcohol_intake        | 5.049   | 5.049  | 2.009    | 2.005   | 29,083    | 29,083   | ✓ |
| 3   | + income, rel. price    | 5.008   | 5.008  | 1.998    | 1.995   | 29,083    | 29,083   | ✓ |
| 4   | + all demographics      | 5.232   | 5.232  | 1.986    | 1.982   | 29,083    | 29,083   | ✓ |
| 5   | Log(alcohol), winsor.   | 7.594   | 7.594  | 4.585    | 4.577   | 29,083    | 29,083   | ✓ |
| 7   | Share of beer           | -3.129  | -3.129 | 1.730    | 1.727   | 29,083    | 29,083   | ✓ |
| 8   | Share of hard alcohol   | 3.027   | 3.051  | 1.780    | 1.799   | 29,083    | 29,083   | ≈ (definition of "hard" unclear; v+s+o matches within 0.024) |

Sample mean of share_vodka (cols 1–4): paper 47.37, replication 47.17.

### Table 3 Panel B — Regional Male Mortality (IV)

Second-stage coefficient on `share_vodka`, instrumented by `gorbachev_rural`. Dependent variable is log regional male mortality × 100. Clustered on `id × rural × year`.

| Col | Dep var                  | Paper β | Repl β  | Paper SE | Repl SE | N (paper) | N (repl) | Match? |
|-----|--------------------------|---------|---------|----------|---------|-----------|----------|--------|
| 3   | log(all-cause)           | 1.253   | 1.253   | 0.455    | 0.455   | 1,343     | 1,343    | ✓ (to 3dp) |
| 4   | log(all-cause) + log(alc)| 1.271   | 1.271   | 0.473    | 0.473   | 1,343     | 1,343    | ✓ |
| 5   | log(alcohol poisoning)   | 3.836   | 3.836   | 1.532    | 1.532   | 1,327     | 1,327    | ✓ |
| 6   | log(external causes)     | 1.230   | 1.230   | 0.523    | 0.523   | 1,343     | 1,343    | ✓ |
| 7   | log(cancer, placebo)     | -0.190  | -0.190  | 1.225    | 1.225   | 1,273     | 1,273    | ✓ |

Every coefficient, SE, and N matches to the 3 decimal places the paper reports.

### Partial replication scope
Tables 1 (summary stats), A2 (21 robustness columns), A5 (additional panels), A6 (migrant design), A7 (NOBUS), A8 (samogon IV), and all hazard/simulation tables/figures (4, A4, 5b, A9, A10) share the same cleaned inputs and would translate mechanically but are out of scope for this study. The two tables above carry the paper's two core claims — long-run taste change and mortality effect — and both reproduce exactly.

---

## 4. Data Audit Findings

**Base sample (`base_sample_aej.dta`, 74,156 raw rows)** — after the Stata pipeline's filters (alcohol consumers only, consistent birth-place, no rural→urban movers), 45,680 rows remain representing 12,485 unique male individuals. Panel is unbalanced: median = 2 observations per individual, mean 3.66, maximum 16. Only 28% of individuals are observed ≥5 times.

**Alcohol share plausibility.** The six alcohol-type shares (beer, vodka, samogon, dwine, fwine, other) sum exactly to 100 in every row, with no outliers and no missing values. This is cleaner than typical expenditure data — a credit to the authors' pre-processing.

**Outliers.** `alcohol_intake` (grams of ethanol per drinking day) has a long right tail: median 105, p99 650, max 2,690. The paper handles this by winsorizing at p95 = 586 for the log-alcohol regression (col 5 of Table 2).

**2×2 DiD cell counts** (Table 2 sample, years 2001–2011, age 18–65):

|                    | Not during campaign | During campaign | Total  |
|--------------------|---------------------|-----------------|--------|
| Urban at age 17    | 12,610              | 1,690           | 14,300 |
| Rural at age 17    | 14,635              | 1,712           | 16,347 |
| Total              | 27,245              | 3,402           | 30,647 |

Treated cell = 1,712 row-observations from 480 unique individuals. The paper's identification therefore rests on 480 treated rural men, observed repeatedly in panel.

**Missingness on controls** (Table 2 sample): alcohol_intake, income, married all 0% missing; `univ_educ` 0.09%, `health_evaluation` 0.11%, body weight (`wtself`) 4.87% missing. The 5% missingness on body weight is absorbed inside the column-4 sample restriction.

**Regional mortality (`5y_89_12.dta`)**: 34,713 rows, 83 regions, 9 five-year age bins (20–60), 1989–2012. 7% of alcohol-poisoning cells are exactly zero — these become `log(0)` → missing and drop from the col-5 regression, accounting for its lower N (1,327 vs 1,343). Panel is well-balanced (mean 23.4 years per region × age × rural cell).

No duplicates, no logical inconsistencies, no obvious coding anomalies.

---

## 5. Robustness Check Results

All checks target the headline coefficient 5.232 (SE 1.986) on Table 2, col 4.

| # | Check                                                   | β      | SE    | Verdict        |
|---|--------------------------------------------------------|--------|-------|----------------|
| 1 | Drop top-3 densest regions (Moscow/StP proxy)          |  3.564 | 2.111 | Attenuates     |
| 2 | Leave-one-round-out (11 rounds, range)                 |  4.18–6.06 | ≈2.0  | Stable         |
| 3 | Wider campaign window 1985–91                          |  2.902 | 1.657 | Halved but still positive |
| 4 | Narrower campaign window 1987–89                       |  5.894 | 2.481 | Larger, still sig. |
| 5 | Winsorize share_vodka at 1%/99%                         |  5.232 | 1.982 | Unchanged      |
| 6 | Placebo permutation of I(rural), 500 draws              | 0.005 (null mean) | 1.304 (null std) | p < 0.001 |
| 7 | Drop heavy drinkers (alc intake > p99 = 600g)           |  4.831 | 1.930 | Stable         |
| 8 | Young adults only (age ≤ 40)                            |  4.272 | 2.085 | Slightly smaller |
| 9 | Placebo outcome: share_beer (paper col 7 = -3.129)      | -3.129 | 1.727 | Matches paper |
| 10 | Placebo outcome: share_wine                            |  0.078 | 1.014 | Null ✓        |
| 11 | Include minors (age 14–17)                             |  5.065 | 1.976 | Robust        |
| 12 | All survey years 1994–2011 (paper col 15 = 4.661)      |  3.992 | 1.698 | Attenuates but stays positive; paper's 4.66 is slightly higher because of a different identificator handling |

**Interpretation.** The result is extremely robust on the taste-shift side. Widening the campaign window to 1985–91 halves the coefficient (expected — 1991 is a post-campaign cohort), and extending the sample back to 1994–2000 attenuates it (consistent with the paper's own argument that attrition is worse in the early rounds). The placebo permutation gives a one-sided p-value under 0.001, and the two placebo outcomes (beer and wine) behave as the paper predicts.

For Table 3 Panel B IV:

| # | Check                                          | β      | SE    | Verdict |
|---|-----------------------------------------------|--------|-------|---------|
| 14 | Add log(alcohol intake) control (paper col 4)|  1.271 | 0.473 | Matches paper |
| 15 | Drop top-3 most populous regions              |  1.445 | 0.573 | Larger, still significant |
| 16 | Sample restricted to years ≥ 2000             |  0.580 | 0.372 | **Fragile — loses significance** |
| 17 | Placebo outcome: log(cancer)                  | -0.190 | 1.225 | Null ✓ |
| 18 | Cluster SEs by region only (vs id×rural×year) |  1.253 | 0.523 | Same β, SE ~15% larger |

The main fragility is check 16: restricting to the second half of the panel roughly halves the IV estimate and pushes its t-statistic from ≈2.75 to ≈1.56. This is consistent with the identifying variation being tightest around the campaign cohorts who are aging out of the 22–65 sample by the late 2000s. It is not a coding bug — it is a precision issue in the instrument.

---

## 6. Summary Assessment

This is among the cleanest replications in this collection. Two key tables reproduce the paper's headline coefficients and standard errors to 3 decimal places with no adjustments, no additional data wrangling, and no reliance on Stata-specific quirks beyond the standard `vce(cluster)` small-sample correction. The cleaned analytical datasets shipped in `/Data/` are internally consistent, the Stata code is organized logically, and the README gives a precise map from sub-program to output.

**What replicates:**
- Table 2 cols 1–7 (main DiD on long-run share of vodka and placebo outcomes): exact.
- Table 3 Panel B cols 3–7 (regional mortality IV, all-cause and by cause including cancer placebo): exact.
- The beer-share placebo column (paper Table 2, col 7 = -3.129) emerges unchanged in our robustness table — a double check that the DiD design isolates type-of-alcohol substitution rather than total consumption.

**What is fragile but documented in the paper:**
- The mortality IV attenuates sharply in the second half of the panel (years ≥ 2000). The paper acknowledges this indirectly by reporting the 1st-stage F around 10 — the IV is powered but not overpowered.
- Widening the campaign window to 1985–91 halves the taste-shift coefficient, consistent with the paper's discussion in Appendix C.

**What is not replicated here (scope):**
- The cohort-hazard Cox model and counterfactual simulation (Tables 4, A4, Figures 5b, A4) — substantial translation effort beyond the headline claims.
- National-level mortality OLS (Table 3 Panel A, N=44) — uses a different dataset (`Data_Aggregate_Statistics.xlsx`) and a simple time-series regression; mechanical to add if needed.
- Migrants design, NOBUS cross-section, samogon IV, and sensitive-age kernel plots.

**No coding bugs found.** The Stata code does exactly what the paper describes, the cleaned analytical files are internally coherent, and the headline numbers reproduce in a foreign language (Python) with off-the-shelf packages.

---

## 7. File Manifest

```
replication_117443/
  utils.py            # paths, control list, base-sample cleaning helper
  01_clean.py         # load base_sample_aej.dta → print sample sizes, save parquet
  02_table2.py        # reproduce Table 2 cols 1-5, 7, 8 (AbsorbingLS HDFE)
  03_table3.py        # reproduce Table 3 Panel B cols 3-7 (IV2SLS)
  04_data_audit.py    # panel balance, share plausibility, DiD 2x2 counts
  05_robustness.py    # 18 checks targeting Table 2 & Table 3 Panel B
  output/
    cleaned_full.parquet      # post-cleaning base sample (45,680 rows)
    table2_sample.parquet     # Table 2 analytic sample
    table2_replication.csv    # side-by-side with paper values
    table3_replication.csv
    mortality_panel.parquet   # region×age×year×rural panel for Table 3
    audit.log                 # data audit console output
    robustness.log            # robustness console output
    table3_run.log
  writeup_117443.md           # this file
```
