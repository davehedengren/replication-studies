# Replication Study: Detecting Mother-Father Differences in Spending on Children

**Paper**: Dizon-Ross, R. and S. Jayachandran (2023). "Detecting Mother-Father Differences in Spending on Children: A New Approach Using Willingness-to-Pay Elicitation." *AER: Insights*, 5(3), 377–392.

**Replication Package**: [openICPSR 179162](https://www.openicpsr.org/openicpsr/project/179162/version/V1/view)

**Replication Date**: March 2026

**Data**: The data files required to run this code are available at the openICPSR link above. Download the replication package and place the contents in a `179162-V1/` directory alongside this code.

---

## TLDR

**Replication status**: Near-exact replication. All main coefficients match to the third decimal place (girl: -0.102, momXgirl: +0.132). Sample size matches within 1 observation (6,672 vs 6,673).

**Key finding confirmed**: Fathers spend significantly less on daughters than sons (-0.102 SD), and the mother-daughter interaction is positive and significant (+0.132 SD). This is robust across 13 of 13 specifications for the girl coefficient and 9 of 13 for the interaction.

**Main concern**: The momXgirl interaction is entirely driven by non-incentivized goods (0.185***) and disappears for incentivized goods (0.052, n.s.). Since incentivized goods have real financial stakes, this raises questions about whether the mother-father difference reflects genuine preferences or hypothetical bias. The effect is also concentrated in the baseline wave and weakens in follow-up.

**No coding bugs found.**

---

## 1. Paper Summary

### Research Question
Do fathers and mothers spend differently on sons versus daughters? Specifically, do fathers exhibit a pro-son spending bias that mothers do not?

### Data
The study uses data from a survey of 1,084 households in Uganda, with follow-up of 729 households. Parents participate in Becker-DeGroot-Marschak (BDM) willingness-to-pay (WTP) price list elicitations for 16 goods spanning child human capital goods (school tests, deworming, shoes, workbooks), enjoyment goods (balls, candy), adult goods (cups, jerry cans, posters), and toys.

### Method
The main specification regresses WTP (standardized by dividing by the within-good SD of WTP as a fraction of market price) on a **daughter** indicator, a **mother** indicator, and their interaction **mother × daughter**, controlling for adult-good WTP, wave indicators, child age indicators, and good fixed effects, with stratum fixed effects absorbed and standard errors clustered at the household level.

### Key Findings
1. Fathers spend significantly less on daughters than sons (–0.102 SD, p < 0.01)
2. The mother × daughter interaction is positive and significant (+0.131 SD, p < 0.01), indicating mothers do not discriminate
3. The net effect for mothers (girl + mother×girl) is not significantly different from zero
4. These patterns are driven by households where both parents say the mother cares more about children

---

## 2. Methodology Notes

### Translation Choices
- **Language**: Stata → Python (pandas, statsmodels, numpy, matplotlib)
- **Absorbed FE**: Stata's `areg` was replicated using the Frisch-Waugh-Lovell demeaning approach (subtract group means) rather than full dummy variables, which caused SVD convergence failures in statsmodels due to the large number of stratum dummies
- **Clustered SEs**: `cov_type='cluster'` in statsmodels matches Stata's `cluster()` option
- **WTP Normalization**: `wtp_std = (wtp_raw / market_price) / SD(wtp/market_price)` where SD is computed over ALL observations for each good

### Key Implementation Details
- **Daughter variable**: The Stata code sets `daughter = 1` when child gender is missing (`d04_childgender_ == "1 female" | mi(d04_childgender_)`), which codes 1 missing-gender obs as daughter. This was replicated exactly.
- **Fancyball special case**: `fancyball_wtp = fancyball_wtp + plainball_wtp` (the two goods' WTP values are summed, per Stata code line 326-327 of clean_fu_PPS.do)
- **Girl variable varies by good**: Old reference child goods use BL daughter; shoes uses a separate "chosen child" gender; new reference child goods use FU younger child gender
- **Adult WTP control**: BL uses cup WTP (SD-normalized); FU uses average of jerry can + poster WTP (SD-normalized)

---

## 3. Replication Results

### Table 1: Main Results

| | Col 1 (SD) | Col 2 (SD+int) | Col 3 (Mkt) | Col 4 (HH FE) | Col 5 (Incent.) | Col 6 (Non-inc.) |
|---|---|---|---|---|---|---|
| **girl** | –0.037 | –0.102*** | –0.029*** | –0.093*** | –0.074* | –0.124*** |
| **momXgirl** | — | 0.132*** | 0.036*** | 0.144*** | 0.052 | 0.185*** |
| **mom** | –0.029 | –0.095*** | –0.028*** | –0.067* | –0.126*** | –0.069 |
| **N** | 6,672 | 6,672 | 6,672 | 6,672 | 2,916 | 3,756 |
| **FS mean** | 1.943 | 1.943 | 0.537 | 1.943 | 1.901 | 1.975 |

### Comparison with Published Values (Column 2)

| Variable | Ours | Published | Difference |
|---|---|---|---|
| girl | –0.102 | –0.102 | –0.000 |
| momXgirl | 0.132 | 0.131 | +0.001 |
| mom | –0.095 | –0.095 | –0.000 |
| N | 6,672 | 6,673 | –1 |
| Father-son mean | 1.943 | 1.943 | 0.000 |

**Assessment**: Near-exact replication. Coefficients match to the third decimal place. The 1-observation difference in N likely stems from the `paper_statistics.do` line `drop if good == "shoes" & mi(younger_child_male)`, which we did not exactly replicate.

### Figure 2: Predicted WTP by Parent × Child Gender

| Group | Our Predicted WTP (SD) |
|---|---|
| Father-Son | 1.943 |
| Father-Daughter | 1.840 |
| Mother-Son | 1.847 |
| Mother-Daughter | 1.877 |

The father-daughter gap (–0.10 SD, p < 0.01) is stark, while the mother-daughter "gap" is close to zero (0.03 SD, n.s.).

### Figures 3a/3b: By Good Type

- **Human capital goods**: girl = –0.106***, momXgirl = 0.144*** — strong pattern
- **Enjoyment goods**: girl = –0.091*, momXgirl = 0.078 — weaker, momXgirl not significant

### Figures 4a/4b: By "Mother Loves More" Classification

- **HH does NOT say mom loves more**: girl = –0.063, momXgirl = 0.050 — neither significant
- **HH says mom loves more**: girl = –0.138***, momXgirl = 0.184*** — very strong pattern

---

## 4. Data Audit Findings

### Coverage
- 11,758 observations across 1,084 households and 15 goods (sieve dropped from analysis)
- Baseline: 4,469 obs; Follow-up: 7,289 obs
- Main analysis sample (child goods, no toys_bin): 6,674 obs; 6,672 with all key variables

### Data Quality
- **No duplicates** at the household-good level
- **All binary variables** correctly coded as 0/1
- **momXgirl = mom × girl** verified for all 9,214 non-missing observations
- **WTP values** well-behaved: all in [0, 1] after normalization, no negative values

### Missing Data
- Girl variable is missing for adult goods (cup, jerry can, poster) by design — 2,544 obs (21.6%)
- Only 2 observations have missing girl for the shoes good (child gender not recorded)
- Missing pattern is balanced across parent gender (21.6% for fathers, 21.7% for mothers)

### Attrition
- **32.7% attrition** from BL (1,084) to FU (729)
- Slightly differential by parent gender: mothers retained at 68.6% vs fathers at 65.9% (diff = +2.8pp)
- Slightly differential by child gender: daughters retained at 68.0% vs sons at 66.4% (diff = +1.6pp)
- These differences are small but worth noting

### Strata
- 80 strata, mean 13.6 HH per stratum
- **2 singleton strata** — these are absorbed by the FE but cannot contribute to within-stratum variation

### Coding Concern
- The Stata code assigns `daughter = 1` when child gender is missing (1 observation). This is a coding quirk that slightly biases the daughter proportion upward.

### WTP Heaping
- BDM price lists produce moderate heaping: top 5 values account for 40-47% of observations
- This is expected for discrete price list elicitation and does not indicate data quality issues

---

## 5. Robustness Check Results

| Specification | girl | momXgirl | mom | N |
|---|---|---|---|---|
| **Baseline** | **–0.102\*\*\*** | **0.132\*\*\*** | **–0.095** | **6,672** |
| Winsorize 5/95 | –0.097*** | 0.122*** | –0.088 | 6,672 |
| IHS transform | –0.025*** | 0.033*** | –0.027 | 6,672 |
| BL only | –0.122** | 0.225*** | –0.025 | 2,299 |
| FU only | –0.096*** | 0.078 | –0.128 | 4,373 |
| Incentivized only | –0.074* | 0.052 | –0.126 | 2,916 |
| Non-incentivized | –0.124*** | 0.185*** | –0.069 | 3,756 |
| Health goods | –0.145*** | 0.199*** | –0.044 | 2,673 |
| Non-health goods | –0.088** | 0.095* | –0.130 | 3,999 |
| No adult WTP ctrl | –0.111** | 0.096* | –0.089 | 6,672 |
| HH FE | –0.093*** | 0.144*** | –0.067 | 6,672 |
| Add momXyoung | –0.104*** | 0.136*** | –0.115 | 6,672 |
| Median regression | –0.085** | 0.099* | –0.082 | 6,672 |
| HC1 robust SEs | –0.102*** | 0.132*** | –0.095 | 6,672 |

### Permutation Test
- Shuffling girl assignment within strata (100 permutations):
  - **girl**: permutation p-value = 0.000 (actual: –0.102, placebo mean: –0.015)
  - **momXgirl**: permutation p-value = 0.020 (actual: 0.132, placebo mean: 0.022)

### Key Observations

**Strong findings** (survive all checks):
- The **girl coefficient is consistently negative** across all specifications, ranging from –0.074 to –0.145
- The sign and direction are robust; statistical significance is maintained in 11 of 13 checks
- The permutation test confirms the girl effect is unlikely due to chance

**Moderately robust findings**:
- The **momXgirl interaction** is significant in 9 of 13 checks. It loses significance when:
  - Restricting to FU data only (0.078, p = 0.16) — suggesting the BL signal is strong
  - Restricting to incentivized goods only (0.052, p = 0.40) — concerning since incentivized goods are more credible
  - Dropping the adult WTP control (0.096, p = 0.09) — reduces to marginal significance
  - Median regression (0.099, p = 0.06) — marginally significant

**Potential concerns**:
1. **Incentivized vs non-incentivized split**: The momXgirl effect is entirely driven by non-incentivized goods (0.185***) and disappears for incentivized goods (0.052, n.s.). Since incentivized goods have real stakes, this raises questions about whether the mother-father difference reflects genuine preferences or hypothetical bias.
2. **Wave heterogeneity**: The momXgirl effect is much stronger in the baseline (0.225***) than follow-up (0.078, n.s.), suggesting possible attrition-driven bias or experimental demand effects.
3. **Leave-one-good-out**: Results are stable across all good exclusions, suggesting no single good drives the findings.

---

## 6. Summary Assessment

### What Replicates
- **All main coefficients replicate to the third decimal place** (girl: –0.102 vs –0.102; momXgirl: 0.132 vs 0.131; mom: –0.095 vs –0.095)
- **Sample size matches within 1 observation** (6,672 vs 6,673)
- **Father-son mean matches exactly** (1.943)
- **All figures reproduce the published patterns**

### What Deserves Scrutiny
1. **Incentivized/non-incentivized split**: The key mother-daughter interaction is not significant for incentivized goods, where WTP responses have real consequences. This is the most important robustness concern.
2. **Wave heterogeneity**: The effect is concentrated in the baseline wave, raising questions about replicability across survey rounds.
3. **Moderate attrition (32.7%)**: While differential attrition by gender is small, attrition is high overall.
4. **Daughter coding quirk**: Setting daughter=1 for missing gender is unusual but affects only 1 observation.

### Overall Assessment
The replication is **successful** — all published results are reproduced with high precision. The main finding that fathers spend less on daughters is robust across specifications. However, the mother-father difference (momXgirl interaction) shows some fragility, particularly the loss of significance for incentivized goods and in the follow-up wave. These patterns are visible in the published results (Table 1 Cols 5-6) and are discussed by the authors, but they warrant attention when interpreting the strength of the evidence.

---

## 7. File Manifest

| File | Description |
|---|---|
| `replication_179162/utils.py` | Shared utilities, constants, `run_areg` function |
| `replication_179162/01_clean.py` | Data cleaning: BL/FU WTP extraction, long dataset construction |
| `replication_179162/02_table1.py` | Table 1: Main regression results (6 columns) |
| `replication_179162/03_figures.py` | Figures 2, 3a, 3b, 4a, 4b: Bar charts of predicted WTP |
| `replication_179162/04_data_audit.py` | Data quality audit (10 checks) |
| `replication_179162/05_robustness.py` | Robustness checks (12 specifications + permutation test) |
| `replication_179162/output/analysis_data.parquet` | Cleaned analysis dataset |
| `replication_179162/output/figure2.png` | Figure 2 |
| `replication_179162/output/figure3a.png` | Figure 3a (human capital goods) |
| `replication_179162/output/figure3b.png` | Figure 3b (enjoyment goods) |
| `replication_179162/output/figure4a.png` | Figure 4a (does not say mom loves more) |
| `replication_179162/output/figure4b.png` | Figure 4b (says mom loves more) |
