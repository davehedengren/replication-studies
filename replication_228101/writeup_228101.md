# Replication Study: Gender Gaps in Entrepreneurship — Business Networks and Collaborations in Ghana

**Original Paper**: Lambon-Quayefio, Asiedu, Truffa, and Wong (2025). "Gender Gaps in Entrepreneurship: Business Networks and Collaborations in Ghana." *AEA Papers and Proceedings* 115: 495-99.

**Replication Package**: [openICPSR 228101](https://www.openicpsr.org/openicpsr/project/228101/version/V1/view)

**Replication Date**: March 2026

**Data**: The data files required to run this code are available at the openICPSR link above. Download the replication package and place the contents in a `228101-V1/` directory alongside this code.

---

## TLDR

**Replication status**: All main findings replicate successfully. Coefficient magnitudes, SEs, and significance levels are consistent across all tables and figures.

**Key findings confirmed**: Female business owners earn ~GHS 889 less/month (-30% of mean), have smaller business networks, and the networking index predicts profits (610 GHS per SD). The gender profit gap is robust to winsorizing, log transforms, and alternative SE specifications.

**Main concerns**: (1) Adding product fixed effects reduces the female coefficient by ~40% and renders it insignificant in the kitchen-sink model, suggesting much of the gender gap operates through product selection. (2) Quantile regression shows no gender gap at the 10th-25th percentiles — the gap is entirely concentrated at the median and above, consistent with a "glass ceiling" interpretation.

**Code bug found**: The published `output.do` references a variable `profit` that is never created in `clean.do` (which creates `profits_lastmonth` instead). The code cannot run as-is without modification. This is a packaging oversight, not an analytical error — it does not affect any published results.

---

## 1. Introduction and Paper Summary

This paper examines gender disparities in business networks among 1,487 agribusiness owners in Ghana. The key findings are:

1. **Gender profit gap**: Female business owners earn significantly less than males (about GHS 889 less per month, or ~30% of the mean).
2. **Network size gap**: Women have smaller business networks — fewer advice contacts, fewer regular business meetings, and fewer collaborations.
3. **Network composition differs**: Women's networks contain a higher share of other women, but a similar share of friends/relatives.
4. **Networks correlate with profits**: A one-SD increase in the networking index is associated with GHS 610 more in monthly profits.
5. **Mechanisms**: The gender gap in networking is partly explained by education, business location (at home), and caregiving hours.

The original replication package contains three Stata files (`master.do`, `clean.do`, `output.do`) and a custom z-score index program (`zindex.ado`). We translated all code to Python and reproduced the full set of results.

---

## 2. Replication Results

### Code Bug in Original Package

The original `clean.do` creates a variable called `profits_lastmonth` from survey variable `g6`, but `output.do` references `profit` throughout — a variable that is never created. The published Stata code would fail as-is. Our Python replication uses `g6` (aliased as `profit`) and documents this discrepancy. This is likely a minor oversight — the authors probably had `gen profit = g6` in a working version that didn't make it into the final replication package.

### Table 1: Summary Statistics

| Variable | All | Male | Female | Diff | p-value |
|---|---|---|---|---|---|
| Profit last month (GHS) | 2,869 | 3,036 | 2,147 | 889*** | 0.01 |
| Years in operation | 7.09 | 7.28 | 6.36 | 0.92*** | 0.00 |
| Total workers (excl. owner) | 3.28 | 3.49 | 2.45 | 1.03*** | 0.00 |
| Business located at home | 0.14 | 0.12 | 0.21 | -0.08*** | 0.00 |
| Hours worked at business | 22.46 | 22.22 | 23.09 | -0.86 | 0.51 |
| Hours at wage employment | 17.32 | 18.29 | 13.55 | 4.74*** | 0.00 |
| Hours caregiving | 10.60 | 8.80 | 18.44 | -9.64*** | 0.00 |
| Married or in partnership | 0.58 | 0.58 | 0.57 | 0.01 | 0.72 |
| Age | 35.32 | 35.38 | 35.02 | 0.36 | 0.47 |
| College degree | 0.68 | 0.71 | 0.56 | 0.15*** | 0.00 |
| Children | 1.61 | 1.54 | 1.92 | -0.39*** | 0.00 |
| Primary income from business | 0.65 | 0.63 | 0.71 | -0.08** | 0.01 |
| Work for pay | 0.54 | 0.57 | 0.42 | 0.16*** | 0.00 |

**Replication assessment**: Results closely match the published Table 1. All sign directions, significance levels, and magnitudes are consistent. Minor differences in decimal places are expected from Stata vs. Python numerical precision.

### Figure 1a: Network Size by Gender

| Network Variable | Male Mean | Female Mean | Diff | p-value |
|---|---|---|---|---|
| # Business Owners for Advice | 0.714 | 0.608 | -0.106** | 0.031 |
| Meet Other Business Owners | 0.411 | 0.319 | -0.092*** | 0.004 |
| # Business Owners Meet Regularly | 2.011 | 1.232 | -0.779* | 0.055 |
| Business Association Member | 0.186 | 0.148 | -0.038 | 0.128 |
| # Business Collaborations | 6.715 | 4.217 | -2.498*** | 0.001 |
| # Suppliers | 1.403 | 1.431 | 0.028 | 0.798 |
| # Clients | 2.538 | 2.958 | 0.419 | 0.233 |

Women have significantly fewer advice contacts, fewer regular meetings, and fewer collaborations. No significant difference in supplier or client counts.

### Figure 1b: Network Composition Gender Differences

**Female Share (coefficient on female dummy):**

| Variable | Coef | SE | p |
|---|---|---|---|
| Business Owners for Advice | 0.105 | 0.035 | 0.002 |
| Business Owners Meet Regularly | 0.350 | 0.042 | 0.000 |
| Suppliers | 0.120 | 0.032 | 0.000 |
| Business Clients | 0.143 | 0.033 | 0.000 |

**Friends/Relatives Share (coefficient on female dummy):**

| Variable | Coef | SE | p |
|---|---|---|---|
| Business Owners for Advice | 0.105 | 0.035 | 0.003 |
| Business Owners Meet Regularly | -0.015 | 0.050 | 0.761 |
| Suppliers | 0.044 | 0.034 | 0.194 |
| Business Clients | -0.002 | 0.035 | 0.961 |

Women's networks have a significantly higher share of women across all categories. The friends/relatives share shows no significant gender difference (except for advice contacts).

### Table 3: Effect of Networking on Profits

| | (1) | (2) | (3) | (4) | (5) | (6) |
|---|---|---|---|---|---|---|
| Female | -912*** | -831** | -858** | -921*** | -735** | -473 |
| | (334) | (333) | (343) | (336) | (350) | (354) |
| Networking Index | | 610*** | | | 455** | 357* |
| | | (170) | | | (188) | (187) |
| Female Networking Index | | | -161 | | -224 | -120 |
| | | | (153) | | (154) | (157) |
| Friends/Rel. Networking Index | | | | -224 | -117 | -47 |
| | | | | (199) | (193) | (193) |
| Product FE | No | No | No | No | No | Yes |
| R² | 0.004 | 0.013 | 0.010 | 0.012 | 0.017 | 0.057 |
| N | 1,398 | 1,398 | 1,398 | 1,398 | 1,398 | 1,398 |

The overall networking index strongly predicts profits. The female-specific and friends/relatives indices are not individually significant. Adding product fixed effects absorbs much of the female coefficient.

### Table 2: Determinants of Business Network Index

| | (1) | (2) | (3) | (4) | (5) | (6) |
|---|---|---|---|---|---|---|
| Female | -0.140** | -0.114* | -0.127** | -0.172*** | -0.131** | -0.129** |
| | (0.061) | (0.061) | (0.060) | (0.063) | (0.062) | (0.063) |
| College Degree | | 0.183*** | | | 0.195*** | 0.172*** |
| | | (0.057) | | | (0.057) | (0.059) |
| Business at Home | | | -0.164** | | -0.174** | -0.172** |
| | | | (0.069) | | (0.070) | (0.072) |
| Hours Caregiving | | | | 0.004** | 0.004** | 0.004** |
| | | | | (0.002) | (0.002) | (0.002) |
| Product FE | No | No | No | No | No | Yes |
| R² | 0.016 | 0.026 | 0.020 | 0.023 | 0.037 | 0.048 |
| N | 1,436 | 1,436 | 1,436 | 1,436 | 1,436 | 1,436 |

The female network gap is partially mediated by: having a college degree (reduces gap by 19%), operating business from home (reduces by 9%), and caregiving hours (which actually increase the raw gap, suggesting a suppression effect).

---

## 3. Data Quality Findings

### 3.1 Sample Characteristics
- 1,487 total observations; 1,201 male, 278 female, 8 missing gender
- 146 original survey variables expanded to 212 after cleaning

### 3.2 Outlier Analysis
Substantial right skewness in financial variables:
- **Profit**: 125 IQR outliers; max = GHS 120,000 (41x mean). Skewness = 8.0, kurtosis = 111.
- **Sales**: 150 outliers; max = GHS 868,267. Skewness = 17.5, kurtosis = 420.
- **Workers**: max = 95 (skewness = 8.7)
- **Collaborations**: max = 504 (skewness high)

The extreme skewness in profit is a concern for OLS estimation. The top profit value (GHS 120,000) is from a 32-year-old male with 2 years of operation and only 4 workers — potentially unreliable.

### 3.3 Logical Consistency
- 1 case where workers_female > workers_total
- 2 cases with total hours > 168/week (impossible for a 7-day week)
- 42 cases with total hours > 100/week (plausible but high)
- 5 cases where business started before age 10 (likely reporting errors)
- All share variables properly bounded in [0,1]

### 3.4 Missing Data Patterns
Missingness is **systematically higher for women** across most variables:
- Profit: 4.5% missing for males vs 9.7% for females (chi² p=0.001)
- Most survey variables: ~2.3% missing for males vs ~5.4% for females
- This pattern is consistent with the 8 observations with missing gender having mostly missing data across the board (likely incomplete surveys)

### 3.5 Distribution Anomalies
Strong evidence of **heaping** in financial variables:
- 95.4% of profit values divisible by 100
- 74.8% divisible by 500
- 61.3% divisible by 1,000
- 29.3% report zero profit

This is expected for self-reported financial data in developing country surveys but contributes to measurement error.

### 3.6 No Duplicate Cases
All 1,487 case IDs are unique. Some rows share demographic/profit values (203 duplicates on gender+age+years+profit), which is expected given the limited range of discrete variables.

---

## 4. Robustness Check Results

### 4.1 Winsorizing and Trimming
The female profit gap is robust to outlier treatment, though the magnitude decreases with more aggressive treatment:

| Treatment | Female Coef | SE | p |
|---|---|---|---|
| Baseline (levels) | -912*** | 334 | 0.006 |
| Winsorized 1% | -808*** | 298 | 0.007 |
| Winsorized 5% | -612*** | 185 | 0.001 |
| Trimmed 1% | -712*** | 241 | 0.003 |
| Trimmed 5% | -434*** | 145 | 0.003 |

The gap remains significant in all cases. The reduction from -912 to -434 with 5% trimming suggests extreme profit values do amplify the estimated gap, but the core finding holds.

### 4.2 Log and IHS Transforms
| Transform | Female Coef | SE | p |
|---|---|---|---|
| Levels | -912*** | 334 | 0.006 |
| Log(profit + 1) | -0.438* | 0.248 | 0.077 |
| IHS(profit) | -0.459* | 0.269 | 0.088 |
| Log(profit), excl. zeros | -0.299*** | 0.089 | 0.001 |

Significance weakens slightly with log/IHS transforms (p=0.08-0.09) due to the high fraction of zero-profit observations. When zeros are excluded, the log specification is highly significant.

### 4.3 Alternative Standard Errors
| SE Type | SE(female) | p |
|---|---|---|
| HC1 (Stata default) | 334 | 0.006 |
| HC3 | 335 | 0.006 |
| OLS (homoskedastic) | 432 | 0.035 |
| Clustered by product | 404 | 0.024 |

Results are robust across all SE specifications, including clustering by product category.

### 4.4 Alternative Index Construction
| Index Method | Female Coef (Table 3 style) | p |
|---|---|---|
| Z-score (original) | -831** | 0.013 |
| Simple [0,1] average | -864** | 0.010 |
| PCA (first component, 27.8% variance) | -842** | 0.012 |

All index construction methods yield very similar results. The first PCA component captures only 27.8% of variance, reflecting the relatively low correlation among network dimensions.

### 4.5 Control Sensitivity
| Controls | Female Coef | p |
|---|---|---|
| None | -889*** | 0.008 |
| Age only | -887*** | 0.008 |
| Years operation only | -886*** | 0.009 |
| All 3 baseline controls | -912*** | 0.006 |
| + product FE | -557* | 0.099 |
| Kitchen sink (all + married + children) | -486 | 0.162 |

The female coefficient is very stable across basic controls but attenuates substantially with product fixed effects (from -912 to -557). This suggests that product selection is an important mediator of the gender profit gap. In the fully saturated model, the coefficient becomes insignificant (p=0.16), raising questions about whether the gender gap operates primarily through product choice.

### 4.6 Sample Restrictions
| Sample | Female Coef | p | N |
|---|---|---|---|
| Full sample | -912*** | 0.006 | 1,398 |
| Complete network data | -909*** | 0.007 | 1,394 |
| Crop sellers only | -857** | 0.037 | 966 |
| Positive profit only | -1,138** | 0.012 | 989 |
| Drop top 1% profit | -712*** | 0.003 | 1,383 |
| Business NOT at home | -1,038*** | 0.006 | 1,209 |
| Business at home | -153 | 0.819 | 189 |

The gap disappears entirely among home-based businesses (N=189), consistent with the paper's hypothesis about business location as a mechanism. The gap is larger among positive-profit firms (-1,138).

### 4.7 Quantile Regression
| Quantile | Female Coef | p |
|---|---|---|
| 10th | 0.000 | 1.000 |
| 25th | 0.000 | 1.000 |
| 50th (median) | -500*** | 0.003 |
| 75th | -796** | 0.012 |
| 90th | -2,708*** | 0.007 |

The gender gap is concentrated in the upper part of the profit distribution. At the 10th and 25th percentiles (where many observations are zero), there is no gender difference. At the median, the gap is GHS 500; at the 90th percentile, it reaches GHS 2,708. This suggests the gender gap is primarily about a "glass ceiling" effect — women are less likely to achieve high profits rather than being uniformly disadvantaged.

### 4.8 Bootstrap Inference
| Method | Coef | SE | 95% CI |
|---|---|---|---|
| Analytical HC1 | -912 | 334 | [-1,567, -258] |
| Bootstrap (1,000 reps) | -928 | 344 | [-1,544, -254] |

Bootstrap confidence intervals are very similar to analytical HC1, confirming the reliability of the standard inference.

### 4.9 Selection Analysis
Gender does not significantly predict having network data (p=0.28-0.48 for all three indices), suggesting that differential missingness in network variables is not driving the results.

---

## 5. Conclusions

### Successful Replication
All main findings from the paper replicate successfully in Python. The coefficient magnitudes, standard errors, significance levels, and sample sizes are consistent across all tables and figures.

### Key Code Issue
The original Stata replication package has a bug where `output.do` references a variable `profit` that is never created in `clean.do` (which creates `profits_lastmonth` instead). This means the published code cannot run as-is without modification.

### Data Quality
The data is generally well-constructed, with a few minor issues:
- Extreme right skew in financial variables (profit, sales)
- Strong heaping in self-reported monetary values
- Systematically higher missingness for women
- A small number of logical inconsistencies

### Robustness Assessment
The paper's central findings are **broadly robust**:
- The female profit gap survives winsorizing, log transforms, alternative SEs, and most sample restrictions
- The networking index construction method does not matter
- Bootstrap inference confirms analytical SEs

**Two important caveats emerge**:
1. **Product selection matters**: Adding product fixed effects reduces the female coefficient by ~40% and renders it insignificant in the kitchen-sink specification. This suggests much of the gender profit gap may operate through women selecting into different (lower-profit) product categories.
2. **The gap is concentrated at the top**: Quantile regression reveals no gender gap at the 10th-25th percentiles (where many report zero profit) but a large and growing gap at the median and above. This is consistent with a "glass ceiling" interpretation.

---

## Appendix: File Inventory

| File | Description |
|---|---|
| `replication/utils.py` | Shared utilities (zindex, regression helpers, constants) |
| `replication/01_clean.py` | Data cleaning (Stata → Python translation) |
| `replication/02_table1.py` | Table 1: Summary statistics with t-tests |
| `replication/03_figures.py` | Figures 1a (bar chart) and 1b (coefficient plot) |
| `replication/04_table3.py` | Table 3: Profit regressions with network indices |
| `replication/05_table2.py` | Table 2: Network index determinants |
| `replication/06_data_audit.py` | Data quality checks |
| `replication/07_robustness.py` | 10 robustness checks |
| `replication/output/` | Generated tables, figures, and reports |
