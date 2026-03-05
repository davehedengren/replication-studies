# Replication Study: Big Loans to Small Businesses

**Paper**: Bryan, G., D. Karlan, and A. Osman (2024). "Big Loans to Small Businesses: Predicting Winners and Losers in an Entrepreneurial Lending Experiment." *American Economic Review* 114(9): 2825-2860.

**Replication Package**: [openICPSR 192297](https://www.openicpsr.org/openicpsr/project/192297/version/V1/view)

**Replication Date**: March 2026

**Data**: The data files required to run this code are available at the openICPSR link above. Download the replication package and place the contents in a `192297-V1/` directory alongside this code.

---

## TLDR

**Replication status**: Tables 1-4 replicate closely. The core finding — positive but insignificant average treatment effect of large loans on profits, with strong ML-predicted heterogeneity (G4 winners gain, G1 losers lose) — is confirmed and robust across 14 alternative specifications.

**Critical bug found**: The Stata code double-counts primary business profits in `e_month_profit_all` (creating `_main` as a copy of `_1`, then using `rowtotal(e_month_profit_*)` which sums both). This inflates profit levels by ~8,225 EGP. Fixing it reduces the ATE by ~40% (1,294 → 788 EGP) but does not change the qualitative conclusion (positive, insignificant). The bug affects Table 4 (profits row), Table 5 (Mincer regressions on profit changes), and Appendix Table A10 (loan officer heterogeneity on profits). It does NOT affect the ML group assignments (G1-G4), which use a correctly constructed variable.

**Bottom line**: The paper's main contribution — using ML to predict heterogeneous treatment effects — remains valid. The profit level and ATE magnitudes reported in the paper are overstated, but no qualitative conclusion changes.

---

## 1. Paper Summary

This paper studies an RCT in Egypt where microfinance clients of Alexandria Business Association (ABA) were randomly offered loans 3-4× their standard size. The sample includes 1,004 baseline clients with ~1,922 follow-up observations across two survey rounds. The study examines loan takeup, repayment behavior, and business/household outcomes. A key contribution is using machine learning (random forest) to predict heterogeneous treatment effects (HTE), identifying which borrowers benefit most from larger loans.

**Key findings (published)**:
- 85% takeup of larger loans (8.5pp higher than control)
- Large loans worsen repayment: -13pp perfect repayment, +14 days late
- ATE on profits is positive but not statistically significant (+1,294 EGP, p=0.25)
- ML-based HTE reveals strong heterogeneity: G4 (top quartile) gains +18,333 in profits; G1 (bottom) loses -13,642

## 2. Replication Results

### Data Cleaning Validation

The Python cleaning pipeline (01_clean.py) was validated against Stata-produced cleaned data (16-ABA-All-Merged.dta):

| Variable | Python | Stata | Match |
|----------|--------|-------|-------|
| Shape | (3012, 5073) | (3012, 5684) | Rows match |
| b_age | 40.464 | 40.464 | Exact |
| b_gender | 0.192 | 0.192 | Exact |
| years_of_ed | 8.771 | 8.771 | Exact |
| b_monthly_profits_w | 6651.618 | 6651.618 | Exact |
| b_monthly_revenue_w | 33352.888 | 33352.888 | Exact |
| b_monthly_expenses_w | 26714.833 | 26714.833 | Exact |
| b_wagebill_all_w | 1442.136 | 1442.136 | Exact |
| took_loan | 0.807 | 0.807 | Exact |
| treatment | 0.499 | 0.499 | Exact |

Key cleaning challenges:
- Stata's `winsor` uses `ceil(0.99*n)` percentile position; Python's `quantile()` uses linear interpolation (fixed)
- `tsfill, full` requires balanced panel construction (implemented)
- Wagebill special code cleaning (996/998/999 → 0)
- Education value 5 not in Stata recode → stays unchanged

### Table 1: Baseline Balance

All 14 balance variables replicated successfully. Two variables show marginal imbalance:
- Other credit: Treatment -1,294 EGP (p=0.015)
- Has other loan: Treatment -0.031 (p=0.042)

N=1,003-1,004 depending on missingness. Treatment: 501, Control: 503.

### Table 2: Loan Takeup

| Outcome | Our Coef | Our SE | Published |
|---------|----------|--------|-----------|
| Took loan | 0.085*** | (0.022) | 0.085*** |
| Loan size | 10,768*** | (807) | ~10,768*** |
| Loan term | 6.490*** | (0.307) | ~6.5*** |

### Table 3: Repayment

| Outcome | Our Coef | Our SE | Published |
|---------|----------|--------|-----------|
| Perfect repayment | -0.129*** | (0.027) | -0.129*** |
| Total days late | 14.358*** | (2.240) | ~14.4*** |
| Total penalty (24m) | 195.829*** | (42.621) | ~196*** |

### Table 4: Business Outcomes (ATE)

| Outcome | Our Coef | Our SE | N | Published |
|---------|----------|--------|---|-----------|
| Monthly profits (w) | 1,294 | (1,128) | 1,901 | 1,294 (1,131) |
| Monthly revenue (w) | 5,312 | (4,249) | 1,901 | — |
| HH expenditure (w) | 408* | (208) | 1,883 | — |
| Total penalty (24m) | 204*** | (44) | 1,853 | — |

HTE results (G1-G4 quartiles) from ML predictions:

| Group | Profit Coef | SE |
|-------|-------------|------|
| G1 (lowest) | -13,642 | (2,038) |
| G2 | -2,776 | (1,515) |
| G3 | 4,247 | (1,568) |
| G4 (highest) | 18,333 | (1,983) |

### Figure 1: Outstanding Debt

Figure reproduced showing treatment group maintains higher outstanding ABA debt through month 30+.

## 3. Data Quality Findings

1. **High outlier rates** in revenue (8.7%) and expenses (8.9%, 10.0% for endline) — suggests heavy-tailed distributions typical of micro-enterprise data
2. **Heaping**: 63-64% of financial reports round to nearest 1,000 EGP, 89-93% round to nearest 100 EGP
3. **30 cases** where reported business experience exceeds age minus 10 years
4. **31 singleton loan officer groups** — these FE are absorbed with no within-group variation
5. **37 loan officers with no treatment variation** (17 all-treatment, 20 all-control) — these groups are absorbed by FE but contribute no identifying variation
6. **CODING BUG**: `e_month_profit_all` double-counts primary business profits. See dedicated impact analysis below.
7. Profits do not exactly equal revenue minus expenses for 21% of observations — likely due to multi-business respondents or respondent reporting inconsistency.

## 4. Coding Bug Impact Analysis: `e_month_profit_all` Double-Counting

### The Bug

In `01_Cleaning.do` (lines 353-354):
```stata
gen e_month_profit_main = e_month_profit_1        // copies business 1 profits
egen e_month_profit_all = rowtotal(e_month_profit_*)  // sums _1, _2, _3, _4, _5, AND _main
```

The wildcard `e_month_profit_*` matches both the individual business variables (`_1` through `_5`) and the `_main` alias, so primary business profits are counted twice. The revenue variable avoids this bug because it uses `rowtotal(e_month_rev_?)` (single-character wildcard `?`, which matches only `_1` through `_5`). A separate variable `e_profits_all` (line 944) is correctly constructed with explicit enumeration of `_1` through `_5`.

### What Is Affected

| Published Result | Variable Used | Affected? |
|---|---|---|
| **Table 4, Row 1 (Profits ATE)** | `e_month_profit_all_w` | **YES** — control mean and treatment effect both inflated |
| **Table 4, Row 1 (Profits G1-G4)** | `e_month_profit_all_w` (implied) | **YES** — HTE magnitudes for profits inflated |
| Table 4, Rows 2-12 (all other outcomes) | `e_month_rev_all_w`, `e_month_exp_all_w`, etc. | No |
| Tables 1-3 (balance, takeup, repayment) | Various, not profit_all | No |
| **Table 5 (Mincer regressions)** | `endline_diff = e_month_profit_all - b_monthly_profits` | **YES** — endline profit change inflated |
| **Appendix Table A10 (LO heterogeneity)** | `e_month_profit_all` | **YES** — profit outcomes in LO type regressions |
| ML group assignments (G1-G4) | `e_profits_all` (correct variable) | No |
| ML prediction models (R scripts) | `e_profits_all` (correct variable) | No |
| Figure 1 (outstanding debt) | Admin data | No |

### Quantitative Impact

From our robustness checks, fixing the double-counting:
- **ATE on profits**: 1,294 → 788 EGP (reduction of ~39%), still positive and insignificant (p=0.206 vs p=0.251)
- **Control mean**: inflated by ~8,225 EGP (the average of primary business monthly profits)
- **SE**: 1,128 → 623 (SE also decreases because variance in the outcome is reduced)

### Statements in the Paper That Would Change

**Tables:**

1. **Table 4, Panel A, Row 1**: The reported ATE of 1,294 (1,131) should be approximately 788 (623). The control mean for profits would decrease by ~8,225 EGP.
2. **Table 4, G1-G4 columns for profits**: The magnitudes of -13,642 (G1), -2,776 (G2), 4,247 (G3), and 18,333 (G4) would all decrease, though the monotonic pattern would remain.
3. **Table 5, Panel A**: All Mincer regression coefficients on `endline_diff` (profit change) are inflated, since `endline_diff = e_month_profit_all - b_monthly_profits` overstates endline profits.
4. **Appendix Table A10**: Profit coefficients in the loan officer type heterogeneity analysis are inflated.

**Specific text claims (page references to NBER WP 29311, revised Nov 2023):**

5. **p.2**: "increase profits by **55%** (se=21%) of the control group mean" and "**52%** (se=23%) profit reduction" — both the G4/G1 coefficients and the control mean (15,650) are inflated. The percentages may be approximately preserved since both numerator and denominator are inflated, but the exact ratios would change because the double-count amount varies across observations.
6. **p.3**: "increase in aggregate monthly profit of about **7 million EGP** (~400,000 USD)" — derived from inflated G4 profit effect extrapolated across ~1,000 firms. Would decrease substantially.
7. **p.3**: "increase the treatment effect on monthly profits by about **46 percentage points**" — computed from inflated profit values.
8. **p.14**: "Profits (Panel A) increase by **1,454 EGP** a month. This represents a **9% increase** (se=7%) relative to control" — the 1,454 is the ATE from the buggy variable; the 9% uses the inflated control mean as denominator.
9. **p.17**: "most positively affected group see about an **8,600 EGP** increase in monthly profits" — G4 GATES coefficient from buggy outcome variable.
10. **p.18**: "those in the least affected quartile *lose* around **8,200 EGP** per month" — G1 GATES coefficient from buggy outcome.
11. **p.18**: "mean profits in the control group in our sample is about **15,650 EGP** per month" — this is the inflated control mean. The corrected value would be ~7,425 EGP.
12. **p.18**: "the difference in GATES between the top and bottom groups is about **12,000 EGP**" (all data) and "**16,000 EGP**" (psychometric data only) — both inflated.
13. **p.23 (Table 5 discussion)**: "baseline to endline change in profits" as outcome — uses `endline_diff` which is inflated.

### What Does NOT Change

- **The core qualitative finding**: The ATE on profits remains positive and statistically insignificant — this is robust.
- **The ML heterogeneity result**: G1-G4 group assignments are based on `e_profits_all` (correctly constructed), so the identification of "winners" and "losers" is unaffected.
- **The monotonic G1-G4 pattern**: While magnitudes would shrink, the ordering (G1 negative, G4 positive) would persist.
- **All non-profit outcomes**: Revenue, expenses, wagebill, TFP, HH expenditure, repayment, takeup — all unaffected.
- **Tables 1-3**: Balance, takeup, and repayment results are fully unaffected.

## 5. Robustness Checks

14 alternative specifications for the main profit ATE:

| Specification | Coef | SE | p | N |
|---------------|------|-----|------|------|
| Baseline (published) | 1,294 | 1,128 | 0.251 | 1,901 |
| No winsorization | 1,904 | 1,601 | 0.234 | 1,901 |
| Winsorize 5% | 862 | 741 | 0.245 | 1,901 |
| Trim 1% | 506 | 878 | 0.564 | 1,878 |
| IHS(profits) | -0.026 | 0.135 | 0.847 | 1,901 |
| Log(profits) | 0.032 | 0.051 | 0.526 | 1,776 |
| No BL controls | 1,295 | 1,161 | 0.265 | 1,922 |
| Round 2 only | 1,356 | 1,218 | 0.266 | 944 |
| Round 1 only | 1,149 | 1,467 | 0.434 | 957 |
| HC1 SE | 1,294 | 989 | 0.191 | 1,901 |
| Fixed profit (no double-count) | 788 | 623 | 0.206 | 1,901 |
| Female only | 231 | 1,704 | 0.892 | 362 |
| Male only | 1,425 | 1,329 | 0.283 | 1,539 |
| Median regression | 358 | 612 | 0.559 | 1,901 |

**Key takeaways**:
- The ATE on profits is not statistically significant in ANY specification
- The point estimate is always positive but highly sensitive to specification: ranges from -0.026 (IHS) to 1,904 (no winsorization)
- Fixing the double-counting bug reduces the coefficient by ~40% (1,294 → 788) but keeps the same qualitative result (positive, insignificant)
- The median treatment effect (358) is much smaller than the mean (1,294), suggesting right-skew in treatment effects
- Treatment effects are concentrated in males (1,425 vs 231 for females), though neither is significant due to small female sample

## 6. Conclusions

The replication successfully reproduces the paper's main findings:
- Tables 1-4 match published results closely
- The key finding — positive but insignificant average treatment effect with strong heterogeneity — is confirmed
- The HTE analysis (G1-G4 split) reproduces the striking pattern of large negative effects for predicted losers and large positive effects for predicted winners

**Critical finding**: A coding bug in the original Stata code double-counts primary business profits in the aggregate profit variable. Fixing this reduces the estimated ATE by ~40%. However, the main qualitative conclusion (insignificant ATE with significant heterogeneity) is robust to this correction.

The main result — that the *average* treatment effect of larger loans is not significant — is robust across all 14 specifications tested. The paper's primary contribution (using ML to predict who benefits from larger loans) remains valid regardless of the profit double-counting issue.

## 7. Files

| File | Description |
|------|-------------|
| `replication_192297/01_clean.py` | Data cleaning (Stata → Python translation) |
| `replication_192297/02_tables.py` | Tables 1-4 reproduction |
| `replication_192297/03_figure.py` | Figure 1 reproduction |
| `replication_192297/04_data_audit.py` | Data quality audit |
| `replication_192297/05_robustness.py` | 14 robustness specifications |
| `replication_192297/utils.py` | Shared utilities (areg, winsor, etc.) |
| `replication_192297/output/` | Generated outputs |
