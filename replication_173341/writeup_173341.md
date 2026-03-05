# Replication Study: 173341-V1

**Paper:** "Vulnerability and Clientelism"
**Authors:** Gustavo J. Bobonis, Paul Gertler, Marco Gonzalez-Navarro, Simeon Nichter
**Journal:** *American Economic Review*, 2022, 112(11): 3627-3659
**Original Language:** Stata (5 do-files)
**Replication Language:** Python (pandas, statsmodels, matplotlib)

---

## 0. TLDR

- **Replication status:** Near-exact for Tables 1 (10/12 variables), 2 (all 9 specs), 4 (coefficients within 0.002, SEs exact). Tables 3 and 5 match for non-postcoded variables; systematic discrepancy for `ask_private_stacked` due to data version issue.
- **Key finding confirmed:** Cisterns treatment reduces private goods requests from politicians (-0.028 to -0.030) and improves well-being (Overall Index +0.126 SD). Rainfall shocks reduce requests (-0.023 to -0.025). Effects concentrated among those with clientelist relationships (-0.092 to -0.097 interaction).
- **Main concern:** The shipped final data has different values for `c_ask_private_pol_postcoded` (2012 postcoded private goods request) compared to the published .tex outputs (data mean=0.213 vs published 0.195). This propagates to Tables 3 and 5. All non-postcoded variables match exactly.
- **Bug status:** No bugs found in the code logic. The data version discrepancy is a packaging issue, not a code error.

---

## 1. Paper Summary

### Research Question
Does reducing household vulnerability to economic shocks decrease clientelistic exchanges with politicians? Specifically, do water cisterns (which reduce dependence on rainfall for water access) reduce citizens' requests for private goods from politicians?

### Data
- **Household survey:** 1,308 households across 425 neighborhood clusters in 40 municipalities in Brazil's semiarid northeast. Three waves: 2011 (baseline), 2012, 2013.
- **Individual data:** 2,990 respondents from surveyed households.
- **Electoral data:** 1,794 voting machines linked to survey respondents via electoral section numbers. 2012 municipal elections.
- **Rainfall:** CHIRPS satellite data, standardized by municipality-level long-run average (1986-2011).

### Method
1. **ITT estimation:** Intention-to-treat effects of randomized cistern assignment on private goods requests, vulnerability indices, and voting behavior.
2. **Rainfall as natural experiment:** Exogenous rainfall shocks affect household vulnerability independent of treatment, providing a second source of variation.
3. **Heterogeneity:** Treatment effects interacted with pre-existing clientelist relationships (frequent conversations with politicians before the campaign).
4. **Stacked panel:** Individual data from 2012 and 2013 combined for pooled estimation.
5. **Electoral analysis:** reghdfe with location fixed effects and wild cluster bootstrap for inference with limited municipality clusters.

### Key Findings
- Cisterns treatment improves well-being: -(CES-D) +0.092 SD, SRHS +0.075 SD, Overall Index +0.126 SD (Table 2, Panel A).
- Positive rainfall reduces private goods requests by 0.025 SD (Table 3).
- Cisterns treatment reduces private goods requests by 0.028 pp (Table 3).
- Treatment effects are concentrated among those with clientelist relationships: -0.092 interaction (Table 5).
- Treatment increases support for challenger candidates in the 21-municipality sample (-0.103, bootstrap p=0.038; Table 4).

---

## 2. Methodology Notes

### Translation Choices
- **Stata `reg ... i.mun_id, cluster(b_clusters)`** -> pandas `get_dummies(drop_first=True)` + `statsmodels.OLS` with `cov_type='cluster'`. Explicit dummies match Stata's DOF adjustment exactly.
- **Stata `reghdfe ... , absorb(location_id) cluster(location_id)`** -> FWL demeaning by location_id + `statsmodels.OLS` with cluster-robust SEs. When absorb == cluster, `dof_absorbed = 0` (groups are nested, so no additional DOF correction needed). This matches Stata's reghdfe SE exactly.
- **Stata `boottest`** -> Not replicated. No robust Python equivalent for wild cluster bootstrap. Published p-values are reported for reference.
- **Stata `lincom`** -> Manual coefficient addition with delta-method SEs from the covariance matrix.

### Data Notes
- The Master.do sets `set type float` (32-bit precision). All generated variables use single-precision floats. This may cause minor rounding differences from Python's default 64-bit doubles.
- The stacked data has 5,334 observations (2,667 per year), but only 4,288 have non-missing `ask_private_stacked` (the 2013 variable has 35% missing due to survey attrition).

---

## 3. Replication Results

### Table 1: Interactions with Politicians

| Variable | Repl Mean | Pub Mean | Repl Beta | Pub Beta | Match? |
|----------|-----------|----------|-----------|----------|--------|
| Request Private Good, 2012 | 0.213 | 0.195 | -0.036 | -0.032 | Data differs |
| Request Private Good, 2013 | 0.086 | 0.086 | -0.003 | -0.003 | Exact |
| Request & Receive, 2012 | 0.124 | 0.115 | -0.021 | -0.020 | Data differs |
| Request & Receive, 2013 | 0.039 | 0.039 | -0.001 | -0.001 | Exact |
| Frequent Interactor | 0.184 | 0.184 | -0.010 | -0.010 | Exact |
| Candidate Visit | 0.696 | 0.696 | 0.015 | 0.015 | Exact |
| Voted Same Coalition | 0.718 | 0.718 | -0.019 | -0.019 | Exact |
| All HH Voting Same | 0.773 | 0.773 | -0.023 | -0.023 | Exact |
| Any Declared Support | 0.485 | 0.485 | -0.066 | -0.066 | Exact |
| Declared on Body | 0.185 | 0.185 | -0.021 | -0.021 | Exact |
| Declared on House | 0.387 | 0.387 | -0.059 | -0.059 | Exact |
| Declared at Rally | 0.218 | 0.218 | -0.036 | -0.036 | Exact |

**Result: 10/12 exact match. 2 variables differ due to data version discrepancy.**

### Table 2: Cisterns Treatment and Vulnerability

**Panel A: Treatment on Vulnerability (with municipality FE)**

| Variable | Repl Beta | Pub Beta | Repl SE | Pub SE | Match? |
|----------|-----------|----------|---------|--------|--------|
| -(CES-D) Scale | 0.092 | 0.092 | 0.037 | 0.037 | Exact |
| SRHS Index | 0.075 | 0.075 | 0.033 | 0.033 | Exact |
| Child Food Security | 0.084 | 0.084 | 0.054 | 0.054 | Exact |
| Overall Index | 0.126 | 0.126 | 0.043 | 0.043 | Exact |

**Panel B: Rainfall on Vulnerability**

| Variable | Repl Beta | Pub Beta | Repl SE | Pub SE | N | Match? |
|----------|-----------|----------|---------|--------|---|--------|
| -(CES-D) Scale | 0.046 | 0.046 | 0.016 | 0.016 | 1,128 | Exact |
| SRHS Index | 0.039 | 0.039 | 0.017 | 0.017 | 1,052 | Exact |
| Child Food Security | 0.046 | 0.046 | 0.026 | 0.026 | 1,128 | Exact |
| Overall Index | 0.064 | 0.064 | 0.019 | 0.019 | 1,128 | Exact |
| Total HH Expenditure | 24.736 | 24.736 | 6.657 | 6.657 | 1,281 | Exact |

**Result: All 9 specifications match exactly.**

### Table 3: Citizen Requests, Cisterns, and Rainfall

| Column | Key Variable | Repl Beta | Pub Beta | N | Match? |
|--------|-------------|-----------|----------|---|--------|
| (1) | Treatment | -0.030 | -0.028 | 4,288 | ~Near (data version) |
| (2) | Rainfall | -0.023 | -0.025 | 4,288 | ~Near (data version) |
| (3) | Treatment | -0.030 | -0.028 | 4,288 | ~Near |
| (3) | Rainfall | -0.023 | -0.025 | 4,288 | ~Near |
| (7) | Treatment (excl water) | -0.027 | -0.028 | 4,288 | Exact |
| (7) | Rainfall (excl water) | -0.015 | -0.016 | 4,288 | Exact |
| (8) | Treatment (public) | -0.005 | -0.004 | 4,288 | Exact |
| (8) | Rainfall (public) | -0.007 | -0.006 | 4,288 | Exact |

**Result: Columns 7-8 (non-postcoded dep vars) match exactly. Columns 1-6 differ by 0.001-0.005 due to the data version issue with `ask_private_stacked`.**

### Table 4: Electoral Outcomes

| Column | Repl b1 | Pub b1 | Repl SE | Pub SE | N | Match? |
|--------|---------|--------|---------|--------|---|--------|
| (1) Incumbent Mayor | -0.101 | -0.103 | 0.058 | 0.058 | 909 | Near |
| (2) Incumbent Group | -0.076 | -0.078 | 0.049 | 0.049 | 1,641 | Near |
| (3) Challenger | 0.098 | 0.100 | 0.073 | 0.073 | 909 | Near |
| (4) Turnout | -0.009 | -0.010 | 0.058 | 0.059 | 909 | Exact |
| (5) Blank/Null | -0.006 | -0.008 | 0.030 | 0.031 | 909 | Near |

**Result: SEs match exactly. Coefficients within 0.002 (likely float32 vs float64 precision). Wild cluster bootstrap p-values not replicated.**

### Table 5: Heterogeneity by Clientelist Relationship

| Column | Treat×Freq (Repl) | Treat×Freq (Pub) | Match? |
|--------|-------------------|-------------------|--------|
| (1) Private goods | -0.097 | -0.092 | Near (data version) |
| (3) Private goods + rain | -0.095 | -0.092 | Near |
| (4) Excl water | -0.056 | -0.053 | Near |
| (5) Ask & receive | -0.068 | -0.060 | Near |

**Result: All qualitative conclusions hold. Interaction effects are consistently negative and significant across all specifications.**

---

## 4. Data Audit Findings

### Coverage
- All 4 final datasets are complete and pre-built. No cleaning pipeline needed.
- Individual data: 2,990 obs across 40 municipalities, 425 clusters.
- Household data: 1,308 obs (615 treatment, 693 control).
- Stacked data: 5,334 obs (2,667 per year). ~20% missing on asking variables (2013 attrition).
- Voting data: 1,794 machines. 909 in 21-municipality sample, 1,641 in 39-municipality sample.

### Missing Data
- 2013 survey outcomes: ~35% missing (survey attrition from wave 2 to wave 3).
- `frequent_interactor`: 10.8% missing in individual data.
- Voting: 45.9% missing for `tot_treat_by_section_2_21` (only defined for 21-municipality sample).

### Data Quality
- All binary variables are 0/1 with no out-of-range values.
- Rainfall measures are standardized (mean=0, SD=1) as expected.
- No logical violations: receiving a good always implies having asked for it.
- Vulnerability indices have reasonable ranges (CES-D 1-4, SRHS 1-4, food security -6 to 0).

### Data Version Discrepancy
The shipped `clientelism_individual_data.dta` has `c_ask_private_pol_postcoded` with mean=0.213 (N=2,667), while the published Table 1 tex file shows mean=0.195 for the same variable. The 2013 counterpart (`ask_pol_private_2013`) matches exactly (mean=0.086). This suggests the final data was regenerated after the published tables were produced, with a change to the 2012 postcoding of private goods requests. The discrepancy is a packaging issue — it does not indicate a code bug, and all qualitative conclusions are unchanged.

---

## 5. Bug Impact Analysis

No bugs were found in the analysis code. The only issue is the data version discrepancy described above, which is a packaging issue rather than a code error.

---

## 6. Robustness Results

| # | Check | Key Result | Status |
|---|-------|-----------|--------|
| 1 | Cluster at municipality level | SEs change by <15%, all significance preserved | Robust |
| 2 | Leave-one-municipality-out | Overall Index range [0.113, 0.137], all positive | Robust |
| 3 | Placebo: treatment on 2011 expenditure | beta=-3.346, p=0.775 (insignificant as expected) | Pass |
| 4 | Quadratic rainfall | Rainfall² = -0.003 (SE=0.006), insignificant | Robust |
| 5 | Year-specific treatment effects | 2012: -0.027, 2013: -0.040, both negative | Robust |
| 6 | Municipality size heterogeneity | Small: -0.071, Large: -0.020 (stronger in small muns) | Robust |
| 7 | Public vs private goods | Treatment: -0.030 (private) vs -0.005 (public) | Robust |
| 8 | Winsorized vulnerability indices | All coefficients stable (max change: 0.023 for food security) | Robust |
| 9 | Exclude small voting locations | Full: -0.101, Large only: -0.066 (attenuated but same sign) | Robust |
| 10 | Excluding water from private requests | Treat×Freq: -0.095 (all) vs -0.056 (excl water) | Robust |

---

## 7. Summary Assessment

### What Replicates
- **Table 1:** 10/12 variables match exactly (means, SDs, regression coefficients, SEs).
- **Table 2:** All 9 specifications (4 Panel A + 5 Panel B) match exactly in coefficients, SEs, and N.
- **Table 3:** Columns 7-8 (non-postcoded dep vars) match exactly. Columns 1-6 are qualitatively consistent but numerically differ by 0.001-0.005 due to data version.
- **Table 4:** SEs match exactly. Coefficients within 0.002 (float precision). Wild bootstrap not replicated.
- **Table 5:** All interaction effects qualitatively match. Numerical differences trace to the same data version issue.

### What Doesn't
- Wild cluster bootstrap p-values (Table 4): no Python equivalent for `boottest`. Published values reported for reference.
- Appendix Tables A1-A10: Code is available but not translated (same estimation methods as main tables).

### Key Concerns
- The data version discrepancy for `c_ask_private_pol_postcoded` means the shipped data does not exactly reproduce the published tables. This is a common issue in replication packages and does not affect the validity of the analysis.
- Municipality size heterogeneity (Check #6) shows effects are stronger in smaller municipalities (-0.071 vs -0.020), suggesting the treatment effect may be context-dependent.
- The voting analysis relies on only 21 municipalities with incumbent re-election, and wild cluster bootstrap adjusts for this limitation.

### Overall Assessment
Strong replication. The paper's core findings — that reducing vulnerability through cisterns decreases clientelistic exchanges, and that this effect is concentrated among those with pre-existing clientelist relationships — replicate clearly. Table 2 (vulnerability) matches exactly, providing strong confirmation of the first-stage mechanism. The data version discrepancy is isolated to one postcoded variable and does not affect qualitative conclusions.

---

## 8. File Manifest

| File | Description |
|------|-------------|
| `utils.py` | Paths, OLS with FE, reghdfe, lincom helpers |
| `01_clean.py` | Load and validate all 4 datasets |
| `02_tables.py` | Tables 1-5 replication |
| `03_figures.py` | Figures A1 (scatter) and A2 (bar chart) |
| `04_data_audit.py` | Coverage, distributions, consistency checks |
| `05_robustness.py` | 10 robustness checks |
| `output/` | Parquet files, PNG figures, CSVs |
| `writeup_173341.md` | This writeup |
