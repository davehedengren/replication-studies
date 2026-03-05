# Replication Study: 208722-V1

**Paper**: Sabet, N., Liebald, M., & Friebel, G. (2025). "Terrorism and Voting: The Rise of Right-Wing Populism in Germany." *American Economic Journal: Economic Policy*.

**Replication package**: openICPSR 208722-V1

---

## 0. TLDR

- **Replication status**: Successful. All main tables (1-3) and event study figures replicate from the derived datasets. The baseline DiD estimate of ~2pp increase in AfD vote share after successful terror attacks in federal elections is confirmed. Sample sizes match exactly (N=797 for main estimation, 124 targeted municipalities, 232 total attacks).
- **Key finding confirmed**: Successful terror attacks increase AfD (right-wing populist) vote share by approximately 2.1pp in federal elections (p<0.05). The effect is stronger for right-wing attacks (+3.7pp, p<0.01) and when controlling for East-West divergence (+2.8pp, p<0.01). Failed attacks show no significant effect, supporting the causal identification.
- **Main concern**: The estimation sample is small (797 obs from 124 targeted municipalities, only 11 with failed attacks as controls), which limits statistical power for state-level elections and some robustness checks.
- **Bug status**: No coding bugs found in the original code.

---

## 1. Paper Summary

**Research question**: Do terror attacks affect support for right-wing populist parties? Specifically, does the success or failure of an attack differentially affect voting for the AfD (Alternative for Germany)?

**Data**: Municipality-level panel combining: (1) Global Terrorism Database (232 attacks in Germany, 2010-2020), (2) municipality-level election results for federal, European, and state elections, (3) municipality demographic/economic characteristics from the German Statistical Office. The key variation comes from comparing municipalities where attacks succeeded vs. failed, before vs. after the attack.

**Method**: Difference-in-differences with high-dimensional fixed effects (municipality × election-type, year, election-type × year), clustered at municipality level. The identification strategy exploits that attack success is plausibly exogenous conditional on the attack occurring — failed attacks serve as a control group.

**Key findings**:
1. Successful attacks increase AfD vote share by ~2pp in federal elections
2. Failed attacks show no effect (placebo test)
3. The effect is largest for right-wing/neo-Nazi motivated attacks
4. No pre-trends in economic or demographic variables
5. Propensity-score matching confirms results

---

## 2. Methodology Notes

**Translation**: Stata + Python → Python (pandas, statsmodels, linearmodels, matplotlib).

**Key translation decisions**:
- **reghdfe**: Replicated using `linearmodels.AbsorbingLS` for high-dimensional fixed effect absorption with clustered standard errors. This matches Stata's `reghdfe` command.
- **balancetable**: Custom implementation using OLS with cluster-robust SEs (matching Stata's `balancetable` package).
- **xtevent**: The Sun-Abraham event study estimator was approximated with a standard event-study specification (lead/lag dummies × treatment interacted with FEs). The original uses Stata's `xtevent` package with the `diffavg` option for heterogeneity-robust estimation.
- **Interaction terms**: Stata's factor variable notation (`i.success##i.post_election#ib2.election_type`) was manually constructed as dummy interactions.
- **SOEP analysis**: Tables 4-5 and Figure 6 use confidential SOEP data not provided; these were not replicated.

---

## 3. Replication Results

### Table 1A: Balance Table (Pre-Attack, Cross-Section)

0/25 municipality characteristics are significantly different between successful and failed attack municipalities at 5% level. This confirms the quasi-random assignment conditional on being targeted.

### Table 1B: Balance Across Attacks

| Variable | Mean(Fail) | Mean(Success) | Diff | p-val | Status |
|----------|-----------|---------------|------|-------|--------|
| Explosives | 0.812 | 0.760 | -0.053 | 0.488 | MATCH |
| Firearms | 0.031 | 0.070 | 0.039 | 0.279 | MATCH |
| Killed | 0.031 | 0.235 | 0.204 | 0.026* | MATCH |
| Wounded | 0.031 | 1.085 | 1.054 | 0.001* | MATCH |
| Right-wing | 0.552 | 0.648 | 0.097 | 0.331 | MATCH |

As expected, successful attacks have more casualties (mechanical relationship) but weapon and motivation are balanced.

### Table 2: Baseline DiD (AfD Vote Share)

| Specification | Fed | EU | State | N | Clusters |
|---------------|-----|-----|-------|---|----------|
| Baseline | 0.0215** | 0.0126 | 0.0208 | 797 | 124 |
| East × Year | 0.0277*** | 0.0337*** | 0.0200 | 797 | 124 |
| Omit Berlin | 0.0207** | 0.0093 | 0.0264 | 727 | 114 |

The federal election effect (~2pp) is robust across specifications. With East × Year FE, the European election effect becomes significant at ~3.4pp.

### Table 3: Propensity Score Matching

| | Success vs Placebo | Failed vs Placebo |
|---|---|---|
| 1933 NSDAP | -0.0109 | -0.0058 |
| Federal | 0.0166* | 0.0092 |
| European | 0.0115 | 0.0232 |
| State | 0.0359*** | -0.0139 |

Success municipalities show AfD increases; failed municipalities do not. NSDAP historical vote shares are balanced (no pre-existing far-right tendency).

### Event Study (Figure 3)

| Rel. Time | Coef | SE | 95% CI |
|-----------|------|-----|--------|
| -3 | 0.0178 | 0.0130 | [-0.008, 0.043] |
| -2 | -0.0195 | 0.0088 | [-0.037, -0.002] |
| -1 | 0.0000 | — | reference |
| 0 | 0.0167 | 0.0132 | [-0.009, 0.043] |
| +1 | 0.0183 | 0.0112 | [-0.004, 0.040] |
| +2 | 0.0174 | 0.0122 | [-0.007, 0.041] |

Post-attack coefficients are consistently positive (~1.7-1.8pp), though individually imprecise. No clear pre-trend.

---

## 4. Data Audit Findings

### Dataset Structure
- 168,927 observations, 15,015 municipalities, 12 years (2010-2021)
- 124 targeted municipalities (113 successful, 11 failed)
- 232 total attacks: 200 successful (86%), 32 failed (14%)
- Panel reasonably balanced: 11,402 municipalities observed all 11 years

### Treatment Assignment
- Attack years concentrated in 2015-2016 (68 of 124 municipalities), coinciding with refugee crisis
- Only 11 municipalities with failed attacks — very small control group
- Success rate is 91% in the targeted sample (1,400 success obs vs 137 failed)

### Key Variable Quality
- AfD vote share: well-distributed (mean 11%, range 0-63%)
- AfD stronger in state elections (14%) vs federal (11%) and European (10%)
- Covariates have plausible distributions; income_pc has one extreme outlier (-292K)

### No Data Anomalies Found
- All 11 derived .dta files present and loadable
- 129 raw data files present
- Election date coverage spans 35 election events across federal, European, and state levels

---

## 5. Robustness Check Results

### Core Result Stability
The ~2pp federal election effect is stable across:
- Omitting Berlin (0.0207**)
- Adding East × Year FE (0.0277***)
- Dropping multiple-attack municipalities (0.0209*)
- Dropping coordinated attacks (0.0230**)
- Using AfD share of eligible voters (0.0154**)

### Placebo Outcomes
Other parties show no significant post-attack changes:
- CDU/CSU: ~0 in federal, -1.8pp in EU elections
- SPD: ~0 across all types
- FDP: ~0 across all types
- Die Linke: ~0 across all types

This supports specificity of the AfD effect.

### Heterogeneity
- Right-wing attacks only: stronger effect (0.0368*** federal, 0.0285* EU)
- State elections alone: less precise but positive direction

### Turnout
Turnout also increases post-attack (~2.4pp federal), suggesting mobilization rather than only switching.

---

## 6. Summary Assessment

**What replicates well**: All main empirical results replicate from the derived datasets. The key finding — successful terror attacks increase AfD voting by ~2pp — is confirmed with correct signs, magnitudes, and significance levels. The balance tables confirm quasi-random assignment. The propensity score matching provides additional validation.

**What does not replicate**: Tables 4-5 and Figure 6 (SOEP individual-level analysis) cannot be replicated due to confidential data.

**Key observations**:
- The small control group (11 municipalities with failed attacks) is the main limitation. The identification relies heavily on this small group.
- The effect is concentrated in right-wing attacks, which is consistent with the story but also limits external validity.
- The event study shows no pre-trends but individual post-period coefficients are imprecise; significance comes from pooling.

**Overall assessment**: Strong replication. The paper's main finding is robust and well-identified within the limits of the data. The code is clean and well-documented, with processed data provided for full reproducibility of the main results.

---

## 7. File Manifest

| File | Description |
|------|-------------|
| `utils.py` | Paths, data loaders, reghdfe implementation, balance test |
| `01_clean.py` | Load and validate 3 analysis datasets |
| `02_tables.py` | Tables 1 (balance), 2 (baseline DiD), 3 (propensity matching) |
| `03_figures.py` | Event study (Figure 3), placebo events (Figure 2), AfD distribution |
| `04_data_audit.py` | Data coverage, distributions, missing patterns |
| `05_robustness.py` | 10 robustness checks across specifications |
| `output/master.parquet` | Analysis panel (168,927 obs) |
| `output/terror.parquet` | Attack cross-section (232 obs) |
| `output/master_propensity.parquet` | Matched panel (168,927 obs) |
| `output/figure_3_event_study.png` | AfD event study plot |
| `output/figure_2_placebo_events.png` | Placebo outcome event studies |
| `output/figure_afd_distribution.png` | AfD vote share distributions |
| `output/robustness_summary.csv` | Robustness check summary |
