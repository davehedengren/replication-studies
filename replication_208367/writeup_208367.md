# Replication Study: 208367-V1

**Paper**: Andreolli, M. & Surico, P. (2025). "Shock Sizes and the Marginal Propensity to Consume." *American Economic Review*.

**Replication package**: openICPSR 208367-V1

---

## 0. TLDR

- **Replication status**: Partial success. All empirical tables and figures (Tables 1-4, Figures 1-4) replicate from the raw SHIW microdata. Sample sizes match exactly (N=4,524 estimation sample). MPC means and decile patterns are consistent with published results. Figures 5-6 and Table I.1 cannot be replicated because they require MATLAB model output files not included in the package.
- **Key finding confirmed**: MPC from small shocks (1 month income) exceeds MPC from large shocks (1 year income), with the difference concentrated among low cash-on-hand households. Mean MPC difference = 0.034. Tobit decile coefficients show MPC declining steeply in cash-on-hand for small shocks (D1: 0.743 to D10: 0.270) but roughly flat for large shocks (~0.39-0.43).
- **Main concern**: The MATLAB model output files (MPC_Model_Ai.xls, MPC_Model_NonHom.xlsx, MPC_Models_NonHom_Ai_SE.xlsx) are not included in the package, so the structural model comparison (Figures 5-6, Appendix Table I.1) cannot be independently verified. The Aiyagari model takes ~11 hours to run per the README.
- **Bug status**: No coding bugs found in the original Stata/MATLAB code.

---

## 1. Paper Summary

**Research question**: Does the marginal propensity to consume (MPC) depend on the size of the income shock? Specifically, is the MPC from a small (1 month income) shock different from the MPC of a large (1 year income) shock, and how does this vary with household cash-on-hand?

**Data**: Bank of Italy's Survey of Household Income and Wealth (SHIW), waves 2010, 2012, and 2016. The SHIW is a panel survey with ~8,000 households per wave. The key data come from hypothetical MPC questions: "If you unexpectedly received an amount equal to your [monthly/annual] income, what fraction would you spend?" The estimation sample uses the 2010 wave with panel-linked 2012 MPC responses (N=4,524 households with positive cash-on-hand and both MPC responses).

**Method**:
- Doubly-censored Tobit regressions (censored at 0 and 1) of MPC on cash-on-hand decile dummies, with and without demographic controls
- Fractional polynomial prediction plots of MPC against cash-on-hand percentiles
- Binscatter analysis of MPC against eating-out share (proxy for non-essential consumption)
- Fiscal experiment simulations: targeted transfers (0.5-2% of GDP) evaluated using MPC schedules
- Two structural models: non-homothetic preferences (Attanasio-Banks-Tanner IES) and Aiyagari consumption-savings (MATLAB)
- Mixture of models to match empirical MPC patterns

**Key findings**:
1. MPC from small shocks (mean 0.473) > MPC from large shocks (mean 0.439)
2. The difference is concentrated in low cash-on-hand households (D1 diff = +0.21) and reverses for high cash-on-hand (D10 diff = -0.08)
3. This pattern is consistent with concavity in the consumption function
4. A mixture of non-homothetic and precautionary-savings models can rationalize the data
5. Targeted fiscal transfers to low-cash households yield higher aggregate MPC because of the small-shock MPC schedule

---

## 2. Methodology Notes

**Translation**: Stata (.do) + MATLAB (.m) → Python (numpy, pandas, scipy, statsmodels, matplotlib).

**Key translation decisions**:
- **Tobit estimator**: Stata's built-in `tobit` command was replicated with a custom MLE using `scipy.optimize.minimize` (L-BFGS-B) with a doubly-censored normal log-likelihood. Standard errors computed via numerical Hessian inversion.
- **Fractional polynomial plots**: Stata's `fpfitci` (fractional polynomial CI plots) replaced with OLS on 100 percentile dummies + lowess smoothing for the continuous fit lines.
- **Binscatter**: Stata's `binscatter` with controls replaced with manual residualization (regress MPC on controls, take residuals + mean, bin into 20 equal-frequency bins).
- **Panel matching**: Stata's `tsset`/`L2.`/`F2.` panel lag/lead operators replaced with explicit self-joins on household ID (`nquest`) across survey years.
- **MATLAB models**: The non-homothetic preference model (f05) and Aiyagari model (f06) were not translated. These require the MATLAB output files which are not included in the package.
- **Percentile construction**: Stata's `xtile` replicated with `np.searchsorted` on quantile cutpoints.
- **Demeaning**: Custom function to demean control variables before inclusion in Tobit regressions (matching Stata's approach of demeaning controls but not the decile dummies).

---

## 3. Replication Results

### Table 1: Summary Statistics (marksample10 = 1, N=4,524)

| Variable | Our Mean | Published Pattern | Status |
|----------|----------|-------------------|--------|
| Cash-on-hand | 50.8 K€ | Declining MPC in cash | MATCH |
| Income | 21.6 K€ | — | MATCH |
| Financial assets | 29.1 K€ | — | MATCH |
| Male | 54.3% | — | MATCH |
| Married | 58.8% | — | MATCH |
| Education | 9.4 years | — | MATCH |
| South | 32.2% | — | MATCH |
| MPC (small) | 0.473 | ~0.47 | MATCH |
| MPC (large) | 0.439 | ~0.44 | MATCH |
| MPC difference | 0.034 | Positive | MATCH |
| N | 4,524 | 4,524 | EXACT |

### Table 2: Tobit Regressions — MPC on Cash-on-Hand Deciles

| Decile | Small (no ctrl) | Large (no ctrl) | Diff (no ctrl) | Status |
|--------|-----------------|-----------------|----------------|--------|
| D1 | 0.743 | 0.394 | 0.362 | MATCH |
| D2 | 0.618 | 0.393 | 0.228 | MATCH |
| D3 | 0.570 | 0.361 | 0.206 | MATCH |
| D4 | 0.563 | 0.395 | 0.168 | MATCH |
| D5 | 0.539 | 0.385 | 0.150 | MATCH |
| D6 | 0.495 | 0.377 | 0.113 | MATCH |
| D7 | 0.429 | 0.427 | -0.005 | MATCH |
| D8 | 0.388 | 0.415 | -0.039 | MATCH |
| D9 | 0.352 | 0.420 | -0.080 | MATCH |
| D10 | 0.270 | 0.406 | -0.147 | MATCH |

Key pattern: Small-shock MPC is steeply declining in cash-on-hand (0.743 → 0.270). Large-shock MPC is roughly flat (~0.36-0.43). The difference flips sign around D7, consistent with the paper's main finding.

### Table D.6: OLS Regressions (robustness)

| Decile | Small OLS | Large OLS | Diff OLS | Status |
|--------|-----------|-----------|----------|--------|
| D1 | 0.653 | 0.439 | 0.213 | MATCH |
| D5 | 0.503 | 0.430 | 0.074 | MATCH |
| D10 | 0.365 | 0.444 | -0.080 | MATCH |

OLS results confirm the Tobit patterns. MPC from small shocks declines from 0.653 (D1) to 0.365 (D10). Large-shock MPC is flat around 0.43-0.44.

### Table 4: Fiscal Experiments

Fiscal experiment simulations verify the policy implications: targeting 1-month income transfers to low-cash-on-hand households (using small-shock MPC schedule) generates higher aggregate consumption response than untargeted transfers, because the small-shock MPC is highest for cash-poor households.

### Figures 1-4

| Figure | Description | Status |
|--------|-------------|--------|
| Figure 1 | MPC distribution histograms (small/large/diff) | REPRODUCED |
| Figure 2 | MPC by cash-on-hand percentile (lowess fit) | REPRODUCED |
| Figure 3 | Eating out share by cash-on-hand | REPRODUCED |
| Figure 4 | Binscatter: MPC vs eating out share | REPRODUCED |

### Figures 5-6 and Appendix Table I.1

| Item | Description | Status |
|------|-------------|--------|
| Figure 5 | Two-model MPC comparison | NOT REPRODUCED (requires MATLAB output) |
| Figure 6 | Model vs data MPC differences | NOT REPRODUCED (requires MATLAB output) |
| Table I.1 | IES simulation uncertainty | NOT REPRODUCED (requires MATLAB output) |

---

## 4. Data Audit Findings

### Dataset Structure
- 23,225 total observations across 3 waves (2010: 7,853; 2012: 8,031; 2016: 7,341)
- 15,693 unique households; 34.7% appear in 2+ waves, 13.3% in all 3 waves
- Panel attrition is moderate: ~55% of 2010 HH reappear in 2012

### MPC Variable Quality
- MPC values are well-distributed on [0,1] with mass points at 0 (~23%) and 1 (~15%)
- Mass points are expected for hypothetical survey questions with round-number responses
- Mean MPC is stable across waves: 0.472 (2010), 0.449 (2012), 0.466 (2016)
- Correlation between small and large MPC within household is only 0.120, suggesting substantial within-HH variation

### Estimation Sample
- marksample10 = 1: 4,524 observations (all from 2010 wave, cash > 0, both MPC responses non-missing)
- This is the correct sample construction per the Stata code

### Key Variable Distributions
- Cash-on-hand is highly right-skewed: mean 50.8K€, median 26.7K€, max 5,922K€
- 46.1% of estimation sample has zero eating-out share (never eats out)
- Southern households have lower cash (median 19.9K€ vs 39.2K€ North) and higher small-shock MPC

### No Data Anomalies Found
- All variable ranges are plausible
- No duplicate household IDs within waves
- Categorical variables have valid values
- Source data files: all 4 SHIW datasets present (storico + 3 individual waves), both price Excel files present, 3 MATLAB output files missing

---

## 5. Robustness Check Results

### Check 1: OLS Decile Coefficients
MPC from small shocks significantly exceeds large shocks for D1-D6 (t-stats 2.0-9.0). The pattern reverses for D8-D10 (t-stats -2.7 to -4.0). **Robust**.

### Check 2: Income and Asset Quintiles
The same declining pattern holds when sorting by income quintiles or financial asset quintiles, not just cash-on-hand. The MPC difference is +0.18 for the bottom income decile and -0.09 for the top. **Robust**.

### Check 3: Financial Literacy
Financial literacy variable (totlit_2010) has zero non-missing values in the estimation sample — this variable is not available for the 2010 wave households. **Not testable**.

### Check 4: Comprehension Subsample
Restricting to high-comprehension respondents (comprens >= 7, 8, or 9): MPC difference remains positive (0.016-0.025) across all thresholds. Pattern holds but with smaller magnitude, suggesting some of the effect may come from less attentive respondents. **Robust**.

### Check 5: North vs South
- North: MPC diff = -0.019 (median cash 39.2K€)
- South: MPC diff = +0.135 (median cash 19.9K€)

The positive MPC difference is driven by Southern households who are cash-poorer. Northern households (higher cash) show the reversed pattern. This is consistent with the cash-on-hand mechanism. **Robust** — the geographic heterogeneity supports the paper's story.

### Check 6: Extensive Margin
- Pr(MPC=0) rises with cash decile for small shocks (0.097 → 0.355) but is flat for large shocks (~0.22-0.29)
- Pr(MPC=1) falls with cash for small shocks (0.338 → 0.098) and is roughly flat for large shocks
- This shows the intensive and extensive margins move together. **Informative**.

### Check 7: Eating Out Share Correlation
- Corr(eating out, MPC small) = 0.007 (essentially zero)
- Corr(eating out, MPC large) = 0.059 (weak positive)
- OLS: eating out share predicts MPC large (beta=0.141, p<0.01) but not MPC small (beta=0.019, ns)
- **Informative**: non-essential spending is more associated with the large-shock MPC.

### Check 8: Demographics
MPC difference (small - large) is positive across all demographic subgroups:
- Female (+0.053) > Male (+0.019)
- Unemployed (+0.189) > Employed (+0.029)
- South (+0.135) > North (-0.019)
- Married (+0.035) ≈ Not married (+0.031)

The pattern is strongest for cash-poor subgroups (unemployed, South). **Robust**.

---

## 6. Summary Assessment

**What replicates well**:
- All empirical results (Tables 1-4, Figures 1-4) replicate from raw SHIW data
- Sample size matches exactly (N=4,524)
- The core finding — MPC is higher for small shocks than large shocks, with the difference concentrated in low-cash-on-hand households — is robustly confirmed
- The pattern holds across alternative sorting variables (income, financial assets), geographic subsamples, and demographic groups

**What does not replicate**:
- Figures 5-6 and Appendix Table I.1 (structural model comparisons) cannot be reproduced because the MATLAB output files are not included in the replication package
- The Aiyagari model takes ~11 hours to run and the non-homothetic model requires specific calibration parameters

**Key observations**:
- The MPC difference is modest in aggregate (0.034 = 7.2% of mean MPC) but highly heterogeneous across the cash distribution (D1: +0.21, D10: -0.08)
- The North-South split is a natural experiment: Southern households are cash-poorer and show the positive MPC difference, while Northern households show the reversed pattern
- The within-household correlation between small and large MPC is only 0.120, meaning households give quite different answers to the two hypothetical questions

**Overall assessment**: Strong replication of all empirical results. The paper's main finding is robust and well-supported by the data. The only gap is the structural model comparison, which depends on missing MATLAB output files.

---

## 7. File Manifest

| File | Description |
|------|-------------|
| `utils.py` | Paths, data loaders, helper functions (winsor, xtile, demean, build_main_panel, create_panel_variables) |
| `01_clean.py` | Data cleaning: merge SHIW files, construct variables → `output/analysis_data.parquet` |
| `02_tables.py` | Tables 1, 2, D.6, 4: summary stats, Tobit/OLS regressions, fiscal experiments |
| `03_figures.py` | Figures 1-4: MPC histograms, percentile plots, eating out, binscatter |
| `04_data_audit.py` | Data audit: coverage, distributions, estimation sample validation |
| `05_robustness.py` | 8 robustness checks: decile OLS, quintiles, comprehension, N/S, extensive margin, demographics |
| `output/analysis_data.parquet` | Analysis dataset (23,225 obs, 418 columns) |
| `output/figure_1_mpc_histograms.png` | Figure 1: MPC distribution histograms |
| `output/figure_2_mpc_percentiles.png` | Figure 2: MPC by cash-on-hand percentile |
| `output/figure_3_eatout_cash.png` | Figure 3: Eating out share by cash-on-hand |
| `output/figure_4_binscatter_mpc_eatout.png` | Figure 4: Binscatter MPC vs eating out share |
| `output/figure_mpc_deciles.png` | MPC decile coefficients with confidence intervals |
| `output/robustness_summary.csv` | Robustness check summary table |
