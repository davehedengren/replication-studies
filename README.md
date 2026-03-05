# Replication Studies

Python replications of economics papers from openICPSR replication packages. Each study translates the original code (typically Stata, R, or MATLAB) into Python and verifies published results, audits data quality, and runs robustness checks.

## Summary Statistics

| Metric | Count |
|--------|-------|
| Papers assessed | 32 |
| Completed replications | 17 |
| Skipped (data unavailable) | 15 |
| Bugs found | 3 |
| Bugs affecting conclusions | 0 |

---

## Completed Replications

### Full / Near-Exact Replications

| ID | Paper | Authors | Journal | Status | Bug? |
|----|-------|---------|---------|--------|------|
| 212821 | Underestimating Learning by Doing | Horn & Loewenstein | AEJ: Micro 2025 | Exact (all stats match to 2dp) | None |
| 226781 | Trade, Value Added, and Productivity Linkages | de Soyres & Gaillard | AER 2025 | Exact (4 tables match to 4sf) | None |
| 237010 | Temporary Layoffs and Cyclical Unemployment | Gertler, Huckfeldt, Trigari | AER 2025 | Exact (12+ tables from pre-computed) | None |
| 179162 | Mother-Father Differences in Spending on Children | Dizon-Ross & Jayachandran | AER: Insights 2023 | Near-exact (coefficients to 3dp) | None |
| 208722 | Terrorism and Voting in Germany | Sabet, Liebald, Friebel | AEJ: Policy 2025 | Full (Tables 1-3, event studies) | None |
| 239496 | Persistently Low Global Fertility | Geruso & Spears | JEP 2026 | Full (all replicable figures) | None |
| 239791 | Family Institutions and the Fertility Transition | Gobbi, Hannusch, Rossi | JEP 2026 | Full (24/24 figures, R² to 3dp) | **Yes** (minor) |
| 241085 | Antitrust Enforcement in Labor Markets | Prager | JEP 2026 | Full (Figure 1, clusters match) | None |
| 173341 | Vulnerability and Clientelism | Bobonis, Gertler, Gonzalez-Navarro, Nichter | AER 2022 | Near-exact (Table 2 exact, Tables 1,3-5 within 0.002) | None |

### Partial Replications (data constraints)

| ID | Paper | Authors | Journal | Status | Bug? |
|----|-------|---------|---------|--------|------|
| 192297 | Big Loans to Small Businesses | Bryan, Karlan, Osman | AER 2024 | Tables 1-4 replicate | **Yes** (profit double-count) |
| 208367 | Shock Sizes and the MPC | Andreolli & Surico | AER 2025 | Tables 1-4, Figs 1-4 (model needs MATLAB) | None |
| 219907 | Labor Market Power and Self-Employment | Amodio, Medina, Morlacco | AER 2025 | OLS/IV match qualitatively | None |
| 221423 | Income Inequality in the Nordic Countries | Mogstad, Salvanes, Torsvik | AER 2025 | Tables 1,3 + Figs 2,3 (PIAAC data missing) | **Yes** (minor) |
| 227802 | Childcare Laws for Women's Empowerment | Anukriti et al. | AEA P&P 2025 | TWFE consistent with synthdid | None |
| 228101 | Gender Gaps in Entrepreneurship (Ghana) | Lambon-Quayefio et al. | AEA P&P 2025 | All findings replicate | **Yes** (packaging) |
| 238484 | The Price of War | Federle et al. | AER 2025 | Table 1 exact; LP IRFs qualitative | None |
| 238658 | NC Falsification Tests for IV | Danieli, Nevo, Oster | REStat 2025 | Literature survey exact; tests replicate | None |

---

## Key Findings by Paper

### 212821 — Underestimating Learning by Doing
People systematically underpredict their future task performance. Prediction errors are negative (mean = -0.42 to -1.39 across studies). Performers underpredict more than outside observers. Exceptionally clean data: 100% complete, zero missing values.

### 226781 — Trade, Value Added, and Productivity Linkages
Intermediate goods trade intensity predicts GDP comovement (beta = 0.066, p < 0.05); final goods trade does not. Extensive margin drives the result. All 4 empirical tables match to 4 decimal places. One of the cleanest replication packages encountered.

### 237010 — Temporary Layoffs and Cyclical Unemployment
Temporary layoffs account for 37-78% of unemployment increases across recessions. TL workers have much higher re-employment rates (0.427 vs 0.227 for job losers). The 2020 COVID recession was dominated by TL (78%) with rapid recall.

### 179162 — Mother-Father Differences in Spending on Children
Fathers spend significantly less on daughters than sons (-0.102 SD). The mother-daughter interaction is positive (+0.132 SD), indicating mothers don't discriminate. **Concern**: The interaction is entirely driven by non-incentivized goods and disappears for goods with real financial stakes.

### 192297 — Big Loans to Small Businesses
Positive but insignificant average treatment effect of large microfinance loans on profits, with strong ML-predicted heterogeneity (top quartile gains, bottom loses). **Bug found**: Stata code double-counts primary business profits via `rowtotal`, inflating the ATE by ~40%. Fixing it reduces ATE from 1,294 to 788 EGP but doesn't change qualitative conclusions. ML group assignments are unaffected.

### 208367 — Shock Sizes and the MPC
MPC from small income shocks (1 month) exceeds MPC from large shocks (1 year), with the difference concentrated among low cash-on-hand households. Tobit decile coefficients show MPC declining steeply in cash-on-hand for small shocks (D1: 0.743 to D10: 0.270) but flat for large shocks.

### 208722 — Terrorism and Voting in Germany
Successful terror attacks increase AfD (right-wing populist) vote share by ~2pp in federal elections. Failed attacks show no effect (placebo test). Effect is strongest for right-wing/neo-Nazi motivated attacks. **Concern**: Small sample (797 obs, only 11 failed-attack controls).

### 219907 — Labor Market Power and Self-Employment
Higher employer concentration (HHI) increases self-employment and reduces wages in Peruvian manufacturing. IV estimate of inverse labor supply elasticity: 0.455 (p < 0.01), implying significant monopsony power. **Concern**: Only 2.5% of workers (7,637/307,566) have HHI data after merge.

### 221423 — Income Inequality in the Nordic Countries
Nordic countries have Gini of 0.27 (disposable income) vs US 0.39 and OECD 0.31. Nordic redistribution gap (market - disposable Gini) is 0.12 vs 0.08 for US. High union density (50-67%), employment rates (82-88%), and public sector employment (25-31%) distinguish Nordic labor markets. Table 1: 60/60 values match. Table 3: 8/8 match. **Minor issue**: Code uses 2021 public employment data rather than stated 2019.

### 227802 — Childcare Laws for Women's Empowerment
Childcare law enactment increases female labor force participation by ~1.4-1.6 pp. Affordability provisions show strongest effects (~2.7-3.2 pp). Effects grow over time to ~2.7 pp by year 5. **Concern**: Effect is heavily driven by Sub-Saharan Africa; dropping SSA eliminates significance.

### 228101 — Gender Gaps in Entrepreneurship (Ghana)
Female business owners in Ghana earn ~30% less than males and have smaller business networks. The networking index predicts profits (GHS 610 per SD). **Concern**: Adding product fixed effects reduces the gender coefficient by ~40% and renders it insignificant, suggesting much of the gap operates through product selection. **Packaging bug**: Published code references undefined variable `profit` (should be `profits_lastmonth`).

### 238484 — The Price of War
Wars cause persistent GDP declines at war sites (~6-8% after 8 years), with effects propagating to belligerents and third parties through trade and proximity channels. 694 war sites across 502 wars (1816-2024).

### 238658 — NC Falsification Tests for IV
Negative control falsification tests can detect IV validity violations that standard overidentification tests miss. ADH's China Shock IV fails the NC test. Deming's school choice lottery passes. Literature survey: only 51% of IV papers (72/140) report any falsification test.

### 239496 — Persistently Low Global Fertility
Where completed cohort fertility has fallen below 1.9, no subsequent cohort has rebounded above 2.1 — "0-for-24" countries. Two-thirds of world population lives in countries with TFR below replacement. About 38% of CCF decline is due to rising childlessness; 62% is smaller families among parents.

### 239791 — Family Institutions and the Fertility Transition
Countries with "good" family institutions (monogamy, partible inheritance) have dramatically higher R-squared when regressing TFR changes on development indicators (0.816 vs 0.245). **Bug found**: R code computes GDP differences in levels rather than logs for Figure 4 despite the variable name `dlgdppc`. R-squared gap changes from 0.571 to 0.519 with the fix — qualitative conclusion unchanged.

### 241085 — Antitrust Enforcement in Labor Markets
Occupation clusters based on worker transition probabilities frequently cross standard SOC major-group boundaries, suggesting formal classifications may not reflect actual labor market substitutability. Weighted-average transition correlation of 0.058 replicates exactly.

### 173341 — Vulnerability and Clientelism
Randomized water cisterns in Brazil's semiarid northeast reduce private goods requests from politicians (-0.028 pp) and improve well-being (Overall Index +0.126 SD). Rainfall shocks independently reduce requests (-0.025). Effects concentrated among those with clientelist relationships (-0.092 interaction). Treatment reduces incumbent vote share (-0.103, bootstrap p=0.038). Table 2 (vulnerability) matches exactly; small discrepancies in Tables 3/5 trace to a data version issue in the shipped package.

---

## Bugs Found

| Paper | Bug | Severity | Impact on Conclusions |
|-------|-----|----------|----------------------|
| 192297 | Double-counts primary business profits in `rowtotal` | Moderate | ATE inflated ~40%; qualitative results unchanged |
| 239791 | Figure 4 uses level GDP differences instead of log | Minor | R² changes 0.571 → 0.519; conclusion unchanged |
| 228101 | Code references undefined variable `profit` | Packaging | Code won't run as-is; analytical results unaffected |
| 221423 | Uses 2021 public employment instead of stated 2019 | Minor | ~1pp difference; rankings unchanged |

---

## How This Was Made

This project was produced by [Claude Code](https://claude.ai/claude-code) (Anthropic's AI coding agent) working from a set of written instructions. A human selected the papers, downloaded the replication packages, and reviewed the output. Claude Code did everything else: reading the original code, translating it to Python, running the scripts, debugging errors, comparing results to published values, and writing the writeups.

### Process

Each replication package was downloaded from [openICPSR](https://www.openicpsr.org/openicpsr/) and unzipped into this directory. Packages can be found at:

```
https://www.openicpsr.org/openicpsr/project/{ID}/version/V1/view
```

For example, paper 226781 is at https://www.openicpsr.org/openicpsr/project/226781/version/V1/view.

Claude Code was given the following instructions (saved in [`instructions.txt`](instructions.txt)) and worked through papers sequentially:

**Phase 1 — Orientation:** Read the README, published paper PDF, and all source code. Identify the original language, datasets, main tables/figures, and estimation strategy.

**Phase 2 — Translate & Reproduce:** Create a `replication_{ID}/` directory. Write `utils.py` with shared paths and helpers, then translate each analysis script to Python (`01_clean.py`, `02_tables.py`, `03_figures.py`). Match the original sample construction exactly. Compare every output to published values.

**Phase 3 — Data Audit:** Write `04_data_audit.py` checking coverage, distributions, logical consistency, missing data patterns, panel balance, and duplicates.

**Phase 4 — Robustness Checks:** Write `05_robustness.py` with 6-12 alternative specifications tailored to each paper (e.g., leave-one-out, alternative SEs, placebo tests, subgroup heterogeneity, winsorization, alternative functional forms).

**Phase 5 — Writeup:** Write `writeup_{ID}.md` with a structured TLDR, paper summary, methodology notes, side-by-side replication results, data audit findings, bug impact analysis (if applicable), robustness results, and file manifest.

### Feasibility Assessment

Before attempting each paper, Claude Code assessed whether replication was feasible by checking:
- Whether the required data was included or available for download
- Whether the data required restricted/confidential access
- Whether the raw data size was manageable (papers with 20+ GB of raw data and no pre-built intermediates were skipped)
- Whether the computational pipeline was translatable to Python (papers requiring only MATLAB/Dynare DSGE models with no empirical component were skipped)

Of 31 papers assessed, 15 were skipped — most commonly because the core data required restricted institutional access (e.g., FSRDC, JPMCI, hospital administrative records, commercial databases).

### What Claude Code Did Well
- Translating Stata and R estimation code to Python equivalents
- Matching sample sizes and coefficients to published values
- Identifying bugs in original code (3 found across 16 papers)
- Running systematic robustness checks beyond what the original papers report
- Producing structured, comparable writeups across all papers

### Limitations
- No Python equivalent for some estimators (e.g., `synthdid`, Driscoll-Kraay SEs, GAM-based tests) — closest available methods were used with differences documented
- MATLAB/Dynare structural models were not translated; only pre-computed outputs were verified
- High-dimensional fixed effect absorption in Python (`linearmodels.AbsorbingLS`) occasionally produces slightly different SEs than Stata's `reghdfe` due to convergence tolerance differences

---

## Skipped Papers

| ID | Reason |
|----|--------|
| 219181 | FSRDC restricted-access data required |
| 220321 | JPMCI restricted-access data required |
| 201464 | Aberdeen restricted-access data required |
| 231821 | Chinese administrative restricted data required |
| 235542 | Norwegian register restricted data required |
| 214121 | GSS geographic identifiers restricted |
| 213741 | DHS, POEA migration, Census data all restricted |
| 234421 | Chilean hospital confidential data not provided |
| 229322 | CRISM proprietary mortgage/credit data; SQL to restricted DB |
| 223281 | FAME commercial data + UK SecureLab restricted |
| 225841 | Confidential CoreLogic/CBRE data + MATLAB/Dynare model |
| 213241 | No data included in package; all external sources |
| 237164 | oTree experimental platform only |
| 206261 | 20+ GB raw microdata; 10-hour build pipeline; no intermediates |
| 199083 | 23 GB raw data; 20-step pipeline; infeasible at scale |

---

## Repository Structure

Each completed replication follows a standard structure:

```
replication_{ID}/
  utils.py              # Paths, constants, helper functions
  01_clean.py           # Data loading and validation
  02_tables.py          # Main table replication
  03_figures.py         # Main figure replication
  04_data_audit.py      # Coverage, distributions, missing data
  05_robustness.py      # Alternative specifications
  writeup_{ID}.md       # Detailed writeup with comparisons
  output/               # Parquet files, PNG figures, CSVs
```

The original replication packages are in `{ID}-V1/` directories.

## Translation Patterns

All replications translate original code (Stata/R/MATLAB) to Python using pandas, statsmodels, linearmodels, scipy, and matplotlib. Key mappings:

- **Stata `areg`/`reghdfe`** → `linearmodels.AbsorbingLS` or Frisch-Waugh-Lovell demeaning + OLS
- **Stata clustered SEs** → `cov_type='cluster'` in statsmodels
- **Stata `ivreghdfe`** → Manual 2SLS on FE-demeaned variables
- **R `fixest::feols`** → `linearmodels.AbsorbingLS`
- **R `synthdid`** → TWFE DiD (no robust Python equivalent)
- **Stata `tobit`** → Custom MLE with `scipy.optimize.minimize` (L-BFGS-B)
- **RData files** → `pyreadr.read_r()`

## Requirements

```
numpy pandas statsmodels linearmodels scipy matplotlib
openpyxl pyreadr geopandas
```
