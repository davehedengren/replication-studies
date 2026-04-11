# Replication Studies

Python replications of economics papers from openICPSR replication packages. Each study translates the original code (typically Stata, R, or MATLAB) into Python and verifies published results, audits data quality, and runs robustness checks.

The data-collection layer (the crawler that finds and downloads replication packages from openICPSR, and that fetches the corresponding paper PDFs from open-access sources) lives in [`aer-replication-crawler/`](aer-replication-crawler/). The merged tracker of every project we've ever looked at is in [`project_log.csv`](project_log.csv) (1,081 rows, 261 downloaded packages).

## Summary Statistics

| Metric | Count |
|--------|-------|
| Papers assessed | 52 |
| Completed replications | 34 |
| Skipped (data unavailable) | 18 |
| Bugs found | 5 |
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
| 113192 | Disrupting Education? (Mindspark CAL) | Muralidharan, Singh, Ganimian | AER 2019 | Near-exact (Table 2 ITT + Table 8 IV within 0.02 SD) | None |
| 117443 | Alcohol Tastes and Mortality (Russia) | Kueng & Yakovlev | AEJ: Policy 2020 | Exact (Tables 2 & 3 Panel B to 3dp) | None |
| 119381 | Friend-Based Ranking | Bloch & Olckers | AEJ: Micro 2022 | Near-exact (Table 1 9/10 cells, all Fig 6–9 counts exact) | None |
| 125321 | Can Technology Solve the Principal-Agent Problem? (China pollution monitoring) | Greenstone, He, Jia, Liu | NBER WP / AER: Insights 2022 | Exact (Table 1 Panels A & B all cells to 0.1) | None |
| 125561 | How Effective Are Monetary Incentives to Vote? (Peru) | Gonzales, León-Ciliotta, Martínez | AEJ: Applied 2022 | Exact (Tables 1–5, 42/42 coefficients & SEs to 3dp) | None |
| 131341 | Risk Exposure and Acquisition of Macroeconomic Information | Roth, Settele, Wohlfart | AER: Insights 2022 | Exact (Table 2 Panels A–C and Table 3 Panels A–D match to 3dp) | None |
| 133501 | Teenage Driving, Mortality, and Risky Behaviors | Huh & Reif | AEJ: Applied 2023 | Exact (Table 1 Panels A+B all 48 estimates to 3dp) | None |
| 140161 | Checking and Sharing Alt-Facts | Henry, Zhuravskaya, Guriev | AEJ: Policy 2022 | Exact (Table 2 col 1 & Table 3 col 1 match to 3dp) | None |
| 141541 | Top of the Batch: Interviews and the Match | Echenique, Gonzalez, Wilson, Yariv | AER: Insights | Exact (Table 1 + 11 appendix tables, 936/936 cells to 0.1 pp) | None |
| 145561 | M Equilibrium: Beliefs and Choices in Games | Goeree & Louis | JET/arXiv 2021 | Exact (Tables 2, 4, 5 + OA Table 1, all to 3–4dp) | **Yes** (Table 5 typo) |
| 146041 | Human Capital and Macro-Economic Development | Rossi | JEL 2020 | Exact (Tables 1–4 all cells to 3dp) | None |

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
| 120281 | Knowledge Diffusion, Trade and Innovation | Cai, Li, Santacreu | AEJ: Macro 2022 | Gravity step exact (162/162 cells); MATLAB calibration out of scope | None |
| 127341 | Optimal Lockdown in a Commuting Network | Fajgelbaum, Khandelwal, Kim, Mantovani, Schaal | AER: Insights 2021 | Gravity elasticities exact (κ=1.53, ε=0.45 to 3sf); MATLAB SEIR/Hamiltonian out of scope | None |
| 136342 | Distortions and the Structure of the World Economy | Caliendo, Parro, Tsyvinski | AEJ: Macro 2022 | Stata measurement exact (833k τ, 24k TFP cells match to 1e-6); MATLAB counterfactuals out of scope | None |
| 140141 | Population Aging and Structural Transformation | Cravino, Levchenko, Rojas | AEJ: Macro 2022 | Table 1 all 12 cells match to ≤0.06 pp; US CES micro pipeline (16h+) out of scope | None |
| 145161 | Time Use and Gender in Africa in Times of Structural Transformation | Dinkelman & Ngai | JEP 2022 | Figs 1–5 reproduce; Table 1 USA1920s + Morocco exact; SA 2000/2010 within ±1 hr; Ghana/SL data not shipped | **Yes** (minor) |
| 146381 | Synthetic Difference in Differences | Arkhangelsky, Athey, Hirshberg, Imbens, Wager | AER 2021 | Table 1 SDID/SC/DID/DIFP exact (±0.05); MC + simulation tables out of scope | None |

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

### 113192 — Disrupting Education? (Mindspark CAL)
A 4.5-month after-school computer-assisted-learning program in Delhi raises math scores by 0.37 SD and Hindi by 0.24 SD (paper reports 0.36 and 0.22). Dose-response IV gives 0.0067 SD/day math and 0.0043 SD/day Hindi — essentially identical to the paper's 0.0065 and 0.0040. Main ITT and IV replicate from scratch in Python; the package ships no Stata code (data-only). **Concern**: Treatment attrition is 4.8 pp higher than control (p ≈ 0.07). Lee (2009) bounds of [0.23, 0.44] math and [0.19, 0.40] Hindi keep the effect meaningfully positive even under worst-case trimming.

### 119381 — Friend-Based Ranking
Theoretical paper on ranking individuals from friend-based comparisons; empirical application uses 75 Karnataka village networks (Banerjee et al. 2013) and 633 Indonesian hamlet networks (Alatas et al. 2016). All 13 per-village statistics recompute to within 2×10⁻¹⁵ of the authors' shipped values. Table 1 matches 9/10 cells exactly; the published India "Support" mean (0.85) disagrees with the authors' own shipped data (0.82), which is a paper–package discrepancy, not a code bug. Every headline count in the paper's description of Figures 6–9 — including the Figure 9 "162 of 213 networks" support-beats-partition comparison and the 45 / 127 theorem-1 counts — reproduces exactly. Placebo Erdős–Rényi graphs matched on n and density have much lower support shares (0.37 vs 0.81 India; 0.76 vs 0.96 Indonesia), confirming the real networks are substantially more triangular than random. No coding bugs found.

### 120281 — Knowledge Diffusion, Trade and Innovation across Countries and Sectors
A multi-sector Eaton-Kortum model with endogenous R&D and knowledge spillovers, calibrated to 19-country patent-citation and trade data and used to compute dynamic welfare gains from a 40% trade liberalization (average 44.6%, ranging from 17.5% in the US to 124% in Slovenia). The paper's single empirical step — the sector-level gravity regression that produces the bilateral trade costs feeding the calibration — replicates exactly in Python: all 162 coefficients (9 regressors × 18 tradable sectors) match the shipped Stata output to 7 decimals, and the reconstructed `lD_inj` trade cost matrix matches `lD_inj1919.dta` to 1e-6 across 6,156 off-diagonal pairs. The structural calibration, welfare counterfactuals, knowledge-diffusion estimation, and Section 6 sub-models all live in a ~60-file MATLAB pipeline and are out of scope for a Python port. **Concern**: The shipped gravity panel covers only 19 countries, but the paper's welfare figures plot 29+ economies including SVN, POL, MEX, CHN, IND — none of the raw inputs for those are in the replication package, so the headline welfare numbers are not independently verifiable without first obtaining the extended-country data. No coding bugs found.

### 125321 — Can Technology Solve the Principal-Agent Problem? (China pollution monitoring)
Automating China's air-quality monitoring network caused reported daily PM₁₀ to jump 34.9 μg/m³ (≈35%) at each station's automation date, while satellite-based AOD (an unmanipulable benchmark) shows no discontinuity — clean evidence that pre-automation underreporting was a data-quality artifact, not real pollution variation. Every Table 1 cell replicates to within 0.2 μg/m³ using the Python port of `rdrobust`: Panel A PM₁₀ row (34.69 / 27.58 / 63.26 / 57.16 vs pub 34.7 / 27.5 / 64.7 / 57.1), Panel A AOD row (all within 0.002 and insignificant), and all 35 event-study coefficients in Panel B match to 0.05. 15 robustness checks all survive, including two placebo cutoffs (180 days before and after automation) that are cleanly null. **Concern**: the pooled 34.9 headline is carried by the Wave-1 deadline cohort — dropping it attenuates the effect to 15.8 μg/m³ (−55%). The paper's own Panel B is transparent about the deadline cohorts doing the heavy lifting, but the abstract glosses over this. No coding bugs found.

### 125561 — How Effective Are Monetary Incentives to Vote? (Peru)
Peru's 2006/2010 reform gave 1,755 districts three different abstention-fine values, creating clean within-district variation that the paper exploits with a district × election×province×2006-category TWFE design, weighted by 2001 voters and clustered by province. A 10-sol lower fine (~US$7) cuts turnout by 0.5 pp (elasticity 0.030, SE 0.005), with a registration elasticity of −0.045 concentrated entirely in 18–20-year-olds (−0.276, SE 0.043), consistent with young first-time voters fraudulently registering in low-fine districts. All 42 coefficients and 42 standard errors in Tables 1 (A+B), 2, 3, 4, and 5 replicate to ±0.001 — an exceptionally clean package. I hand-wrote a Gauss-Seidel two-way FE demeaner and a Liang-Zeger cluster SE with nested-FE DOF correction because `linearmodels.AbsorbingLS` does not match reghdfe's iterative singleton drop; subtracting district and province×date×category FEs as "nested in cluster" (K contribution = 0) is what brings the SEs exactly onto the published values. **Concern**: the headline 0.030 elasticity is concentrated in 2016 — dropping that election cycle cuts the baseline from 0.030 to 0.011 (−64%). The paper is transparent about this ("gradual learning / voltage drop" in Section IIIE and Table 2 col 1), but a reader should treat the average elasticity as a time-weighted object that grew from 0.011 in 2011 to 0.049 in 2016. Permutation placebo (shuffle fine within election, 200 reps) gives p < 0.005; leave-one-region-out range is 0.026–0.032; the top-2 vote share is a clean placebo null (β = 0.005, SE 0.011). No coding bugs found.

### 127341 — Optimal Lockdown in a Commuting Network
A SEIR + spatial trade model applied to Seoul, Daegu, and NYC Metro finds that spatially-targeted Covid lockdowns achieve 20–58% lower economic losses than uniform lockdowns, with the gap growing in viral severity; optimal policy restricts inflows to central districts early in NYM and Daegu but relaxes gradually in Seoul. The paper's one reduced-form step — the equation (20) gravity regression on Seoul district-to-district credit-card data — replicates to 3 significant figures: (σ−1)κ₁ = 1.529 vs 1.53 published, (σ−1)ε = 0.447 vs 0.45 published, SEs 0.064/0.065 vs 0.066/0.067, sample N = 75,625 matches exactly. Table A.1 summary statistics (districts, populations, first-case and lockdown dates for all three cities) match the paper verbatim. The Seoul panel is perfectly balanced (625 OD pairs × 851 days, zero missing cells). Twelve robustness checks all survive: κ is stable at 1.50–1.62 across LOO, winsorization, and subsample splits; the permutation placebo collapses ε to 0.009. **Concern**: Equation (20) is written with ln χ but the shipped Stata code uses χ in levels — the level specification gives the 0.45 the paper quotes, while the log specification gives 0.37. Notation/spec choice, not a bug, but a literal reader of equation (20) would get a different number. The ~30-script MATLAB structural pipeline (SEIR calibration, Hamiltonian backward-resolution, Pareto frontier, optimal lockdown maps) is not replicated — ~9.9 GB of .mat state files, multi-day compute, and the key outputs are reported only at one-decimal precision. No coding bugs found.

### 130626 — Optimal Targeted Lockdowns in a Multi-Group SIR Model
Acemoglu, Chernozhukov, Werning, Whinston (AER: Insights 2021) study differentiated lockdown policies in a multi-group SIR model calibrated to COVID-19 age-specific fatality rates. Their headline result is that targeting the oldest age group with a strict, long lockdown while easing restrictions on the young can cut mortality from ~1% to ~0.5% at the same economic cost (or cut economic losses from ~37% to ~25% of a year's GDP at the same ~0.02% mortality target). **Skipped** — the paper is a pure structural optimal-control exercise with no empirical component: four Jupyter notebooks that call the `gekko` NLP solver to compute Pareto frontiers from hand-coded epidemiological parameters, no dataset, no regression tables, no standard errors. This does not fit the replicate-an-empirical-result workflow this project uses; running the notebooks would just reproduce the authors' own figures from their own code. Analogous to the skipped structural portion of 127341 (Fajgelbaum et al. commuting-network lockdown paper), but without that paper's reduced-form gravity step to fall back on.

### 147524 — The Government Spending Multiplier in a Multi-Sector Economy
Bouakez, Rachedi, and Santoro (AEJ, forthcoming) build a calibrated multi-sector New Keynesian DSGE model with 57 industries, BEA input-output linkages, sector-specific Calvo price rigidity, and capital adjustment costs, and find that the aggregate value-added government spending multiplier is 0.74 in the multi-sector baseline versus 0.42 in the one-sector limit — a 75 percent amplification that they attribute primarily to I-O linkages rather than heterogeneity in price stickiness. At the ZLB the multi-sector multiplier rises to 1.98 (vs 1.07), and a short empirical section (Table 4) validates the model's prediction that upstream sectors respond more strongly to an aggregate spending shock using Ramey-Zubairy fiscal news as an IV. **Skipped** — the replication package is a Dynare/MATLAB pipeline with ~90 model variant subdirectories under `Models/` (baseline `Het`, `Het_ZLB`, `Het_CRRA`, `Het_StickyWage`, `Het_InflTarg_*`, plus matching `1Sect_*` comparators), all driven by `RunMainScript_Model.m`, and Dynare has no Python equivalent that would let us reproduce the multipliers without hand-reimplementing every `.mod` file. The one empirical table that could be run in Python is missing two of its four input files: `Data/RZDAT.xlsx` (Ramey-Zubairy quarterly fiscal series) and `Data/CentralityTable.xlsx` (the authors' paper-specific sectoral upstreamness measure computed from the BEA I-O matrix) are not in the shipped package, and `Tables/`, `Figures/`, and `InTextNumbers/` are all empty. Even building the Table 4 panel from scratch would cover at most ~3 percent of the paper's empirical content while leaving all of Tables 1-3, 5-6, every figure, and the entire ZLB / sensitivity sweep unverified. Fits the "MATLAB-only structural model" infeasibility criterion; closest repo analogue is 225841 (also DSGE + missing empirical data).

### 141001 — Fiscal Rules and the Sovereign Default Premium
Hatchondo, Martinez, and Roch (AEJ: Macro, forthcoming) build an Eaton-Gersovitz sovereign-default model with long-term bonds and evaluate two classes of fiscal rules — a debt-brake and a spread-brake — in a monetary union calibrated to Spain. Their headline claim is that a spread-brake strictly dominates a debt-brake at the same average debt level: it delivers lower spread volatility, higher welfare (0.6% of consumption in the benchmark), and is the constrained-efficient rule across 13 economies that vary default costs, exclusion probabilities, and recovery parameters. **Skipped** — the replication package is Fortran plus MATLAB with no empirical component. Thirteen separate `aej_code_*.f90` solvers (~13 hours runtime each on the authors' hardware, so ~150–200 CPU-hours for a full recomputation) produce value, policy and bond-price functions on discrete grids, and `figures_tables_aej.m` reads a pre-built `workspace_replication_aej.mat` to generate every table and figure in the paper. The only files in `Data/` are a small IMF fiscal-rules count series for the descriptive Figure 1 and a six-row Spain calibration target sheet — no regressions, no panel, nothing to estimate in Python. Fits the "MATLAB-only structural model with no empirical component" infeasibility criterion explicitly called out in the driver instructions; in practice worse, since the core solver is Fortran and MATLAB is just the reporting layer. Closest analogue in the repo is 130626 (also structural, no empirical step).

### 131341 — Risk Exposure and Acquisition of Macroeconomic Information
Roth, Settele, and Wohlfart (AER: Insights 2022) run an online experiment with 1,008 US full-time employees: respondents are shown either ACS- or CPS-derived data on how the unemployment rate changed for people "like them" over the Great Recession, and the noise-difference between the two signals identifies an exogenous shift in perceived personal recession exposure. A 1-pp larger shown-vs-alternative signal raises perceived next-recession unemployment risk by 0.49 pp and the probability of choosing the SPF recession forecast by 0.6 pp (2SLS: 1.2 pp per 1 pp of perceived risk, about a 5% increase on the 25% baseline). Every coefficient, SE, N, R², and first-stage F in Table 2 Panels A–C and Table 3 Panels A–D replicates verbatim to 3 decimal places; only two Panel C IV standard errors in insignificant columns drift by 0.001 due to a small-sample correction difference between `ivreg2` and `linearmodels.IV2SLS`. The first stage is extremely robust: it survives HC3 SEs, occupation clustering, dropping all controls, winsorizing the outcome, trimming signal outliers, and a 500-draw permutation placebo. **Concern**: the reduced-form effect on recession-forecast demand is driven entirely by the CPS arm (β = 0.008, p < 0.01); in the ACS arm the point estimate flips to −0.003 (p ≈ 0.5). The pooled estimate is valid under the paper's identification logic, but the arm-specific asymmetry is worth reporting. No coding bugs found.

### 133501 — Teenage Driving, Mortality, and Risky Behaviors
Huh and Reif (AEJ: Applied 2023) use a regression discontinuity at the minimum legal driving age (MDA) to estimate that driving eligibility raises teenage mortality by 5.84 deaths per 100,000 (15%), motor vehicle fatalities by 4.92 per 100,000 (44%), and — surprisingly — female drug-poisoning deaths by 0.747 per 100,000 (76%). The LATE implies new teen drivers face 6–9× the per-mile crash risk of a typical adult. Every single estimate in Table 1 — all 9 Add Health first-stage coefficients and all 39 mortality coefficients across 3 scenarios — replicates to three decimals using the Python port of `rdrobust`, after a two-line monkey-patch to fix a NumPy shape bug in `rdrobust-py 1.4` that crashed `float()` on the `covs=firstmonth` path. **Concern**: the female poisoning result's magnitude is fragile — at a 24-month bandwidth it shrinks 72% (0.75 → 0.21), and under a uniform (non-triangular) kernel at the baseline bandwidth it shrinks 55% (→ 0.34). The CCT MSE-optimal bandwidth is the most favorable specification. Sign and significance survive every alternative and all 50 placebo cutoffs, so the effect is real, but readers should treat 0.747 as "the point estimate at the narrowest data-driven window" rather than a stable magnitude. The MVA and all-cause results, by contrast, are extraordinarily robust across 12 specifications and placebo exercises. Dropping the Dong (2015) `firstmonth` dummy is the most consequential analyst choice in the paper — it cuts the MVA estimate 29% — but dropping the month-0 cell entirely recovers 4.77, consistent with the dummy doing exactly what it claims. No bugs in the authors' Stata code.

### 136342 — Distortions and the Structure of the World Economy
Caliendo, Parro, and Tsyvinski propose closed-form sufficient statistics that separate sectoral TFPs from input-output distortions in a CES production + CES consumption model of the world economy, then apply them to the World Input-Output Database (40 countries × 35 sectors × 17 years, 1995–2011). The paper's headline claim is that the elasticity of world real GDP to *internal* (within-country) distortions is an order of magnitude larger than to *external* (cross-border) distortions — 0.41 for China, 0.33 for the US, 0.15 for Japan. The Stata measurement step (`genFrictions.do`) replicates to machine precision against every packaged intermediate: α, β, γ, 833,000 normalized internal distortions τ, and 24,395 normalized TFPs — max |Python − Stata| is 2.6e-6, correlation 1.000000 everywhere. Figure 11 (histograms of annual distortion growth) and Figure 12 (manufacturing vs services TFP index + sector ranking) replicate qualitatively: services TFP hits 1.127 in 2011 vs manufacturing 1.092 (both 1996=1), the top manufacturing sectors are Petroleum / Plastics / Basic Metals / Food as in the paper, and the top services sectors are Post-Telecom / Education / Health as in the paper. **Concern**: all of the paper's *policy* conclusions — the world-GDP elasticities that make the abstract — come out of a MATLAB counterfactual solver (`CPT_application.m`, `Dinprime.m`, `GMC.m`, `Phat.m`) that requires MATLAB and a Python rewrite of a nested fixed-point trade equilibrium. I verified the *inputs* to that solver but not its outputs, so readers should treat "measurement verified" as necessary but not sufficient. One measurement-level oddity: Water Transport (sector 24) is a clear outlier in gross-output-weighted TFP growth (−2.45%/yr vs paper ~+1%), driven by a handful of small-economy cells. No coding bugs found in the Stata source.

### 140141 — Population Aging and Structural Transformation
Cravino, Levchenko, and Rojas (AEJ: Macro 2022) document that population aging is tightly linked to the rise of the service sector. The paper's Table 1 cross-country panel regression shows a 1 pp rise in the 65+ population share is associated with a 1.3 pp rise in the value-added share of services and a 1.5 pp rise in the employment share, conditional on log GDP per capita and its square. Using a 20-country OECD panel (1970–2007, EUKLEMS + WDI + Maddison), the Python replication reproduces all 12 Table 1 slopes to within 0.06 pp and all SEs to within 0.02; R² match to three decimals. Sample is N = 745 vs published N = 707 — the 38-row delta is exactly Luxembourg, and dropping LUX recovers N = 707 and β = +1.325 (pub +1.309). **Concern**: the headline cross-country result is fragile to adding year fixed effects (β collapses from +1.28 to +0.39 and loses significance) because both aging and services-share are shared upward trends over the 1970–2007 OECD panel. The paper's preferred specification correctly omits year FEs (the theoretical channel would be absorbed), but readers should treat 1.3–1.5 pp as an upper bound on the causal cross-country effect. A 500-draw within-country permutation test rejects the null at p < 0.002, and results are robust to dropping USA, LUX, and JPN/KOR. **Out of scope**: the 17-hour Stata pipeline that rebuilds the BLS Consumer Expenditure Survey microdata (1982–2016, 7.5 GB raw) and the two-sector PIGL structural model that decomposes the US service-share rise into aging / real income / relative prices / residual taste. Only the cross-country empirical block is replicated. No coding bugs found.

### 140161 — Checking and Sharing Alt-Facts
Henry, Zhuravskaya, and Guriev (AEJ: Policy 2022) run an online randomized experiment with ~2,537 French Facebook users during the May 2019 European elections. Participants are shown two false statements about the EU attributed to Rassemblement National leaders and then assigned to one of three arms: Alt-Facts (no correction), Imposed Fact-Check (forced to see a fact-check), or Voluntary Fact-Check (offered the option to view one). The headline claim is that both imposed and voluntary fact-checking reduce the rate at which participants agree to share the false statements on their real Facebook accounts by ~27–30% (14.7% → 10.2% → 10.8%), and that the two treatment regimes have statistically indistinguishable average effects — i.e. *offering* the choice is as effective as *forcing* the exposure. Every Wave-1 number I attempted matches to the 3rd or 4th decimal: Table 2 col 1 replicates as Imposed = −0.0451 (SE 0.0160) vs pub −0.045 (0.016) and Voluntary = −0.0380 (0.0162) vs pub −0.038 (0.016); Table 3 col 1 gives Voluntary = −0.0284 (0.0163) vs pub −0.028 (0.016); the three treatment means come out 14.69% / 10.18% / 10.89%; the 39% voluntary-viewing rate replicates as 39.1%. Sample size is N = 2,534 vs published 2,537 (a 0.12% difference from a `keep if n > 4` idiom in the Stata pipeline that drops the first four data rows — harmless here). Twelve robustness checks all leave the baseline intact (probit ≈ LPM, permutation p < 0.013 for both arms, HC3 ≡ HC1, MLP-voter placebo null, pre-treatment-variable placebo null, 30-bin time jackknife range −0.053 to −0.036). **Concern**: an undiscussed gender heterogeneity — *imposed* fact-checking reduces men's sharing (−0.062, p = 0.008) but not women's (−0.028, p = 0.19), while *voluntary* fact-checking reduces women's sharing (−0.052, p = 0.01) but not men's (−0.024, p = 0.33). The paper's "Imposed ≈ Voluntary" framing is correct only in the population average; the policy lever that works differs by gender. One data-quality note: one respondent reports 10¹⁶ Facebook friends, benign under the paper's log-friends transformation. No coding bugs found.

### 117443 — Alcohol Tastes and Mortality (Russia)
Russia's 1986–90 anti-alcohol campaign permanently shifted the drinking preferences of rural men who turned 17 during the campaign: they drink 5.2 pp more vodka as adults two decades later (paper: 5.232, SE 1.986; replication: 5.232, SE 1.982). The paper's IV estimate — a 1-pp increase in share of vodka raises log regional male mortality by 1.25 — also reproduces exactly (paper and replication: 1.253, SE 0.455, N = 1,343), as does the cancer placebo null and the alcohol-poisoning, external-cause, and "+log(alcohol intake)" columns. Every coefficient, SE, and N in Table 2 (cols 1–7) and Table 3 Panel B (cols 3–7) matches to 3 decimal places. **Concern**: The mortality IV is fragile to restricting the panel to years ≥ 2000 — the point estimate drops from 1.25 to 0.58 and loses significance, consistent with instrument strength coming from the first decade of the panel. No coding bugs found; the Stata pipeline is exceptionally clean.

### 145161 — Time Use and Gender in Africa in Times of Structural Transformation
Dinkelman and Ngai (*JEP* 2022) stitch together macro and micro evidence to argue that African women's structural transformation runs from agriculture directly into services (largely skipping manufacturing), that FLFP is high in most of SSA but below 30% in North Africa and does not rise with GDP, and that African housewives spend 32–47 hours a week in home production — comparable to US housewives in the 1920s–1960s — with cooking alone accounting for 20–52% of home-production time. The replication rebuilds all five figures and every replicable column of Table 1 in Python: Figure 1 (11-country sectoral employment scatter from GDCC+PWT) reproduces; Figure 2b weekly market/home-hours cells match the paper's bubble positions for Uganda 2005 (47.4/22.1), Tanzania 2014 (41.5/17.1), Ghana 2009 (40.9/16.7), SA 2000 (37.2/11.4), SA 2010 (31.9/12.9), and Algeria 2012 (42.5/7.1); Figure 3's SSA-2017 segments match within 0.7 pp. Table 1 USA 1920s and Morocco 2011 columns replicate exactly to the last printed digit; South Africa 2000/2010 match within ±1 hr on every activity and ±2 hr on the total; USA 1965/2010 (MTUS) activity cells match within ±1 hr. **Concern**: Ghana 2009 and Sierra Leone 2003 time-use microdata are registered-access and not shipped, so nearly half of Table 1 is not independently verifiable (the authors disclose this in `master.do` but not in the published paper). **Bug found** (moderate): `timeuse_US_UK.do` contains `gen married=(cohab==0|cohab==1) \ replace married=1 if cohab==.`, and because `cohab` is 100% missing for the US1965 MTUS extract, every US1965 observation is forced to married=1 — the paper's reported "79% married" US1965 housewife share is not reproducible from the shipped code. Two additional cosmetic Stata bugs exist: a dead `childinhouse` branch in the SA 2000 cleanup (both `==1` and `==0` routes assign to `care`) and a `code==250` overwrite in SA 2010 (code 250 is assigned to cooking then immediately overwritten to firewater). None of the three bugs change the paper's substantive conclusions; the US1965 married-share is a sample-features cell, not a substantive result. Leave-one-country-out on Figure 1 shows Mauritius and South Africa are the biggest leverage points but the agric↓ / services↑ / manuf-flat pattern holds under every drop. Morocco housewife robustness: total weekly hours are stable at 45.7 across every age band, education filter, weighting scheme, and winsorization; relaxing the housewife definition to unmarried women or women with some paid market work only lowers the total by 2–3 hours. Out of scope: any structural estimation — this is a descriptive JEP piece with no regressions or standard errors.

### 145561 — M Equilibrium: A Theory of Beliefs and Choices in Games
Goeree and Louis (arXiv 1811.05138v2, April 2021) propose M-equilibrium, a set-valued solution concept for normal-form games that only requires monotonicity (better options are chosen more often) and consequential unbiasedness (expected-payoff rankings of beliefs match those of choices), and show it dominates Nash, logit-QRE, HQRE/SQRE, level-k and Cognitive Hierarchy at organizing experimental choice and belief data from 2×2 Asymmetric Matching Pennies games and four symmetric 3×3 games (DS1, DS2, "No-Logit", Kohlberg–Mertens). The replication reads the authors' MATLAB `.mat` data files with `scipy.io.loadmat` and re-implements everything in Python: Table 2's best-response fractions match all 18 cells to the 3rd decimal (Row avg .582 vs .58, Col avg .723 vs .72), Online Appendix Table 1's diagonal-covariance Hotelling p-values match every pairwise cell, Table 4's logit-QRE matches all five λ̂ (.0375/.0000/.0078/.0696/.0124 vs pub .0376/.0000/.0078/.0696/.0125), all five fit log-likelihoods, and all 16 out-of-sample cells verbatim, and Table 5's μ-equilibrium matches all five ε̂ (.382/.378/.522/.290/.384) and all four per-game fit log-likelihoods plus all 16 out-of-sample cells to ±0.1 log-points. **Bug found** (cosmetic): Table 5's pooled-μ fit log-likelihood is published as −1529.9 but is actually −1538.2 — the four out-of-sample cells in the pooled row (−429.8, −450.1, −479.5, −178.8) sum to exactly −1538.2, and by construction the pooled fit at ε̂ must equal this sum, so the published number is a typo. A grid search over ε ∈ [0.05, 0.99] confirms no ε produces anything near −1529.9. The replication MATLAB package would output −1538.2; the typo entered somewhere between MATLAB and the typeset table. **Concern**: the μ-equilibrium log-likelihood surface has multiple local optima for DS1 (ε=.382 at x₀=.3, ε=.529 at x₀∈{.1,.5,.7} with logL 38.7 points worse) and DS2 (ε=.378 at x₀=.3, ε=.990 at x₀∈{.1,.5,.7} with logL 78.3 points worse). The published estimates are the correct global optima, but the authors' MATLAB code hard-codes x₀=.3 with only a one-line code comment — not discussed in the paper — so a naive re-implementation with a different seed would miss the global peak for half the games. All ten robustness checks (belief-shuffle placebo drops Row BR from .582 to .504, leave-one-session-out grand avg in [.642,.662], subject-cluster bootstrap CIs cover every published estimate, μ beats QRE by +40/+78/+28/+8 log-points per game) confirm the paper's substantive claims without qualification. Typo doesn't change any conclusion — μ still beats QRE by 246 log-points in the pooled fit with the corrected value.

### 146041 — Human Capital and Macro-Economic Development: A Review of the Evidence
Federico Rossi (*Journal of Economic Literature* 2020) reviews three lines of evidence on whether human-capital differences can explain the 30–40× cross-country GDP-p.w. gap. His central empirical object is a CES-aggregate relative skill efficiency, irAQ₅₃ = (w₅/w₃)^(σ/(σ-1)) × (H₅/L₃)^(1/(σ-1)), normalized to US 2000 = 1 and built from 12-country IPUMS microdata. The baseline elasticity of log irAQ₅₃ wrt log GDP p.w. is 1.408 (SE 0.394, σ=1.5) — an order of magnitude larger than the migrant-wage-based Q elasticity of 0.105 from Hendricks-style comparisons, so once you take the CES calibration seriously, relative-skill-efficiency *does* rise steeply with development and can account for a meaningful share of the GDP gap. Every cell in Tables 1 (6 columns × 12 countries + elasticity row), 2 (8 specs × 5 columns of coefs+SEs), 3 (8 Q/AQ elasticity columns + all 6 θ_Q/θ_AQ ratios), and 4 (India development accounting, 4 methods × 4 σ values) replicates to 3 decimal places against the published numbers; the only non-obvious step is the sector-level σ transformation σ_sec = (σ − χ/3)/(1 − χ) where χ is a US-2000 payment-share heterogeneity index (computed here as 0.0751, giving σ_sec = 1.595). **Concern**: the 1.408 headline rides on a 12-country cross-section. Leave-one-country-out moves it to [1.14, 1.97]; restricting to the 4 highest-dispersion countries (India, Indonesia, US, Canada) gives 1.55; the σ=1.3 variant blows up to 2.44 while σ=2.0 collapses to 0.63 (σ enters as 1/(σ-1), so low σ levers everything). A 5000-draw permutation test gives p = 0.004 and the w₅/w₃ placebo outcome matches the published −0.138, so the effect is real — but readers should treat 1.4 as "the point estimate under σ=1.5 and the hours-weighted variant" rather than a tight number. Table 4's India development-accounting column shows the cross-method spread that the paper itself emphasizes: Jensen σ=1.5 gives y_India / y_US = 0.698, Caselli-Ciccone σ=1.5 gives 0.104, and the Mincerian + CC variants all converge around 0.11–0.16 — i.e. the accounting conclusion depends heavily on which human-capital aggregator you adopt. No coding bugs found; the paper's 31-do-file Stata pipeline (~3,300 lines) is well-organized, and all pre-built intermediate `.dta` files in `temp/` are internally consistent.

### 146381 — Synthetic Difference in Differences
Arkhangelsky, Athey, Hirshberg, Imbens, and Wager (*AER* 2021) propose SDID, a panel-data ATT estimator that combines SC-style unit reweighting with DID-style time reweighting, solved as a weighted two-way fixed-effects regression on Frank-Wolfe-fitted ω and λ. The replication ports the whole `R/solver.R` + `R/synthdid.R` + `R/vcov.R` stack to Python and re-runs the canonical California Proposition 99 application (39 states, 1970–2000, California treated 1989). Every point estimate in the paper's Table 1 replicates to ±0.05 packs/capita: SDID −15.60 (pub −15.6), SC −19.62 (−19.6), DID −27.35 (−27.3), DIFP −11.10 (−11.1); the placebo SE (Algorithm 4, 200 reps) matches to ±0.15 for SDID (8.42 vs 8.4) and DIFP (9.62 vs 9.5), and drifts ~1–2 for SC and DID within expected 200-rep Monte Carlo noise. Figure 1's SDID time weights concentrate on 1986/87/88 exactly as shown in the paper, and Nevada is the top donor (ω = 0.124). **Out of scope**: the MC (matrix completion) estimator requires the `MCPanel` R package's platform-dependent CV'd nuclear-norm solver (the authors themselves flag this in REPLICATION.md), and simulation Tables 2–4 / Figure 2 take "a few days on an 8-core machine" per the authors' own documentation — neither fits an automated single-day replication, and both are flagged explicitly. Eleven robustness checks all survive: leave-one-top-donor-out stays in [−17.1, −15.2], in-space placebo p ≈ 0.026, regularization sensitivity η_ω ∈ [0.25×, 4×] keeps τ ∈ [−17.8, −12.4], placebo-in-time checks produce small nulls (|τ| ≤ 3) at fake treatment years 1980/82/84/86, and disabling the ω intercept pulls τ toward the pure-SC value (−18.75). The jackknife SE (Algorithm 3) is not applicable because the paper's canonical application has only a single treated unit. No coding bugs found; the `synthdid` R implementation is clean and every translated Python value matches.

### 141541 — Top of the Batch: Interviews and the Match
Echenique, Gonzalez, Wilson, and Yariv (AER: Insights) ask whether building the interview stage into deferred acceptance (Int-DA) can meaningfully beat running standard DA on artificially truncated rank lists (Tr-DA), the de facto NRMP setup. Using a 5×5 common-value grid (λ_D, λ_H) × six balanced N values up to 1700 × four interview-capacity k values × one unbalanced (500, 600) design, they show Int-DA Pareto-dominates Tr-DA on every matching-outcome cell (top-1 share 33–49% vs 0.2–31%; unmatched 5.6–8.2% vs 26–96% at NRMP scale) and essentially matches full-DA on stability (same-partner-under-proposer-change ≥ 99.8% in every NRMP-rescaled cell vs 42–99.5% for full DA). The replication consumes the shipped Python→R-produced summary CSVs and re-implements the Mathematica table/figure pipeline in Python: all 858 appendix-table cells (6 balanced N + 4 k + unbalanced) and all 78 cells of NRMP-rescaled Table 1 replicate to ≤ 0.1 pp — a 936 / 936 exact match after correctly transcribing the conditional-on-matched rescalings for Panel B (`samepartner`, `identical`, and the blocking-pair counts are all normalized by the matched / unmatched share, as the Mathematica `MakeTableN` function does explicitly). Twelve robustness checks all confirm the paper: 0/450 Pareto violations on matching outcomes, 0/25 violations at k=2 and in the unbalanced design, Int-DA stability ≥ 96.6% everywhere on the grid (≥ 99.6% at every N ≥ 500), matched blocking-pair share ≤ 2.2%, and the aggregate rank-gap statistic shrinks monotonically from +119.1 at N=50 to +27.5 at N=1700 at (λ_D, λ_H) = (0.5, 0.5). **Concern**: Table 1's NRMP-scale numbers rely on a log-linear logit extrapolation from at most N=1700 to sub-markets as large as n=9127; the log-N slopes for matching outcomes are small (~0.02) and OLS-in-percent vs logit rescaling agree to < 0.01 pp, so the functional form is empirically innocuous — but nothing in the shipped data directly validates the model above N ≈ 1700. No coding bugs found; the Python simulator, R assembly notebook, and Mathematica table/figure code all transcribe cleanly.

---

## Bugs Found

| Paper | Bug | Severity | Impact on Conclusions |
|-------|-----|----------|----------------------|
| 192297 | Double-counts primary business profits in `rowtotal` | Moderate | ATE inflated ~40%; qualitative results unchanged |
| 239791 | Figure 4 uses level GDP differences instead of log | Minor | R² changes 0.571 → 0.519; conclusion unchanged |
| 228101 | Code references undefined variable `profit` | Packaging | Code won't run as-is; analytical results unaffected |
| 221423 | Uses 2021 public employment instead of stated 2019 | Minor | ~1pp difference; rankings unchanged |
| 145161 | `married=1` fallback in MTUS US1965 arm forces everyone to married | Minor | US1965 sample-features cell unreproducible; activity hours unaffected |
| 145561 | Table 5 pooled-μ fit logL published as −1529.9; correct value is −1538.2 | Cosmetic | 0.5% of log-likelihood; doesn't change sign, rank, or any prose claim |

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
| 130626 | Structural SIR optimal-control model; no empirical component |
| 141001 | Structural Fortran+MATLAB sovereign-default model; no empirical component |
| 147524 | Multi-sector DSGE in Dynare/MATLAB (~90 model variants); Table 4 missing RZDAT.xlsx + CentralityTable.xlsx |

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
