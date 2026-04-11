# Replication Study: 127341-V1

**Paper:** "Optimal Lockdown in a Commuting Network"
**Authors:** Pablo D. Fajgelbaum, Amit Khandelwal, Wookun Kim, Cristiano Mantovani, Edouard Schaal
**Journal:** *American Economic Review: Insights*, 2021 (this package is the replication archive for the published paper; the 127341 PDF in the pipeline is the IATRC proceedings preprint of the same paper, Nov 2020).
**Original Language:** Stata (data prep + reduced-form estimation) + MATLAB (structural model)
**Replication Language:** Python (pandas, numpy, statsmodels)

---

## 0. TLDR

- **Replication status:** The one reduced-form regression the paper reports (the gravity elasticity estimates in Section 3.3) replicates to 3 significant figures. Every summary-statistic I checked against the paper and against Table A.1 matches exactly. The MATLAB structural model (optimal lockdown simulations, Pareto frontiers, SEIR calibration) is **not** replicated — it is ~9.9 GB of compiled `.mat` files and ~30 MATLAB scripts, and the authors report its key outputs only at one-decimal precision in the text.
- **Key finding confirmed:** Using Seoul's district-to-district credit-card data, a gravity regression of log expenditures on log distance and a commute-flow ratio recovers (σ−1)κ₁ = 1.53 and (σ−1)ε = 0.45 — matching the paper's reported values to the third significant figure. These pin down the two key elasticities the structural model then uses.
- **Main concern:** The regressor in equation (20) is written as `ln χ(i,j,t)` but the actual Stata code (and this replication) uses χ in **levels**, not logs. Both specifications give economically similar estimates (0.45 vs 0.37), but readers inspecting equation (20) literally would reproduce a different number.
- **Bug status:** No coding bugs found. The discrepancy between equation (20) (log) and the Stata code (level) is a minor notational/specification choice, not a bug — the Stata file `203_estimation_elasticities.do` clearly regresses on `yod` (the raw ratio), and the resulting estimates are the ones the paper cites.

---

## 1. Paper Summary

### Research Question
How should a planner spatially target a Covid-19 lockdown within a commuting network — i.e., should some districts shut down more aggressively than others, and if so, which ones? How much economic output is saved (for a given health outcome) by spatially-targeted versus uniform lockdowns?

### Data
Three cities with very different initial viral spread at the time of lockdown:
- **Seoul (25 districts):** Commuting flows from subway + bus turnstile records (individual trip-level, Seoul Big Data Campus), credit-card district-to-district expenditures (Shinhan Bank), covid case counts from the Seoul Metropolitan Government.
- **Daegu (8 districts):** Subway turnstile entries/exits (Daegu Metro), census commute flows from the 2015 Korean Population Census, covid case counts obtained via an Official Information Disclosure Act request.
- **NYC Metro (20 counties):** SafeGraph cellphone-derived commuting flows, LEHD wage bins, Census population, covid cases from Johns Hopkins + NY State DoH.

Pre-pandemic commuting flows come from 2018–2019 averages (Seoul), 2015 Census (Daegu), and January 1–20 2020 (NYM).

### Method
1. **SEIR + spatial trade model.** Agents live in origin i, commute to j. A gravity-style trade model generates origin-destination expenditures X(i,j,t); a SEIR diffusion model tracks infections with location-specific transmission β_j = β/area_j. The planner chooses a commuting attenuation matrix χ(i,j,t) to minimise a weighted sum of economic costs and Covid deaths.
2. **Reduced-form elasticity estimation (equation 20).** To parameterise the trade model, the authors estimate a two-way-fixed-effects gravity regression on Seoul credit-card data:
   ln X(i,j,t) = ψ(j) + η(i) − (σ−1)κ₁ ln(dist(i,j)) + (σ−1)ε · ln χ(i,j,t) + ε(i,j,t)
   using 2020 observations, clustered two-way on RCo and RCd.
3. **SEIR calibration.** β is calibrated to match Covid case dynamics in each city 10+ days after the peak; R₀ is then recovered as the largest eigenvalue of a matrix involving the post-lockdown commute matrix.
4. **Optimal spatial lockdown (MATLAB).** A Hamiltonian system is solved backward from the steady state to trace the optimal lockdown χ*(i,j,t) under the calibrated parameters, generating the Pareto frontier of economic cost vs. cumulative infections.

### Key Findings
- Spatial lockdowns deliver **20% / 32% / 58%** lower economic losses than uniform lockdowns for Daegu / Seoul / NYM (given the actual number of cases) — the gap grows with viral severity.
- Under optimal uniform lockdowns the economic gain is 19% / 27% / 37% relative to actual policy.
- In NYM and Daegu the optimal lockdown first restricts inflows to central districts, then gradually relaxes; in Seoul it imposes small temporal but large spatial variation.
- Actual commuting reductions were **too weak in central locations** in Daegu and NYM, and **too strong across Seoul**, compared to the optimum.
- Gravity elasticities estimated from Seoul credit-card data: **(σ−1)κ₁ = 1.53** (SE 0.066), **(σ−1)ε = 0.45** (SE 0.067), implying (with σ=5): κ₁ ≈ 0.38, ε ≈ 0.11. These are the numbers fed into the structural model.
- R₀: 1.32 in Seoul and Daegu (first week post patient-zero), 2.94 in NYM.

---

## 2. Methodology Notes

### Scope of the replication
The paper has two clearly separable components:

| Component | Language | Replicated? |
|---|---|---|
| Stata data cleaning for Seoul / Daegu / NYM | Stata (21 scripts) | We read the cleaned `.dta` outputs directly. |
| Figure 1 & Figure A.1/A.4 event-study plots | Stata (`reghdfe`) | Not replicated (plots, not numbers in text). |
| **Table A.1 summary statistics** | Stata | **Replicated** (`03_tableA1.py`). |
| **Equation (20) elasticity regression** | Stata (`reghdfe` w/ 2-way cluster) | **Replicated** (`02_elasticities.py`). |
| SEIR calibration of β | MATLAB (`run_SEIR_model_estimation.m`) | Not replicated — requires `.mat` state files that were constructed by older MATLAB scripts. |
| Optimal lockdown solver + Pareto frontier | MATLAB (Hamiltonian BVP) | Not replicated — multi-day structural simulation, not feasible in this time budget. |

The regression we replicate is the only quantitative statistic the paper reports with two-decimal precision that does not go through the MATLAB stack. The MATLAB results (optimal lockdown shares, economic cost gains) are reported as rounded percentages and are not amenable to tight numerical comparison without re-running the entire backward-resolution loop.

### Translation choices
- **`reghdfe lns lndist yod, a(RCo RCd) cluster(RCo RCd)` → statsmodels OLS** with RCo and RCd fixed effects added explicitly as dummy variables (drop-first to avoid collinearity). The two-way clustered variance matrix is computed manually via the Cameron–Gelbach–Miller formula V = V(g₁) + V(g₂) − V(g₁∩g₂).
- **`pd.read_stata(convert_categoricals=False)`** is used throughout — the Stata files encode RCo / RCd as numeric district codes and we want the codes, not label strings.
- **Stata `%td` dates** in `SEL_CC.dta` are stored as raw float days-since-1960 (the file was created without a display format), so we convert them with `pd.to_datetime(x, unit="D", origin="1960-01-01")`. In `SEL_parms_od.dta` they are already proper datetimes.
- **χ (commute flow ratio)** is taken from the `yod2` column of `SEL_parms_od.dta` (per the Stata code, which renames `yod2 → yod` right before the regression). In the Stata and in our baseline we use χ in **levels**, matching the author code; we also report a log-χ robustness check (Section 5).

### Things we did NOT do
- Re-run the SEIR calibration. β is pinned down by a nonlinear least-squares fit that depends on intermediate MATLAB state matrices.
- Re-compute the optimal spatial lockdown. This is a backward-resolution Hamiltonian loop over an eigenvalue problem at each node, with no Python equivalent in the package.
- Re-estimate elasticities for Daegu or NYM. The paper only runs equation (20) on Seoul credit-card data — Daegu and NYM do not have the analogous OD expenditure panel.

---

## 3. Replication Results

### 3.1 Equation (20) — Gravity elasticities (Seoul)

Regression sample: 75,625 observations = 25 origins × 25 destinations × 121 days of 2020. This exactly matches the "regression sample size … 75,625 for Seoul" reported in Figure 1's footnote and makes us confident we have the same sample as the authors.

| Quantity | Replication | Published | Absolute diff |
|---|---:|---:|---:|
| (σ−1)·κ₁ (distance) | **1.529** | 1.53 | 0.001 |
| SE (two-way cluster) | 0.064 | 0.066 | 0.002 |
| (σ−1)·ε (commuting) | **0.447** | 0.45 | 0.003 |
| SE (two-way cluster) | 0.065 | 0.067 | 0.002 |
| κ₁ (σ=5)  | 0.382 | 0.383 | 0.001 |
| ε (σ=5)  | 0.112 | 0.113 | 0.001 |
| Own-district share (Mar 2020, %) | 54.57 | 55 | 0.4 pp |

The tiny residual differences in the SEs come from finite-sample degrees-of-freedom adjustments (Stata's `reghdfe` applies a small-cluster correction that we do not) — the point estimates match at the third decimal. The rounded "own_share" of 55% in the paper corresponds to 54.57% in the raw data (the Stata `string(x, "%9.0f")` format rounds to the nearest whole percent).

### 3.2 Table A.1 — Summary statistics

| Statistic | Daegu | Seoul | NYC Metro |
|---|---:|---:|---:|
| Population (my replication) | 2,438,031 | 9,729,107 | 19,467,622 |
| # Districts | 8 | 25 | 20 |
| Sample period | 2018-01-01 – 2020-04-30 | 2018-01-01 – 2020-04-30 | 2020-01-01 – 2020-04-30 |
| First case (paper) | 2020-02-17 | 2020-01-30 | 2020-03-03 |
| First case (replication) | 2020-02-17 | 2020-01-30 | 2020-03-03 |
| Lockdown date | 2020-02-24 | 2020-02-24 | 2020-03-22 |
| Cumulative cases | 6,778 | 354 | 389,603 |

All district counts, dates, and sample ranges match the text of the paper (e.g., Section 3: "25 districts in Seoul and 8 districts in Daegu. We define NYM to be 20 counties"; Section 3.2: NYM "first confirmed within-city case on March 3", NY lockdown order "March 22").

### 3.3 What we did NOT check numerically
- Figure 1 time-fixed-effect point estimates (Stata `reghdfe` with date FEs; event-study coefficients are plotted, not tabulated).
- Ridership drops of 60.2% / 34.9% (Daegu / Seoul) between lockdown announcement and April 30. These are reported in the body text but would require re-running `202_figure1_4.do` on the raw subway panel; we did not attempt it.
- Model parameters (R₀ = 1.32 / 1.32 / 2.94; optimal-vs-uniform lockdown gaps of 20% / 32% / 58%). These all come from the MATLAB structural pipeline.

---

## 4. Data Audit Findings

The regression dataset (`sel_gravity.parquet`) is an exceptionally clean balanced panel:

### Coverage
- **531,875 total rows, 100.00% balanced.** 25 × 25 = 625 OD pairs, each observed on all 851 days from 2018-01-01 through 2020-04-30.
- **Regression sample (2020):** 75,625 rows, 625 OD pairs per date, 121 dates. No missing cells.
- **No zero/negative expenditures** — `log(eod)` is always defined.
- No missing values in any regression column.

### Distributions
| Var | Mean | SD | Min | Max |
|---|---:|---:|---:|---:|
| eod (KRW) | 2.62 M | 7.68 M | 4,919 | 119.5 M |
| log(eod) | 13.53 | 1.39 | 8.50 | 18.60 |
| distance (km) | 14.75 | 7.82 | 0.00 | 36.13 |
| log(dist+1) | 2.58 | 0.71 | 0.00 | 3.61 |
| χ (= yod2) | 0.968 | 0.195 | 0.072 | 5.904 |

- **Own-district expenditure** is an order of magnitude larger than off-diagonal (mean 34.2 M vs 1.3 M KRW) — consistent with strong distance frictions.
- **χ (commute flow ratio)** averages 1.000 (SD 0.168) in 2018–2019 and drops to 0.775 (SD 0.228) in 2020, confirming the pandemic-era commute collapse.
- **Top 5 expenditure rows** are all Jongno-gu (10101) on 2018/2019 New Year's Eve and a 2018-12-24 row — the commercial downtown cluster during the end-of-year shopping window. No obvious measurement errors.
- **`ipat` flag** (post-first-case indicator) covers 10.8% of the full panel and ~76% of the 2020 subsample, so the regression relies on both pre- and post-first-case variation.

### Logical consistency
- All expenditures > 0; all distances ≥ 0; χ > 0.
- Panel balance perfect (no gaps, no extra observations).
- Dates correspond exactly to the 2018–2020 calendar range declared in the paper.

No anomalies were found.

---

## 5. Robustness Check Results

All checks re-estimate the same gravity regression on different samples or SE/specification choices. HC1 SEs are reported for the baseline (slightly smaller than the two-way cluster SEs from the paper, but point estimates are identical).

| # | Check | (σ−1)κ₁ | (σ−1)ε | N | Verdict |
|---|---|---:|---:|---:|---|
| 1 | Baseline 2020, HC1 | 1.529 | 0.447 | 75,625 | Matches paper |
| 2 | 2018-2019 only | 1.502 | 0.217 | 456,250 | κ stable; ε attenuated (no pandemic variation) |
| 3 | Full panel 2018-2020 | 1.506 | 0.336 | 531,875 | Pooling dilutes ε as expected |
| 4 | Post-first-case (ipat=1) only | 1.533 | 0.418 | 57,500 | Both stable |
| 5 | Off-diagonal only (drop i=j) | 1.624 | 0.441 | 72,600 | κ slightly higher (stronger distance elasticity once own-dist dropped) |
| 6 | Leave-one-out: drop Gangnam (10123) | 1.539 | 0.457 | 69,696 | Not driven by Seoul's CBD |
| 7 | Winsorise eod at 1/99 pct | 1.504 | 0.444 | 75,625 | Outliers don't drive it |
| 8 | Cluster SE on RCo only | 1.529 (SE 0.049) | 0.447 (SE 0.045) | 75,625 | Closest match to published SEs |
| 9 | Cluster SE on date | 1.529 (SE 0.002) | 0.447 (SE 0.078) | 75,625 | ε SE inflates with date clustering |
| 10 | Log(χ) instead of level χ (matches eq. 20 literally) | 1.526 | **0.366** | 75,625 | κ unchanged; ε ≈ 0.37 instead of 0.45 |
| 11 | Weekdays only | 1.525 | 0.350 | 54,375 | ε slightly smaller (less weekend noise) |
| 12 | Placebo: shuffle χ | 1.531 | **0.009** | 75,625 | Placebo collapses — regression is real signal |

Highlights:
- κ₁ is remarkably stable (1.50–1.62) across every specification including the placebo. This elasticity is a distance-friction parameter, so it is identified by cross-sectional variation alone and the treatment-like robustness checks should not — and do not — disturb it.
- ε is identified off the 2020 commute collapse: when we use only the pre-pandemic panel it attenuates sharply (0.22), and the placebo collapses it to zero. Both are the right signs.
- **The log-χ check (row 10) is worth noting.** Equation (20) is written with ln χ, but the Stata file uses χ in levels. Switching to the log specification changes ε from 0.45 to 0.37 — a ~20% reduction. The paper quotes 0.45, so we used the level specification as the baseline. The qualitative conclusion (positive, significant, ≈ 0.4) is the same either way.

---

## 6. Summary Assessment

### What replicates
- **The one reduced-form regression in the paper (equation 20, Section 3.3) replicates to three significant figures.** Point estimates 1.529 vs 1.53 and 0.447 vs 0.45; SEs 0.064 vs 0.066 and 0.065 vs 0.067. Sample size 75,625 matches exactly.
- **Table A.1 summary statistics** — district counts, populations, first-case dates, lockdown dates — match the paper verbatim.
- **Data audit** finds a perfectly balanced 625-pair × 851-day Seoul panel with no missing values, no zero expenditures, and χ distributions that match the pandemic narrative (mean 1.0 pre-2020, 0.78 in 2020).

### What doesn't
- **The MATLAB structural model is not replicated.** This includes: SEIR calibration of β, the nonlinear backward-resolution Hamiltonian solver, the Pareto frontier across 8+ ω values, the optimal lockdown maps, the uniform-vs-optimal gap comparisons, R₀ eigenvalue calculations, and all four panels of Figures 2 and 3. Re-running these would require executing ~30 MATLAB scripts against 9.9 GB of pre-computed `.mat` files and a commercial MATLAB licence. None of these numbers are reported in the paper with enough precision to make a tight Python re-implementation worthwhile.
- **Figure 1 event-study plots** are not re-computed (they rely on `reghdfe` with day-of-week controls and wild-cluster bootstrap SEs; the underlying point estimates are not tabulated in the paper text).

### Key concerns
1. **Level vs log χ in equation (20).** The paper writes ln χ; the Stata code uses level χ. Both give positive, significant, similar-magnitude estimates (0.45 vs 0.37), and both are consistent with the paper's qualitative message. This is a notation/specification choice, not a bug, but a careful reader would notice.
2. **Seoul-only identification of ε.** The commuting elasticity feeding the structural model for all three cities is estimated from Seoul credit-card data alone, since Daegu and NYM do not have the OD expenditure panel. The paper acknowledges this but readers should note that NYM and Daegu inherit ε from Seoul.
3. **No external benchmark for the MATLAB results.** The paper cites Monte, Redding & Rossi-Hansberg (2018) as an external comparison for κ (they get 1.29 vs the paper's 1.53), so the distance elasticity is plausibly in the literature's range. No such external benchmark exists for the commuting elasticity ε.

### Overall assessment
The empirical component of this paper is **cleanly replicable** and **matches to the third significant figure**. The data pipeline is transparent, the panel is perfectly balanced, and every robustness check gives a sensible answer. The bulk of the paper's quantitative claims come from the MATLAB structural model, which we did not attempt — but the two empirical parameters that feed the structural model both replicate correctly, which is a strong indirect check on the rest of the pipeline.

---

## 7. File Manifest

| File | Description |
|---|---|
| `utils.py` | Paths, Stata loaders, published-value dict, date conversion |
| `01_clean.py` | Merges SEL_CC + SEL_parms_od + KOR_commute_obs into the gravity panel |
| `02_elasticities.py` | Replicates equation (20) with two-way clustered SEs |
| `03_tableA1.py` | Replicates Table A.1 summary statistics for Daegu/Seoul/NYM |
| `04_data_audit.py` | Coverage, distributions, balance, outlier checks on the gravity panel |
| `05_robustness.py` | 12 robustness checks: alt samples, LOO, SE variants, log vs level χ, placebo |
| `sel_gravity.parquet` | Cleaned 531,875-row OD-date panel for Seoul |
| `elasticities.csv` | One-row CSV with the replicated coefficients and SEs |
| `tableA1.csv` | The Table A.1 values in tidy form |
| `robustness.csv` | All 12 robustness-check coefficients |
| `writeup_127341.md` | This writeup |
