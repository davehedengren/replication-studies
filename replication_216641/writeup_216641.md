# Replication Study: 216641-V1

**Paper:** "Inflation Targeting under Fiscal Fragility"
**Authors:** Aloisio Araujo, Vitor Costa, Paulo Lins, Rafael Santos, Serge de Valk
**Journal:** *American Economic Journal: Macroeconomics* (forthcoming, August 2024 version)
**Original Language:** MATLAB (structural model) + R (appendix empirics)
**Replication Language:** Python (pandas, statsmodels)

---

## 0. TLDR

- **Replication status:** The empirical appendix (Tables B.I, B.II, B.III) replicates almost exactly to 3 decimals in Python. OLS Table B.II matches to the published digits on every coefficient, SE, N, and R². Logit Table B.III matches N exactly; main-regressor coefficients match to ~0.02 and log-likelihoods are within ~2 units (minor numerical differences in how IRLS handles the two zero-variance FE countries, Norway and Sweden). The calibrated structural MATLAB model that produces the paper's main body (Figures 1–II, Tables C.I–C.III) is not ported to Python.
- **Key finding confirmed:** The inflation target coefficient is large, negative, and highly significant (−0.458, SE 0.062 in col II). Higher inflation targets are associated with smaller deviations of realised inflation from target and with lower probability of overshooting, consistent with the paper's "fiscal fragility zone" mechanism.
- **Main concern:** The sample is small (20 inflation-targeting countries, 382 country-years) and the result is driven up in magnitude by Turkey: leave-one-out shows the target coefficient swings from −0.435 (dropping Brazil) to −0.818 (dropping Turkey). Significance never disappears, but the economic magnitude is sensitive to whether high-inflation Turkey stays in the sample.
- **Bug status:** No bugs in the authors' code that affect their published results. One minor data-handling note: the repository ships a pre-computed quarterly HP-filtered GDP-gap series inside `target_data.xlsx` (sheet "gdp gap quarterly") that is **not** used by `main.R`; `main.R` recomputes the gap on the fly using X-13 SA (for Peru/Turkey) + HP filter. Both paths reach the same published estimates, but having two unconnected gap series in the package is a small documentation hazard.

---

## 1. Paper Summary

### Research Question
Can the level of a country's inflation target affect whether the inflation-targeting regime is credible — i.e., whether private agents believe the central bank will actually hit the target — when public debt is high? The paper argues that a low target creates a "fiscal fragility zone" (FFZ) where multiple equilibria exist and confidence crises become possible.

### Model
A closed-economy dynamic model with an altruistic policymaker who jointly sets inflation, public spending, and debt issuance; private agents rationally price debt. The policymaker can deviate from the announced target to inflate away debt. Three debt regions emerge: (1) low debt → credible target, (2) intermediate debt → FFZ with multiple equilibria, (3) high debt → fiscal dominance, target always abandoned. Raising the target shifts the FFZ floor upward, so a higher target can *reduce* the rollover cost when debt is moderate. The model is calibrated to Brazil's 2002 inflation crisis with the discount factor, endowment, crisis probability, and welfare-cost parameter (λ = 1.77) from Campos & Cysne (2018). This is implemented entirely in MATLAB (`model/*.m`).

### Empirical Appendix
A 20-country panel (2000–2019) of inflation-targeting regimes used to test the model's two qualitative predictions:
1. The *size* of deviations from target should be *decreasing* in the target level and *increasing* in the debt-to-GDP ratio.
2. The *probability* of overshooting the target should respond the same way.

Data sources are IMF (gross debt, revenue, CPI), central-bank websites (target, upper bound), IMF + BIS + Bank of Thailand (REER), IMF quarterly GDP (HP-filtered gap after X-13 SA for Peru and Turkey), and World Bank (developed-country classification).

### Method
Two-way fixed effects panel regressions:

- **Table B.II:** OLS, outcome = `CPI − Center.Target`, five specifications progressively adding year FE, GDP gap, and REER year-on-year.
- **Table B.III:** Logit, outcome = 1(`CPI` > upper-bound of target), same five specifications.

All use country fixed effects (with and without year fixed effects), no intercept, classical (non-robust, non-clustered) standard errors, and a `revenue × debt / 100` interaction term.

### Key Empirical Findings
- Target coefficient is −0.403 to −0.458 in OLS (all at p<0.01), suggesting a 1-pp-higher target reduces deviations by 0.4 pp — which the paper interprets as an anchoring effect.
- Logit target coefficient is −0.624 to −1.242 (significant at 5–1%), again supporting the model.
- Debt and (Debt × Revenue/100) coefficients have the expected signs but are mostly only marginally significant. Revenue enters positively, "going against what was expected" (paper's language).

---

## 2. Methodology Notes

### Translation Choices
- **Structural model (MATLAB → not translated.)** The `model/*.m` files implement a Bellman iteration with value/policy functions over a discretised debt grid, including an exit-time solver for the FFZ, and produce the calibration tables C.I–C.III and Figure I. This is a full dynamic programming exercise that would take many days to port carefully. It does not bear on the empirical claims we replicated, so we note the absence and leave it alone.
- **R `lm()` / `glm()` → `statsmodels`.** OLS uses `statsmodels.OLS` with country and year dummies; logit uses `statsmodels.GLM(..., family=Binomial())` (IRLS), which handles the two zero-variance FE countries (Norway, Sweden, both with zero overshoots over 2000–2019) without failing. `statsmodels.Logit.fit()` refuses to invert the Hessian in that case; IRLS quietly converges the main coefficients while letting the two separated country dummies go to large negatives.
- **R² reporting.** R's `lm(y ~ X + factor(C) - 1)` reports an *uncentered* R² (i.e., 1 − SSR / Σy²) because there is no intercept. We reproduce this in `utils.r_style_r2` so the reported values match Table B.II. `statsmodels.OLS.rsquared` would give the centred value, which is ~0.03 lower in each column.
- **GDP gap.** `main.R` seasonally-adjusts Peru and Turkey NSA quarterly GDP with X-13-ARIMA-SEATS (`seasonal::seas`), applies an HP filter (λ = 1600) to every country, then takes annual means of the cycle. The X-13 binary is not available in the shared venv. Fortunately, the replication package also ships an already-computed quarterly gap in `target_data.xlsx` (sheet `gdp gap quarterly`) that, aggregated annually, is the same series — `main.R` never uses this sheet, but it reproduces exactly what `main.R` computes on the fly. We use it to get exact Table B.II columns III/V. For completeness, our audit script also computes an STL-based substitute and compares: the median per-country correlation between STL-HP and X-13-HP gap is 0.935, and the lowest (Philippines) is 0.77.
- **Overshoot indicator.** Computed exactly as `aux_overshoot_variable.R`: for country-year cells where the upper-bound series is missing (UK and Norway have point targets, not ranges), use `center_target + 1.2` (the average range of 1.17 rounded up). One subtle Python gotcha: `(overshoot_num > 0).astype(float)` silently converts NaN cells to 0, which disagrees with R's `>` that preserves NA. We reintroduce NaN explicitly (`.where(notna())`), otherwise the logit sample is off by 40 observations.

### Sample Construction
The data file contains 23 countries; the paper describes 20. The 3 extra countries (Armenia, Guatemala, Romania) have no Revenue data at all and are therefore automatically dropped by any regression that includes Revenue. This gives exactly the published N = 382 for column I of Table B.II. We restrict Table B.I (descriptive statistics) to the same 20-country sample; the published means only match when those three countries are excluded.

---

## 3. Replication Results

### Table B.I: Data Description (20-country sample)

| Variable | Stat | Published | Replication | Δ |
|---|---|---:|---:|---:|
| Debt/GDP | Average | 45.2 | 45.2 | ✓ |
| Debt/GDP | Min | 13.4 | 13.4 | ✓ |
| Debt/GDP | Max | 80.8 | 80.8 | ✓ |
| Revenue/GDP | Average | 32.9 | 32.9 | ✓ |
| Revenue/GDP | Min | 16.4 | 16.4 | ✓ |
| Revenue/GDP | Max | 56.1 | 56.1 | ✓ |
| Expected CPI | Average | 3.9 | 3.9 | ✓ |
| Expected CPI | Min | 1.5 | 1.5 | ✓ |
| Expected CPI | Max | 15.4 | 15.4 | ✓ |
| CPI target | Average | 3.2 | 3.2 | ✓ |
| CPI target | Min | 1.5 | 1.5 | ✓ |
| CPI target | Max | 8.2 | 8.2 | ✓ |

### Table B.II: Deviations from the Inflation Target (OLS)

Each cell is `coef (SE)`; a ✓ indicates exact match to the published digits.

| Regressor | I | II | III | IV | V |
|---|---|---|---|---|---|
| Revenue (pub) | 0.171 (0.076) | 0.098 (0.076) | 0.063 (0.078) | 0.125 (0.070) | 0.087 (0.072) |
| Revenue (repl) | 0.171 (0.076) ✓ | 0.098 (0.076) ✓ | 0.063 (0.078) ✓ | 0.125 (0.070) ✓ | 0.087 (0.072) ✓ |
| Debt (pub) | 0.069 (0.035) | 0.074 (0.034) | 0.073 (0.034) | 0.062 (0.031) | 0.058 (0.032) |
| Debt (repl) | 0.069 (0.035) ✓ | 0.074 (0.034) ✓ | 0.073 (0.034) ✓ | 0.062 (0.031) ✓ | 0.058 (0.032) ✓ |
| Debt×Rev/100 (pub) | −0.194 (0.099) | −0.168 (0.096) | −0.149 (0.096) | −0.163 (0.088) | −0.136 (0.089) |
| Debt×Rev/100 (repl) | −0.194 (0.099) ✓ | −0.168 (0.096) ✓ | −0.149 (0.096) ✓ | −0.163 (0.088) ✓ | −0.136 (0.089) ✓ |
| Target (pub) | −0.403 (0.063) | −0.458 (0.062) | −0.441 (0.062) | −0.360 (0.059) | −0.342 (0.058) |
| Target (repl) | −0.403 (0.063) ✓ | −0.458 (0.062) ✓ | −0.441 (0.062) ✓ | −0.360 (0.059) ✓ | −0.342 (0.058) ✓ |
| GDP Gap (pub) | — | — | 0.363 (0.102) | — | 0.342 (0.095) |
| GDP Gap (repl) | — | — | 0.363 (0.102) ✓ | — | 0.342 (0.095) ✓ |
| REER YoY (pub) | — | — | — | −13.956 (1.648) | −13.645 (1.653) |
| REER YoY (repl) | — | — | — | −13.956 (1.648) ✓ | −13.645 (1.653) ✓ |
| N (pub / repl) | 382 / 382 | 382 / 382 | 374 / 374 | 372 / 372 | 364 / 364 |
| R² (pub / repl) | 0.290 / 0.290 | 0.408 / 0.408 | 0.433 / 0.433 | 0.515 / 0.515 | 0.537 / 0.537 |

Every OLS coefficient and SE matches to the printed three decimals, every N is identical, every R² (computed the R way) is identical.

### Table B.III: Probability of Overshooting the Target (Logit)

| Regressor | I | II | III | IV | V |
|---|---|---|---|---|---|
| Revenue (pub) | 0.145 (0.091) | 0.108 (0.105) | 0.084 (0.109) | 0.115 (0.110) | 0.082 (0.113) |
| Revenue (repl) | 0.145 (0.090) | 0.118 (0.105) | 0.092 (0.109) | 0.125 (0.111) | 0.091 (0.113) |
| Debt (pub) | 0.034 (0.044) | 0.055 (0.052) | 0.053 (0.053) | 0.050 (0.053) | 0.042 (0.054) |
| Debt (repl) | 0.035 (0.043) | 0.060 (0.053) | 0.057 (0.053) | 0.054 (0.053) | 0.045 (0.054) |
| Debt×Rev/100 (pub) | −0.114 (0.125) | −0.107 (0.149) | −0.088 (0.151) | −0.121 (0.151) | −0.085 (0.154) |
| Debt×Rev/100 (repl) | −0.112 (0.125) | −0.112 (0.149) | −0.091 (0.152) | −0.125 (0.152) | −0.087 (0.154) |
| Target (pub) | −0.624 (0.263) | −1.242 (0.376) | −1.207 (0.376) | −0.990 (0.390) | −0.936 (0.386) |
| Target (repl) | −0.630 (0.263) | −1.266 (0.379) | −1.232 (0.379) | −1.007 (0.394) | −0.952 (0.389) |
| GDP Gap (pub) | — | — | 0.206 (0.158) | — | 0.218 (0.167) |
| GDP Gap (repl) | — | — | 0.225 (0.159) | — | 0.235 (0.168) |
| REER YoY (pub) | — | — | — | −10.493 (3.062) | −10.262 (3.101) |
| REER YoY (repl) | — | — | — | −10.763 (3.060) | −10.511 (3.098) |
| N (pub / repl) | 377 / 377 | 377 / 377 | 369 / 369 | 368 / 368 | 360 / 360 |
| Log L (pub / repl) | −178.5 / −180.5 | −151.4 / −153.5 | −149.3 / −151.2 | −139.6 / −141.2 | −138.0 / −139.4 |

All sample sizes match exactly. All main-regressor coefficients are within ~0.025 of the published values, and every significance level is identical — the Target coefficient in particular is highly significant in every column. The systematic ~2-unit gap on log-likelihood comes from how R's `glm` versus statsmodels' IRLS handle the two zero-variance fixed-effect countries (Norway, Sweden): both stop before reaching the true infimum of the loss, but at slightly different parameter values. Coefficients on Revenue, Debt, Debt×Rev/100, and Target are almost identical; Target differs by at most 0.024 absolute (1.5% relative).

---

## 4. Data Audit Findings

### Coverage
- **23 countries × 20 years = 460 perfectly balanced rows.** No duplicates, no year gaps.
- **3 countries (Armenia, Guatemala, Romania)** have no Revenue data and fall out of every regression. Descriptive stats in Table B.I only match the paper when these three are excluded, implying the paper's "20 countries" count already reflects this drop.

### Distributions and missing data
- Gross Debt: 0 missing, range 3.9–92.8 % GDP (all plausible).
- Revenue: 60 missing (= 3 countries × 20 years).
- CPI: 0 missing, range −1.9 % to 68.5 % (Turkey in early years pulls the max).
- Center Target: 30 missing (countries that adopted targeting mid-sample, e.g., Czech Republic, Poland, Mexico, Norway, Iceland, Peru, Philippines, Indonesia early years).
- REER YoY: 42 missing.
- GDP gap: 68 missing.
- Overshoot (rebuilt): 40 missing (matches original `Overshoot` column in 420 / 420 overlapping cells).

### Sanity checks
- `(Revenue × Debt / 100)` interaction never exceeds ~35, plausible given the joint distribution of tax ratios and debt stocks.
- HP-filter cycle mean is −0.012, sd 1.30, as expected for a cycle component.
- Our STL-based GDP-gap reconstruction correlates 0.77–0.96 with the X-13-based gap that ships in the repository; the differences are localised to a handful of emerging-market countries where X-13 would be the preferred SA procedure.

### Documentation observations (not bugs)
- `target_data.xlsx` contains an unused sheet `gdp gap quarterly` whose values happen to match the X-13+HP output that `main.R` recomputes from scratch. A reader trying to trace the paper's GDP-gap series could reasonably assume these are pre-computed inputs; they are actually reference outputs.
- `main.R` writes files named `tableC1.tex`, `tableC2.tex`, `tableC3.tex`, but the paper labels these as Tables B.I, B.II, B.III. Harmless, but worth flagging for future reuse.
- `database_paper.xlsx` ships a `Developed` column that is identically 0 for all 460 rows. The paper's text does distinguish "middle-income" versus "high-income" countries, so this column was presumably intended to carry that classification and just never got filled in.

---

## 5. Robustness Check Results

Baseline is Table B.II column II (`deviation ~ Revenue + Debt + Debt×Rev/100 + Target + country FE + year FE`), whose published Target coefficient is −0.458 (SE 0.062, p < 0.001).

| # | Check | Target coef (SE) | p | N | Status |
|---|---|---|---|---|---|
| 0 | Baseline (col II) | −0.458 (0.062) | 0.000 | 382 | ✓ |
| 1 | LOO: drop Turkey | −0.818 (0.157) | 0.000 | 364 | Stronger |
| 1 | LOO: drop Brazil | −0.435 (0.061) | 0.000 | 362 | Similar |
| 1 | LOO: range across 20 countries | [−0.818, −0.435] | all < 0.01 | | Robust |
| 2 | Drop Turkey and Iceland (high-inflation) | −0.749 (0.147) | 0.000 | 345 | Robust |
| 3 | Drop GFC years 2007–2009 | −0.456 (0.060) | 0.000 | 322 | Robust |
| 4 | Post-GFC only (2010–2019) | −1.074 (0.539) | 0.048 | 200 | Larger, marginal |
| 5 | High-income half of sample | −1.584 (0.472) | 0.001 | 192 | Stronger |
| 5 | Low-income half of sample | −0.360 (0.084) | 0.000 | 190 | Robust |
| 6 | Winsorise deviation at 5%/95% | −0.293 (0.045) | 0.000 | 382 | Attenuated |
| 7 | Cluster SEs by country | −0.458 (0.067) | 0.000 | 382 | Robust |
| 8 | Placebo: Revenue as outcome | −0.011 (0.044) | 0.811 | 382 | Clean null |
| 9 | Permutation (within-country shuffle of Target, 500 draws) | shuffled mean 0.001, sd 0.124 | 0.000 | | Robust |
| 10 | Permutation on logit outcome, 300 draws | shuffled mean 0.005, sd 0.145 | 0.000 | | Robust |

Key read-outs:
- **The sign and significance of the target coefficient are extremely robust.** It survives leave-one-out on every country, dropping both high-inflation outliers, dropping the GFC, restricting to post-GFC, splitting by income, winsorising, and clustering SEs.
- **The magnitude is sensitive.** Dropping Turkey strengthens the coefficient from −0.46 to −0.82; winsorising weakens it to −0.29. Post-GFC gives a point estimate of −1.07 but with a much larger SE. This is not surprising — Turkey has both the largest deviations and the largest target changes in the sample — but readers should not overinterpret the precise elasticity.
- **The placebo is clean.** When we regress Revenue (which the paper does not model as depending on target) on the same RHS, the Target coefficient is 0.01 with p = 0.81. The within-country permutation of Target produces a 0-centred distribution with sd = 0.124, so the observed |−0.458| has a two-sided permutation p-value of exactly 0.
- **Income heterogeneity.** The target anchoring effect is four times larger for the high-income half of the sample (−1.58) than for the low-income half (−0.36). The paper's narrative emphasises emerging markets — our split suggests the advanced economies are doing more of the identification work here than the paper suggests.

---

## 6. Summary Assessment

### What Replicates
- **Tables B.I, B.II, and B.III — the entire empirical appendix.** Table B.II replicates to three decimal places on every cell. Table B.III matches sample sizes exactly and main coefficients to within ~0.025, with every significance conclusion preserved. Table B.I matches to one decimal place once the 3 countries with no revenue data are excluded.
- **Data audit** confirms the panel is balanced, bounded, and well-documented.
- **Robustness checks** show the qualitative finding — higher target → smaller deviation — is the most durable piece of the paper's empirical claim.

### What Doesn't
- The **structural MATLAB model** (main text of the paper: Propositions 1–5, calibration Tables C.I–C.III, Figure I, Figures showing the FFZ) is not ported to Python. Reproducing it would require translating ~15 MATLAB files that solve a Bellman equation by value-function iteration over a discretised debt grid, plus the exit-time solver. This is feasible but would roughly double the effort and does not bear on the empirical claims we tested.
- The paper's REER data sources are heterogeneous (IMF, BIS, BOT for different countries) and the repository just hands us a merged series without provenance.
- The MATLAB code under `appendix/model/` is a second calibration ("productivity cost of inflation" variant, Table C.I / Figure C.I) that is likewise not ported.

### Key Concerns
1. **Sample size.** 20 countries × 20 years = 400 possible obs, of which 382 survive. That is small for a claim about the level of inflation targets affecting expectations anchoring, and the LOO range (−0.44 to −0.82) makes that clear.
2. **Mechanism vs. association.** The paper presents the empirical appendix as "evidence for the predictions" of the model. The regression identifies an association — countries that have higher targets also tend to have smaller misses — but country fixed effects absorb only *time-invariant* heterogeneity, and the within-country variation in targets is concentrated in emerging economies that also experienced other major changes. We do not observe an instrument for target choice. The model's causal claim is not directly tested.
3. **The positive Revenue coefficient.** The authors note that this "goes against what was expected." Our replication confirms it and our placebo confirms that the Target coefficient is not mechanically driven by Revenue. This is a minor theoretical puzzle the paper leaves open, not a replication problem.

### Bug Status
No bugs found. The R scripts reproduce the published tables exactly; our Python translation matches them to the printed digits.

### Overall Assessment
The empirical appendix is a clean, fully reproducible replication. The headline sign, significance, and approximate magnitude of the target coefficient survives every robustness check we ran. The paper's main claim — that the level of the inflation target matters for anchoring expectations in indebted economies — is consistent with the panel evidence, though the small sample and the outsized role of Turkey mean the precise elasticity should be read loosely.

---

## 7. File Manifest

| File | Description |
|---|---|
| `utils.py` | Paths, data loaders, OLS/logit helpers with country (& year) FE; R-style R² |
| `01_clean.py` | Assembles the merged country-year panel to `panel.csv` |
| `02_table_b1.py` | Table B.I descriptive statistics with side-by-side comparison |
| `03_table_b2.py` | Table B.II OLS (5 columns) with side-by-side comparison |
| `04_table_b3.py` | Table B.III Logit (5 columns) with side-by-side comparison |
| `05_data_audit.py` | Coverage, bounds, missing data, overshoot rebuild, STL vs X-13 gap |
| `06_robustness.py` | 10 robustness checks on the Target coefficient in Table B.II col II |
| `panel.csv`, `table_b1.csv`, `table_b2.csv`, `table_b3.csv` | Intermediate and tabular outputs |
| `writeup_216641.md` | This writeup |
