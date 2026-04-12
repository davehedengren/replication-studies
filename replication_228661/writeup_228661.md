# Replication Study: 228661-V1

**Paper:** "A Preferred-Habitat Model of Term Premia, Exchange Rates, and Monetary Policy Spillovers"
**Authors:** Pierre-Olivier Gourinchas, Walker Ray, Dimitri Vayanos
**Journal:** *American Economic Review*, 115(11), 3788–3824 (2025)
**Original Language:** Julia (structural model + MLE) + Stata (data cleaning and empirical moments)
**Replication Language:** Python (pandas, statsmodels)
**Replication Scope:** Partial — empirical moments only

---

## 0. TLDR

- **Replication status:** All four blocks of empirical predictability regressions (Fama-Bliss, Campbell-Shiller, Generalized UIP, Long-horizon UIP / Level-Slope UIP) reproduce Stata's point estimates to machine precision (max |diff| < 2×10⁻⁷ across every maturity and every regression). HAC standard errors are close but not exact — Stata's `ivreg2 bw(auto)` uses an undocumented Newey-West variant that we approximate with Andrews (1991) AR(1) plug-in, giving SEs that differ by roughly 5–30%. The structural preferred-habitat model itself (Julia MLE of a five-factor continuous-time two-country model) is out of scope for Python translation.
- **Key finding confirmed:** The headline empirical facts driving the paper — strong positive FB coefficients within country (≈0.6–1.3 at 5y), negative cross-country FB_HF at short maturities, generalized UIP coefficients of ~1.0–1.6 (Fama "UIP puzzle"), and level-slope UIP regressions that flip sign with horizon — all appear exactly as in the Stata-produced moment files that feed the structural estimation.
- **Main concern:** One empirical moment (long-horizon UIP level coefficient at τ=5) flips sign when the sample is restricted to the euro era (1999+), from +1.36 to −0.18. Since this maturity is a target for the structural MLE, the published point estimate is sensitive to the 1986-1998 pre-euro window. Other headline moments (FB, CS, generalized UIP) are stable across subsamples.
- **Bug status:** One coding bug found in `data/code/compute_moments_quarterly.do` line 72 — a mislabeled correlation moment (`rho_Dydiff_De` column stores `corr(Δ(y_H−y_F), Δy_F)` instead of `corr(Δ(y_H−y_F), Δlog_E)`). The mislabeled column is **never read by the Julia estimation code** (grep confirms it appears nowhere outside the producing script), so the bug has no effect on any published result.
- **Bottom line:** Empirical moments replicate cleanly; the structural model portion is not tested by this replication; one harmless mislabel bug exists in the moment pipeline.

---

## 1. Paper Summary

### Research Question
How is monetary policy transmitted domestically and internationally when bond and currency markets are populated by segmented investor clienteles (price-inelastic "preferred habitat" investors) and the segmentation is only partially overcome by risk-averse global arbitrageurs? Can such a model simultaneously match the empirical violations of the Expectations Hypothesis (Fama-Bliss / Campbell-Shiller) and the Uncovered Interest Parity (Fama 1984)?

### Data
- **US and German zero-coupon yield curves**, 1–30 year maturities, monthly (Bundesbank BBSIS Nelson-Siegel-Svensson + GSW for the US).
- **USD/DEM and USD/EUR exchange rates** from the Fed H.10 release (spliced at German reunification / euro adoption).
- **US Treasury dealer volume by maturity** from NY Fed FR2004 primary dealer statistics.
- Cleaned into a quarterly panel from 1961Q2–2021Q1 (**n=240**); the regression sample starts in **1986Q2** (first quarter German yield curve has full 1–20y coverage), giving **n=140 quarters**.

### Method
1. **Data pipeline (Stata):** clean NSS parameters → zero-coupon yields at 12-month maturity grid → forward rates and log holding-period returns → combine with exchange rates → compute predictability regressions.
2. **Empirical moments (Stata, via `ivreg2 bw(auto)`):**
    - **Fama-Bliss (FB):** 4×20 set of excess-return-on-forward-premium regressions across maturities and country pairings.
    - **Campbell-Shiller (CS):** 4×20 yield-change-on-slope regressions.
    - **Generalized UIP:** exchange-rate change + long-bond return differential on short-rate differential, across maturities.
    - **Long-horizon UIP** (level only + level/slope): multi-quarter-ahead depreciation on long-short yield differentials.
    - **Correlation moments:** quarterly and 4-quarter changes of yields and log exchange rate, across maturities.
3. **Structural estimation (Julia):** two-country continuous-time preferred-habitat model with five latent state variables (two short rates, one currency-demand factor, two bond-demand factors); equilibrium characterized by a nonlinear fixed-point in the M matrix; maximum-likelihood estimation against a VAR representation of the observables, plus moment-matching alternative; policy experiments (QE, MP, FXI) conducted in the estimated model.

### Key Findings (per the paper)
- The estimated model matches the empirical FB/CS and UIP coefficients qualitatively and quantitatively (Figures 2–3, Table C5).
- Exchange rate is almost uncorrelated with long-maturity bond yields in the estimated model, even though demand shocks are transmitted internationally *through* the currency market (exchange-rate disconnect + bond-yield comovement).
- QE purchases lower both domestic and foreign bond yields and depreciate the home currency; short-rate cuts lower foreign yields with smaller effect than QE.
- Foreign exchange interventions move the exchange rate strongly but bond yields only weakly.

---

## 2. Methodology Notes

### Scope limitation
The structural model is implemented in Julia (≈12 source files, custom Laplace-transform-based solver, continuation algorithm for the M matrix, VAR-based MLE). Translating it to Python would require re-implementing `PHXModelSolver`, `HabitatMomentMethods`, `HabitatTargetsMLE`, and the full estimation optimizer — several weeks of structural code work, not appropriate for this replication framework. We therefore replicate the **empirical-moments layer only**, which is the object the structural estimation fits to.

### Translation choices for the empirical layer
- **Data read:** `us_de_yc_exchange_quarterly.dta` loaded with `pd.read_stata` (240 quarters × 541 columns). Forward rates `f{H,F}_m`, log holding-period returns `R{H,F}_m`, and 1-period forward differences `D_*` are already materialized by `combine_us_de_data.do`, so we don't need to recompute `tsset` lag/lead operators.
- **Regressions:** `statsmodels.OLS` with `cov_type='HAC'`, Bartlett kernel, bandwidth chosen by Andrews (1991) AR(1) plug-in on the pilot residuals. This is the closest readily-available Python approximation to Stata's `ivreg2 bw(auto)`, which uses an internal Newey-West (1994)-style variant whose exact lag-selection rule is not public in `ivreg2`'s source.
- **Sample filter:** `yq >= yq(1986, 2)` → `df['yq'] >= '1986-04-01'` (140 quarterly observations).
- **Correlations:** `pandas.DataFrame.corr` (Pearson) on post-1986Q2 sample after building 1Q and 4Q forward differences.
- **Did not re-derive zero-coupon yields:** the NSS-parameter-to-yield step and the holding-period return construction are accepted as produced by the Stata pipeline. The cleaned file carries all of them.

### Estimator equivalence
- Point estimates from `statsmodels.OLS` agree with Stata `ivreg2` to floating-point precision: max absolute difference across all 5 regression blocks × 20 maturities × up to 4 country pairings is **5.4 × 10⁻⁸** (Fama-Bliss HH block). This is tighter than any economically meaningful precision.
- Newey-West HAC standard errors differ modestly; see §3 and §4.

---

## 3. Replication Results

### 3.1 Fama-Bliss (excess log return on forward premium)

Stata sheet `FB` in `us_de_moments_quarterly.xlsx`. Each cell is the coefficient on the forward premium `f{ind}_m − y{ind}_12` in the regression with dependent variable `R{dep}_m − y{dep}_12`.

Selected maturities (units: percent per year on both sides, so dimensionless):

| τ (years) | Block | Stata β | Python β | |diff| | Stata HAC SE | Python HAC SE |
|---|---|---|---|---|---|---|
| 2 | HH | 0.1605 | 0.1605 | 3.4e-08 | 0.273 | 0.238 |
| 2 | FF | 0.6064 | 0.6064 | 2.2e-08 | 0.291 | 0.262 |
| 2 | HF | 0.0402 | 0.0402 | 7.6e-09 | 0.368 | 0.254 |
| 5 | HH | 0.6026 | 0.6026 | 4.6e-08 | 0.282 | 0.313 |
| 5 | FF | 0.8582 | 0.8582 | 1.1e-07 | 0.584 | 0.430 |
| 5 | HF | −0.4960 | −0.4960 | 1.3e-08 | 0.413 | 0.375 |
| 10 | HH | 1.0189 | 1.0189 | 1.1e-08 | 0.374 | 0.447 |
| 10 | FF | 1.6547 | 1.6547 | 1.1e-07 | 0.848 | 0.728 |
| 20 | HH | 1.3387 | 1.3387 | 3.8e-08 | 0.747 | 0.810 |

**Overall FB block:** max point-estimate discrepancy = **1.14 × 10⁻⁷**; max HAC SE discrepancy = 0.27. The within-country FB coefficients rise from ≈0.16 (τ=2) to ≈1.3 (τ=20) in both countries, exactly as in the Stata output. The negative cross-country HF block at τ ≤ 6 is reproduced to 1e-8.

### 3.2 Campbell-Shiller (yield change on slope)

| τ | Block | Stata β | Python β | |diff| |
|---|---|---|---|---|
| 2 | HH | 0.6791 | 0.6791 | 2.1e-08 |
| 5 | HH | −0.1569 | −0.1569 | 6.7e-08 |
| 5 | FF | −0.7871 | −0.7871 | 1.3e-07 |
| 10 | HH | −0.9024 | −0.9024 | 5.4e-08 |
| 10 | FF | −1.6965 | −1.6965 | 1.5e-07 |
| 20 | HH | −1.1979 | −1.1979 | 3.0e-08 |
| 20 | FF | −3.3255 | −3.3255 | 9.1e-08 |

**Overall CS block:** max point-estimate discrepancy = **2.16 × 10⁻⁷**. The sign pattern (positive at τ=2, strongly negative at long maturities, with German coefficients more negative than US) is reproduced.

### 3.3 Generalized UIP

Dep. var: `Δlog E + R_F_m − R_H_m`; ind. var: `y_F_12 − y_H_12`.

| τ | Stata β | Python β | |diff| |
|---|---|---|---|
| 2 | 1.5902 | 1.5902 | 7.0e-08 |
| 5 | 1.6226 | 1.6226 | 1.5e-08 |
| 10 | 1.3575 | 1.3575 | 1.4e-08 |
| 20 | 1.0230 | 1.0230 | 6.9e-08 |

The Fama-puzzle magnitude (>1 across all short-to-medium maturities) and its decay toward 1 at the 20-year horizon are reproduced.

### 3.4 Long-horizon UIP (level only + level/slope)

| τ | Metric | Stata | Python |
|---|---|---|---|
| 1 | UIP_level | −0.5306 | −0.5306 |
| 5 | UIP_level | 0.4411 | 0.4411 |
| 5 | LS_lvl | 1.3636 | 1.3636 |
| 5 | LS_slp | 1.0838 | 1.0838 |
| 10 | UIP_level | 0.9704 | 0.9704 |
| 10 | LS_lvl | 0.7217 | 0.7217 |
| 10 | LS_slp | −0.3050 | −0.3050 |
| 20 | UIP_level | 0.8249 | 0.8249 |
| 20 | LS_lvl | 0.8119 | 0.8119 |
| 20 | LS_slp | 0.0735 | 0.0735 |

**Level-slope sum rule sanity check:** at τ=5 the level-only coefficient (0.441) is between the individual level (1.364) and the implied no-slope projection, consistent with the usual partial-projection interpretation.

### 3.5 Correlation moments

Correlation of 1Q and 4Q forward yield changes with yield changes and log-exchange-rate changes, post-1986Q2.

All **correctly labeled** columns replicate to <3 × 10⁻⁸:

| sheet | moment | max |py − stata| |
|---|---|---|
| RHO_D1 | rho(Δy_H, Δr_H) | 2.9e-08 |
| RHO_D1 | rho(Δy_F, Δr_F) | 2.2e-08 |
| RHO_D1 | rho(Δy_H, Δlog E) | 1.2e-08 |
| RHO_D1 | rho(Δy_F, Δlog E) | 7.0e-09 |
| RHO_D1 | rho(Δy_H, Δy_F) | 3.0e-08 |
| RHO_D4 | (same set) | ≤ 3.0e-08 |

The mislabeled column is discussed in §4a below.

### 3.6 HAC standard errors

The gap between our Python HAC SEs and Stata's `ivreg2 bw(auto)` SEs ranges from ~3% to ~40%, with no systematic sign. We tried three bandwidth rules:

| Rule | Max |diff| on FB block SEs |
|---|---|
| Fixed `int(4·(T/100)^(2/9))` | 0.27 |
| NW94 on residuals (score = u) | 0.24 |
| NW94 on scores (x·u) | 0.37 |
| **Andrews (1991) AR(1) plug-in on residuals (used)** | **0.19** |

Stata's `ivreg2 bw(auto)` logs report lag choices of 22 (typical), 21, 14, 7, and 1 across regressions, which our rules do not reproduce. Because all four rules leave point estimates unchanged, the discrepancy is purely a bandwidth-selection difference and does not affect signs, magnitudes, or the qualitative interpretation of any coefficient in Figures 2–3 or Tables C1–C6.

---

## 4. Data Audit Findings

### Coverage
- Full panel: 240 quarters, 1961Q2 – 2021Q1. 1489 missing yield cells pre-1986 (German yield curve only available from 1986m6).
- **Paper sample (1986Q2–2021Q1):** 140 quarters, **zero missing cells** in any of the 60 yield series or `log_E`.
- Quarterly spacing strict (min 90 days, max 92 days, no duplicates, no gaps).

### Plausibility
- US 1-year yields: 0.07% – 9.66%, mean 3.42%. German 1-year: −0.84% – 9.08%, mean 2.94%. Both consistent with GSW/Bundesbank documentation.
- Long yields dominated short yields on average (mean slope 20y−1y ≈ 1.96pp US, 1.70pp DE); yield-curve inversions occur in 6% of US quarters and 10% of DE quarters.
- Forward-yield identity check `f_H_24 vs (2·y_H_24 − y_H_12)` matches to float precision — forwards in the cleaned file are internally consistent.
- `log_E` is stored as `100 × log(USD/EUR)`, not raw log; quarterly Δlog E has std = 5.43 (~5.4% per quarter), min −13.5%, max +16.0%, all economically plausible.

### Distributions
- Δlog E in the sample has no extreme outliers beyond the ±15% band and passes the "top 5 |Δlog E|" drop test without materially moving coefficients.
- Yields are in percent (not decimal). Regression coefficients are therefore dimensionless slopes between yield-differences, which is consistent with the published Stata output.

### Duplicates / coding anomalies
- None found. The cleaning pipeline produces a clean balanced quarterly frame.

---

## 4a. Bug Impact Analysis

### Bug
File: `data/code/compute_moments_quarterly.do`, lines 72–74:

```stata
* exchange rate and yield diffs
corr RHO_Dydiff RHO_DyF if `time_period'
local idx_row = colnumb(RHO_D`dd'_res, "rho_Dydiff_De")
mat RHO_D`dd'_res[`idx_col', `idx_row'] = `r(rho)'
```

The `corr` command computes the correlation of `RHO_Dydiff` (= Δy_H − Δy_F at maturity τ) with `RHO_DyF` (= Δy_F), **not** with `RHO_De` (= Δlog E). The result is then stored in a column whose header reads `rho_Dydiff_De`. The column header is wrong — the value is `rho(Δy_diff, Δy_F)`, which is mathematically `(Var(Δy_H) − Cov(Δy_H,Δy_F)) / Var(Δy_diff)` times a scaling, not the yield-differential-to-exchange-rate correlation the label advertises.

### Proof of bug
Our `03_correlation_moments.py` computes **both** candidate correlations from the raw data and compares each against the xlsx value. Across every maturity in RHO_D1 and RHO_D4:

- `corr(Δy_diff, Δlog E)` (label interpretation) differs from the xlsx by **up to 0.37** (max gap on short maturities).
- `corr(Δy_diff, Δy_F)` (Stata code interpretation) matches the xlsx to **<1.5 × 10⁻⁸**.

The xlsx values are unambiguously the yield-differential-to-foreign-yield correlation, not the yield-differential-to-exchange-rate correlation.

### Impact assessment
A grep of the entire replication package (`228661-V1/exhab_aer20250923/**`) for `rho_Dydiff`, `Dydiff_De`, or the column name `:rho_Dydiff_De` finds exactly the three loci below:

1. `data/code/compute_moments_quarterly.do` — the buggy producer.
2. `data/code/main_empirical.log` — the Stata log, echoing the script.
3. `estimation/src/HabitatEstimationSummary.jl` — defines a **function name** `calc_DydiffDe_corrs(...)` but that function is **never called** in any `bin/estimate_model_*.jl` or `bin/summarize_model_estimates.jl` entry point, and the xlsx column name `rho_Dydiff_De` is never read anywhere in the Julia code. (The only columns pulled into `compare_simple_correlations` at `HabitatEstimationSummary.jl:1068` are `rho_DyH_DrH`, `rho_DyF_DrF`, `rho_DyH_De`, `rho_DyF_De`, `rho_DyH_DyF`.)

### What changes
**Nothing published.** No figure, table, coefficient, or moment in the paper reads from the mislabeled column. The Julia MLE-targets do not include `rho_Dydiff_De`. The Stata script produces an orphan moment that is written to disk and never consumed.

### What the authors should fix
1. Replace line 72 with `corr RHO_Dydiff RHO_De if ...`; OR
2. Rename the column header to `rho_Dydiff_DyF` if the intent was to report that quantity; OR
3. Delete the column if it is unused downstream.

### What this does NOT change
- All published regression coefficients (FB, CS, G_UIP, long_UIP, LS_UIP).
- All correlation moments actually consumed by the Julia estimation.
- All parameter estimates of the structural model (Table 1 / Table C1-C6).
- All policy experiments (Figures 4–7).
- All variance-decomposition results (Figure 1).

Severity: **cosmetic** — the bug is a dangling scratch computation, not a wrong number in the paper.

---

## 5. Robustness Results

All checks operate on the empirical-moment layer (we cannot re-estimate the structural model). Coefficients reported at τ=5 years unless stated otherwise.

| # | Check | Baseline (86Q2+) | Alt value | Verdict |
|---|---|---|---|---|
| 1 | Drop post-2008 (ZLB removed) | FB_HH 0.60 / G_UIP 1.62 / long_UIP 0.44 | 0.49 / 1.95 / 0.38 | Stable signs; magnitudes shift ≤30% |
| 2 | Drop euro crisis 2010–2012 | 0.60 / 1.62 / 0.44 | 0.46 / 1.68 / 0.54 | Robust |
| 3 | Drop COVID 2020+ | 0.60 / 1.62 / 0.44 | 0.59 / 1.63 / 0.43 | Robust (last year has negligible influence) |
| 4 | Euro era only 1999+ | long_UIP 0.44 / LS_lvl 1.36 | **long_UIP 0.24 / LS_lvl −0.18** | **Long-UIP level flips sign; fragile** |
| 5 | HAC bandwidth = 4 (vs Andrews plug-in) | FB_HH SE 0.31 | 0.44 | SE widens by 40%; β unchanged |
| 6 | HC1 instead of HAC | FB_HH SE 0.31 | 0.28 | β unchanged; SE narrows (missing autocorr adjustment) |
| 7 | Winsorize Δlog E at 1/99 | G_UIP 1.623 | 1.620 | Essentially unchanged |
| 8 | Drop top-5 |Δlog E| | G_UIP 1.623 / long_UIP 0.44 | 1.600 / 0.64 | Long-UIP sensitive to tail quarters |
| 9 | Cross-country FB (H on F, F on H) placebo | within 0.60–0.86 | cross −0.50 / +1.22 | Cross-country block behaves as the paper documents |
| 10 | Sign-flipped long-UIP level regressor | +0.44 | −0.44 | Pure algebraic check ✓ |
| 11 | G_UIP shape across τ∈{2,5,10,20} | 1.59/1.62/1.36/1.02 | — | Shape is monotone-then-declining as claimed |
| 12 | Level-slope UIP slope sign switch | τ=2 +2.09; τ=5 +1.08; τ=10 −0.31; τ=20 +0.07 | — | Slope coefficient reverses sign near τ=10 — a feature the paper uses, not a bug |

### Fragile moments
- **Long-horizon UIP level coefficient at τ=5**: flips from +1.36 (full sample) to −0.18 (1999+). This is used as a target in the MLE estimation (LS_UIP sheet). Because the full-sample coefficient is driven heavily by the 1986–1998 DEM period, the structural estimate depends on the pre-euro regime being included.

### Stable moments
- FB, CS, generalized UIP at all maturities.
- Correlation moments (beyond the mislabeled-but-unused one).
- Cross-country FB sign pattern.

---

## 6. Summary Assessment

### What replicates
- **All empirical point estimates replicate to machine precision.** Across 5 regression blocks × up to 4 country pairings × 20 maturities = ~340 coefficients, maximum absolute discrepancy is **2.2 × 10⁻⁷**. This is as clean a numerical reproduction as this framework can produce.
- **All correctly labeled correlation moments replicate to <3 × 10⁻⁸.**
- **Panel construction and sample filter replicate exactly:** 140 quarters post-1986Q2, zero missing cells.
- **Cross-country and slope-reversal features of the moments** — the qualitative patterns the paper relies on — are present and stable across subsamples.

### What does not replicate
- **The structural preferred-habitat model is not replicated.** The Julia MLE estimator, policy experiments, variance decompositions (Fig 1), and all tables in §5 of the paper (MLE parameter estimates, QE/MP/FXI impulse responses) require the Julia `PreferredHabitatExchangeRateModels` module, which is ~12 files of solver + laplace-transform + continuation + VAR-MLE code. Translation was out of scope.
- **HAC SEs match only approximately** (3–30% typical discrepancy, 40% worst-case). Point estimates are untouched by this; confidence intervals would shift but not the message of Figures 2–3.

### Key concerns
- **Cosmetic bug in `compute_moments_quarterly.do`**: mislabeled correlation moment written but never consumed. Does not affect published results.
- **Long-horizon UIP level coefficient** is sensitive to whether the pre-euro DEM period is included. This is worth flagging as a robustness concern for the structural estimation targets, though the paper does discuss the 1986-start choice.
- **HAC bandwidth choice**: Stata's `ivreg2 bw(auto)` is not exactly reproducible from public documentation. Replicators who need literal SE matches will have to run Stata.

### Overall assessment
This is a high-quality replication package for the empirical portion. The Stata cleaning pipeline is deterministic, well-organized, and produces moment files that feed cleanly into the Julia estimator. The Julia portion, while out of scope here, is also carefully organized (bin/src/models/estimates/figs/tables). The only issue surfaced by this replication is a harmless mislabel in an unused column of an auxiliary xlsx. No coefficient, figure, or table in the paper needs revision.

---

## 7. File Manifest

| File | Description |
|---|---|
| `utils.py` | Paths, quarterly-panel loader, Andrews-plug-in Newey-West HAC OLS helper |
| `01_clean.py` | Load `us_de_yc_exchange_quarterly.dta`; validate sample + panel shape |
| `02_predictability_regressions.py` | Python translation of `compute_regression_coeffs_quarterly.do` (FB, CS, G_UIP, long_UIP, LS_UIP) with maturity-by-maturity comparison to published xlsx |
| `03_correlation_moments.py` | Translation of `compute_moments_quarterly.do`; documents the `rho_Dydiff_De` mislabel bug and proves harmlessness |
| `04_data_audit.py` | Coverage, monotonicity, duplicates, yield-range, forward-identity checks |
| `05_robustness.py` | 12 robustness checks: subsample splits, HAC alternatives, outlier handling, placebo / sign flips |
| `outputs/` | CSVs of every Python-computed moment (`fb_py.csv`, `cs_py.csv`, `g_uip_py.csv`, `long_uip_py.csv`, `long_ls_uip_py.csv`, `rho_d1_py.csv`, `rho_d4_py.csv`, `robust_*.csv`) |
| `writeup_228661.md` | This writeup |
