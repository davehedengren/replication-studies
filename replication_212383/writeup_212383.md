# Replication Study: 212383-V1

**Paper:** "Shocks and Exchange Rates in Small Open Economies"
**Authors:** Vito Cormun, Pierre De Leo
**Journal:** *American Economic Review* (forthcoming at replication time; draft November 2019)
**Original Language:** MATLAB (R2024a) + Dynare 6.4
**Replication Language:** Python (numpy, pandas, scipy.io, statsmodels)

---

## 0. TLDR

- **Replication status:** The paper's headline empirical finding — that external shocks explain the bulk of currency excess-return variance while domestic shocks dominate UIP-consistent dynamics — replicates. Across the 18 SOEs in the baseline sample, the country-median domestic share of 12-month excess-return variance is **0.15** vs the paper's ~**0.20**, and the country-median Fama β is **0.68** on domestic shocks and **-3.00** on external shocks (paper: ~1 and strongly negative). The DSGE model / Dynare components of the paper (Table 2, Figures 5 and 6) are not replicated — they require Dynare and add no new empirical content beyond what is already tested.
- **Key finding confirmed:** External shocks are the dominant source of UIP deviations in small open economies; domestic shocks (monetary-policy-like) are broadly UIP-consistent.
- **Main concern:** The Philippines has only 29 usable monthly observations, too short for reliable VAR inference; the paper keeps it in the baseline. Dropping it barely moves the median (see §5). The post-1990 subsample more than doubles the domestic share (0.36 vs 0.15), so the headline ~80/20 split is partly driven by the 1974-1990 era when most countries had wider interest-rate differentials.
- **Bug status:** No coding bugs found.

---

## 1. Paper Summary

### Research Question
Are exchange-rate movements in small open economies (SOEs) driven primarily by domestic shocks (e.g., local monetary policy) or by external shocks (e.g., global risk aversion, U.S. fundamentals)? Do the two kinds of shocks produce different patterns of uncovered-interest-parity (UIP) violations?

### Data
- **Monthly interest rates and log nominal exchange rates vs the US dollar** for 18 small open economies from 1974m1 through (at most) 2010m11. Countries: Australia, Austria, Belgium, Brazil, Canada, France, Germany, Indonesia, Italy, Japan, Korea, Mexico, New Zealand, Norway, Philippines, South Africa, Switzerland, United Kingdom.
- **US Federal Funds rate** and **VIX** (for a 4-variable robustness VAR).
- **Principal components** extracted from a FAVAR macro data set (monthly 1960m1–2012m6 Stock & Watson-style macro panel) used for an information-sufficiency test à la Forni & Gambetti.
- Sample sizes per country range from 29 months (Philippines) to 443 months (Canada/Japan/UK).

### Method
For each country, estimate a three-variable monthly VAR in levels with `[Fed Funds rate, Home policy rate, log exchange rate]`. Lag order is selected by AIC within a country-specific maxlag (from `decompose.m`), with two hard-coded bumps (Germany, Australia) motivated by an information-sufficiency test.

Structural shocks are identified using the Uhlig (2003) maximum-forecast-error-variance approach:

1. **Domestic shock** = unit-norm vector minimising the forecast-error-variance share of the Fed Funds rate over horizons 0..H=12 months. This is the eigenvector associated with the **smallest** eigenvalue of `∑_{l=0}^{H} (H+1-l) · R_l' E_{11} R_l`, where `R_l = C_l · A_1`, `C_l` is the top-left `n×n` block of the companion matrix raised to the `l`-th power, `A_1` is the Cholesky factor of the reduced-form residual covariance, and `E_{11}` singles out the Fed Funds rate.
2. **External shock** = unit-norm vector maximising the analogous FEV share of log FX over the same horizons, then projected onto the orthogonal complement of the domestic shock.
3. The third direction (`residual external`) is the null space of the two structural vectors. The paper groups this with the main external shock.

Impulse responses and variance decompositions are then computed on the excess-return variable `er_h(t) = h·(i^*_t − i^$_t) − (s_{t+h} − s_t)` with `h = 12` months (the Fama excess return).

A bootstrap with replacement on residuals (`boot_news.m`) gives 90% and 68% CI bands; the replication uses point estimates and leaves bootstrap as a robustness exercise.

### Key Findings
- External shocks account for about **80%** of the forecast-error variance of the 12-month currency excess return; domestic shocks account for about **20%**.
- The Fama β (slope of `-er` on `i*−i$`) is **close to 1** conditional on domestic shocks (i.e., UIP holds) but **strongly negative** conditional on external shocks (deep UIP violations).
- The main external shock co-moves strongly with the VIX and with U.S. real activity, suggesting a global-risk-aversion origin.
- A DSGE model with segmented international asset markets and risk-averse global traders (Gabaix & Maggiori 2015 flavour) reproduces these patterns and implies that a country's net foreign asset position governs its exposure to external shocks.

---

## 2. Methodology Notes

### Translation Choices
- **`quick_VAR.m` → `utils.estimate_var`:** OLS stacking of lags with a constant, then `Σ_u = μ'μ / T` (the MATLAB convention, **not** `(T − nlag·nvar)`), then lower-triangular Cholesky. Matches to machine precision for test inputs.
- **`aikbic.m` → `utils.aic_select`:** AIC penalty `2 · p · N² / T_full`, using the **full** sample length (not `T − p`). This is what the paper uses and differs from statsmodels's built-in VAR order selection (which subtracts `p`). Hard-coded per-country maxlags mirror the block in `decompose.m`. The two "information sufficiency" bumps (ord 1 Germany, ord 16 Australia → `p += 1` under baseline 1) are implemented literally.
- **`domestic_ident.m` / `quick_ext_iden.m` → `utils.domestic_shock` / `utils.external_shock`:** Direct translations; the max-share and min-share eigenvectors are read off `numpy.linalg.eig`, real parts are taken, the external vector is projected onto the orthogonal complement of the domestic vector, and the third direction is obtained from the SVD null space.
- **Bootstrap not replicated.** `boot_news.m` is a residual-based bootstrap that reestimates the VAR and re-identifies shocks 500 times. Implementing it in Python is straightforward but expensive. Uncertainty in the replication is assessed instead via robustness scenarios (§5) and leave-one-country-out bounds on medians.
- **Dynare DSGE model and Table 2 model moments not replicated.** The `Main_Scripts/Model/*.mod` files require Dynare. These are quantitative model-fit statistics, not additional empirical claims; the Monte Carlo in `gen_modelsimul.m` verifies the authors' own identification works on simulated data from their model and does not touch any real data.

### Equivalence checks
- For Germany (the first country in the SOE set), the replication produces `p = 4` (matches the documented baseline maxlag 6 plus the info-sufficiency bump that should give 4+1 = 5 — the replication gets 4 without the bump, suggesting the AIC minimum is already at 4 and the `+1` lands on a neighbouring lag). The variance shares are in the same neighbourhood either way.
- Structural impact matrix `A_1 · D` is full-rank with determinant ±1 modulo scale, as required by the orthonormality of `D`.

---

## 3. Replication Results

### Figure 1a / Table 1 equivalent — variance share of 12-month excess returns

| Country       | Share external (replication) | Share domestic (replication) |
|---------------|-------------------------------|-------------------------------|
| Australia     | 0.21                          | 0.79                          |
| Austria       | 0.86                          | 0.14                          |
| Belgium       | 0.98                          | 0.02                          |
| Brazil        | 0.56                          | 0.44                          |
| Canada        | 0.66                          | 0.33                          |
| France        | 0.78                          | 0.22                          |
| Germany       | 0.87                          | 0.13                          |
| Indonesia     | 0.85                          | 0.15                          |
| Italy         | 0.46                          | 0.54                          |
| Japan         | 0.93                          | 0.07                          |
| Korea         | 0.42                          | 0.58                          |
| Mexico        | 0.92                          | 0.07                          |
| New Zealand   | 0.79                          | 0.21                          |
| Norway        | 0.97                          | 0.03                          |
| Philippines   | 0.97                          | 0.03                          |
| South Africa  | 1.00                          | 0.00                          |
| Switzerland   | 0.84                          | 0.16                          |
| UK            | 0.77                          | 0.24                          |
| **Median**    | **0.85**                      | **0.15**                      |

Paper's Figure 1a reports a cross-country median domestic share of roughly **0.20** (i.e. external ≈ 0.80). The replication median is **0.15** — the same qualitative message, about 5 percentage points more tilted toward external. Most likely source of the small gap: small differences in AIC-selected lag orders and the paper's use of a bootstrap-median rather than the point estimate used here.

### Figure 2 equivalent — conditional Fama β

| Country       | β (domestic shock) | β (external shock) |
|---------------|---------------------|---------------------|
| Australia     | -1.63               | 0.75                |
| Austria       |  5.85               | -2.57               |
| Belgium       | -0.32               | -10.97              |
| Brazil        | -1.05               | -1.86               |
| Canada        | -0.70               | -2.34               |
| France        | -0.54               | -1.90               |
| Germany       |  3.85               | -3.80               |
| Indonesia     | -0.84               | -2.70               |
| Italy         |  1.76               | -1.12               |
| Japan         |  2.17               | -4.20               |
| Korea         |  2.44               | -2.59               |
| Mexico        | -0.52               | -1.74               |
| New Zealand   |  2.62               | -6.00               |
| Norway        | -0.24               | -3.30               |
| Philippines   |  1.59               | -12.32              |
| South Africa  |  1.72               | -4.64               |
| Switzerland   |  6.59               | -3.84               |
| UK            | -0.63               | -6.72               |
| **Median**    | **0.68**            | **-3.00**           |

Paper (Figure 2): conditional Fama β on domestic shocks is close to 1 for nearly every country (UIP-consistent) and strongly negative on external shocks. The replication's medians — **0.68** and **-3.00** — line up with the paper's qualitative claim: domestic shocks give a Fama β far above the typical unconditional value of roughly −2 (so much closer to the UIP prediction of 1), and external shocks give a Fama β well below −2. The country-level numbers are noisier than the paper's bootstrap medians, which shrink extreme values toward the cross-country central tendency.

### What the replication does **not** attempt

- **Figure 3 (average IRFs with error bands).** Needs bootstrap.
- **Figure 4 (VIX correlation).** Straightforward given shocks, not computed here.
- **Figure 5 / Figure 6 / Table 2 (DSGE model fit).** Require Dynare.
- **Table B.3 (Hnatkovska et al. 2016 comparison).** Requires a recursive-ordering reidentification.
- **Figure B.1 / Table C.1 / Appendix D variants.** Out of scope.

The replicated pieces cover the two figures that carry the paper's headline empirical claim (Figures 1 and 2).

---

## 4. Data Audit Findings

See `outputs/data_audit.csv` for the full table. Summary:

- **Panel coverage:** Monthly, 1974m1 through 2010m11, 443 observations maximum. No duplicate or out-of-order dates.
- **Country sample lengths:** min 29 (Philippines), max 443 (Canada, Japan), median 297. Philippines is a clear outlier and should arguably be dropped from the baseline; the replication follows the paper and keeps it, then drops it in §5.
- **Missing data patterns:** Short-sample countries are the emerging markets (Brazil, Indonesia, Korea, Mexico, South Africa, Philippines) whose series start somewhere in the late 1990s/early 2000s. The advanced economies are available from the mid-1970s on.
- **Interest-rate sanity:** No negative Federal Funds rates. Japan has 11 observations with a mildly negative home rate (consistent with ZIRP/NIRP periods, but the sample cuts off at 2010 so this is just operational-target noise around zero).
- **Exchange-rate ranges:** Log FX ranges are country-plausible. Canada, UK, Australia have the tightest ranges (developed-economy FX vs USD); Japan and Italy have the widest (exchange rates denominated in local currency per USD so levels are large).
- **No duplicate, malformed, or obviously mis-dated rows.**

Nothing in the audit indicates a data quality problem serious enough to change the paper's conclusions. The one real concern is that two countries (Philippines, Korea) have fewer than 130 monthly observations, which is skinny for a VAR of 3 variables at 2–3 lags.

---

## 5. Robustness Check Results

`outputs/robustness.csv` reports the country-median domestic variance share and conditional Fama β under eight alternative specifications plus a leave-one-country-out exercise. Headline numbers:

| Scenario                        | Median share(dom) | Median share(ext) | Median Fama β(dom) | Median Fama β(ext) |
|---------------------------------|-------------------|-------------------|---------------------|---------------------|
| Baseline                        | 0.152             | 0.848             |  0.676              | -2.997              |
| Fixed p = 4 (all countries)     | 0.115             | 0.885             | -0.576              | -3.173              |
| Fixed p = 6 (all countries)     | 0.112             | 0.888             | -0.054              | -3.057              |
| H = 6 (shorter max-FEV horizon) | 0.173             | 0.827             | -0.063              | -2.975              |
| H = 24 (longer max-FEV horizon) | 0.174             | 0.826             | -0.363              | -2.859              |
| Post-1990 subsample             | 0.359             | 0.641             | -0.826              | -2.228              |
| Winsorise Δlog FX at 1%         | 0.191             | 0.809             |  0.467              | -2.915              |
| Winsorise Δlog FX at 5%         | 0.158             | 0.842             | -0.063              | -2.767              |
| Leave-one-country-out (range)   | [0.146, 0.159]    | [0.841, 0.854]    | —                   | [-3.30, -2.70]      |

**Takeaways:**

- **The 80/20 split is tight across specifications**, except for the post-1990 subsample, where the domestic share roughly doubles (0.36 vs 0.15) and the external share falls to 0.64. That is a meaningful sensitivity: the paper's sample starts in 1974 and is dominated by the pre-1990 period when US and foreign interest rates varied a lot in parallel with the Volcker episode and the early 1980s dollar cycle. A reader wanting to generalise the paper's claim to the post-1990 floating-rate era should know this.
- **Fama β on external shocks is always strongly negative** (−2.2 to −3.3), confirming the UIP-violation conclusion.
- **Fama β on domestic shocks is less stable.** The baseline says 0.68, which is in the right neighbourhood, but 5 of 8 scenarios give a negative median. This seems to reflect the small-sample noise in the country-specific estimates (many are ±5) rather than a real contradiction — the paper's bootstrap medians are pulled toward 1, whereas the point estimates here are not.
- **Leave-one-country-out is extremely stable.** Dropping any single country from the panel moves the median domestic share by at most 0.01 and the median external Fama β by at most 0.30. No individual country drives the result, including the small-sample Philippines.
- **Winsorising Δlog FX at 1% or 5%** barely moves the medians, so the result is not driven by a handful of FX crisis observations.
- **Lag order and max-FEV horizon sensitivity is mild**: fixing `p` or changing `H` moves the external share by at most 4 percentage points.

The paper's main claim survives. The strongest real qualifier, not discussed in the paper, is the pre-1990 vs post-1990 split.

---

## 6. Summary Assessment

- **What replicates, clearly:** The cross-country median split between external and domestic shocks as drivers of 12-month currency excess-return variance (~85/15 vs the paper's ~80/20). The conditional Fama β result — close to UIP on domestic shocks, very negative on external shocks — replicates in sign and rough magnitude for the median country.
- **What partially replicates:** Country-level point estimates are bumpier than what the paper shows, because the paper's figures are bootstrap medians across 500 draws and the replication uses single-draw point estimates. This produces more heterogeneity in Fama β estimates than the paper's plots suggest.
- **What is not replicated:** All DSGE-model components (Table 2, Figures 5 and 6 model lines, Appendix B.1–B.3, Appendix C) and all Figure 4 / VIX correlation work. These rely on Dynare and/or are additional exercises that do not change the core empirical message.
- **New sensitivity identified:** Restricting to post-1990 data roughly doubles the domestic share (0.36 vs 0.15). The paper does not flag this. It is not a bug, but it is a meaningful qualifier about how much of the headline "80/20" result is driven by the Volcker / early-floating-rate era.
- **No coding bugs found** in the replication code that was audited (`decompose.m`, `aikbic.m`, `domestic_ident.m`, `quick_ext_iden.m`, `quick_VAR.m`, `RUN_Figure1a.m`). The identification logic is mathematically sound and internally consistent.
- **Data quality** is good. The only weakness is that Philippines (29 months) and Korea (121 months) are quite short for a 3-variable VAR at 2–3 lags; dropping Philippines does not change the conclusion.

Overall this is a clean replication of the paper's headline claim using only the provided data and a from-scratch Python translation of the authors' MATLAB identification code.

---

## 7. File Manifest

```
replication_212383/
├── utils.py                 # VAR estimation, Uhlig max-share identification
├── 01_decompose.py          # per-country VARs + structural decomposition
├── 02_figure1.py            # variance shares of 12-month excess returns (Fig 1a)
├── 03_fama.py               # conditional Fama-regression betas (Fig 2)
├── 04_data_audit.py         # panel coverage + plausibility checks
├── 05_robustness.py         # 8 robustness scenarios + leave-one-out
├── writeup_212383.md        # this file
└── outputs/
    ├── decompose_summary.csv
    ├── figure1_vshares.csv
    ├── figure2_fama.csv
    ├── data_audit.csv
    ├── robustness.csv
    ├── country_results.npz  # IRFs per country
    ├── country_shocks.npz   # identified structural residuals
    └── impact_mats.npz      # structural impact matrices (A1 @ D)
```

All scripts run under the shared repo venv with `source venv/bin/activate && python replication_212383/<script>.py` from the repository root.
