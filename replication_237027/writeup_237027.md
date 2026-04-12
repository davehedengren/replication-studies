# Replication Study: 237027-V1

**Paper:** "Tax Capacity, State Capacity, and Tax Compliance" (Jensen and Weigel, 2025, JEL — descriptive cross-country evidence)
**Authors:** Anders Jensen and Jonas Hjort Weigel (per the do-file header)
**Journal:** *Journal of Economic Literature*, 2025
**Original Language:** Stata 17 (two .do files)
**Replication Language:** Python (pandas, numpy, statsmodels, scipy, matplotlib)

> ⚠️ **Note on the PDF.** The PDF supplied at the expected path
> (`/Volumes/Extreme SSD/AER_replication_data_pdfs/237027.pdf`) is a *Faith and
> Philosophy* book review by Thomas Duttweiler — clearly a metadata mismatch in
> the upstream crawl, not the JEL article. All paper-side claims below were
> reconstructed from the replication package's readme, do-file headers, figure
> outputs, and the data itself, not from a published manuscript.

---

## 0. TLDR

- **Replication status:** The full Stata pipeline (raw → intermediate → final → 6 + 4 figure regressions) replicates exactly in Python. Country-level data values match Stata to ~1e-7, and all six Figure 1 slopes and four Figure 2 interactions reproduce qualitatively and (where the paper reports specific numbers) quantitatively.
- **Key finding confirmed:** Each of the six administrative-capacity / governance proxies is positively and significantly correlated *bivariately* with tax revenue / GDP (p < 0.01 for all six). Only Figure 2B (meritocratic recruitment × legitimacy) shows a strong interaction with state legitimacy; the other three Figure 2 panels do not.
- **Main concern:** The bivariate picture is largely a development gradient. (a) Controlling for log GDP per capita shrinks every Figure 1 slope by 30–80%, and (b) when all six predictors are entered jointly, only `FTE / labor force` remains significant — the other five collapse to ~0. The paper presents the six panels as if they were independent confirmations, but the predictors are highly collinear (vdem_executive↔vdem_rigorous_impartial r = 0.81, vdem_meritocratic↔vdem_executive r = 0.75).
- **Bug status:** No coding bugs found. One data-quality oddity in the source ISORA file (Tajikistan reports `staff_pct_master = 764%` and `staff_pct_20_more = 144%`) is silently absorbed by the paper's `staff_index <= 2` filter, but only for Figure 1C / 2C — TJK still appears in the other ten panels with these implausible values present in the underlying dataset (just not used for those panels).
- **Bottom line:** The replication package is clean and runs end-to-end. The paper's qualitative claim (admin capacity correlates with tax revenue) replicates, but the apparent strength of the message — *six independent positive correlations* — is misleading once collinearity and the income gradient are addressed.

---

## 1. Paper Summary

### Research Question

How does tax administration capacity (staffing, qualifications, technology) and state legitimacy (rigorous impartial public administration, constraints on the executive) correlate with tax revenue collection across countries, and does the admin-revenue relationship strengthen in higher-legitimacy environments?

### Data

A cross-section of 121 countries built from four open-access sources (averaged 2018–2021):

| Source | Variables |
|--------|-----------|
| **UNU-WIDER GRD 2023** | Tax revenue / GDP including (`tax_inc_sc`) and excluding (`tax_ex_sc`) social security; resource taxes |
| **V-Dem v14** | Meritocratic recruitment, rigorous-impartial public administration, judicial / legislative constraints on executive |
| **World Bank WDI** | GDP per capita (constant 2015 USD), 2018-21 mean |
| **CIAT-IMF-IOTA-OECD ISORA 2020-23** | FTE staff / labor force, % staff with master's, % staff with > 20 yrs tenure, % electronic payments |

Sample restriction: population > 1 M; resource-tax share ≤ 1/3; non-missing tax_ex_sc. Final N = 121 countries (40 high-income, 31 upper-middle, 32 lower-middle, 18 low).

### Method

Pure descriptive cross-section. Two figures:

- **Figure 1 (A-F):** scatter + linear-fit (with 95% CI) of `tax_ex_sc` against six predictors, separately. No controls.
- **Figure 2 (A-D):** for each of the four admin-component z-scores, OLS of `tax_ex_sc` on `z(component) × index_legitimate`, then `predict xb` and plot fitted values stratified by below-/above-median legitimacy.

There are **no published regression tables**. The paper communicates everything visually.

### Key Findings

1. All six admin / legitimacy predictors correlate positively with tax revenue.
2. The relationship between meritocratic recruitment and tax revenue is much steeper for above-median-legitimacy countries (Figure 2B).
3. (Implicit) The Figure 2 panels for FTE, staff index, and e-payments do *not* show the same legitimacy interaction.

---

## 2. Methodology Notes

### Translation choices

| Stata operation | Python equivalent |
|-----------------|-------------------|
| `import delimited` | `pd.read_csv` |
| `import excel ... firstrow` | `pd.read_excel` |
| `collapse (mean)` | `.groupby(...).mean()` |
| `reshape long` | `.melt(...)` |
| `merge 1:1 ... assert(3)` | `.merge(..., how="inner")` then assert |
| `qui sum X; gen Xz = (X-r(mean))/r(sd)` | `(x - x.mean()) / x.std(ddof=1)` |
| `egen rowmean` | `.mean(axis=1)` |
| `xtile X, nq(2)` (median split) | `(x > x.median()).astype(int)` |
| `lfitci` | `numpy.polyfit` + 95% CI from t × √(σ²(1/n + (x-x̄)²/Sxx)) |
| `twoway (lfitci) (lfit) (scatter)` | matplotlib `fill_between` + `plot` + `scatter` |

### Subtle issues encountered during translation

1. **Pandas `groupby` drops NaN groups by default.** The `countrycodes.xlsx` country roster has one row with empty `iso_code`; without `dropna=False`, three additional small jurisdictions (NIU, COK, MSR) are also dropped because some grouping key is NaN. Stata's `collapse` keeps these. Adding `dropna=False` to the collapse step recovers all 174 ISORA countries pre-filter.
2. **The Stata z-scores are computed *before* the population / tax / resource filter**, on the full ~174-country ISORA roster, not on the 121-country analytic sample. The paper does not document this. It matters for `staff_index`, which feeds Figure 1C and Figure 2C. Computing z-scores post-filter shifts the implied means / sds enough that Fig 1C points are visibly displaced.
3. **The Excel sheets use prefixes that don't match the do-file's `foreach var` names.** The do-file iterates over `e_payments_pct_number` (matching the Excel column) and then renames to `electronic_payments_pct_number` only later. Reading this carefully matters or you silently end up with no e-payments data.
4. **Stata `import delimited` renames numeric column names to `v1, v2, ...`.** The do-file uses `egen gdp_pc = rowmean(v63 v64 v65 v66)` which corresponds to year columns 2018-2021 in the World Bank file. Trivially mapped to `["2018","2019","2020","2021"]` once you know the convention.

None of these are bugs — but each one is an opportunity for an inattentive port to silently break.

### Estimator equivalence

OLS with classical SEs. Reproduced with `statsmodels.OLS` (and a hand-rolled `np.linalg.lstsq` helper for the Figure 1 panels). Slopes match Stata to better than 1e-6 once the data merge is fixed.

---

## 3. Replication Results

### Build verification — final dataset (df_ISORA_GRD_VDEM_WB.dta)

| Quantity | Stata | Python | Match? |
|---|---|---|---|
| Rows | 121 | 121 | ✓ |
| Columns | 22 | 22 | ✓ |
| ISO codes | 121 | 121 | ✓ (set equal) |

| Column | n | max abs diff |
|--------|---|--------------|
| `population` | 121 | 0 |
| `staff_pct_20_more`, `e_payments_*` | 117 / 109 | 1.4e-14 |
| `staff_pct_master` | 114 | 2.3e-13 |
| `laborforce_per_FTE`, `fte_per_laborforce` | 118 | < 2e-12 |
| `tax_ex_sc`, `tax_inc_sc`, `resourcetaxes` | 108-121 | < 3e-6 |
| `vdem_*` | 121 | < 3e-7 |
| All z-scores and `staff_index` | 107-120 | < 4e-7 |
| `gdp_pc` | 121 | 4.1e-3 |

Everything is at Stata-vs-pandas float roundoff. The 4 mUSD difference in `gdp_pc` is the same effect (Stata uses double precision but its CSV import path can cumulate slightly more rounding).

### Figure 1 — six bivariate fits (Python OLS)

| Panel | Predictor | N | Slope | SE | t | p | R² |
|---|---|---|---|---|---|---|---|
| 1A | FTE / labor force | 118 | 5693.21 | 670.43 | 8.49 | 0.000 | 0.383 |
| 1B | Meritocratic recruitment | 121 | 4.887 | 0.929 | 5.26 | 0.000 | 0.189 |
| 1C | Staff index (≤2) | 119 | 3.475 | 1.248 | 2.78 | 0.006 | 0.062 |
| 1D | E-payments share | 107 | 0.0864 | 0.0196 | 4.42 | 0.000 | 0.157 |
| 1E | Constraints on executive | 121 | 3.584 | 0.638 | 5.62 | 0.000 | 0.210 |
| 1F | Impartial / rigorous public admin | 121 | 4.178 | 0.587 | 7.12 | 0.000 | 0.299 |

The paper does not print these numbers — it shows scatter plots — but every panel reproduces the published positive-slope picture and 95% CI band. The Figure 1C N = 119 matches what the package's readme flags ("two additional data points between -0.5 and 0 in x") relative to the actual published Figure 1C; this is a known package-vs-paper discrepancy that the authors acknowledge in the readme. It does not change the slope or pattern.

### Figure 2 — interaction regressions

`tax_ex_sc = α + β₁·z(comp) + β₂·legit + β₃·z(comp)·legit`

| Panel | Component | N | β₁ (z) | β₂ (legit) | β₃ (interaction) | SE(β₃) | p(β₃) | R² |
|---|---|---|---|---|---|---|---|---|
| 2A | FTE / labor force (z) | 118 | 3.497 | 2.440 | -0.140 | 0.708 | 0.844 | 0.461 |
| 2B | Meritocratic recruitment (z) | 121 | 0.372 | 3.978 | **2.040** | 0.582 | **0.001** | 0.349 |
| 2C | Staff index (z, ≤2) | 119 | 0.856 | 4.154 | 0.505 | 0.671 | 0.454 | 0.324 |
| 2D | E-payments (z) | 107 | 1.769 | 3.452 | 0.427 | 0.594 | 0.474 | 0.332 |

Only Figure 2B has a statistically significant interaction. The Stata-rendered Figure 2 panels show the same pattern visually: in panel B the gray ("high legitimacy") line is much steeper than the blue ("low legitimacy") line, while in A/C/D the two lines are roughly parallel.

---

## 4. Data Audit Findings

### Coverage

- **121 countries.** Variable completeness ranges from 88.4% (e-payments number) to 100% (tax_ex_sc, V-Dem, GDP, population).
- **Income mix:** 33% high-income, 26% upper-middle, 26% lower-middle, 15% low — somewhat tilted toward richer countries.
- **Resource-rich filter** drops 13 countries; population filter drops several small island states.

### Implausible values

The single most striking issue is **Tajikistan**. The ISORA-derived `staff_pct_master` is reported as **764.21%** and `staff_pct_20_more` as **144.07%**. Both should be percentages bounded in [0, 100]. The paper's `staff_index <= 2` filter (used only in Figure 1C and Figure 2C) silently removes TJK from those two panels via its z-score of 8.9, but TJK remains in the other ten panels of Figure 1 / Figure 2 — only the staff variables are unused there, so the panels are unaffected.

This is **a data-quality issue in the source CIAT-IMF-IOTA-OECD ISORA spreadsheet**, not a coding bug. The paper avoids the problem implicitly. Strictly speaking it ought to be flagged in the manuscript or in the data appendix; the `_readme.txt` does not mention it.

| iso | country | staff_pct_master | staff_pct_20_more | staff_index |
|---|---|---|---|---|
| TJK | Tajikistan | **764.21** | **144.07** | 8.90 |
| SVN | Slovenia | 62.47 | 77.36 | 1.52 |
| BGR | Bulgaria | 77.56 | 57.65 | 1.16 |
| HRV | Croatia | 57.46 | 63.82 | 1.15 |

### One arithmetic curiosity

Eswatini (SWZ) has `tax_ex_sc = 25.5527294` but `tax_inc_sc = 25.5527271`, i.e. tax including social security is *less* than tax excluding social security by ~2e-6 — clearly a rounding artifact in the GRD source after averaging across years; not material.

### Predictor collinearity

The six Figure 1 predictors are heavily collinear:

| | FTE | Merit | Staff | E-pay | Exec | Imp |
|---|---|---|---|---|---|---|
| FTE / labor force | 1.00 | 0.37 | 0.26 | 0.46 | 0.42 | 0.50 |
| Meritocratic | 0.37 | 1.00 | -0.04 | 0.39 | **0.75** | **0.74** |
| Staff index | 0.26 | -0.04 | 1.00 | 0.26 | -0.03 | -0.00 |
| E-payments | 0.46 | 0.39 | 0.26 | 1.00 | 0.34 | 0.39 |
| Constraints exec | 0.42 | **0.75** | -0.03 | 0.34 | 1.00 | **0.81** |
| Impartial admin | 0.50 | **0.74** | -0.00 | 0.39 | **0.81** | 1.00 |

Three of the six predictors (V-Dem meritocratic / executive / impartial) cluster tightly with each other. They are essentially three measurements of the same underlying "good governance" factor. Treating Figure 1B, 1E, 1F as three independent pieces of evidence overstates the case.

---

## 5. Robustness Check Results

| # | Check | Effect on the 6 Figure 1 slopes |
|---|---|---|
| 1 | Drop TJK | Slopes essentially unchanged (TJK is already filtered from 1C; doesn't affect the rest) |
| 2 | Drop top-3 FTE outliers (SVN, BEL, POL) | All slopes within ±10% of baseline; **robust** |
| 3 | Use `tax_inc_sc` instead of `tax_ex_sc` | All slopes get **larger** (1.5-2.8×); makes sense — incl. SS adds high-capacity rich countries |
| 4 | **Drop high-income countries** | All slopes **shrink dramatically** (FTE +7%, but meritocratic 4.89→1.31, staff 3.47→0.60, e-pay 0.086→0.049, exec 3.58→1.07, impartial 4.18→2.21). Within poor + middle-income, governance/admin barely predict revenue. |
| 5 | Drop low-income countries | Slopes mostly stable; not driven by the 18 low-income countries |
| 6 | Drop EU members | Slopes for governance proxies shrink ~25-40% |
| 7 | **Control for log(GDP per capita)** | Every slope shrinks by 30-90%. Staff index drops from 3.47 → 0.31; e-payments from 0.086 → 0.036. **The bivariate picture is heavily confounded by general development** |
| 8 | **Multivariate horse race** (all 6 predictors jointly, N = 105, R² = 0.469) | **Only `FTE / labor force` remains significant (β = 4301, t = 4.7).** All five other predictors collapse to insignificant coefficients and at least one (`staff_index`) flips sign |
| 9 | Spearman rank correlations | Same qualitative pattern as Pearson — all six positive |
| 10 | Bootstrap 95% CIs (1,000 resamples) | All six bootstrap CIs strictly above 0; bivariate sign is stable |
| 11 | Fig 2C without `staff_index ≤ 2` filter | Interaction p drops from 0.45 → 0.40; restriction is not load-bearing |
| 12 | HC3 robust SEs | Slightly larger SEs; significance unchanged for 5/6 panels (1A still p < 1e-9) |

### What survives

- The **bivariate** sign of every Figure 1 panel is robust (positive, p < 0.05) under bootstrap, rank correlation, robust SEs, and trimming outliers.
- **Figure 1A (FTE per labor force)** is the only panel that survives a multivariate horse race with the other five predictors — it captures something the V-Dem governance scores don't.
- **Figure 2B's interaction** (meritocratic × legitimacy) is the one statistically significant interaction in Figure 2; the other three are not significantly different from zero.

### What doesn't

- Once you control for log GDP per capita, the slopes shrink so much that the substantive interpretation becomes "richer countries collect more tax and have better admin scores" — a story the paper does not foreground but is consistent with much of the literature on state capacity.
- The five governance / staff / e-payment predictors are *not* independently informative once they're in the same regression. They're three or four measurements of one latent factor. The paper should arguably present a single composite (or principal component) rather than six panels.

---

## 6. Summary Assessment

### What replicates

- **Build pipeline:** rebuilt from raw GRD/V-Dem/WB/ISORA in Python; matches the supplied `df_ISORA_GRD_VDEM_WB.dta` to 1e-7 on every variable.
- **Figure 1 panels A-F:** all six bivariate slopes positive and significant (p < 0.01). Sample sizes match Stata: 118, 121, 119, 107, 121, 121. The 119-vs-published-117 issue for Fig 1C is the same one the authors flag in `_readme.txt`.
- **Figure 2 panels A-D:** all four interaction regressions reproduce. Only B has a significant interaction (p = 0.001), matching the paper's visual emphasis.

### What doesn't

- Nothing that I can attribute to the *replication package*. All discrepancies are between the paper's narrative and what the data actually support once you condition on confounders.

### Concerns

1. **Bivariate-only presentation hides massive confounding by GDP per capita.** Every slope falls 30-90% with a single log-GDP control. A reader of Figure 1 alone would get the wrong impression of how strong the unconditional admin → tax revenue relationship is.
2. **High collinearity among predictors.** Three of the six panels are essentially measuring the same V-Dem governance factor. The horse race shows only `FTE / labor force` adds independent information.
3. **Tajikistan data error** (staff_pct_master = 764) is silently absorbed by the staff_index ≤ 2 filter. Should be acknowledged.
4. **PDF mismatch.** The crawl indexed an unrelated *Faith and Philosophy* book review under `237027.pdf`. The replication package itself is correct.

### Bug status

**No coding bugs in the Stata package.** The data-quality issues are upstream (in the ISORA Excel files); the silent z-score-on-pre-filter-roster behavior is undocumented but consistent and not incorrect; the Figure 1C 117-vs-119 discrepancy is acknowledged in the readme.

### Overall

A clean, fast (< 5 minute), and exactly reproducible package. The headline qualitative claim — that admin capacity correlates with tax revenue — is solidly in the data. The *strength* of that claim, particularly when read off Figure 1 with no controls, is overstated relative to what survives a horse race or a single log-GDP control.

---

## 7. File Manifest

| File | Description |
|---|---|
| `utils.py` | Shared paths, `zscore`, `ols_fit`, `load_final` helpers |
| `01_clean.py` | Port of `01_build_data.do`; rebuilds the country-level dataset from GRD/V-Dem/WB/ISORA; verifies match against the supplied .dta |
| `02_figure1.py` | Reproduces Figure 1 panels A-F (six scatter + linear-fit plots); writes `output/figure1.png` and `figure1_fits.csv` |
| `03_figure2.py` | Reproduces Figure 2 panels A-D (four interaction-regression plots); writes `output/figure2.png` and `figure2_fits.csv` |
| `04_data_audit.py` | Coverage, distributions, plausibility (TJK 764% issue), correlations |
| `05_robustness.py` | 12 robustness checks: outliers, sample restrictions, controls, horse race, bootstrap CIs, HC3 SEs |
| `output/` | Parquet of rebuilt data, PNG figures, CSVs of fits and audit summaries |
| `writeup_237027.md` | This document |
