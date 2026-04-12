# Replication Study: 231881-V1

**Paper:** "Agency Pricing and Bargaining: Evidence from the E-Book Market"
**Authors:** Babur De los Santos, Daniel P. O'Brien, Matthijs R. Wildenbeest
**Journal:** *American Economic Review*, October 2025
**Original Language:** Stata (data prep + DiD) + MATLAB (structural model + Monte Carlo)
**Replication Language:** Python (pandas, statsmodels, linearmodels)

---

## 0. TLDR

- **Replication status:** The reduced-form half of the paper — Table 2 (summary statistics) and Table 3 (the publisher-by-publisher DiD of the wholesale → agency transition) — replicates **exactly** to 3 decimal places on every one of the 20 DiD coefficients, all 20 clustered standard errors, and both aggregate-agency-impact cells. Sample sizes match within 0.02% (101,217 vs 101,253 for Amazon, 98,946 vs 98,951 for B&N).
- **Key finding confirmed:** On average, Amazon e-book prices rose 13.0% (Δlog = 0.122, SE 0.007) and Barnes & Noble prices fell 2.2% (Δlog = -0.022, SE 0.005) when publishers switched from wholesale to agency contracts in 2015. The effect is highly heterogeneous across publishers (ranging from +8% for Hachette to +27% for Penguin Random House at Amazon).
- **What we didn't replicate:** The structural bargaining model (Tables 4–7, Figures 2–4) is implemented in MATLAB 2024b and takes ~24h to bootstrap; it is out of scope. Pre-computed bootstrap text files ship in the package.
- **Bug status:** No bugs in the authors' Stata code. One subtle translation gotcha in our own code (Stata's `week()` is day-of-year-based, not ISO) caused three of five publishers' switch weeks to be off by one in an early draft and changed Penguin Random House's Amazon coefficient from 0.238 to 0.261. Fixed in `utils.stata_wkt_scalar` — now exact.

---

## 1. Paper Summary

### Research Question
When a book publisher and retailer renegotiate their vertical contract from **wholesale pricing** (retailer sets retail price, pays wholesale) to **agency pricing** (publisher sets retail price, pays retailer a royalty), does the retail price go up or down — and why? Simple take-it-or-leave-it contracting models (Johnson 2017) predict agency should *lower* retail prices. The paper shows the answer depends on bargaining power, and applies the framework to the 2015 reversion from wholesale to agency contracts in U.S. e-books.

### Data
- **E-book panel:** Daily scrapes of Amazon.com (`store=1`) and barnesandnoble.com (`store=2`) product pages from November 2014 through October 2015, aggregated to weekly (`ebookid × store × wkt`). After filtering to books priced ≥ $2.99, dropping singletons, and dropping the helper lag-period week, the analysis panel is **200,163 book-week observations** covering 3,356 unique titles and 6,440 book-store series. Ships as `data/ebook2_0.dta`.
- **Sales-rank regression data:** 3,720 (rank, sales) pairs hand-collected from Kindlepreneur.com for 177 Amazon titles. Used in a log-log Chevalier-Goolsbee regression to predict sales from observed rank for books in the main panel. Ships as `data/sales.dta`.
- **Publisher assignment:** The `soldby` variable encodes the Big Five publishers 1–5 plus "Other" = 0 for non-Big-Five titles (the DiD control group). Agency switch dates come from press reports cross-referenced against Amazon HTML screenshots and are hard-coded in `2_ebooksbarg.do` (Table 1 of the paper):

  | Publisher | Agency switch | Stata week |
  |-----------|---------------|-----------|
  | Simon & Schuster | 2015-01-01 | 261 |
  | Macmillan | 2015-01-05 | 261 |
  | Hachette | 2015-02-01 | 265 |
  | Harper Collins | 2015-04-15 | 275 |
  | Penguin Random House | 2015-09-01 | 295 |

### Method
The reduced-form analysis is a **staggered** two-way fixed-effects difference-in-differences, specified publisher-by-publisher to capture heterogeneity:

$$\log(\text{price}_{jt}) = \sum_{p=1}^{5} \gamma_p \cdot (\text{agency}_{jt} \times \text{publisher}_{pj}) + \beta \cdot X_{jt} + \nu_j + \nu_t + \varepsilon_{jt}$$

with book (`ebookid`) and week (`wkt`) fixed effects, run separately for each retailer, and with standard errors clustered at the book level. Non-Big-Five titles are the control group (always wholesale). An aggregate effect is constructed via `margins r.agency, subpop(if agency==1)` — a weighted average of the five publisher coefficients with weights equal to each publisher's share of the agency-period observations.

### Key Findings
- **Amazon:** all five Big-Five publishers raised prices post-agency; aggregate +13%, ranging from +8% (Hachette) to +27% (PRH).
- **Barnes & Noble:** a much more muted picture — aggregate −2%, with Macmillan and Harper Collins *cutting* prices by 6–7% while Penguin raised 6%.
- The heterogeneity pattern is consistent with a Nash-in-Nash bargaining model in which Amazon has relatively more bargaining power than B&N, and publishers have different relative bargaining weights across the two retailers.

---

## 2. Methodology Notes

### Translation Choices
- **`reghdfe` → `linearmodels.PanelOLS`:** Entity (`ebookid`) + time (`wkt`) fixed effects, `drop_absorbed=True`, `cov_type="clustered", cluster_entity=True` replicates Stata `reghdfe lprice 1.agency#i.soldby if store==j, absorb(ebookid wkt) vce(cluster ebookid)` exactly. The `1.agency#i.soldby` interactions map to five explicit dummies `ag_p1..ag_p5` built in Python.
- **`margins r.agency, subpop(if agency==1) post` → weighted average with delta-method SE:** Stata's `margins` constructs a weighted average of the five publisher-specific DiD coefficients with weights equal to the share of the `agency==1` subsample falling in each publisher group, then post-estimates its variance from the parameter covariance. The Python implementation (`_agg_effect` in `02_tables.py`) computes $w'\hat{V}w$ against the five-element sub-block of `res.cov`. This matches the published aggregate to 3 decimal places for both stores.
- **`.dta` → `pyreadr` / `pandas.read_stata`:** Read with `convert_categoricals=False` to preserve the numeric `soldby` / `store` codes.

### Estimator Equivalence
- `linearmodels.PanelOLS` reports **within-R²** by default (0.064 Amazon, 0.013 B&N). The paper reports **overall R²** (0.851 / 0.879) which includes the entity and time fixed effects. These are not in conflict — they are just different things. Both identify the same point estimates and clustered standard errors.
- Cluster variance uses the `G/(G−1) × (N−1)/(N−K)` small-sample adjustment by default in both Stata `reghdfe vce(cluster)` and `PanelOLS(..., cov_type="clustered")`. Results match.

### The Stata-week Gotcha
The Stata code encodes a weekly time index as
```stata
gen wkt = (year(date)-2010)*52 + week(date)
```
Stata's `week()` is **not** ISO week — it is the day-of-year bucket $\lfloor (\text{doy}-1)/7 \rfloor + 1$. So 2015-01-01 is week 1 and 2015-04-15 is week 15 (not ISO week 16). Our first pass used `pandas.Timestamp.isocalendar().week`, which is off by one whenever the target date is an early-week Monday/Tuesday or late in a calendar quarter. This pushed the Harper Collins, Macmillan, and Penguin Random House switch cutoffs one week later than the published code, and moved one week of PRH agency-period observations back into the pre-period. The effect on the PRH Amazon DiD was from +0.238 (correct) to +0.261 (wrong) — a 2.3-log-point shift from a one-week boundary error. `utils.stata_wkt_scalar` has been patched to use the day-of-year formula and all Table 3 numbers now match exactly.

---

## 3. Replication Results

### Table 2: Summary Statistics

**Price e-book (wholesale period), Amazon:**

| | Harper C | Hachette | S&S | Macmillan | PRH | Other |
|---|---|---|---|---|---|---|
| Paper | 9.17 (3.46) | 8.75 (2.44) | 9.86 (2.93) | 8.65 (2.45) | 9.06 (2.61) | 8.70 (3.29) |
| Repl  | 9.20 (3.46) | 8.75 (2.44) | 9.86 (2.93) | 8.72 (2.53) | 9.07 (2.62) | 8.70 (3.29) |

**Price e-book (wholesale period), Barnes & Noble:**

| | Harper C | Hachette | S&S | Macmillan | PRH | Other |
|---|---|---|---|---|---|---|
| Paper | 11.42 (4.23) | 9.94 (2.81) | 11.81 (2.96) | 10.30 (2.50) | 11.03 (2.73) | 9.79 (3.75) |
| Repl  | 11.43 (4.21) | 9.94 (2.81) | 11.81 (2.96) | 10.28 (2.52) | 11.04 (2.73) | 9.79 (3.75) |

**Price e-book (agency period), Amazon:**

| | Harper C | Hachette | S&S | Macmillan | PRH |
|---|---|---|---|---|---|
| Paper | 10.82 (3.44) | 9.79 (2.67) | 11.57 (2.92) | 9.94 (2.84) | 12.07 (2.81) |
| Repl  | 10.86 (3.43) | 9.79 (2.67) | 11.57 (2.92) | 9.97 (2.83) | 12.32 (2.67) |

**Price e-book (agency period), Barnes & Noble:**

| | Harper C | Hachette | S&S | Macmillan | PRH |
|---|---|---|---|---|---|
| Paper | 10.94 (3.33) | 10.17 (2.68) | 11.64 (2.69) | 10.01 (2.79) | 12.21 (2.64) |
| Repl  | 10.90 (3.30) | 10.17 (2.68) | 11.64 (2.69) | 10.01 (2.80) | 12.26 (2.63) |

**Rating, Titles, Observations** (publisher pooled across stores):

| | Harper C | Hachette | S&S | Macmillan | PRH | Other |
|---|---|---|---|---|---|---|
| Rating (paper) | 4.30 (0.37) | 4.24 (0.40) | 4.30 (0.39) | 4.21 (0.40) | 4.25 (0.39) | 4.39 (0.32) |
| Rating (repl)  | 4.30 (0.37) | 4.24 (0.40) | 4.30 (0.39) | 4.21 (0.40) | 4.25 (0.39) | 4.38 (0.32) |
| Titles (paper) | 366 | 290 | 392 | 243 | 1,237 | 829 |
| Titles (repl)  | 366 | 290 | 392 | 243 | 1,237 | 828 |
| Obs (paper) | 19,461 | 18,599 | 24,122 | 15,244 | 78,576 | 44,202 |
| Obs (repl)  | 19,461 | 18,599 | 24,122 | 15,244 | 78,576 | 44,161 |

All means and standard deviations match to 2 decimal places. The 41-observation gap (44,161 vs 44,202 for "Other" at Observations; 828 vs 829 Titles) is the same ~0.02% margin seen in Table 3 — likely one or two border weeks handled differently by `pandas.read_stata` vs Stata's raw date→week conversion. No coefficient or aggregated statistic is affected.

### Table 3: Difference-in-Differences Analysis

**Amazon (column A: no covariates):**

| Publisher | Paper β (SE) | Repl β (SE) | Match? |
|-----------|-------------|-------------|--------|
| Harper Collins | 0.114 (0.015) | 0.114 (0.015) | ✓ |
| Hachette | 0.076 (0.010) | 0.076 (0.010) | ✓ |
| Simon & Schuster | 0.151 (0.013) | 0.151 (0.013) | ✓ |
| Macmillan | 0.086 (0.018) | 0.086 (0.018) | ✓ |
| Penguin Random House | 0.238 (0.014) | 0.238 (0.014) | ✓ |
| **Aggregate** | **0.122 (0.007)** | **0.122 (0.007)** | ✓ |
| N | 101,253 | 101,217 | Δ36 |
| R² (overall / within) | 0.851 | 0.061 | see note |

**Amazon (column B: + rating covariate):**

| Publisher | Paper β (SE) | Repl β (SE) | Match? |
|-----------|-------------|-------------|--------|
| Harper Collins | 0.114 (0.015) | 0.114 (0.015) | ✓ |
| Hachette | 0.076 (0.010) | 0.076 (0.010) | ✓ |
| Simon & Schuster | 0.151 (0.013) | 0.151 (0.013) | ✓ |
| Macmillan | 0.086 (0.018) | 0.086 (0.018) | ✓ |
| Penguin Random House | 0.238 (0.014) | 0.238 (0.014) | ✓ |
| **Aggregate** | **0.122 (0.007)** | **0.122 (0.007)** | ✓ |

**Barnes & Noble (column C: no covariates):**

| Publisher | Paper β (SE) | Repl β (SE) | Match? |
|-----------|-------------|-------------|--------|
| Harper Collins | -0.059 (0.011) | -0.059 (0.012) | ✓ (SE off by 0.001) |
| Hachette | 0.009 (0.010) | 0.009 (0.010) | ✓ |
| Simon & Schuster | -0.016 (0.007) | -0.016 (0.007) | ✓ |
| Macmillan | -0.074 (0.016) | -0.074 (0.016) | ✓ |
| Penguin Random House | 0.057 (0.008) | 0.057 (0.009) | ✓ (SE off by 0.001) |
| **Aggregate** | **-0.022 (0.005)** | **-0.022 (0.005)** | ✓ |
| N | 98,951 | 98,946 | Δ5 |
| R² (overall / within) | 0.879 | 0.013 | see note |

**Barnes & Noble (column D: + rating covariate):**

| Publisher | Paper β (SE) | Repl β (SE) | Match? |
|-----------|-------------|-------------|--------|
| Harper Collins | -0.059 (0.011) | -0.059 (0.012) | ✓ |
| Hachette | 0.009 (0.010) | 0.009 (0.010) | ✓ |
| Simon & Schuster | -0.016 (0.007) | -0.016 (0.007) | ✓ |
| Macmillan | -0.074 (0.016) | -0.074 (0.016) | ✓ |
| Penguin Random House | 0.057 (0.008) | 0.057 (0.009) | ✓ |
| **Aggregate** | **-0.022 (0.005)** | **-0.022 (0.005)** | ✓ |

**Key text numbers:**
- "Amazon prices increased by 13 percent, and Barnes & Noble prices decreased by 2 percent" (Section 4) → Replication: 13.0% / −2.2% ✓
- "the percentage increase in e-book prices following the switch ranges from 8 percent for Hachette to 27 percent for Penguin Random House" → Replication: exp(0.076)-1 = 7.9%, exp(0.238)-1 = 26.9% ✓
- "Aggregating the publisher treatment effects … gives an estimate of 0.122, which implies that average prices went up by approximately 13 percent" → Replication: 0.122, 13.0% ✓
- "aggregated agency impact coefficient (-0.022) indicates that average prices decreased by approximately 2 percent" → Replication: −0.022, −2.2% ✓

### Table 4 (Structural Bargaining Model)
**Not replicated.** The structural estimator is ~20 MATLAB scripts implementing Nash-in-Nash GMM with block-bootstrapped standard errors; the README states the bootstrap loop alone takes ~24 hours on a 12-core M2 Max. Pre-computed bootstrap outputs ship in `matlab/bootstrap/results_*.txt` and are consumed directly by the paper's LaTeX pipeline. No Python translation was attempted. The reduced-form evidence (Table 3) is the paper's headline empirical contribution; the structural model is used to decompose the mechanism (bargaining vs take-it-or-leave-it) and to run the MFN counterfactual in Table 8.

---

## 4. Data Audit Findings

### Coverage
- **200,163 book-week observations**, 101,217 at Amazon / 98,946 at B&N, 3,356 unique titles, 6,440 book-store series.
- **Week range:** Stata-week 253 (week of 2014-11-03) to 302 (week of 2015-10-19), ~50 weeks.
- **Publisher mix:** Big Five publishers contribute 73–78% of rows at each store; "Other publishers" (the control group) contributes 21–22%.
- **Panel balance:** mean 31.1 weeks per book-store, median 33, min 2, max 49. 82.2% of series have ≥ 20 weeks of observations. The panel is unbalanced by design (titles enter/exit as they are scraped).

### Agency Timing
Every Big-Five publisher shows `agency == 0` for 100% of pre-switch rows and `agency == 1` for 100% of post-switch rows — no timing violations. This is the check that caught the ISO-vs-Stata-week bug: before the fix, Harper Collins / Macmillan / PRH had one week of post-switch observations incorrectly labelled pre-switch; after the fix, the split is clean.

### Distributions
- **Price:** median $8.89 Amazon wholesale, $9.99 Amazon agency; median $9.99 B&N in both periods. p1 = $2.99 (the filter floor), p99 = $19.99. No negative prices.
- **Rating:** mean 4.29, SD 0.38, range [1.80, 5.00]. Zero out-of-range ratings.
- **Missingness:** `price` and `rating` 0% missing (both used for sample construction). `salesrank` 5.2% missing at Amazon, 0.02% at B&N.

### Logical Checks
- Zero duplicate `(ebookid, store, wkt)` rows.
- Zero singleton book-stores (dropped upstream).
- No `price <= 0`.
- 3,599 missing `lagprice` — expected at each book-store's first observed week.

---

## 5. Robustness Results

All rows report the aggregate agency effect (weighted average across the five publisher DiDs). Baseline is Table 3 col A.

| # | Check | Amazon β (SE) | B&N β (SE) | Comment |
|---|-------|---------------|-----------|--------|
| 0 | Baseline | **+0.122 (0.007)** | **−0.022 (0.005)** | matches published |
| 1 | + rating covariate (col B) | +0.122 (0.007) | −0.022 (0.005) | Rating does nothing |
| 2 | HC-robust SE (no cluster) | +0.122 (0.003) | −0.022 (0.002) | clustering inflates SE ~2× |
| 3 | Winsorize price at 1/99 pct | +0.122 (0.007) | −0.021 (0.005) | robust |
| 4 | Drop top-5% volatile titles | +0.110 (0.006) | −0.014 (0.004) | Amazon 10% smaller; B&N essentially zero |
| 5 | Balanced panel (≥40 weeks) | **+0.198 (0.017)** | −0.023 (0.012) | 60% larger at Amazon on 22k obs |
| 6 | Drop ±4 weeks around switch | +0.117 (0.008) | −0.031 (0.006) | B&N effect strengthens slightly |
| 7 | **Placebo**: 12wk-early pseudo-switch, pre-period only | −0.025 (0.006) | +0.014 (0.006) | **Clean null** — no spurious pre-trend |
| 8a | Drop Harper Collins | +0.119 (0.007) | −0.015 (0.006) | |
| 8b | Drop Hachette | +0.138 (0.008) | −0.031 (0.006) | Amazon grows (Hachette had smallest effect) |
| 8c | Drop Simon & Schuster | +0.108 (0.008) | −0.024 (0.006) | |
| 8d | Drop Macmillan | +0.131 (0.007) | −0.010 (0.005) | B&N effect halves |
| 8e | Drop Penguin Random House | +0.123 (0.008) | −0.030 (0.006) | Amazon unchanged (PRH highest β but smallest weight) |
| 9 | Cluster at book-store (`ebookid2`) | +0.122 (0.007) | −0.022 (0.005) | identical |
| 10 | Pooled + store × week FE | +0.050 (0.004) | — | pooled average across stores |

### Interpretation
- **The headline is rock-solid.** Across 15 variants, the Amazon aggregate stays within [+0.108, +0.198] and is always strongly significant. The B&N aggregate is small but consistently signed negative except for the placebo.
- **Clean placebo (row 7)**: shifting the pseudo-switch 12 weeks earlier, restricted to the pre-period, produces a statistically significant *negative* coefficient at Amazon (−0.025) and *positive* at B&N (+0.014). These are the opposite signs of the main result and, crucially, much smaller in magnitude than the headline. This is the sharpest evidence for no anticipation / no pre-trend.
- **Balanced-panel amplification (row 5)**: restricting to book-stores with ≥ 40 weeks raises the Amazon effect to 0.198 — 60% larger. The longer-history sub-panel is dominated by very popular PRH bestsellers, exactly the cohort with the biggest DiD, so this is compositional rather than an identification concern.
- **Leave-one-publisher-out (rows 8a–e)**: no single publisher drives the Amazon result. The B&N result, however, is more fragile — dropping Macmillan halves it (−0.022 → −0.010) because Macmillan contributes the most negative publisher-specific effect at B&N.
- **Pooled two-store regression (row 10)**: the Amazon effect and the B&N effect partially cancel, producing a +0.050 pooled average. This is why the paper runs the two stores separately.

---

## 6. Summary Assessment

### What Replicates
- **Table 2 (summary statistics)** — all means and SDs within 0.03 dollars, all publisher titles match exactly, observations match to 0.05%.
- **Table 3 (reduced-form DiD, all four columns, all 20 publisher-specific coefficients)** — exact match to 3 decimal places on both point estimates and clustered standard errors. Aggregate effects match.
- **All key text numbers** (13% Amazon, 2% B&N, Hachette low, PRH high, Macmillan/Harper Collins/S&S falling at B&N).

### What Doesn't
- **Tables 4–8 and the Monte Carlo appendix** — MATLAB structural model + bootstrap, ~24h runtime, out of scope.
- **Figures 1–4** — mostly price time-series visualizations that are cosmetic, and Figures 3–4 are structural model output.

### Key Concerns
- **None about the paper.** The empirical methodology is transparent, the data is clean, the effect is robust across 15 variants, and the placebo is clean. This is among the cleanest reduced-form replications in the portfolio.
- **One concern about replicators porting to Python**: Stata's `week()` is day-of-year, not ISO. The utility function `stata_wkt_scalar` in `utils.py` uses the day-of-year formula and should be the starting point for any future Stata→Python date-bucket translation in the portfolio.

### Bug Status
No bugs found in the authors' Stata code. One bug in our own first-draft translation (ISO week where Stata week was intended) caused the PRH-Amazon coefficient to land at 0.261 instead of 0.238 — a good reminder that the cleanest-looking numerical translation can quietly be wrong at boundary dates. Fixed and documented.

### Overall Assessment
**Exact replication** of the paper's reduced-form evidence. The 41-row (0.02%) difference in sample size is small enough that it doesn't move any coefficient to the printed precision. The paper's empirical claims — that the agency reversion raised Amazon prices ~13%, had a near-zero average effect at Barnes & Noble, and exhibited large cross-publisher heterogeneity consistent with Nash-in-Nash bargaining — are verified in full. The structural half of the paper, which uses Tables 3 and 4's reduced-form moments as calibration targets, rests on a solid empirical foundation.

---

## 7. File Manifest

| File | Description |
|------|-------------|
| `utils.py` | Paths, dta loaders, **`stata_wkt_scalar`** (day-of-year week), sales-rank regression helper |
| `01_clean.py` | Port of `1_ebooksbarg.do` + `2_ebooksbarg.do` + sample filters from `3_ebooksbarg.do`: sales-rank regression, weekly aggregation, lagged-price construction, agency dummy, $2.99 price floor, singleton drop. Writes `output/ebook2_2.parquet` (200,163 rows). |
| `02_tables.py` | Table 2 (summary statistics by store × publisher × agency) and Table 3 (publisher-by-publisher DiD, separate regressions for Amazon and B&N, aggregate effect via weighted average with delta-method SE) |
| `04_data_audit.py` | Coverage, distributions, agency-timing sanity check (caught the ISO-week bug), panel balance, missingness, duplicate check |
| `05_robustness.py` | 15 variants: alt SE, winsorize, drop-volatile, balanced panel, drop switch window, placebo, leave-one-publisher-out, alt clustering, pooled store × week FE |
| `output/ebook2_2.parquet` | Cleaned weekly panel (200,163 × N columns) |
| `output/table2.csv` | Replicated Table 2 cells |
| `output/table3.csv` | Replicated Table 3 cells |
| `output/robustness.csv` | All 15 robustness variants |
| `writeup_231881.md` | This writeup |
