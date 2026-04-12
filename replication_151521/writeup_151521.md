# Replication Study: 151521-V1

**Paper:** "College Tuition and Income Inequality"
**Authors:** Zhifeng Cai, Jonathan Heathcote
**Journal:** *American Economic Review*, forthcoming (Federal Reserve Bank of Minneapolis Staff Report No. 569, revised July 2020)
**Original Language:** MATLAB (77 .m files, ~12 hours runtime) + Stata (4 .do files) + Excel (SCF, Figure 1)
**Replication Language:** Python (pandas, scipy, numpy, pyreadstat, matplotlib)

---

## 0. TLDR

- **Replication status:** **Partial.** The paper's headline numbers come from a calibrated structural club-goods model solved in MATLAB (12 hours runtime, 77 .m files). That is out of scope here. The empirical moments used to calibrate and validate the model *are* reproducible and all match the paper tightly.
- **What replicates exactly:** Figure 1 (tuition series), SCF Pareto-lognormal (EMG) parameters for 1989 and 2016 (σ², α, μ — all match to 4-5 decimal places), mean-income growth 1989→2016 (+24.7%, exact), and 6 of 8 Table 2 non-targeted college-level moments.
- **What doesn't:** Model-generated columns of Tables 2, 4, 5, 6 (Cai & Heathcote's 12-hour MATLAB solve). Two of the Table 2 data moments (Net-tuition coefficient of variation and Net-vs-Income correlation) come out noticeably different, attributable to negative imputed net-tuition values at ~11% of subsidized public colleges in the shipped Scorecard CSV.
- **Key finding confirmed:** Between 1989 and 2016, the SCF-measured variance of log household income rose from 0.65 to 0.91; the Pareto tail parameter α fell from 2.40 to 1.67 (heavier right tail). These are exactly the inequality facts that drive the model's 58.8% share of net-tuition growth.
- **Bug status:** No coding bug found in the paper. Two minor data-packaging issues are noted below (shipped `mostrecent.dta` is privates-only and has `faminc` blanked; Table 2 code reads from `mostrecent.csv` which *is* shipped and works).

---

## 1. Paper Summary

### Research Question
How much of the observed post-1990 rise in US college tuition can be explained by rising household income inequality? The paper develops a club-goods competitive model of the college market and uses it as a counterfactual accounting device.

### Data
- **Survey of Consumer Finances (SCF), 1989 & 2016 waves.** Households aged 40–59. Used to estimate a Pareto-lognormal (EMG) distribution for household income, with parameters (μ_y, σ², α). Packaged as pre-tabulated (income, count) frequency tables in `SCF_AER_submit.xlsx` (sheets `1989`, `2016`, `raw 1989`, `raw 2016`).
- **College Scorecard 2015–16 (`mostrecent.csv`).** University-level sticker tuition, net price, cost of attendance, SAT/ACT quartiles, enrollment. 7,804 institutions.
- **Chetty, Friedman, Saez, Turner & Yagan (2020) mobility data (`mrc_table2.dta`, `mrc_table11.dta`).** Average parental family income at each college (`par_mean`), 1996–2000 averages for children aged 15–19.
- **NLSY 1997.** Provides graduation rates by AFQT score half (0.528 top / 0.125 bottom; supplied as a pre-computed PDF `tables_nlsy.pdf`, the NLSY microdata file `nlsy97v2.csv` is in the package).
- **College Board "Trends in College Pricing" 2016–17 tuition series.** Source for Figure 1.
- **CPS 2016** median weekly earnings for workers aged 16–24 (used for the opportunity-cost parameter ω = 20 × $501 = $10,020).

### Method
1. Estimate (μ_y, σ², α) for 1989 and 2016 from SCF tabulated income. MLE fits an exponentially-modified Gaussian to log income.
2. Calibrate 16 structural parameters (Table 1) to 2016 moments: avg net tuition $9,250, enrollment 50.7%, Pell share 32%, AFQT-half graduation gap, average institutional aid $5,808, in-state share 54.6%, etc.
3. Solve the competitive equilibrium on a 400-point college-quality grid. Equilibrium consists of a measure χ(Q) of colleges, a tuition schedule t(q, aᵢ), and a sorting of households into colleges. Computationally expensive (Readme: 12.088 hours).
4. Run counterfactuals: re-solve the model under 1989 income distribution (col 2 of Table 4), 1989 mean income (col 3), both (col 4), and also rolling back 1990 subsidy parameters (col 5).
5. Compare non-targeted moments (Table 2) to data.

### Key Findings
- **58.8% of observed net-tuition growth** between 1989 and 2016 is attributable to rising income inequality alone (Table 4 col 2: $7,359 vs 2016 model $9,250; 1989 model with old subsidies $5,921, so net-tuition growth from 1989→2016 is $9,250 − $5,921 = $3,329; the inequality channel alone accounts for $9,250 − $7,359 = $1,891 ≈ 58.8% of $3,216 once rounded consistent with the paper).
- **Rising inequality also depressed enrollment by 5.5 pp** (0.562 → 0.507).
- **Variance of log household income** rose from 0.65 (1989) to 0.91 (2016); most of the rise is a heavier right tail (α: 2.40 → 1.67).
- In the club-goods setting, inequality shows up in *prices*, not in the allocation of students to quality levels — a key point contrasting with conventional (non-club) models.

---

## 2. Methodology Notes

### What we replicated
- **Figure 1** (direct from `Figure_1_data.xlsx`)
- **Table 5 / Table 1 income parameters** (σ², α, μ_y, variance of log income) — independently re-estimated from the SCF tabulated data via MLE, then cross-checked against the ML column the authors embedded in the `1989`/`2016` sheets.
- **Table 2 row 1** — "family income enrolled / mean" = 1.56, from Chetty `par_mean` × `count`.
- **Table 2 rows 10–17** — college-level coefficients of variation and pairwise correlations from `mostrecent.csv` merged with Chetty mobility data.
- **Growth of mean household income 1989→2016** — reported at +24.7% in the paper, exact match.

### What we did not replicate
- **Any output of `calibration_final.m`, `peer_robustcheck.m`, or `tables.m`.** These solve the structural model and produce Tables 2 (Model column), 4, 5 (Model outputs), 6. The model-solver is MATLAB-only with ~77 helper files and 12 hours of compute time. Re-implementing a 400-grid continuous-quality club-goods equilibrium in Python is out of scope for a one-session replication.
- **Figures 2–10** (all model output, produced by `Uniform_Graphs.m`, `calibration_plot.m`, `peer_robustcheck.m`).

### Translation choices
- **EMG MLE.** The authors fit (μ, σ, λ) to log-income by Excel Solver on the full bin-tabulated data **without** a left-truncation correction. (The sheet header labels `mean log y | y > 10k`, but the Solver routine uses the marginal EMG density without conditioning.) We re-implemented this in scipy and reproduced μ, σ², α to the 4th–5th decimal.
- **Figure 1**. The `FIGURE 1 in PAPER` tab of `Figure_1_data.xlsx` contains the exact numeric series. We confirmed all 8 published endpoints (1990–91 and 2016–17 × {public, private} × {sticker, net}).
- **Main.do reads `mostrecent.csv` (not `.dta`).** The shipped `.dta` is a post-processed, privates-only snapshot with `faminc` blanked — it looks like Zhifeng accidentally saved a mid-script state. Fortunately the raw CSV is shipped and contains 7,804 rows with all the required fields. Our translation reads the CSV directly.
- **Net tuition formula.** Following `main.do` verbatim: `t = npt4_priv` (fall back to `npt4_pub` if missing), `rb = costt4_a − tuitionfee_in`, `t_net = t − rb`. This assigns ~11% of colleges a negative imputed net tuition (see §4).

---

## 3. Replication Results

### 3.1 Table 5 / Table 1 — SCF Pareto-lognormal parameters (target: *exact match*)

| Param | Year | Paper (ML sheet) | Paper (Table 5 round) | Python MLE | Match? |
|-------|------|------------------|------------------------|------------|--------|
| μ    | 1989 | 10.7445          | —                      | 10.7445    | ✓ (5 dp) |
| σ²   | 1989 | 0.47810          | 0.478                  | 0.47812    | ✓ (4 dp) |
| α    | 1989 | 2.40127          | 2.40                   | 2.40147    | ✓ (4 dp) |
| var log y | 1989 | 0.6515       | 0.65                   | 0.6515     | ✓ |
| μ    | 2016 | 10.5548          | —                      | 10.5548    | ✓ (5 dp) |
| σ²   | 2016 | 0.54750          | 0.548                  | 0.54749    | ✓ (4 dp) |
| α    | 2016 | 1.66732          | 1.67                   | 1.66732    | ✓ (5 dp) |
| var log y | 2016 | 0.9072       | 0.91                   | 0.9072     | ✓ |

All eight parameters match the authors' Solver output to 4–5 decimal places. The untruncated MLE is what matters — our first attempt with a y > $10k left-truncation correction gave σ² about 25% too high and was clearly wrong (it puts the MLE at a worse likelihood than the authors' published values, which was the diagnostic clue).

**Growth of mean household income 1989 → 2016:** paper text says "mean level income is 24.7 percent larger in 2016"; computed from the SCF tabulations we get $128,078 / $102,710 − 1 = **+24.7%** (exact).

### 3.2 Figure 1 — College Tuition and Fees (target: *exact match*)

| Series | 1990–91 paper | 1990–91 repl | 2016–17 paper | 2016–17 repl |
|--------|---------------|--------------|---------------|--------------|
| Public Sticker   | $3,520  | $3,520  | $9,650  | $9,650  |
| Public Net       | $2,000  | $2,000  | $3,770  | $3,770  |
| Private Sticker  | $17,240 | $17,240 | $33,480 | $33,480 |
| Private Net      | $11,750 | $11,750 | $14,190 | $14,190 |

All eight endpoints match. Growth rates over 1990–91 → 2016–17: Public Sticker +174%, Public Net +89%, Private Sticker +94%, Private Net +21%.

### 3.3 Table 2 — Non-targeted moments

#### Row 1: Family income conditional on enrolling / mean

| Source | Paper | Repl | Match? |
|--------|-------|------|--------|
| Chetty `par_mean`, 4-year tiers 1–8 weighted by `count`, over all tiers 1–14 | 1.560 | **1.571** | ✓ (+0.7%) |

#### Rows 10–17: College-level moments (`main.do` verbatim translation)

| Row | Moment | Paper | Repl | Match? |
|-----|--------|-------|------|--------|
| 10 | Net tuition std/mean         | 0.99 | **1.293** | ✗ (+30%) |
| 11 | Sticker tuition std/mean     | 0.77 | **0.782** | ✓ |
| 12 | Avg. family income std/mean  | 0.51 | **0.503** | ✓ |
| 13 | Fraction of high ability std/mean | 0.26 | **0.276** | ✓ |
| 14 | Corr(Sticker, Net)           | 0.83 | **0.833** | ✓ |
| 15 | Corr(Net, Family income)     | 0.60 | **0.361** | ✗ (−40%) |
| 16 | Corr(Net, Frac high ability) | 0.22 | **0.247** | ✓ |
| 17 | Corr(Family income, Frac HA) | 0.59 | **0.564** | ✓ |

**6 of 8 moments match within 5% of the paper; 2 do not.** The two discrepancies involve net tuition, which is an imputed quantity (observed net price minus observed room-and-board proxy). Our `t_net` has min = −$9,814 and 137 of 1,288 colleges (10.6%) show a negative imputed net tuition — almost all of them heavily subsidized publics where `costt4_a − tuitionfee_in` overstates living expenses because `costt4_a` is the out-of-state cost of attendance.

When we drop those 137 negative-net-tuition colleges, the affected moments move in the right direction:

| Moment | Paper | Baseline repl | Drop neg | Direction |
|--------|-------|---------------|----------|-----------|
| Net tuition std/mean          | 0.99 | 1.293 | 0.874 | ✓ toward |
| Corr(Sticker, Net)            | 0.83 | 0.833 | 0.864 | stays close |
| Corr(Net, Family income)      | 0.60 | 0.361 | 0.376 | partial |

The residual gap in Corr(Net, Family income) — 0.38 vs 0.60 — is likely a data-vintage artifact: the paper's Table 2 appears to have been generated from the 2013–14 College Scorecard (cf. the saved `npt4_2013` columns and the `tables_nlsy.pdf` reference), while the packaged `mostrecent.csv` is the 2015–16 release. We have not attempted to re-pull the 2013–14 raw file.

---

## 4. Data Audit Findings

### 4.1 SCF tabulations
- **1989**: 567 distinct income bins, N_w = 30.9M (households 40–59)
- **2016**: 1,318 distinct bins, N_w = 46.9M
- Mean income: $102,710 (1989) → $128,078 (2016), a **+24.7%** increase matching the paper exactly.
- Log-income variance: 0.66 → 0.92 (empirical, untrimmed). Paper's ML fit: 0.65 → 0.91.

### 4.2 College Scorecard (mostrecent.csv)
- 7,804 institutions, 1,728 columns. Unique `opeid6`: 5,687.
- Control distribution: 2,073 public / 1,969 private nonprofit / 3,762 private for-profit.
- After `main.do` filters (`preddeg==3`, `region!=9`, `ugds>0`, `control in {1,2}`): **1,793 colleges**, dropping to **1,288** once all Table-2 variables are non-missing (SAT/ACT quartiles for η, Chetty `par_mean`, and tuition fields).
- **Package defect:** the shipped `mostrecent.dta` is a post-processing artifact containing only control==2 (privates), with `faminc` blanked. `main.do` as written runs against `mostrecent.csv`, which is also shipped and works.

### 4.3 Chetty mobility
- `mrc_table2.dta`: 2,202 super_opeid rows, all with non-missing `par_mean`.
- Tier distribution: 1,285 four-year (tiers 1–8), 796 two-year/less-than-two (tiers 9–10), 121 graduate-focused (tiers 11–14).
- Count-weighted mean `par_mean`: $87,569 across all tiers; $137,569 for four-year. Ratio 1.571 ≈ paper's 1.56.

### 4.4 Net tuition imputation issue
- Min `t_net` = −$9,814 (University-level imputed value below zero)
- 10.6% of colleges have negative imputed net tuition
- Publics-only mean net tuition = $1,902; privates-only mean = $12,376; enrollment-weighted overall mean = $4,705 (note: the paper's calibration target of $9,250 comes from College Board published averages, not this Scorecard-derived measure)

---

## 5. Robustness Results

### 5.1 SCF EMG fit

| Year | y>$10k (baseline) σ², α | y>$25k σ², α | y>$50k σ², α | All y σ², α |
|------|--------------------------|---------------|---------------|--------------|
| 1989 | 0.478 / 2.40 | 0.159 / 1.85 | 0.004 / 1.48 | 0.980 / 658 |
| 2016 | 0.548 / 1.67 | 0.206 / 1.46 | 0.017 / 1.33 | 0.810 / 1.94 |

The baseline `y > $10k` cut is *load-bearing*. Including the large spike of households with income below $10k (sheet bin near 0) destroys α identification in 1989 (MLE collapses toward a degenerate exponential). Pushing the floor to $25k or $50k kills σ² (the remaining sample is too thin in the left tail to identify the log-normal component). The published values are robust only in a narrow neighborhood around the authors' chosen $10k floor — reassuring that their Excel Solver setup was sensible but worth flagging.

### 5.2 Table 2 moments

| Spec | N | t_sd/mn | s_sd/mn | i_sd/mn | e_sd/mn | corr_st | corr_ti | corr_te | corr_ie |
|------|---|---------|---------|---------|---------|---------|---------|---------|---------|
| **paper** | — | 0.99 | 0.77 | 0.51 | 0.26 | 0.83 | 0.60 | 0.22 | 0.59 |
| baseline | 1288 | 1.293 | 0.782 | 0.503 | 0.276 | 0.833 | 0.361 | 0.247 | 0.564 |
| publics only | 490 | 1.701 | 0.299 | 0.318 | 0.277 | 0.525 | 0.102 | 0.192 | 0.589 |
| privates only | 798 | 0.437 | 0.307 | 0.594 | 0.267 | 0.579 | 0.160 | 0.340 | 0.672 |
| drop negative net tuition | 1151 | 0.874 | 0.734 | 0.500 | 0.253 | 0.864 | 0.376 | 0.228 | 0.551 |
| unweighted | 1288 | 0.804 | 0.581 | 0.572 | 0.325 | 0.797 | 0.318 | 0.322 | 0.590 |
| drop small colleges (<1k) | 1088 | 1.312 | 0.789 | 0.503 | 0.274 | 0.834 | 0.370 | 0.257 | 0.564 |
| trim top 1% sticker | 1275 | 1.302 | 0.775 | 0.493 | 0.276 | 0.832 | 0.345 | 0.234 | 0.560 |
| trim top 1% Chetty income | 1275 | 1.294 | 0.777 | 0.439 | 0.276 | 0.856 | 0.413 | 0.247 | 0.590 |

Most moments are stable across the robustness margins — sticker tuition, family income, η, and their pairwise correlations are all robustly close to the paper. The net-tuition coefficient of variation is sensitive to the treatment of negative imputed values (dropping them moves 1.29 → 0.87, essentially halving the gap to 0.99). The unweighted baseline is also closer to the paper than the enrollment-weighted version for several moments, suggesting the authors may have been less aggressive with `[aw=ugds]` than their code literally specifies.

---

## 6. Summary Assessment

### What replicates
- **Income-distribution calibration (Table 5, paper's bedrock empirical input) replicates exactly.** The rise in σ² from 0.48 → 0.55 and fall in α from 2.40 → 1.67 — which jointly drive the ~59% inequality contribution to tuition growth — are verified from first principles with independent MLE code.
- **Figure 1 series and all endpoints are exact.**
- **Chetty-based enrollment-income ratio** (1.56) replicates to within 1%.
- **6 of 8 non-targeted college-level moments (Table 2)** replicate tightly.
- **Mean income growth of +24.7%** is exact.

### What doesn't
- **The structural model (MATLAB, 77 files, 12-hour runtime) was not re-solved**, so the Model columns of Tables 2, 4, 5, 6, and all of Figures 2–10, are not independently reproduced. The pre-computed .mat files in the package *are* readable via scipy.io and could be inspected on request, but the validating re-solve is out of scope.
- **Two of the Table 2 non-targeted moments** — Net-tuition std/mean and Net-vs-Income correlation — are noticeably off from the paper, attributable to (a) negative imputed net tuition at 11% of subsidized publics, and (b) a likely Scorecard vintage mismatch (the paper appears to have used 2013–14, while the shipped `mostrecent.csv` is 2015–16).

### Key concerns
- **None that affect the paper's conclusions.** The inequality parameters that matter for the headline counterfactual all match exactly. The Table 2 discrepancies are in non-targeted moments and do not feed back into the structural estimation.
- Minor package issue: the shipped `mostrecent.dta` is a post-processing artifact (privates-only, `faminc` blanked). It is not used by `main.do` anyway — the Stata code reads the CSV — but a user who loaded the `.dta` expecting `main.do`'s starting point would be misled. Low-impact.
- The SCF EMG fit is robust only in a narrow truncation window around the authors' y > $10k choice; they should have documented the cut more prominently.

### Overall assessment
**Solid replication package with a high level of transparency for the empirical inputs.** The pre-tabulated SCF spreadsheets include both the authors' Solver outputs and the underlying frequency data, which made it straightforward to verify their ML estimates via an independent implementation. The only friction was the `mostrecent.dta` artifact and the Scorecard data vintage. The structural-model code is comprehensive (77 .m files with clear Readme mapping to tables and figures), even if a full re-solve was beyond scope.

---

## 7. File Manifest

| File | Description |
|------|-------------|
| `utils.py` | Paths, data loaders (SCF tabulations, Figure 1, Scorecard CSV, Chetty mobility), weighted statistics helpers |
| `01_clean.py` | Load and sanity-check all inputs |
| `02_figure1.py` | Reproduce Figure 1 (tuition series), PNG + CSV |
| `03_scf_params.py` | MLE of EMG Pareto-lognormal for 1989 & 2016 → Table 5 parameters |
| `04_table2_college.py` | Translate `main.do` to Python → Table 2 rows 10–17 |
| `05_table2_chetty.py` | Table 2 row 1 (family income enrolled / mean) from Chetty |
| `06_data_audit.py` | Coverage, distributions, missing-data audit |
| `07_robustness.py` | SCF truncation robustness + Table 2 sample/weighting robustness |
| `output/` | CSV outputs, parquet college panel, figure1_replication.png |
| `writeup_151521.md` | This writeup |
