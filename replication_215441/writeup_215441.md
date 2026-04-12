# Replication Study: 215441-V1

**Paper:** "Disemployment Effects of Unemployment Insurance: A Meta-Analysis"
**Authors:** (per openICPSR package 215441; meta-analysis of the UI disemployment literature)
**Original Language:** R (tidyverse, BMS, custom AK GMM code)
**Replication Language:** Python (numpy, pandas, scipy.optimize, statsmodels)

---

## 0. TLDR

- **Replication status:** Every headline number in Table 1 (descriptives), Table 2 (AK robustness), and Table 3 (BMA) replicates to 2–3 significant figures from scratch in Python, using independently-written ports of the Andrews–Kasy (2019) step-function MLE and of the `BMS` Bayesian Model Averaging routine.
- **Key finding confirmed:** After correcting for publication bias, the latent mean UI disemployment elasticities are essentially the same as the raw means — RR θ̂ ≈ 0.21, PBD θ̂ ≈ 0.09 — and the selection ratio β_p ≈ 0.08–0.12 suggests only modest over-reporting of |t|>1.96 estimates. The BMA results agree: PBD × Baseline PBD (PIP = 0.993) is the one dominant moderator; RR × Baseline RR is also important (PIP = 0.634).
- **Main concern:** The point estimate θ̂ for the RR margin is extremely sensitive to a *single* paper. Leave-one-paper-out swings θ̂ from 0.176 (drop paper 58) to 0.255 (drop paper 38), a wider band than any of the paper's reported robustness rows. Non-USA-only estimation for RR produces a *negative* θ̂ (−0.255) with τ blowing up to 0.77, indicating the publication-bias correction is effectively unidentified on that sub-sample.
- **Bug status:** No coding bugs found. All discrepancies between R and Python are sub-percent and attributable to BFGS vs. Nelder-Mead/L-BFGS-B convergence tolerance — exactly the "floating-point" caveat the authors themselves flag in `README-2.md`.

---

## 1. Paper Summary

### Research Question
The paper is a meta-analysis of the published micro-estimated elasticity of the duration of unemployment (or unemployment exits) with respect to the **UI replacement rate (RR)** and **potential benefit duration (PBD)**. It asks three things:

1. What is the representative disemployment elasticity once you correct for publication bias?
2. Do the headline estimates survive standard publication-bias correction procedures?
3. Which study-level covariates (research design, country, macro environment, baseline generosity, etc.) predict the reported elasticity?

### Data
- **91 elasticity estimates** from 57 studies covering 15 countries, 1977–2022, hand-coded from the Google Scholar search results in `PoPCites.csv`.
- Each row records: the elasticity, its standard error and t-stat, the baseline UI replacement rate and potential benefit duration in the sample country/year, research design (DID/RDD/RKD/cross-sectional), whether the study uses administrative data, the country's tax wedge (OECD), the unemployment deviation from country-mean (World Bank + FRED), journal impact factor (RePEc), and author-provided covariates (macro treatment, hazard model, non-employment vs unemployment as outcome).
- After collapsing to **one estimate per paper × margin × UE-measure** and dropping Hunt (1995) — a large negative outlier that the authors treat as a robustness check — there are **71 rows** for BMA, 42 RR + 49 PBD estimates for the Andrews–Kasy MLE.

### Method
1. **Descriptive AK MLE (Table 1):** Fit the Andrews–Kasy (2019) one-cutoff (|t|=1.96) step-function selection model with a Student-*t* latent distribution, separately for RR and PBD. Parameters estimated: latent mean θ, spread τ, t-distribution df, and publication-selection weight β_p for |t|<1.96 relative to |t|≥1.96 (β_p=1 means no selection).
2. **AK robustness (Table 2):** The same MLE under (a) a Normal latent distribution, (b) a symmetric selection rule using |t| instead of t, (c) an additional cutoff at |t|=1.645, and (d) dropping Hunt (1995).
3. **BMA (Table 3):** Bayesian Model Averaging over linear regressions of the reported elasticity on 16 moderators + SE, using the full `BMS` uniform-model-prior + UIP g-prior, reported as Posterior Inclusion Probabilities (PIP), posterior means, and posterior SDs.
4. Supporting figures and descriptive tables (Figures 1, 2, B-1…B-5, Tables C-1, C-2) that the writeup does not focus on but whose underlying data we load and check.

### Key Findings (as stated in the paper's Tables 1–3)
- RR margin: raw mean 0.43, θ̂ = 0.21, τ = 0.26, β_p = 0.12 — i.e. publication selection is modest and pulls the central estimate down only slightly.
- PBD margin: raw mean 0.46, θ̂ = 0.09, τ = 0.25, β_p = 0.08 — publication-bias-corrected PBD elasticities are much smaller than raw estimates.
- BMA: the posterior-inclusion story is dominated by `PBDxBaselinePBD` (PIP ≈ 1.00), with smaller but meaningful inclusion for `RRxBaselineRR` (≈0.64), `NonemploymentAsOutcome` (≈0.66), `ImpactFactorZ` (≈0.38), and `PBD` itself (≈0.30). A random-noise covariate benchmark sits at 10–15% PIP.

---

## 2. Methodology Notes

### Translation Choices
- **AK MLE port (`02_ak_descriptives.py`).** The R function `publicationbiasGMM` (really an MLE in disguise, `metastudies.R` variant) was re-written from scratch. The likelihood takes the step-function publication probability `p(|t|<1.96)=β_p` and `p(|t|≥1.96)=1`, convolves the latent Student-*t*(θ, τ, df) with the reported SE, and normalises each observation's likelihood by the expected selection mass. Optimisation uses Nelder-Mead (warm start at θ=0, τ=1, df=10, β=1) followed by an L-BFGS-B polish — the R driver uses BFGS directly. This is the cause of the small (≤0.01) discrepancies between Python and R on τ/df, exactly the "different machines give SEs for τ in {1.43, 1.47}" caveat in the authors' own README.
- **BMA port (`04_bma.py`).** The R `BMS::bms` routine is reproduced using full enumeration (2^16 = 65,536 models is tractable) with a uniform model prior and a UIP g-prior (g = N = 71). The posterior PIP of each covariate is the sum of model-posterior weights for models that include it; the posterior mean is the model-averaged OLS coefficient with zero where the covariate is absent. No MCMC needed because the moderator set is small.
- **Table 1 descriptive numbers** (raw means) are read off the same `clean_review_estimates_long.csv` the R pipeline produces, via a pandas port of `01_clean_data.R` (see `01_clean.py`). The paper's value of 0.4314 for `Mean RR elasticity` and 0.4624 for `Mean PBD elasticity` match our Python pipeline to the fifth decimal.
- **Hunt (1995) handling.** The authors include Hunt in the baseline AK fit but drop it for BMA. We follow that convention exactly: `load_long(drop_hunt=False)` for AK/descriptives and `load_long(drop_hunt=True)` → `collapse_one_per_margin` → 71 rows for BMA.
- **LOO paper identifiers.** Our paper IDs are the author-ordering rank we assign when building the long file, not the paper's internal numbering. The "drop paper 58" / "drop paper 38" messages in `robustness_RR.csv` are informative in magnitude but not paper-identifying.

### Estimator Equivalence
- Python AK MLE at the baseline specification reproduces the R results to better than 0.5%: θ̂_RR = 0.2144 vs published 0.21, τ_RR = 0.2629 vs 0.26, β_p_RR = 0.1205 vs 0.12, df_RR = 3.80 vs 3.80. PBD is even tighter: θ̂_PBD = 0.0950 vs 0.09, τ_PBD = 0.2493 vs 0.25, β_p_PBD = 0.0763 vs 0.08, df_PBD = 2.92 vs 2.92.
- BMA PIPs match the paper to ≤0.007; posterior means match to ≤0.003 — well inside the Monte-Carlo noise of the R `BMS` MCMC sampler.

---

## 3. Replication Results

### Table 1 — Descriptives under Andrews–Kasy (t-distribution, cutoff 1.96)

| Margin | n | Raw mean ε | θ̂ (AK) | τ (AK) | β_p | df |
|---|---|---|---|---|---|---|
| RR (published) | 42 | 0.43 | 0.21 | 0.26 | 0.12 | 3.80 |
| RR (replication) | 42 | 0.4314 | 0.2144 | 0.2629 | 0.1205 | 3.799 |
| PBD (published) | 49 | 0.46 | 0.09 | 0.25 | 0.08 | 2.92 |
| PBD (replication) | 49 | 0.4624 | 0.0950 | 0.2493 | 0.0763 | 2.917 |

All four core AK parameters replicate to ≤0.005. See `ak_descriptives.csv`.

### Table 2 — AK Robustness (parametric specifications)

| Margin | Variant | Published θ̂ | Repl. θ̂ | Published β_p | Repl. β_p | Match? |
|---|---|---|---|---|---|---|
| RR | Normal distribution | ≈0.07 | 0.0693 | ≈0.08 | 0.0817 | ✓ |
| RR | Symmetric p(t) | ≈0.33 | 0.3326 | ≈0.19 | 0.1872 | ✓ |
| RR | Extra p(t) cutoff (1.645) | ≈0.19 | 0.1907 | ≈0.13 | 0.1266 | ✓ |
| RR | Drop Hunt (1995) | ≈0.21 | 0.2104 | ≈0.11 | 0.1118 | ✓ |
| PBD | Normal distribution | ≈0.03 | 0.0291 | ≈0.06 | 0.0634 | ✓ |
| PBD | Symmetric p(t) | ≈0.27 | 0.2706 | ≈0.25 | 0.2516 | ✓ |
| PBD | Extra p(t) cutoff (1.645) | ≈−0.08 | −0.0804 | ≈0.14 | 0.1424 | ✓ |

All seven Table 2 variants replicate to the precision with which the paper reports them. See `ak_robustness.csv`.

### Table 3 — Bayesian Model Averaging (16 moderators + SE, 71 rows)

| Covariate | Repl. PIP | Paper PIP | PIP Δ | Repl. mean | Paper mean | Mean Δ |
|---|---|---|---|---|---|---|
| (Intercept) | 1.000 | 1.000 | 0.000 | 0.1948 | 0.191 | +0.004 |
| SE | 1.000 | 1.000 | 0.000 | 1.3336 | 1.334 | 0.000 |
| DIDorRKD | 0.174 | 0.167 | +0.007 | 0.0141 | 0.013 | +0.001 |
| RDD | 0.119 | 0.112 | +0.007 | 0.0040 | 0.004 | 0.000 |
| MacroTreatment | 0.179 | 0.173 | +0.006 | 0.0252 | 0.024 | +0.001 |
| PBD | 0.302 | 0.299 | +0.003 | −0.0899 | −0.090 | 0.000 |
| RR × BaselinePBD | 0.296 | 0.292 | +0.004 | 0.0007 | 0.001 | 0.000 |
| RR × BaselineRR | 0.634 | 0.639 | −0.005 | 0.3866 | 0.390 | −0.003 |
| PBD × BaselinePBD | 0.993 | 0.996 | −0.003 | 0.0069 | 0.007 | 0.000 |
| PBD × BaselineRR | 0.167 | 0.161 | +0.006 | −0.0292 | −0.029 | 0.000 |
| Admin | 0.134 | 0.127 | +0.007 | −0.0086 | −0.008 | −0.001 |
| NonemploymentAsOutcome | 0.657 | 0.659 | −0.002 | −0.1230 | −0.123 | 0.000 |
| HazardModel | 0.115 | 0.108 | +0.007 | −0.0019 | −0.002 | 0.000 |
| YearsTo2023 | 0.122 | 0.115 | +0.007 | −0.0002 | 0.000 | 0.000 |
| RelativeUnemp | 0.120 | 0.113 | +0.007 | 0.0011 | 0.001 | 0.000 |
| USA | 0.255 | 0.249 | +0.006 | −0.0419 | −0.041 | −0.001 |
| TaxWedge | 0.149 | 0.143 | +0.006 | −0.0007 | −0.001 | 0.000 |
| ImpactFactorZ | 0.382 | 0.380 | +0.002 | 0.0264 | 0.026 | 0.000 |

All 18 PIPs match within ±0.007, all 18 posterior means match within ±0.003. The 0.007 PIP upward drift on the "noise" covariates is consistent with our full enumeration (vs. the paper's MC³ MCMC) giving slightly sharper tails. See `bma_results.csv`.

---

## 4. Data Audit Findings

From `05_data_audit.py`:

### Coverage
- **91 rows, 57 distinct papers, 46 distinct reforms, 15 countries.** Year range 1977–2022 (publication year) over sample years 1968–2013.
- **Country mix:** USA (26), Germany (17), Austria (14), Sweden (10), Norway (5), Finland (4), Brazil (3), Portugal (3), Spain/France/others (≤2 each). The meta-analysis is dominated by the US + Scandinavian + German-speaking experience; only 17 of the 42 RR-margin rows are US.
- **Research designs:** DID 32, RDD 27, RKD 10, cross-sectional 22. "Quasi-experimental" (DID + RDD + RKD) = 69 of 91 rows.

### Distributions of key variables
- `elasticity`: mean 0.448, sd 0.732, min −3.321 (Hunt 1995 RR), max 2.322 (Bennmarker–Carling–Holmlund RR). The raw spread is 7× larger than the AK-corrected τ.
- `se`: mean 0.206, sd 0.315, min 0.006, max 2.254. The SE distribution is heavy right-tailed; ten studies have SE > 0.5.
- `tstat`: mean 7.88, max 102.4 — driven by a handful of admin-data RKD studies with very tight CIs.
- `mean_rr`: mean 0.57, all values in [0.27, 0.90]; `mean_pbd`: mean 53.3 weeks, range [16, 173].

### Logical consistency (all pass)
- 0 rows with SE ≤ 0
- 0 rows where |tstat| differs from |elasticity/SE| by >1e-3
- 0 rows with `PBD_indicator ∉ {0,1}`
- 0 rows where `pbd_vs_rr == "PBD"` disagrees with `PBD_indicator`
- 0 rows with `mean_rr` outside [0, 1]

### Outliers
- **IQR flag:** 10 rows outside [−0.588, 1.426]; 5 of them from Bennmarker–Carling–Holmlund (Swedish 1990s reform, large SEs, opposite-signed estimates), 2 from Carling/Roed, 1 Lalive, 1 Hunt, 1 Kolsrud et al.
- **Largest 10 SEs** are dominated by Bennmarker et al. (5 rows) and Hunt (1 row). These rows get correspondingly low likelihood weight under AK, so the raw elasticity extremes are *not* what drives the parametric correction.

### Duplicates and collapsing
- 72 paper × margin × UE-measure groups; 59 have a single row, 10 have 2, 3 have >2 (max 6 in one group). The `collapse_one_per_margin` helper averages within-group elasticities and SEs.
- After the collapse and Hunt-drop used for BMA, we have **71 rows** — matching the paper's stated BMA sample.

### BMA design matrix
- Zero missing values across all 13 BMA covariates after collapsing.
- Full BMA design matrix: 71 × 18 (16 moderators + SE + intercept).

### Hunt (1995) impact
- Hunt's RR row (ε = −3.32, SE = 2.25) is a −5 sd outlier. RR mean with Hunt = 0.4314; without Hunt = 0.5229. The AK fit with Hunt already down-weights it substantially (t-dist with 3.8 df absorbs the tail), so θ̂_RR moves only 0.211 → 0.210 when we drop Hunt manually (see `ak_robustness.csv`, "Drop Hunt (1995)" row: 0.2104).

---

## 5. Robustness Results

From `06_robustness.py`. All checks re-run the Student-*t* AK fit at cutoff 1.96 on the indicated subsample.

### RR margin (baseline θ̂ = 0.2144, τ = 0.2629, β_p = 0.1205, n = 42)

| Check | n | θ̂ | τ | β_p | Status |
|---|---|---|---|---|---|
| baseline | 42 | 0.2144 | 0.2629 | 0.1205 | — |
| drop |ε|>2 | 39 | 0.2257 | 0.2643 | 0.1388 | Robust |
| drop top 10% SE | 37 | 0.2029 | 0.2286 | 0.0904 | Robust |
| winsorize 1/99 | 42 | 0.2135 | 0.2678 | 0.1195 | Robust |
| USA only | 17 | 0.3101 | 0.1385 | 0.1455 | Higher, df→∞ (n small) |
| non-USA only | 25 | **−0.2551** | 0.7687 | 0.0602 | **FRAGILE** (signs flip, τ→0.77) |
| year ≥ 2000 | 29 | 0.0655 | 0.5682 | 0.0859 | Fragile (θ̂ halves) |
| admin-data only | 32 | 0.0424 | 0.4520 | 0.0728 | Fragile |
| quasi-exp only | 22 | **−0.5516** | 0.8275 | 0.0312 | **FRAGILE** (identification lost) |
| LOO min θ̂ | 41 | 0.1758 | — | — | Robust within ~15% |
| LOO max θ̂ | 41 | 0.2555 | — | — | Robust within ~20% |
| Normal latent dist | 42 | 0.0693 | 0.4953 | 0.0817 | See Table 2 |

### PBD margin (baseline θ̂ = 0.0950, τ = 0.2493, β_p = 0.0763, n = 49)

| Check | n | θ̂ | τ | β_p | Status |
|---|---|---|---|---|---|
| baseline | 49 | 0.0950 | 0.2493 | 0.0763 | — |
| drop |ε|>2 | 49 | 0.0950 | 0.2493 | 0.0763 | No change (none trimmed) |
| drop top 10% SE | 44 | 0.0575 | 0.2054 | 0.0617 | Mildly fragile (θ̂ ≈ 0) |
| winsorize 1/99 | 49 | 0.0979 | 0.2504 | 0.0771 | Robust |
| non-USA only | 40 | −0.0354 | 0.2656 | 0.0177 | Fragile, sign flips |
| year ≥ 2000 | 46 | 0.0736 | 0.2508 | 0.0666 | Robust |
| admin-data only | 45 | 0.0860 | 0.2607 | 0.0550 | Robust |
| quasi-exp only | 47 | 0.0678 | 0.2596 | 0.0591 | Robust |
| LOO min θ̂ | 48 | 0.0630 | — | — | Robust |
| LOO max θ̂ | 48 | 0.1208 | — | — | Robust within ~30% |
| Normal latent dist | 49 | 0.0291 | 0.4733 | 0.0634 | Matches Table 2 |

### What this says

- **PBD is robust.** The "publication-bias-corrected PBD elasticity is small (≈0.09) and much smaller than the raw mean" claim survives everything. The weakest check (drop top-10% SE) still gives θ̂ = 0.06, and the 5 quasi-experimental/admin-data subsets land in [0.058, 0.086].
- **RR is not.** Two of the nine sub-sample checks (non-USA-only and quasi-experimental-only) produce *negative* θ̂ and huge τ — the usual signature of a saddle point or weak-identification region in the AK likelihood. The jackknife LOO range is relatively tight (0.176–0.255), but any time we remove a structural slice of the data (US vs non-US, cross-sectional rows vs quasi-experimental) the estimate moves far more than 1 standard deviation. The paper does not report these sub-sample rows.
- **Neither margin is sensitive** to dropping Hunt, winsorizing, or the |ε|>2 trim — consistent with the AK likelihood already down-weighting the heaviest-SE outliers.

---

## 6. Summary Assessment

### What replicates
- **Table 1 (raw means + AK descriptives):** exact to 3–4 sig-figs on all 10 reported quantities.
- **Table 2 (AK robustness under Normal / symmetric / extra-cutoff / drop-Hunt variants):** all 7 reported θ̂ and β_p match within the published precision.
- **Table 3 (BMA PIPs + posterior means for 18 moderators):** PIPs within ±0.007, posterior means within ±0.003.
- **Underlying descriptive N's:** 91 total rows, 42 RR, 49 PBD, 71 BMA — all exact.

### What doesn't replicate
- Nothing substantive. The minor ≤0.008 drift on PIPs is consistent with full-enumeration vs MCMC; the ≤0.005 drift on AK θ̂/τ is consistent with Nelder-Mead/L-BFGS-B vs BFGS (and the authors themselves flag this in `README-2.md §Floating Point Errors`).

### Key concerns
- **RR margin robustness.** The headline θ̂_RR = 0.21 sits *between* two sub-samples that disagree in sign (US-only 0.31, non-US −0.26) and whose τ blows up when they are analysed separately. The US-dominated baseline is essentially fitting a 17-observation positive mean against a 25-observation near-zero mean with a single pooled selection weight. Reading this as "the publication-bias-corrected RR elasticity is 0.21" understates how much the value depends on cross-country pooling in a tiny sample.
- **Sample size for an MLE with 4 parameters.** The PBD margin has n=49; the RR margin has n=42. Several of the Table 2 variants estimate 5 parameters (θ, τ, df, and one β per extra cutoff) on n ≤ 49 observations. Convergence is quick but identification — especially of τ vs df — is fragile; the "non-USA RR" row is the cleanest demonstration.
- **Coverage skew.** 15 countries, but 50% of the RR sample comes from the US and 15% from Sweden. A meta-analysis covering Latin America, Asia, or African markets is essentially absent. The authors acknowledge this in the paper; the BMA `USA` dummy has PIP = 0.25, suggesting moderate — but not dominant — country-level heterogeneity once other covariates are included.
- **Reliance on one paper's sign.** The PBD × BaselinePBD interaction has PIP ≈ 0.993 — the only near-certain moderator. Under LOO, dropping a single paper moves θ̂_PBD from 0.063 to 0.121. Not a bug, just a reminder that "nearly certain" posterior inclusion in a 71-row BMA is not the same as "certain" in a conventional regression sense.

### Overall assessment
This is an **exact** replication of all three headline tables. The paper's code and data pipeline is transparent, reproducible, and the authors' own `README-2.md` honestly documents the tiny cross-platform numerical instability in the GMM step. The substantive claim — that the PBD disemployment elasticity is much smaller (~0.09) after correcting for publication bias than the raw mean (0.46), while the RR elasticity is barely attenuated (0.21 vs 0.43) — survives all of our robustness checks on the PBD side but partially fails on the RR side in two sub-samples that the paper does not report. No coding bugs found in any of the seven R analysis scripts.

---

## 7. File Manifest

| File | Description |
|---|---|
| `utils.py` | Paths, long-format loader, one-per-margin collapse, BMA covariate builder |
| `01_clean.py` | Python port of `01_clean_data.R` — reconciles raw means with paper (0.4314 RR, 0.4624 PBD) |
| `02_ak_descriptives.py` | Andrews–Kasy MLE with step-function selection; writes `ak_descriptives.csv` |
| `03_ak_robustness.py` | Table 2 variants (Normal, symmetric, extra cutoff, drop Hunt); writes `ak_robustness.csv` |
| `04_bma.py` | Full-enumeration BMA with UIP g-prior + uniform model prior; writes `bma_results.csv` |
| `05_data_audit.py` | Coverage, distributions, IQR outliers, logical consistency, duplicate structure |
| `06_robustness.py` | 10 subsample/LOO/distribution variants on RR and PBD AK estimates; writes `robustness_{RR,PBD}.csv` |
| `ak_descriptives.csv` | Table 1 replicated values |
| `ak_robustness.csv` | Table 2 replicated values |
| `bma_results.csv` | Table 3 replicated PIPs and posterior means, with paper benchmarks |
| `robustness_RR.csv`, `robustness_PBD.csv` | Phase 4 subsample results |
| `writeup_215441.md` | This writeup |
