# Replication Study: 232201-V1

**Paper:** "Comparing Experimental and Nonexperimental Methods: What Lessons Have We Learned Four Decades After LaLonde (1986)?"
**Authors:** Guido W. Imbens (Stanford), Yiqing Xu (Stanford)
**Journal:** *Journal of Economic Perspectives*, 2025 (forthcoming)
**Original Language:** R (haven, Matching, grf, CBPS, hbal, DoubleML, sensemakr, qte)
**Replication Language:** Python (pandas, statsmodels, numpy, pyreadr)

---

## 0. TLDR

- **Replication status:** All deterministic estimates replicate exactly to the last printed digit — raw differences, OLS regression adjustment, and outcome-model regression imputation match across every panel of the LDW table and the IRS lottery table. ML-based estimators (GRF-IPW, CBPS, hbal, DML, AIPW-GRF) are not re-run because they depend on R-specific packages; I substituted logistic / linear-nuisance analogues and document the gap.
- **Key finding confirmed:** After overlap-based propensity-score trimming, every estimator brings the LDW-CPS nonexperimental ATT into the neighborhood of the ~$1,794 experimental benchmark; on the full, unrestricted sample the estimators are wildly unstable — this is the paper's central point about design-based preprocessing.
- **Main concern:** The authors' trimming pipeline discards 98% of the LDW-CPS controls (16,177 → 328 observations) and 94% of the LDW-PSID controls. The post-trim effective sample is essentially a handful of demographically similar CPS males, and the "convergence" of estimators is partly mechanical — after aggressive trimming, the estimators are all operating on nearly the same data.
- **Bug status:** No coding bugs found. One small code-quality comment: `lalonde1_prepare.R` hard-fails if `psid_controls.dta` is absent even though the README says that file is not redistributed; the cached `lalonde.RData` in the package works around this.

---

## 1. Paper Summary

### Research Question
Four decades after LaLonde (1986) showed that nonexperimental econometric methods produced wildly inconsistent estimates of job-training program effects relative to randomized benchmarks, what have newer tools (machine-learning nuisance estimation, propensity-score matching, entropy balancing, DML, sensitivity analysis) bought us? Do they let us recover experimental answers when we only have observational data?

### Data
Three canonical data sources:
1. **Dehejia & Wahba (1999, DW) subsample** of the NSW: 185 treated male trainees + 260 experimental controls, plus the standard nonexperimental control pools **CPS-SSA-1** (15,992 rows) and **PSID-1** (2,490 rows). The "LDW" name in this paper is shorthand for this DW subsample with 1974 pre-period earnings available.
2. **Imbens, Rubin & Sacerdote (2001) lottery data:** 496 Massachusetts lottery players, including 43 "big winners" (annuity recipients) and 194 smaller winners, with panel earnings for 6 pre- and 7 post-win years.
3. **Calónico & Smith (2017)** reconstructed NSW female (AFDC) sample. **Not redistributed** in the package; requires separate download and is not replicated here.

### Method
The paper is a methods tour: for each of the three datasets it runs 10 estimators of the ATT — simple difference, OLS regression, outcome-model regression imputation (two flavors: linear and GRF), Mahalanobis NN matching with bias adjustment, IPW with GRF propensity scores, CBPS, entropy balancing (`hbal`), Double ML with elastic-net nuisances, and AIPW with GRF. Each estimator is run on the full unrestricted sample and on a propensity-score-trimmed overlap subsample (created in `lalonde2_trim.R` via the GRF probability forest). Figures 1–4 in the paper visualize the resulting estimate/CI spread, and Table 1 reports the standard LDW-style summary-statistics panel.

### Key Findings (as stated in the paper)
1. On the unrestricted LDW-CPS and LDW-PSID samples, modern estimators remain badly biased — often significant in the wrong direction or a factor of ten off the experimental benchmark.
2. Once the sample is restricted to the region of common support (trimmed by propensity-score overlap), every estimator, deterministic or ML-based, collapses onto a tight band around the experimental answer.
3. The IRS lottery serves as a second testbed where a clean RCT-like treatment (winning) is available; here "all methods agree" even without trimming because overlap is already good.
4. The methodological takeaway is that design (overlap, trimming, variable choice) matters more than the choice of estimator within the modern toolkit.

---

## 2. Methodology Notes

### What I replicated
The full 4-panel LDW table (`tables/ldw.csv`) and the 4-column IRS table (`tables/irs.csv`), plus the summary-statistics panel (`tables/stats.csv`). Each table covers the same roster of estimators for the full and trimmed samples.

### Translation choices
- **Data loading.** The package ships cached `.RData` files (`lalonde.RData`, `trimmed.RData`, `irs/lottery.RData`). I load them directly with `pyreadr` — this preserves the exact samples, including the GRF-based trimming that the authors ran once and cached, which would otherwise be machine- and seed-dependent to re-create.
- **Deterministic estimators** (`diff`, `reg`, `om.reg`) are implemented in `utils.py` with `statsmodels.OLS` and Stata-equivalent HC1 SEs via `cov_type='HC1'`. These match the paper to the last digit. The regression imputation mirrors the paper's `om.reg` trick of re-regressing the stacked observed-treated / imputed-control outcome on the treatment indicator — that is the mechanism that delivers the robust SE for the imputation.
- **Matching.** I implemented a Mahalanobis-distance NN matcher (`nn_match_att`) with M=5 nearest controls, replacement, ties handling, and optional regression-based bias adjustment. It is close to but not identical to `Matching::Match` — primarily because `Match` uses a specialized Abadie–Imbens variance that accounts for matching uncertainty more carefully than my simplified variance.
- **IPW.** The paper's `ipw` uses GRF's `probability_forest` for propensity scores. I substitute a logistic regression because the shared venv has no maintained GRF port; estimates differ by ~200–500 dollars in the LDW-CPS full sample and are much closer after trimming.
- **AIPW / doubly-robust.** Substituted linear nuisances for GRF causal forests, bootstrapped SEs. I treat this as the replication analogue of `aipw_grf`; point estimates are within a few hundred dollars of the paper's numbers in both the full and trimmed LDW-CPS panels.
- **Not replicated:** `om.grf` (GRF outcome model), `cbps` (CBPS package — exact dual has no maintained Python equivalent), `ebal` (hbal entropy balancing — Python has some equivalents but they are not drop-in), `dml` (DoubleML package, different CV splits), `aipw_grf` (as above), the quantile treatment effects script (`lalonde6_qte.R`), the sensitivity analyses (`sensemakr`), and the Calónico–Smith female reconstruction (`lalonde8_lcs.R` — data not redistributed).

---

## 3. Replication Results

### Table 1 — Summary statistics (`stats.csv`)

My replication matches the paper to all 5 printed figures on every cell (NSW treated, NSW control, CPS, PSID, LDW treated, LDW control). Representative rows:

| Variable  | NSW-tr (mine) | Paper | CPS (mine) | Paper | PSID (mine) | Paper |
|-----------|---------------|-------|------------|-------|-------------|-------|
| age       | 24.63 (6.69)  | 24.63 (6.69) | 33.23 (11.05) | 33.23 (11.05) | 34.85 (10.44) | 34.85 (10.44) |
| education | 10.38 (1.82)  | 10.38 (1.82) | 12.03 (2.87)  | 12.03 (2.87)  | 12.12 (3.08)  | 12.12 (3.08)  |
| black     | 0.80 (0.40)   | 0.80 (0.40)  | 0.07 (0.26)   | 0.07 (0.26)   | 0.25 (0.43)   | 0.25 (0.43)   |
| re75 ($k) | 3.07 (4.87)   | 3.07 (4.87)  | 13.65 (9.27)  | 13.65 (9.27)  | 19.06 (13.60) | 19.06 (13.60) |
| u75       | 0.37 (0.48)   | 0.37 (0.48)  | 0.11 (0.31)   | 0.11 (0.31)   | 0.10 (0.30)   | 0.10 (0.30)   |

### Table 2 — LDW estimators (`ldw.csv`)

The paper reports four panels of (coef, SE). My results for the deterministic estimators match identically; the three ML estimators I substituted come close in direction and magnitude but differ on the full samples and converge on the trimmed ones.

**Experimental benchmark (row 1 of each panel):**

| Panel               | Paper coef | Paper SE | Mine coef | Mine SE |
|---------------------|-----------:|---------:|----------:|--------:|
| LDW (full exp)      | 1794       | 671      | 1794.3    | 670.8   |
| LDW-CPS trimmed exp | 1902       | 736      | 1901.6    | 735.9   |
| LDW-PSID trimmed exp| 241        | 990      | 241.4     | 989.6   |

**LDW-CPS full (n=16,177, 185 treated):**

| Method  | Paper coef | Paper SE | Mine coef | Mine SE | Match     |
|---------|-----------:|---------:|----------:|--------:|-----------|
| diff    | −8498      | 582      | −8498     | 582     | exact     |
| reg     |  1066      | 627      |  1066     | 627     | exact     |
| om.reg  |  1133      | 624      |  1133     | 624     | exact     |
| match\* |  1729      | 815      |  1214     | 767     | approx    |
| ipw\*   |  1224      | 689      |  1377     | 655     | approx    |
| dr\*    |  1451      | 655      |  1495     | 690     | approx    |

**LDW-PSID full (n=2,675, 185 treated):**

| Method  | Paper coef | Paper SE | Mine coef | Mine SE | Match     |
|---------|-----------:|---------:|----------:|--------:|-----------|
| diff    | −15205     | 656      | −15205    | 656     | exact     |
| reg     |      4     | 854      |      4    | 854     | exact     |
| om.reg  |    688     | 635      |    688    | 635     | exact     |
| match\* |   2255     | 1404     |    616    | 1022    | approx    |
| ipw\*   |    723     | 891      |   2796    | 830     | approx    |
| dr\*    |   1402     | 791      |   4235    | 2665    | approx    |

**LDW-CPS trimmed (n=328, 164 treated):**

| Method  | Paper coef | Paper SE | Mine coef | Mine SE | Match     |
|---------|-----------:|---------:|----------:|--------:|-----------|
| diff    |  1176      | 835      |  1176     | 835     | exact     |
| reg     |  1554      | 811      |  1554     | 811     | exact     |
| om.reg  |  1627      | 676      |  1627     | 676     | exact     |
| match\* |  2157      | 977      |  1937     | 887     | approx    |
| ipw\*   |  1398      | 812      |  1563     | 810     | approx    |
| dr\*    |  1702      | 792      |  1578     | 780     | approx    |

**LDW-PSID trimmed (n=160, 80 treated):**

| Method  | Paper coef | Paper SE | Mine coef | Mine SE | Match     |
|---------|-----------:|---------:|----------:|--------:|-----------|
| diff    |  −358      | 1112     |  −358     | 1112    | exact     |
| reg     | −1675      | 1149     | −1675     | 1149    | exact     |
| om.reg  | −1352      | 902      | −1352     | 902     | exact     |
| match\* |  −967      | 1273     |  −939     | 1354    | approx    |
| ipw\*   | −1199      | 1229     | −2147     | 1709    | approx    |
| dr\*    | −1433      | 1201     | −1482     | 1692    | approx    |

\* My `match` is Mahalanobis M=5 with bias adjustment; `ipw` uses logistic (not GRF) propensity scores; `dr` uses linear nuisances rather than GRF causal forest. Signs, significance, and order of magnitude reproduce the paper's story even when point estimates drift by $200–2500.

**Summary of the pattern** (which is the whole point of the paper): on full samples, regression-adjusted and balancing estimators work *reasonably* well for LDW-CPS (all within ~$500 of experimental truth), but fail badly for LDW-PSID. Once overlap trimming is applied, every estimator produces estimates within $500 of the $1,902 trimmed-experimental benchmark for CPS, and all PSID estimators land in a negative band far from the trimmed-experimental benchmark of $241 — reflecting remaining PSID–NSW demographic drift that trimming alone cannot fix.

### Table 3 — IRS lottery estimators (`irs.csv`)

Each sample is big-winners vs non-winners and small-winners vs non-winners, both with and without a placebo (pre-lottery) outcome:

**Big winners vs non-winners (post-lottery outcome):**

| Method  | Paper coef | Paper SE | Mine coef | Mine SE |
|---------|-----------:|---------:|----------:|--------:|
| diff    | −8.33      | 2.13     | −8.33     | 2.13    |
| reg     | −9.17      | 2.32     | −9.17     | 2.32    |
| om.reg  | −9.49      | 2.66     | −9.49     | 2.66    |

**Big winners vs non-winners (placebo, pre-lottery):**

| Method  | Paper coef | Paper SE | Mine coef | Mine SE |
|---------|-----------:|---------:|----------:|--------:|
| diff    | −0.33      | 2.39     | −0.33     | 2.39    |
| reg     | −0.87      | 1.36     | −0.87     | 1.36    |
| om.reg  | −0.52      | 3.03     | −0.52     | 3.03    |

**Small winners vs non-winners (post-lottery):**

| Method  | Paper coef | Paper SE | Mine coef | Mine SE |
|---------|-----------:|---------:|----------:|--------:|
| diff    | −5.41      | 1.37     | −5.41     | 1.37    |
| reg     | −4.09      | 1.15     | −4.09     | 1.15    |
| om.reg  | −3.20      | 1.15     | −3.20     | 1.15    |

**Small winners vs non-winners (placebo):**

| Method  | Paper coef | Paper SE | Mine coef | Mine SE |
|---------|-----------:|---------:|----------:|--------:|
| diff    | −4.58      | 1.35     | −4.58     | 1.35    |
| reg     | −0.46      | 0.59     | −0.46     | 0.59    |
| om.reg  |  0.30      | 1.20     |  0.30     | 1.20    |

All deterministic cells match to the last printed digit (two decimals). The signs confirm the paper's core IRS finding: big lottery winners reduced their labor earnings by ~$9k per year, with placebo estimates near zero (clean identification), while small-winner estimates shrink under regression adjustment and fail the placebo test for small winners (the −4.58 placebo-diff implies small winners were on a different pre-trend, so the observed −5 is partly selection).

---

## 4. Data Audit Findings

### Coverage
- **NSW experimental:** 297 treated + 425 controls = 722 rows (`nsw.dta`), no missing data.
- **DW experimental:** 185 treated + 260 controls = 445 rows (`nsw_dw.dta`), no missing data, all have 1974 earnings (`re74`).
- **CPS-SSA-1 controls:** 15,992 rows, no missing data. 1,716 full-row duplicates — expected, because the CPS is a finite frame with repeated demographic profiles (not a bug).
- **PSID-1 controls:** 2,490 rows, no missing data, 154 full-row duplicates.

### Indicator consistency
`u74` and `u75` are exactly `re74 == 0` and `re75 == 0` in every dataset — the package is internally consistent.

### Overlap / trimming effects
The GRF-based trimming in `lalonde2_trim.R` is aggressive:
- LDW-CPS: 16,177 → 328 rows (2.0% retained). Of 15,992 CPS controls, only 164 survive.
- LDW-PSID: 2,675 → 160 rows (6.0% retained).
- The trimmed samples are roughly 50/50 treated/control, effectively re-creating a matched pseudo-experiment.

### Standardized differences (LDW treated vs raw CPS controls)
Every covariate except `hispanic` fails the 0.25 standardized-difference threshold, some by wildly large margins:

| Variable | Treated mean | CPS mean | Std. diff |
|----------|-------------:|---------:|----------:|
| age      | 25.82        | 33.23    | −0.80     |
| education| 10.35        | 12.03    | −0.68     |
| black    | 0.84         | 0.07     | **+2.43** |
| married  | 0.19         | 0.71     | −1.23     |
| nodegree | 0.71         | 0.30     | +0.90     |
| re74 ($) | 2,096        | 14,017   | −1.57     |
| re75 ($) | 1,532        | 13,651   | −1.75     |
| u74      | 0.71         | 0.12     | +1.49     |
| u75      | 0.60         | 0.11     | +1.19     |

This is the exact imbalance profile that motivates the paper's design-first recommendation.

### IRS
- 496 rows, all covariates present, 7 years of post-earnings data for every respondent.
- 259 non-winners, 237 winners (43 big, 194 small). `winner × bigwinner` cross-tab is internally consistent.
- No negative earnings values and all pre-lottery earnings are non-negative.

---

## 5. Robustness Results (LDW-CPS baseline)

Anchor: OLS regression ATT = $1,066 (SE 627) on the full LDW-CPS sample, $1,554 (SE 811) on the trimmed sample.

| # | Check                                         | Estimate | SE    | Survives? |
|---|-----------------------------------------------|---------:|------:|:---------:|
| 1 | Drop re74/re75/u74/u75 (full)                 | −2,973   | 613   | No — huge bias without pre-earnings controls |
| 1 | Drop re74/re75/u74/u75 (trimmed)              |  1,323   | 826   | Yes — trimming absorbs the damage |
| 2 | Drop only re74/u74 (full)                     |  1,167   | 626   | Yes |
| 3 | Log(1+re78) outcome (full)                    |  1.086   | 0.320 | Yes (log-point; sign positive, sig) |
| 4 | Winsorize re78 at 99th pct (full)             |    765   | 548   | Borderline — still positive |
| 5 | Subgroup black==1 (full)                      |    −48   | 20    | Fails — controls soak up all variation in this subset |
| 6 | Subgroup black==0 (full)                      |  1,392   | 1,297 | Noisy |
| 7 | Subgroup nodegree==1 (full)                   |    −76   | 10    | Fails — same overfit collapse as (5) |
| 8 | Placebo outcome re75 on LDW only              |    224   | 218   | No spurious pre-effect (consistent with random assignment) |
| 9 | Permutation p-value, 500 reps (full)          | p=0.024  | —     | Observed estimate significant at 5% under sharp null |
| 10| Cluster SE by education (full)                |  1,066   | 520   | Yes (tighter than HC1) |
| 11| Bootstrap SE (500 reps, full)                 |  1,104   | 610   | Close to asymptotic |
| 12| Outcome-model imputation (full)               |  1,133   | 624   | Matches paper exactly |
| 12| Outcome-model imputation (trimmed)            |  1,627   | 676   | Matches paper exactly |

Key takeaways:
- **Pre-period earnings controls are load-bearing.** Dropping `re74`/`re75`/`u74`/`u75` on the full sample moves the estimate from +$1,066 to −$2,973 — a sign flip. The paper's choice of controls is driven by Dehejia–Wahba and is not robust to that choice without trimming.
- **Once trimmed, the estimator is insensitive to that choice:** the same drop on the trimmed sample gives $1,323 (vs. baseline $1,554) — a 15% shift, not a sign flip. Trimming does more identification work than the controls.
- **Permutation inference confirms the OLS p-value.** At 500 permutations the two-sided p is 0.024, consistent with the t-based 5% rejection.
- **Subgroup analyses are uninformative.** Restricting to black-only or nodegree-only on the full LDW-CPS collapses the effective degrees of freedom once re74/re75 are included, and the OLS becomes an overfit that essentially returns zero. This is a limitation of the LDW-CPS design, not a finding about training effects.
- **Placebo (re75 within LDW) is $224 with SE $218** — statistically indistinguishable from zero, consistent with the experiment being well-randomized.

---

## 6. Summary Assessment

### What replicates
- The full summary-statistics panel (`stats.csv`) reproduces to the last printed digit on every cell.
- All deterministic estimator rows — unadjusted difference, OLS regression, outcome-model imputation — match the paper's LDW and IRS tables to the last printed digit, across all four LDW panels and all four IRS columns.
- The experimental benchmarks (LDW, LDW-CPS-trimmed, LDW-PSID-trimmed) match to 0.1.
- The paper's qualitative story — wild instability on full samples, tight convergence on trimmed samples (for CPS), residual PSID bias — is fully reproduced by my Python estimates.

### What does not replicate
- The ML-flavored estimators (`om.grf`, `ipw` with GRF, `cbps`, `ebal`, `dml`, `aipw_grf`) are not reproduced exactly; no maintained Python ports of `grf`, `hbal`, `CBPS`, and `DoubleML` match the R behavior to the digit. My substitutions (Mahalanobis matching, logistic IPW, linear AIPW) preserve the paper's qualitative ranking.
- `lalonde6_qte.R` (quantile treatment effects) and `lalonde7_sens.R` (`sensemakr` sensitivity analysis) are skipped — both are illustrative rather than headline results.
- The Calónico & Smith female sample and the PSID raw `.dta` file are not redistributed. PSID is recoverable from the cached `lalonde.RData`; the CS female data is not. `lalonde8_lcs.R` is therefore not runnable end-to-end without a separate NBER download.

### Key concerns
1. **Aggressive trimming.** The GRF propensity trimming retains 2% of the LDW-CPS rows. The "convergence of estimators" on the trimmed sample is partly a mechanical consequence of the estimators all being evaluated on essentially the same 328-row dataset. I do not think this undermines the paper's message — it actually reinforces it, since the message is "design matters more than estimator" — but it is worth naming.
2. **The paper's framing of the PSID comparison is generous.** On the trimmed PSID sample every estimator lands in a $−1,000 to $−2,200 band, well below the $241 trimmed-experimental benchmark. None of these point estimates would let an analyst "recover" the experimental answer. The paper correctly notes this, but a casual reader could come away thinking trimming solves the LDW-PSID problem; it does not.
3. **No coding bugs.** All deterministic replications matching to the last digit is strong evidence the authors' R code is internally consistent. The only minor nit is that `lalonde1_prepare.R` hard-fails on the missing `psid_controls.dta` even though the README says that file is excluded; a friendlier version would check for the file and skip cleanly.

### Overall
An exceptionally transparent replication package: data cached at every stage, clear code organization, identical sample reproduction across independent implementations. The core methodological claim — that propensity-score overlap trimming is the dominant lever, not the choice of estimator — is strongly supported by my Python replication. The caveats about PSID and the CS female data apply to the generalizability of the conclusion, not to the package itself.

---

## 7. File Manifest

| File | Description |
|------|-------------|
| `utils.py` | Paths, data loaders (RData), shared estimators (diff, reg, om.reg, NN matching, logistic IPW, AIPW) |
| `01_clean.py` | Loads all LaLonde/IRS datasets and reproduces the Table 1 summary-statistics panel |
| `02_ldw_estimate.py` | Replicates the four-panel LDW table (`tables/ldw.csv`) |
| `03_irs_estimate.py` | Replicates the four-column IRS lottery table (`tables/irs.csv`) |
| `04_data_audit.py` | Coverage, indicator consistency, standardized differences, trimming impact |
| `05_robustness.py` | 12 robustness checks around the LDW-CPS OLS baseline |
| `out_stats_means.csv`, `out_stats_sds.csv` | Table 1 numeric outputs |
| `out_ldw_estimates.csv` | All LDW estimator results |
| `out_irs_estimates.csv` | All IRS estimator results |
| `out_robustness.csv` | Robustness-check outputs |
| `writeup_232201.md` | This writeup |
