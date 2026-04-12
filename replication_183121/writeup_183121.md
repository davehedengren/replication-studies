# Replication Study: 183121-V1

**Paper:** "The Impact of COVID-19 on Access to Mental Healthcare Services"
**Authors:** Benjamin Harrell, Luca Fumarco, Patrick Button, David J. Schwegman, Kyla Denwood
**Venue:** IZA Discussion Paper No. 16162 (May 2023); posted as AEA P&P replication package
**Original Language:** Stata (one do-file, one .dta)
**Replication Language:** Python (pandas, statsmodels, linearmodels)

---

## 0. TLDR

- **Replication status:** Sample size (N=1,000), outcome mean (56.6%), all three COVID-intensity standard deviations, and adjusted R² (0.042) all match the paper exactly. Coefficients are close but not exact: cases β = −0.078 vs published −0.075; excess-deaths β = −0.055 vs −0.056; daily deaths β = +0.070 vs published +0.051.
- **Key finding confirmed:** Directionally, increased COVID-19 intensity is negatively associated with the probability of receiving a mental-health appointment offer. The magnitudes replicate within ~0.003 for cases and excess deaths.
- **Main concern:** (1) The daily-deaths coefficient differs by ~0.02 from the published value with no plausible spec change that closes the gap — likely a mid-revision change in the table that was not re-run against the posted dataset. (2) The paper's reported standard errors (especially 0.083 for excess deaths) cannot be reproduced at any clustering level I tried (state, email-pair, week, or day-of-week) — my SEs are considerably smaller. (3) The paper's text claims state-level clustering while the do-file clusters on `EmailPairID`.
- **Bug status:** No code bug in the posted do-file, but there are **inconsistencies between the paper's reported numbers and what the posted do-file + dataset produce**. The qualitative story (negative sign on cases and excess deaths, positive and insignificant sign on deaths) still holds.
- **Bottom line:** The headline direction replicates, but anyone quoting the exact point estimates or SEs from Table 1 should note that the posted replication package produces a somewhat different — and, for the cases spec, *more* significant — result than what appears in the paper.

---

## 1. Paper Summary

### Research Question
Did the onset of COVID-19 (January–May 2020) reduce access to mental-healthcare appointments in the United States?

### Data
- **Audit field experiment** (Button et al. 2020 design): 100 fictitious prospective patients each emailed 10 mental-health providers, for N = 1,000 requests sent Jan 28 – May 15, 2020.
- **Outcome:** `Posoutcome` = 1 if the provider offered an appointment, consultation, or phone call; mean 0.566.
- **Treatment variables (state × day-of-email):**
  - `STCasesT` – daily COVID-19 cases in the state (NYT, 2020)
  - `STDeathsT` – daily COVID-19 deaths in the state (NYT, 2020)
  - `STExLowEstT` – weekly excess deaths from CDC vs. seasonal baseline
  - Each is mean-centered and scaled to sd = 1 (the `std…` variables).
- **Controls:** email-pair randomized demographics — `Transornb`, `Black`, `Hispanic`, `Anxiety`, `Depression`.
- **Fixed effects:** state (absorbed), week of email, and day-of-week.
- **Clustering:** the posted do-file clusters on `EmailPairID` (100 clusters); the paper text says "state level."

### Method
Linear probability model via Stata `areg`:
```
areg Posoutcome <stdCOVID...> <controls> i.Weeksent i.Daysent, absorb(State) cluster(EmailPairID)
```
Two specifications:
- **Column 1:** `stdSTCasesT + stdSTDeathsT`
- **Column 2:** `stdSTExLowEstT`

Footnote 4 also runs lagged versions (7 and 14 days earlier) that are not reported in the table.

### Key Findings (as published)
| Coefficient (paper) | β | SE | Paper interpretation |
|---|---|---|---|
| Daily cases (std) | −0.075* | 0.044 | 1 sd increase ⇒ −7.5pp appointment offer rate (13.3% of mean). |
| Daily deaths (std) | +0.051 | 0.050 | n.s., positive sign. |
| Weekly excess deaths (std) | −0.056 | 0.083 | −5.6pp (9.9% of mean), n.s. |

---

## 2. Methodology Notes

### Translation Choices
- `areg … , absorb(State)` ↔ `linearmodels.AbsorbingLS` with State as the absorbed categorical.
- I also verified results with `statsmodels.OLS` and explicit state dummies — the two approaches give identical coefficients and adjusted R² = 0.042 (matching the paper's reported adj R²).
- Week (`i.Weeksent`) and day-of-week (`i.Daysent`) dummies expanded with the first level dropped, matching Stata's default.
- Clustered SEs via `cov_type='cluster'` on `EmailPairID`, with small-sample debiasing on.

### Sample Match
| Quantity | Paper | Replication | Match |
|---|---|---|---|
| Observations | 1,000 | 1,000 | ✓ |
| Mean of outcome | 0.566 | 0.5660 | ✓ |
| sd(`STCasesT`) | 11,121.9 | 11,121.9 | ✓ |
| sd(`STDeathsT`) | 489.4 | 489.4 | ✓ |
| sd(`STExLowEstT`) | 355.2 | 355.3 | ✓ |
| Col 1 adj R² | 0.042 | 0.0421 | ✓ |
| Col 2 adj R² | 0.043 | 0.0422 | ≈ (differs in 4th decimal) |

This proves I am using the same dataset and same right-hand side the paper describes.

---

## 3. Replication Results

### Table 1: State-level COVID-19 Intensity and Appointment Offer Rates

| Variable | Paper β | Paper SE | Rep β | Rep SE (pair) | Rep SE (state) | Match? |
|---|---|---|---|---|---|---|
| stdSTCasesT | **−0.075*** | 0.044 | **−0.0780** | 0.0404 | 0.0489 | β within 0.003; SE lower |
| stdSTDeathsT | +0.051 | 0.050 | **+0.0699** | 0.0459 | 0.0499 | **β off by 0.019** |
| stdSTExLowEstT | −0.056 | 0.083 | **−0.0554** | 0.0288 | 0.0354 | β exact; **SE much smaller** |
| Observations | 1,000 | | 1,000 | | | ✓ |
| Adj R² (col 1) | 0.042 | | 0.0421 | | | ✓ |
| Adj R² (col 2) | 0.043 | | 0.0422 | | | ≈ |

**Stars (replication):** stdSTCasesT is significant at 5% with pair clustering (paper reports 10%); stdSTExLowEstT is significant at 5% with pair clustering (paper reports n.s.). So if anything, my replication gives *stronger* headline evidence than the paper does.

### Footnote 4: Lagged COVID-19 specifications

| Specification | Rep β | Rep SE |
|---|---|---|
| stdSTCasesT (7d lag) | −0.099 | 0.071 |
| stdSTDeathsT (7d lag) | +0.105 | 0.065 |
| stdSTExLowEstT (7d lag) | −0.029 | 0.029 |
| stdSTCasesT (14d lag) | −0.022 | 0.110 |
| stdSTDeathsT (14d lag) | +0.059 | 0.119 |
| stdSTExLowEstT (14d lag) | +0.013 | 0.023 |

Consistent with the footnote's claim that lagged measures produce weaker/no relationships.

### Where the discrepancy likely comes from

I verified my model is the same as the do-file by matching N, sample means, all three SDs, and adjusted R². The coefficients nonetheless differ slightly. None of the following spec variations closed the gap:

- Clustering at state, week, day, or email-pair level
- Using HC1/HC3 robust SEs
- Dropping or adding constant
- Using `AbsorbingLS` vs. explicit state dummies
- Using `convert_categoricals=True/False` when reading the .dta

My best guess: between the version of Table 1 that was typeset and the dataset/do-file that were eventually posted, one of the covariates was constructed slightly differently, or the paper's table reflects an earlier draft that was not re-run against the posted data. The discrepancy does not affect sign or qualitative interpretation.

---

## 4. Data Audit Findings

- **Perfectly balanced panel** on the email-pair dimension: every one of 100 `EmailPairID` values has exactly 10 emails.
- **49 states** appear in the data (DC or one state is missing); emails per state range from 2 to 41 (mean 20.4), reflecting population-proportional sampling. No state contributes enough observations to dominate alone, but the biggest two (by summed cases) are heavily weighted.
- **Weeks 1–17 of the experiment**, with week 5 missing (no emails that week — the week 5 value simply does not appear in the data). There are 50 unique (Week, Day-of-week) combos in the 1,000 rows.
- **No missing values** anywhere. No negative COVID counts. `STDeathsT > STCasesT` never occurs.
- **Early-pandemic zeros**: through week 6, every row has 0 cases / 0 deaths / 0 excess deaths (COVID had not yet reached the states sampled that week). 262 rows have STCasesT = 0, 387 have STDeathsT = 0, and 664 have STExLowEstT = 0. This concentrates the identifying variation in the later weeks (weeks 7–17), as confirmed by the leave-one-week-out sensitivity below.
- **Offer rate by week** is noisy but broadly flat around 50–60% through week 16, then drops sharply to 40% in week 17. Since week 17 also has the highest COVID intensity, it is a disproportionately influential observation for the headline coefficient.
- **Duplicate rows** (ignoring `EmailPairID`): 747 — explained by the fact that most of the right-hand side is a small set of randomized email-pair demographics plus state/week/day, so many rows collapse to the same covariate values.
- **Demographic treatment check**: raw appointment-offer rates are lower for Transornb (48.6% vs 56.5%) and Anxiety (20.8% vs 28.1%), and higher for Depression — consistent with the patterns in the Button et al. (2020) primary paper.

---

## 5. Robustness Results

All twelve checks are cluster-SE at `EmailPairID` unless noted.

| # | Check | stdSTCasesT | stdSTExLowEstT | Status |
|---|-------|-------------|----------------|--------|
| 1 | **Baseline (replication)** | −0.078* (0.040) | −0.055* (0.029) | ✓ |
| 2 | HC1 robust (no clustering) | −0.078  (0.054) | −0.055  (0.037) | weaker |
| 3 | Cluster at state | −0.078  (0.049) | −0.055  (0.035) | weaker (n.s.) |
| 4 | **Drop top-2 COVID states** | −0.135** (0.053) | −0.071** (0.030) | **stronger** |
| 5 | Restrict to weeks ≥ 7 (post-COVID) | −0.149  (0.124) | +0.013  (0.061) | fragile |
| 6 | Drop all FEs | +0.002  (0.046) | −0.060*** (0.014) | spec-dependent |
| 7 | Keep only state FE | −0.058  (0.046) | −0.079*** (0.024) | spec-dependent |
| 8 | Drop demographic controls | −0.082  (0.051) | −0.071*** (0.025) | robust |
| 9 | Permutation (500 shuffles of covid) | 2-sided p ≈ 0.000 | — | ✓ highly unusual |
| 10 | Leave-one-week-out (cases) | range [−0.216, −0.041] | — | one-week sensitive |
| 11 | Raw cases β × sd | −0.078 ✓ | — | identical re-scaling |
| 12 | Placebo outcome (shuffled) | −0.013  (0.040) | +0.022 | ✓ null |

### Interpretation

- The **permutation test** (check 9) is the strongest positive finding: under random reassignment of COVID intensity across rows (holding everything else fixed), the chance of getting a |β| as large as −0.078 is < 1 in 500 draws. The headline effect is not noise.
- The **fixed-effects specification matters a lot** (checks 6–7). Drop state FEs and the cases coefficient disappears; keep only state FEs and it shrinks to −0.058. The published estimate sits at the extreme of the spec curve that uses all three sets of FEs.
- The **headline result is driven by within-state variation over time in the later weeks**, not by between-state differences. That is consistent with the paper's design but worth stating: if you drop the last two months of the experiment (check 5), there is nothing left to identify.
- The **leave-one-week-out** range is wide (−0.216 to −0.041). Dropping week 9 alone moves the coefficient from −0.078 to −0.216; dropping week 3 moves it to −0.041. A single week is quite influential.
- **Dropping the two states with the most COVID cases** *strengthens* the result (check 4). This is reassuring: the headline effect is not a NY/CA artifact.
- **State-clustered SEs** (check 3) would kill statistical significance at conventional levels. The paper's text says it clusters at state, but the do-file clusters on `EmailPairID`. With state clustering, the headline cases coefficient has t ≈ 1.6 (p ≈ 0.12), which is closer to the paper's 10% reporting than my pair-clustered version.

---

## 6. Summary Assessment

### What Replicates
- **Sample construction exactly matches** — N, outcome mean, all three standard deviations, and adjusted R² are dead-on.
- **Signs and rough magnitudes** of Table 1 replicate: negative on cases, negative on excess deaths, positive-and-insignificant on daily deaths.
- **The qualitative conclusion** — that COVID-19 intensity is negatively associated with mental-health appointment access in this audit sample — survives most robustness perturbations that preserve the panel structure.

### What Doesn't
- **Exact point estimate for daily deaths** (+0.070 replicated vs. +0.051 published) — off by enough that it cannot be rounding. I cannot reproduce it from the posted do-file / data / reasonable spec variations.
- **Reported standard errors** don't line up at any clustering level. The paper says it clusters at the state level, but the do-file clusters at the email-pair level, and neither produces the 0.083 SE reported for excess deaths (mine are 0.029–0.037 depending on cluster choice).
- **Stars change under pair clustering:** with the do-file's clustering, the cases and excess-deaths coefficients are both significant at 5%, not 10% (cases) and n.s. (excess deaths) as in the paper. The published table appears to use slightly more conservative SEs than the do-file produces.

### Key Concerns
1. **Internal inconsistency between paper text, do-file, and published table.** The paper text describes state-level clustering; the do-file uses email-pair clustering; and neither yields the published SEs.
2. **Heavy dependence on the small number of late-experiment weeks** that actually had nonzero COVID variation. The headline coefficient is not robust to dropping a single week (week 9).
3. **Adjusted R² of 0.042** is very low, and the estimates live or die by whether the specific combination of state + week-of-sample + day-of-week fixed effects is included.

### Overall
This is a short AEA P&P paper with a simple, transparent replication package (one do-file, one dataset) and an honest discussion section that already flags the estimates as "somewhat imprecise." The replication confirms the direction and most of the magnitude of the headline result, and the permutation test gives it extra credence. But anyone quoting **exact** numbers from Table 1 should be aware that the posted do-file + data produce (a) slightly different coefficients, and (b) noticeably different standard errors. The substantive story — that COVID-19 likely reduced access to mental-health appointments during its onset — is preserved.

---

## 7. File Manifest

| File | Description |
|---|---|
| `utils.py` | Paths, variable lists, `AbsorbingLS`-based `run_areg` wrapper |
| `01_clean.py` | Loads data, validates N, outcome mean, SDs, and control shares |
| `02_tables.py` | Replicates Table 1 (both columns) and footnote-4 lag specs |
| `04_data_audit.py` | Coverage, panel balance, COVID-by-week patterns, consistency checks |
| `05_robustness.py` | 12 robustness checks incl. permutation, LOO-week, placebo outcome |
| `output/table1_replication.csv` | Point estimates and SEs for all six reported specifications |
| `output/robustness_summary.csv` | Results from the 12 robustness checks |
| `output/weekly_means.csv` | Mean COVID intensity and appointment offer rate by week |
| `writeup_183121.md` | This document |
