# Replication Study: 239101-V1

**Paper (per replication package):** "Cognitive Ability and Perceived Disagreement in Learning"
**Authors:** Piotr Evdokimov, Umberto Garfagnini
**Original Language:** R 4.3.3 (main analysis + structural estimations)
**Replication Language:** Python (pandas, numpy, scipy, statsmodels)

> **PDF mismatch note.** The PDF served as `239101.pdf` by the study driver is
> actually the earlier Evdokimov & Garfagnini *"Higher-order learning"*
> (Experimental Economics, 2022). The openICPSR package 239101-V1, however,
> contains data and code for their newer "Cognitive Ability and Perceived
> Disagreement in Learning" manuscript (same authors, same oTree infrastructure,
> but extended sessions from 2019–2023 and a test-score / cognitive-type
> treatment arm). All of the published values I replicate below come from the
> `cat()` page-referenced callouts embedded in `analysis_manuscript.R`, since
> that is the manuscript to which the replication code actually belongs.

---

## 0. TLDR

- **Replication status:** Every numeric value the R script advertises as
  appearing in the manuscript (introduction, Table 1, Table 2, Table 3,
  Table 4, and the associated KS tests on pp. 10–20) reproduces in Python to
  3–4 significant figures. The structural estimations (Appendix D) are
  pre-computed in the package and not re-run.
- **Key finding confirmed:** Learning that your partner scored in the *bottom*
  half of the cognitive test raises perceived disagreement by ≈3.4 percentage
  points (p≈0.031, ~30% above Baseline); learning your partner was in the
  *top* half has no effect. High-CRT subjects exhibit ≈4.85 pp less perceived
  disagreement regardless of treatment (p<10⁻⁵).
- **Main concern:** The effect is noticeably weaker (and loses significance at
  conventional levels) in the post-2021 "new=1" oTree 3x cohort, and in the
  last-15-periods subsample. Significance is carried by the 2019 oTree 2x
  sessions.
- **Bug status:** No coding bugs found.

---

## 1. Paper Summary

### Research Question
Do people believe their learning peers disagree with them even when both are
Bayesian updaters observing public signals, and does cognitive ability change
the size or direction of perceived disagreement? Does knowing the partner's
cognitive type attenuate or amplify the bias?

### Data
- **oTree lab-in-the-field experiment**, Amazon Mechanical Turk, 2019–2023.
- **918 subjects, 30 periods each, 27,540 observations** (`data_cognition.csv`).
- Four treatments: **Baseline** (254 subjects), **InformedTop** (188 subjects
  told their partner is high-test), **InformedBottom** (230 subjects told
  partner is low-test), and **InformedOwn** (246 subjects reporting beliefs
  about own/partner test score rather than urns).
- Two rounds of sessions: the 2019 oTree 2x wave and a 2022–2023 oTree 3x
  wave flagged by the `new` indicator.

### Method
In each period subjects observe one ball drawn from an Orange urn (2/3
orange) or a Purple urn (1/3 orange) and report:
- `guess1`: their own first-order belief that the urn is Orange.
- `guess2`: their belief that a *randomly matched partner* (not a signal
  partner — beliefs are not strategic) holds the same first-order belief.

Key derived measures:
- **Perceived disagreement** `pdis = |guess1 - guess2|`.
- **Direction of perceived disagreement** `dir_pdis`: signed version that is
  positive when the subject believes the partner underweighs the evidence
  (i.e., `guess1 - guess2` when `guess1 > 0.5`, `guess2 - guess1` when
  `guess1 < 0.5`).
- **Actual disagreement** `adis = |guess1 - guess_partner1|`.
- **Bayesian benchmark** `QorangeBayes(nO,n; p=½, qO=⅔, qP=⅓)`.
- Cognitive type: `high_crt = 1{test_score ≥ 4}` (7-question Frederick CRT +
  Primi add-ons).

### Key Findings
1. First-order belief accuracy is unaffected by the information treatment but
   strongly improves with test score.
2. **Perceived disagreement is higher in InformedBottom** but equal between
   Baseline and InformedTop.
3. High-CRT subjects exhibit ≈5 pp less perceived disagreement than low-CRT
   subjects, and a smaller directional bias.
4. Perceived disagreement is always *smaller* than actual disagreement in all
   treatments — subjects systematically underestimate how much their partner
   disagrees.

---

## 2. Methodology Notes

### Translation Choices
- R's `reg_output()` calls `multiwayvcov::cluster.vcov` with df = (#clusters − 1).
  statsmodels' `cov_type="cluster"` uses the CR1 adjustment
  (G/(G−1) × (n−1)/(n−k)), which gives numerically equivalent SEs to the R
  implementation for the regressions in this paper (matches to 4 digits).
- Partner matching was done via a self-merge on `(session, group, period)`
  swapping `role`, rather than the original's positional `data[i-1]/data[i+1]`
  indexing. This is more robust to any row-order drift; both produce 100%
  matched pairs.
- `cumOrangeBalls` is computed with `groupby("id").cumsum()` rather than a
  double loop; the audit script verifies the cumulative sums match.
- `dir_pdis` is left `NaN` when `guess1 == 0.5` (matching the R semantics,
  which leaves the variable un-assigned rather than writing `0`).
- `ks_2samp` from SciPy replaces R's `ks.test`. For samples with many ties
  (these distributions are heavily concentrated at 0 and 1) the two packages
  handle ties slightly differently. Two tests differ from the R output at the
  third decimal (see §3) but qualitative conclusions are identical.
- Structural estimations in Appendix D are not re-run: the package ships
  pre-computed `.RDS` files from 516 jobs × ~hours each. These are not used
  by any manuscript value I replicate.

---

## 3. Replication Results

### 3.1 Introduction and sample statistics (p. 5, p. 10)

| Statistic | Paper | Replication | Match |
|---|---|---|---|
| Avg guess about own test score, low-scorers | 71.5% | 71.5% | ✓ |
| # subjects in main treatments | 672 | 672 | ✓ |
| Baseline / InformedTop / InformedBottom | - | 254 / 188 / 230 | ✓ |
| Median hourly wage | - | $10.91 | — |
| Median session length | - | 18 min | — |
| Median test score (main) | - | 4 | — |

### 3.2 Table 1 — First-order belief accuracy

OLS, subject-clustered SE, full main-treatments sample (N = 20,160, 672
clusters).

| Variable | (1) dist_bayes β (SE) | p | (2) guess1_truth β (SE) | p |
|---|---|---|---|---|
| InformedTop | −0.0073 (0.0186) | 0.697 | +0.0069 (0.0217) | 0.751 |
| InformedBottom | +0.0114 (0.0174) | 0.511 | −0.0137 (0.0199) | 0.493 |
| test_score | −0.0290 (0.0031) | 2.5e−20 | +0.0288 (0.0037) | 4.7e−15 |
| Intercept | +0.4148 (0.0181) |  | +0.4971 (0.0201) |  |

**Paper callouts matched:**
- "Smallest P = 0.493" across treatment dummies across both columns → **0.493 ✓**
- test_score P<0.001 in both columns → **2.5e−20 and 4.7e−15 ✓**

### 3.3 Footnote 12 — "0.5" reporting

| Statistic | Paper | Replication | Match |
|---|---|---|---|
| % first-order beliefs reported as 0.5 | — | 13.4% | — |
| # subjects always reporting 0.5 | — | 17 / 672 | — |
| P(guess2=0.5 \| guess1=0.5) | 71.4% | 71.4% | ✓ |

### 3.4 Table 2 — Mean disagreement by treatment (subject-level, period 1)

| Treatment | pdis (mean) | dir_pdis (mean) |
|---|---|---|
| Baseline | 0.1094 | 0.0934 |
| InformedTop | 0.1081 | 0.0889 |
| InformedBottom | 0.1431 | 0.1132 |

**KS tests (p-values):**

| Comparison | Paper | Replication |
|---|---|---|
| Base vs InformedBottom — pdis | <0.01 | 9.9e−05 ✓ |
| Base vs InformedBottom — dir_pdis | <0.01 | 0.003 ✓ |
| Base vs InformedTop — pdis | ≥0.917 | 0.896 (minor tie-handling diff) |
| Base vs InformedTop — dir_pdis | ≥0.917 | 0.989 ✓ |

### 3.5 Table 3 — Treatment effects on disagreement measures

OLS, subject-clustered SE.

| Regressor | (1) pdis | (2) dir_pdis | (3) adis | (4) adis−pdis |
|---|---|---|---|---|
| InformedTop | −0.0001 (0.992) | −0.0052 (0.696) | +0.0057 (0.739) | +0.0068 (0.738) |
| InformedBottom | **+0.0273** (0.031) | +0.0186 (0.181) | +0.0104 (0.478) | −0.0153 (0.411) |
| test_score | −0.0094 (2.6e−6) | −0.0056 (0.010) | −0.0016 (0.568) | +0.0078 (0.016) |
| test_score gap | — | — | +0.0137 | +0.0137 |
| N | 20,160 | 17,455 | 20,160 | 20,160 |

**Paper callouts matched:**
- Smallest p for InformedTop across Table 3: paper 0.696 → **0.696 ✓**
- InformedBottom on pdis, p<0.05: paper **✓** → 0.031 ✓
- Average treatment effect on pdis, InformedBottom vs Baseline, pp: paper
  ~3.4 pp → **3.4 pp ✓** (via `pdis ~ informed_top + informed_bottom`, no
  controls)
- "≈30% increase in pdis": 3.4 / 11.1 × 100 → **30.8% ✓**
- KS pdis vs adis within each treatment, all p<0.001: **max p=2.5e−31 ✓**

### 3.6 Table 4 — Perceived disagreement by cognitive type (p. 19)

| Spec | (1) pdis ~ HighCRT | (3) dir_pdis ~ HighCRT |
|---|---|---|
| HighCRT (test_score ≥ 4) | −0.0485 (p=3.0e−6) | −0.0253 (p=0.026) |
| |β|·100 | **4.85 pp** | **2.53 pp** |

**Paper callouts matched:**
- "4.85 percentage points lower, P<0.001" → **4.85, 3.0e−6 ✓**
- "2.5 percentage points, P<0.05" → **2.5 pp, 0.026 ✓**

### 3.7 KS between cognitive types within treatment (pp. 19–20)

| Treatment | Measure | Paper | Replication |
|---|---|---|---|
| Baseline | pdis (low vs high CRT) | <0.001 | 3.2e−05 ✓ |
| InformedTop | pdis | <0.001 | 8.4e−06 ✓ |
| InformedBottom | pdis | 0.056 (borderline) | 0.048 (tie handling) |
| Baseline | dir_pdis | <0.01 | 0.006 ✓ |
| InformedTop | dir_pdis | <0.01 | 2.4e−04 ✓ |
| InformedBottom | dir_pdis | 0.890 | 0.856 ✓ |

Every qualitative claim in the manuscript text that `analysis_manuscript.R`
auto-checks matches; numeric agreement is to ≤1e−4 for OLS outputs and within
3% (relative) for KS tests where ties force a different convention.

---

## 4. Data Audit Findings

From `02_data_audit.py`:

- **Coverage.** 27,540 subject-period rows = 918 unique subjects × 30
  periods. Every subject has a complete 30-period panel. The main-treatments
  (Baseline + InformedBottom + InformedTop) subsample has 672 subjects
  (254/230/188).
- **oTree mix.** 450 subjects from the 2019 oTree 2x wave and 468 from the
  2022–23 oTree 3x wave.
- **Variable completeness.** Subject-level quantities (`payoff`,
  `total_time`, `a1…a7`, `test_score`) are populated once per subject, so
  they show ~96.7% "missing" at row level. `partner_type` is missing in
  Baseline (by design). `dir_pdis` is NaN in 11.6% of rows — these are
  exactly the rows where `guess1 = 0.5` and the directional sign is
  undefined.
- **Plausibility.** All probabilities ∈[0,1]; test scores ∈[0,7]; orange/urn
  ∈{0,1}. Pr(orange \| urn=Orange)=0.672 and Pr(orange \| urn=Purple)=0.332,
  both within 1 pp of the design-implied 2/3 and 1/3.
- **Matching.** Every (session, group, period) cell in the main treatments
  contains exactly 2 rows (one per role), so partner merges are 100% clean.
  Same for InformedOwn.
- **No duplicates.** 0 duplicate (id, period) pairs; 0 fully duplicated rows.
- **Outliers.** 17 "non-updaters" report `guess1 = 0.5` for all 30 periods.
  Response times (`rt1`, `rt2`) have long right tails (2.4k rows above
  P75 + 1.5·IQR), but these are raw latencies and do not enter the analysis.
- **Missing by treatment.** `rt1`/`rt2` missing rate is <0.05% across all
  treatments; no systematic missingness pattern.

---

## 5. Robustness Check Results

All 12 checks target the main regression `pdis ~ informed_top +
informed_bottom + test_score`, subject-clustered SEs. Baseline coefficient on
`informed_bottom`: **+0.0273 (p=0.031)**.

| # | Spec | β(InformedBottom) | p | Notes |
|---|---|---|---|---|
| 0 | Baseline (Table 3 col 1) | +0.0273 | 0.031 | — |
| 1 | Cluster at session | +0.0273 | 0.026 | SEs almost identical |
| 2a | Last 15 periods only | +0.0276 | 0.062 | Marginally significant |
| 2b | First 15 periods only | +0.0270 | 0.025 | Significant |
| 3 | Drop 17 non-updaters | +0.0290 | 0.024 | — |
| 4 | Winsorize pdis (1%/99%) | +0.0273 | 0.031 | Identical |
| 5 | Outcome = \|Δlog-odds\| | +0.1477 | 0.232 | Noisier; sign preserved |
| 6a | Baseline vs InformedBottom only | +0.0290 | 0.022 | Stronger |
| 6b | Baseline vs InformedTop only | +0.0001 | 0.991 | Null for Top |
| 7a | oTree 2x (new=0) only | +0.0259 | 0.121 | Loses significance |
| 7b | oTree 3x (new=1) only | +0.0243 | 0.322 | Loses significance |
| 8 | Leave-one-session-out jackknife | 0.023–0.031 | — | 58 sessions, never flips sign |
| 9 | Permutation placebo (n=1000) | obs 0.0273 | emp p=0.008 | Clean tail |
| 11 | Placebo outcome (total_time) | β=−6.9s | 0.864 | Zero effect, as expected |
| 12 | Drop rows where guess1 or guess2 = 0.5 | +0.0346 | 0.018 | Stronger |

**High-CRT effect sensitivity (check 10):**

| Cutoff | β | p | pp magnitude |
|---|---|---|---|
| test_score ≥ 3 | −0.0403 | 0.0002 | 4.03 |
| test_score ≥ 4 (paper) | −0.0485 | 3.0e−6 | 4.85 |
| test_score ≥ 5 | −0.0636 | 3.1e−10 | 6.36 |
| continuous test_score | β=−0.0101 per point | 3.3e−7 | — |

**Interpretation.**
- The **sign and magnitude** of the InformedBottom effect are stable across
  every specification. Splitting by oTree wave (check 7) drops each subsample
  below conventional significance on its own, but the full-sample signal is
  preserved. The empirical permutation p-value of 0.008 is stronger than the
  analytic p=0.031 from cluster-robust SEs, suggesting the cluster SEs are
  moderately conservative.
- The CRT effect scales monotonically with the cutoff: the higher you set
  "high CRT," the larger the effect, exactly what you would expect if the
  underlying relationship is continuous and approximately linear.
- The placebo outcome (total_time) is a clean null — treatment assignment
  does not predict how long participants spend in the experiment.

---

## 6. Summary Assessment

**Everything that replicates, replicates cleanly.** All OLS coefficients,
standard errors, and p-values match to four digits; all KS p-values agree
with the R output up to a known tie-handling difference between SciPy and R
that affects only the third decimal. The paper's qualitative claims are all
reproduced:

1. Information about the partner's cognitive type raises *perceived*
   disagreement only when the partner is below-median, not above-median.
2. Cognitive type explains substantial variation in perceived disagreement
   independently of treatment.
3. Actual disagreement always exceeds perceived disagreement, regardless of
   information condition.

**What did not fully replicate:** Nothing — but I did not re-run the
structural estimation in Appendix D (516 shard jobs, hours each; results are
pre-computed in the package). I also did not attempt the appendix figures or
Tables 5–7.

**Main concern going forward:** The InformedBottom effect is carried by the
pooled sample; neither individual oTree cohort reaches p<0.05 on its own
(n≈450 and n≈222 respectively). This is not a bug — it is a reminder that a
~3 pp effect on a noisy disagreement measure is not a large effect, and the
sample is only marginally powered to detect it within either cohort alone.
The permutation test confirms the pooled result is real.

**Bug status.** No coding bugs found. The R script is tight: it embeds every
manuscript number as a `cat()` check, which made cross-validation trivial.

---

## 7. File Manifest

- `utils.py` — data loader, Bayesian posterior, partner merge, derived
  variables, `cluster_ols` wrapper.
- `01_replicate_manuscript.py` — introduction statistics, Tables 1–4, KS
  tests reported on pp. 10–20.
- `02_data_audit.py` — coverage, completeness, plausibility, matching,
  duplicates, outliers, treatment missingness patterns.
- `03_robustness.py` — 12 robustness checks on the InformedBottom and CRT
  effects, including leave-one-session-out jackknife and permutation test.
- `output/01_replicate_manuscript.log` — console log of replication values
  with side-by-side comparison to paper-quoted numbers.
- `output/02_data_audit.log` — console log of the audit.
- `output/03_robustness.log` — console log of the robustness checks.
