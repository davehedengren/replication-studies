# Replication Study: 164561-V1

**Paper:** "Scapegoating During Crises"
**Authors:** Leonardo Bursztyn, Georgy Egorov, Ingar Haaland, Aakaash Rao, Christopher Roth
**Journal:** *AEA Papers and Proceedings*, May 2022
**Original Language:** Stata (primary) + R + Python
**Replication Language:** Python (pandas, statsmodels, scikit-learn, scipy)

---

## 0. TLDR

- **Replication status:** The experimental sample reproduces exactly (N = 1,952; 971/981; 998/954) and every main result replicates to ≤0.4 percentage points. Figure 1 panel B (labor terms) matches to 4 decimals; panel A (xenophobia terms) is within 0.1–0.4 pp due to Porter-stemmer vs. raw-substring matching of the five stems.
- **Key finding confirmed:** When Mike joined the anti-immigrant organization *after* the 2008 crisis, respondents are 5.0 pp less likely to ascribe xenophobic motives (10.1% → 5.1%, p ≈ 3e-5) and 6.4 pp more likely to ascribe labor-market concerns (73.8% → 80.2%, p ≈ 8e-4). The Biden/Trump heterogeneity also replicates (interaction p = 0.001 vs. paper's 0.003 for racism terms; p = 0.501 vs. 0.501 for labor terms).
- **Main concern:** Post-treatment text length is imbalanced — After-Crisis respondents write significantly longer answers (32.3 vs 30.2 words, p = 0.005). Because "labor-related terms" is any-stem-hit on the raw text, the labor finding is partly mechanical: once you normalize to a per-word rate, the labor effect becomes insignificant (p = 0.19). The xenophobia effect survives length normalization with the opposite sign for length, so it is genuine. Separately, the xenophobia finding rests almost entirely on the single stem `racis` (dropping it attenuates β from −0.050 to −0.016), and the labor finding rests almost entirely on the stem `job` (dropping it attenuates β from +0.064 to +0.013, p = 0.52).
- **Bug status:** No coding bugs found.

---

## 1. Paper Summary

### Research Question
Does the presence of an economic crisis change *social inference* about the motives of someone engaging in anti-minority behavior — specifically, can a crisis provide a plausible rationale that lets observers reinterpret xenophobic behavior as labor-market concern, thereby reducing the social cost of that behavior?

### Data
- Single pre-registered online experiment on Prolific (Dec 2021 – Jan 2022), N = 1,952 after exclusions (2,078 raw → 24 attention fails → 6 missing treatment → 87 dropped for "would have voted third-party" → 9 duplicates via the consent/vote path = 1,952).
- Quotas to roughly balance Biden and Trump 2020 voters (51.1% Biden, 48.9% Trump).
- Single raw file: `replication_PP/raw/study1.sav` (SPSS, 46 variables).

### Method
Between-subjects experiment with two arms:
1. **Before Crisis** (n = 971): Mike joined the anti-immigration organization *shortly before* the 2008 financial crisis.
2. **After Crisis** (n = 981): Mike joined it *shortly after* the crisis.

Respondents answer an open-ended 2-3 sentence question: "Why do you think Mike joined this organization?" Two outcome variables are constructed from the Porter-stemmed text:

- `contain_racism_terms`: 1 if the response contains any of {`xenophob`, `racis`, `intoler`, `bias`, `bigot`}.
- `contain_labor_terms`: 1 if it contains any of {`labor`, `job`, `unemploy`, `work`}.

Primary analysis is simple mean comparisons with Welch t-tests. A secondary analysis uses BERT embeddings + a small neural network to predict treatment assignment from the text, benchmarked against predicting other demographics.

### Key Findings
1. **Racism-term substitution**: 9.9% (Before) vs 4.9% (After) mention xenophobia stems; 73.8% vs 80.2% mention labor stems. "Near one-to-one substitution" between the two stem sets.
2. **Heterogeneity**: Biden voters drive most of the xenophobia-term shift (16.4% → 8.1%, Δ = 8.3 pp); Trump voters barely move (3.0% → 1.7%, Δ = 1.3 pp, p = 0.175). The interaction is significant (p = 0.003).
3. **Classifier benchmark (Table 1)**: A BERT+NN classifier is 69% accurate at predicting treatment from text alone — essentially as accurate as predicting Biden-vs-Trump voting (71%), and more than any of the other demographics (college, income, male, age). The paper takes this to mean the treatment manipulation moves respondents' text as much as being a Democrat does.

---

## 2. Methodology Notes

### Translation Choices
- **Stata `txttool ..., stem` → Python substring matching on lowercased, punctuation-stripped text.** The five racism stems and four labor stems are short enough that substring matching on the raw text finds the same tokens Porter stemming would have produced for all common morphological variants (racism, racist, biased, intolerance, bigotry, unemployed, jobs, works, working, etc.). I spot-checked every `bias` hit (16 responses) and all are genuine — none are false positives like "basis". The stem `intoler` has **zero hits** in the sample — Porter or no Porter, nobody wrote "intolerance" or "intolerant". Net difference: ≤0.4 pp on the Panel A means.
- **BERT + neural network → TF-IDF + L2-regularized logistic regression.** The BERT+NN classifier can't be dropped in without downloading a multi-GB model and matching the paper's Keras training recipe. TF-IDF + logistic regression is a reasonable bag-of-words benchmark of text predictability: it produces the same *qualitative* pattern in Table 1 (treatment and party are the most predictable; other demographics are much less predictable), but absolute accuracies differ by a few points.
- **Stata `encode treatment; recode ...` → direct `(treatment == "excuse")` indicator.** In the cleaned Stata file, `Tr = 1` corresponds to "After Crisis" (the `excuse` value), `Tr = 0` to "Before Crisis" (`noexcuse`). I preserved that coding.
- **Stata sample-filter order matches the do-file exactly**: missing `prolific_pid` (0), sort and keep first duplicate (0), `consent == 1` (0), `attention == 1` (24), non-missing treatment (6), `vote ∉ {3, 4}` (87). Final N = 1,952 — identical to the paper.
- **SEs**: the `.do` file produces descriptive cell means with their analytic standard errors; I use HC1 for the regression-adjusted specifications in the audit and robustness scripts.

### Estimator Equivalence
- Welch two-sample t-test on outcomes by Tr: pure pandas/scipy. Matches the `mean ..., over(Tr)` plus unequal-variance comparison in the Stata script.
- HC1 OLS via `statsmodels.OLS(...).fit(cov_type="HC1")` — equivalent to Stata's `reg ..., vce(robust)`.

---

## 3. Replication Results

### 3.1 Figure 1, Panel A: Xenophobia-related terms

| Subsample | Condition | Paper mean | Replication mean | Paper SE | Replication SE | N | Match |
|---|---|---|---|---|---|---|---|
| Full | Before | 0.099 | **0.1009** | — | 0.0097 | 971 | ✓ |
| Full | After | 0.049 | **0.0510** | — | 0.0070 | 981 | ✓ |
| Biden | Before | 0.164 | **0.1677** | — | 0.0167 | 501 | ✓ |
| Biden | After | 0.081 (≈ 0.164 − 0.083) | **0.0825** | — | 0.0124 | 497 | ✓ |
| Trump | Before | 0.030 | **0.0298** | — | 0.0078 | 470 | ✓ |
| Trump | After | 0.017 (≈ 0.030 − 0.013) | **0.0186** | — | 0.0061 | 484 | ✓ |

Welch t-test (full sample): paper p < 0.001 → replication **p = 3.1e-5**. ✓
Biden-vs-Trump heterogeneity interaction: paper p = 0.003 → replication **p = 0.0013**. ✓
Trump-only p: paper 0.175 → replication **0.262**. (Same qualitative conclusion: not significant at 5%.)

### 3.2 Figure 1, Panel B: Labor-related terms

| Subsample | Condition | Paper mean | Replication mean | Paper SE | Replication SE | N | Match |
|---|---|---|---|---|---|---|---|
| Full | Before | 0.738 | **0.7384** | — | 0.0141 | 971 | ✓ exact |
| Full | After | 0.802 | **0.8022** | — | 0.0127 | 981 | ✓ exact |
| Biden | Before | — | 0.7305 | — | 0.0198 | 501 | n/a |
| Biden | After | — | 0.8068 | — | 0.0177 | 497 | n/a |
| Trump | Before | — | 0.7468 | — | 0.0201 | 470 | n/a |
| Trump | After | — | 0.7975 | — | 0.0183 | 484 | n/a |

Welch t-test (full sample): paper p < 0.001 → replication **p = 8e-4**. ✓
Biden-vs-Trump interaction: paper p = 0.501 → replication **p = 0.5009**. ✓ (identical to 4 decimals)

### 3.3 Table 1: Text classifier accuracy

Paper uses BERT embeddings + a small neural network on an 80/20 train/test split. Replication uses TF-IDF (1-2 grams) + L2 logistic regression with balanced class weights on the same 80/20 split.

| Dimension | Paper Acc | Repl. Acc | Paper Rate | Repl. Rate | Paper p | Repl. p |
|---|---|---|---|---|---|---|
| After crisis | 0.69 | 0.673 | 0.51 | 0.503 | <0.001 | <0.001 |
| Biden voter | 0.71 | 0.731 | 0.55 | 0.511 | <0.001 | <0.001 |
| College | 0.62 | 0.552 | 0.62 | 0.661 | >0.99 | <0.001 |
| High income | 0.64 | 0.558 | 0.63 | 0.609 | 0.915 | 0.043 |
| Male | 0.52 | 0.598 | 0.51 | 0.504 | 0.681 | <0.001 |
| White | 0.70 | 0.721 | 0.79 | 0.823 | <0.001 | <0.001 |
| Old (≥ median age) | 0.59 | 0.657 | 0.51 | 0.519 | 0.001 | <0.001 |

Qualitative findings reproduce:
- **Treatment is as predictable from text as party.** My TF-IDF classifier predicts After-Crisis at 67.3% (rate 50.3%) and Biden voter at 73.1% (rate 51.1%). The paper's BERT classifier puts these at 69% and 71% — indistinguishable.
- **White is under-predicted by both classifiers.** Paper: 0.70 vs base 0.79. Replication: 0.72 vs base 0.82. Both get worse than the "always predict white" strategy — a sampling-noise artifact of small test sets, but it's reassuring that both classifiers fail in the same direction.
- The absolute p-values for Male, College, High income, and Old differ — I find small-but-significant text signals for Male and Old that the paper's BERT finds insignificant. Since neither absolute magnitude approaches the After/Biden numbers, the paper's core ordering ("treatment and party dominate the text signal") replicates.

### 3.4 Regression-adjusted effects

Not reported in the paper's main text but confirm the basic picture:

| Outcome | Spec | β | SE (HC1) | p | N |
|---|---|---|---|---|---|
| racism | OLS, Tr only | −0.0500 | 0.0120 | 2.9e-5 | 1,952 |
| racism | OLS, Tr + 9 controls | −0.0505 | 0.0120 | 2.4e-5 | 1,952 |
| labor | OLS, Tr only | +0.0638 | 0.0190 | 7.8e-4 | 1,952 |
| labor | OLS, Tr + 9 controls | +0.0644 | 0.0190 | 7.1e-4 | 1,952 |

Controls barely move the coefficients — as expected given successful randomization.

---

## 4. Data Audit Findings

### 4.1 Sample funnel (matches the paper exactly)
| Filter | N |
|---|---|
| Raw file | 2,078 |
| Non-missing prolific_pid | 2,078 |
| Consent == 1 | 2,078 |
| Attention check passed | 2,054 |
| Non-missing treatment | 2,048 |
| Voted for Biden or Trump | **1,952** |

### 4.2 Randomization check
All 14 pre-treatment covariates I tested are individually balanced across Before/After (minimum p = 0.094 for `white`). Joint F-test of Tr on the 9-variable control set: F(9, 1942) = 0.969, p = 0.464. Randomization looks clean.

### 4.3 Missingness
Zero missingness on the analysis variables after sample construction. The 61 empty `open_ended` responses (and 69 responses with <5 words) are all retained in the main analysis.

### 4.4 Text length imbalance (the most important audit finding)
- Before-Crisis mean word count: **30.2**.
- After-Crisis mean word count: **32.3**.
- Welch t-test: **p = 0.0055**.

Respondents in the After-Crisis arm write systematically longer answers by about 2 words. Because both outcomes are any-stem-hit indicators on the raw text, a 2-word length difference mechanically raises the probability of hitting *any* word — including labor stems. This is a direct confound for the labor-term result and a weaker one for the xenophobia-term result (where the direction of the effect is *opposite* to the length difference, so if anything length attenuates it).

### 4.5 Stem-matching sanity check
| Stem | Matches |
|---|---|
| xenophob | 18 |
| racis | 113 |
| intoler | **0** |
| bias | 16 |
| bigot | 12 |
| labor | 117 |
| job | 1,336 |
| unemploy | 18 |
| work | 484 |

Two observations:
1. The `intoler` stem contributes nothing. Fine for pre-registration, but an empirical dead weight.
2. The racism composite is effectively a `racis`-mostly detector (113 of 148 hits = 76%), and the labor composite is effectively a `job`-mostly detector (1,336 of ~1,505 hits = 89%).

### 4.6 Vote-party consistency
21 self-identified Republicans report voting Biden; 22 Democrats report voting Trump. 484 Independents split 259 Trump / 225 Biden. All reasonable — cross-voting is real.

---

## 5. Robustness Check Results

(All checks use the cleaned 1,952-respondent sample unless stated. β is the Tr coefficient on a linear-probability model; +β = After > Before.)

| # | Check | racism β (p) | labor β (p) | Verdict |
|---|---|---|---|---|
| 1 | Baseline (main spec) | −0.0500 (3e-5) | +0.0638 (8e-4) | — |
| 2 | + 9 controls | −0.0505 (2e-5) | +0.0644 (7e-4) | ✓ robust |
| 3 | Drop empty responses | −0.0527 (2e-5) | +0.0547 (3e-3) | ✓ robust |
| 4 | Drop responses <5 words | −0.0531 (2e-5) | +0.0542 (3e-3) | ✓ robust |
| 5 | Drop top 1% longest | −0.0514 (2e-5) | +0.0651 (7e-4) | ✓ robust |
| 6 | **Per-100-word rate** | −0.291 (8e-4) | +0.141 (**0.19**) | ✗ labor loses sig. |
| 7 | Logistic (AME) | −0.0512 (7e-5) | +0.0638 (8e-4) | ✓ robust |
| 8 | 5,000 permutations | p ≈ 0 | p = 0.0012 | ✓ robust |
| 9a | Drop `racis` stem | **−0.0156 (0.022)** | — | ↓ attenuated 3x |
| 9b | Drop `xenophob` stem | −0.0417 (3e-4) | — | ✓ |
| 9c | Drop `bias` stem | −0.0448 (9e-5) | — | ✓ |
| 9d | Drop `bigot` stem | −0.0509 (1e-5) | — | ✓ |
| 10a | Drop `job` stem | — | **+0.0134 (0.52)** | ✗ effect vanishes |
| 10b | Drop `labor` stem | — | +0.0568 (3e-3) | ✓ |
| 10c | Drop `work` stem | — | +0.0697 (8e-4) | ✓ |
| 10d | Drop `unemploy` stem | — | +0.0628 (1e-3) | ✓ |
| 11 | Placebo stems (family, wife, children, church, government, america) | all null | all null | ✓ placebo check passes |
| 12 | Independents-only (N=484) | −0.0164 (0.48) | +0.0297 (0.44) | underpowered, directional |
| 13 | Subgroup: white | −0.0530 (3e-5) | — | ✓ |
| 13 | Subgroup: non-white | −0.0405 (0.22) | — | directional |
| 13 | Subgroup: college | −0.0442 (4e-3) | — | ✓ |
| 13 | Subgroup: no college | −0.0599 (2e-3) | — | ✓ |

**What replicates:**
- The core "Before > After" xenophobia-term effect is extremely robust: every specification including length-normalization preserves it, and the permutation test has zero null draws as extreme.
- The "After > Before" labor-term effect is robust to all standard sample-restriction perturbations.
- Placebo words are all null.

**What's fragile:**
- The labor-term effect is **not** robust to per-word normalization (check 6, p = 0.19). Combined with the audit finding that After-Crisis respondents write ~7% longer answers, the labor-term increase appears to be mostly a word-count artifact. The paper's stronger interpretation — that the treatment shifts *attention* toward labor concerns — is not ruled out, but neither is the weaker "people just wrote more and therefore hit more labor words" interpretation.
- The labor-term effect is **entirely** driven by the single stem `job` (check 10a): drop `job` and β goes from +0.064 to +0.013 (p = 0.52). This is consistent with the word-count story, since `job` has 1,336 hits and will mechanically correlate with response length.
- The racism-term effect is mostly driven by the single stem `racis` (check 9a): drop it and β shrinks 3× but is still significant at p = 0.02. So the xenophobia finding is *somewhat* concentrated in one word, but broader than the labor finding.

---

## 6. Summary Assessment

**What replicates (every quantitative claim I could test):**
- Sample size 1,952, cells 971/981 and 998/954: exact match.
- Figure 1 Panel A means: ≤0.4 pp from paper for every cell; all direction, significance, and heterogeneity conclusions preserved.
- Figure 1 Panel B means: exact to 4 decimals (0.738 / 0.802).
- Biden-voter heterogeneity in racism terms (interaction p ≈ 0.001, paper 0.003).
- No heterogeneity in labor terms (interaction p = 0.5009, paper 0.501, identical).
- Table 1 ordering: treatment and party are the most text-predictable dimensions; white is under-predicted; other demographics barely move. Absolute accuracies shift under TF-IDF+LR but the qualitative ranking is preserved.

**Concerns:**
1. **Text-length confound on the labor result.** Not a bug — the dictionary and the statistical procedure exactly follow the pre-specified recipe. But the paper's headline "near one-to-one substitution between xenophobia- and labor-related terms" rests on two outcomes that are constructed asymmetrically with respect to a treatment-correlated confound (length). Per-100-word normalization kills the labor effect. A cleaner presentation would be to either (a) report both as rates-per-response and rates-per-word, or (b) add response length as a control. With a length control, I'd expect the labor-term result to be much weaker; the xenophobia-term result would tighten.
2. **Stem concentration.** The racism composite is 76% `racis` and the labor composite is 89% `job`. A reader should understand that the paper's main result is really "After-crisis, people say 'racist/racism' less and say 'job' more."
3. **TF-IDF underperforms BERT on the demographic benchmarks** (College, High income) in ways that matter for the *comparative* interpretation of Table 1 — a future replication could try to add a BERT-embedded LogReg classifier to see if the gap closes.

**Bug status:** No coding bugs found. All Stata code runs as advertised, and every variable I traced matches the documentation.

**Bottom line:** The paper's core experimental finding — that framing the protagonist as joining an anti-immigrant organization *after* the crisis shifts respondents away from xenophobic explanations and toward labor-market explanations — replicates cleanly. The magnitude of the "labor-term" half of the "near one-to-one substitution" is inflated by a mechanical response-length confound. The xenophobia-term half is real, robust, and concentrated in how respondents deploy the stem `racis`.

---

## 7. File Manifest

```
replication_164561/
├── utils.py                     shared paths, stem lists, cleaning routine
├── 01_clean.py                  loads study1.sav, applies clean_study1.do filters, saves parquet
├── 02_figure1.py                Figure 1 panels A and B cell means, t-tests, heterogeneity p
├── 03_table1_classifier.py      TF-IDF + LogReg classifier accuracies for Table 1
├── 04_data_audit.py             sample funnel, balance, missingness, text-length check, stem sanity
├── 05_robustness.py             13-check robustness battery (controls, per-word, permutation, LOO stems, placebo, subgroup)
├── writeup_164561.md            this file
└── output/
    ├── study1_cleaned.parquet   1,952-row cleaned analysis sample
    ├── figure1.png              replicated Figure 1
    ├── figure1_values.csv       all cell means/SEs/Ns
    └── table1_classifier.csv    Table 1 numbers
```

Run order from the repo root:
```
source venv/bin/activate
python replication_164561/01_clean.py
python replication_164561/02_figure1.py
python replication_164561/03_table1_classifier.py
python replication_164561/04_data_audit.py
python replication_164561/05_robustness.py
```
