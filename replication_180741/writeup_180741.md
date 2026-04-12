# Replication Study: 180741-V1

**Paper:** "Enabling or Limiting Cognitive Flexibility? Evidence of Demand for Moral Commitment"
**Authors:** Silvia Saccardo, Marta Serra-Garcia
**Journal:** *American Economic Review*, 113(2), 2023 (DOI: 10.1257/aer.20201333)
**Original Language:** Stata 17
**Replication Language:** Python (pandas, statsmodels, scipy)

---

## 0. TLDR

- **Replication status:** All headline regression and proportion tests replicate to three-decimal precision. Table 1 sample sizes match exactly for NoChoice, and for Choice sub-conditions they match the raw Stata `tab` output in the analysis code — the paper reports slightly smaller numbers in its printed Table 1, apparently after an additional pre-registered attention/quality filter not applied in the released code.
- **Key finding confirmed:** Information order causally shifts self-serving behavior — in the conflict case, advisors assigned to see the incentive first recommend the incentivized product 16.8 pp more often than those assigned to see quality first (79% vs 62%, Z=2.69, p=0.007) — and advisors reveal strict, heterogeneous preferences over information order, with ~30% paying to commit to seeing quality first and ~41% paying for cognitive flexibility.
- **Main concern:** The experimental design is clean and the results are extremely robust. The only minor concern is a mild (non-significant) covariate imbalance in the small NoChoice arm (38.8 vs 35.4 mean age) and a ~6pp "placebo" gap in the no-conflict sub-sample of Table 3 col 1 which, while much smaller than the 19.5pp conflict-sample gap, hints that some of the prefer-incentive-first population is marginally less scrupulous even when no moral conflict is active.
- **Bug status:** No coding bugs found.

---

## 1. Paper Summary

### Research Question
When individuals face a moral dilemma in the role of an advisor and can choose the *order* in which they receive (a) a quality signal about a product and (b) information about their private incentive, do they commit to moral behavior (by asking to see quality first) or seek out cognitive flexibility (by asking to see the incentive first)? How does the order that is actually experienced affect subsequent recommendations and beliefs?

### Experimental Setting
The *advice game*: an advisor chooses between Product A (deterministic $1.20 expected payoff) and Product B (uncertain: $1.60 if high quality, $0.80 if low). The advisor receives a one-ball signal of B's quality and a small incentive ($0.15) attached to one of the two products. When the incentive is for B and the signal is $0 (or vice versa), the advisor faces a *conflict of interest*.

### Experiments and Samples (N = 8,923 advisors across 14 arms)
1. **NoChoice** (N=299 attentive): randomly assigned to see the incentive first or the quality signal first.
2. **Choice** (N ≈ 5,900 main, plus high-stakes robustness): advisors *choose* the information order; preference is implemented with 75% probability. Four main arms: *ChoiceFree*, *ChoiceFree-Professionals*, *IncentiveFirst Costly*, *QualityFirst Costly*.
3. **Choice Stakes** (N=1,472): varies the incentive size ($0.01 / $0.15 / $0.30) to test whether demand for information order responds to incentive magnitude.
4. **Information Architect** (N=498): third-party participants choose the advisor's information order with incentives aligned either with the advisor or the client.

### Method
- Linear-probability models with HC3 (Huber-White) robust standard errors on binary outcomes (recommendation, preference to see incentive first).
- Belief regressions of log-odds beliefs on log-likelihood-ratio dummies (`bad` for conflict signal, `good` for no-conflict signal) with no constant.
- Two-proportion Z-tests for NoChoice descriptive comparisons.

### Key Findings
1. **Information order matters** (NoChoice): In the conflict condition, advisors recommend the incentivized product 79% of the time when assigned to see the incentive first vs 62% when assigned to see quality first. No effect when there is no conflict (89% vs 86%).
2. **Strict, heterogeneous preferences for order**: Roughly 45% vs 55% split in costless conditions; 30% of AMT advisors will pay to see quality first, 41% will pay to see the incentive first.
3. **Effect of realized order depends on preference**: Among advisors *assigned* their preferred order, prefer-incentive-first advisors are 19.5 pp more likely to recommend the incentivized product in the conflict case than prefer-quality-first advisors. Among those *not* assigned their preference, the gap vanishes (0.3 pp).
4. **Beliefs line up**: Advisors assigned to see quality first update more strongly on both conflict and no-conflict signals (βC=0.346 vs 0.267; βNC=0.444 vs 0.324). Differences significant at the 5% level.
5. **Demand for flexibility rises with moral stakes**: Stakes experiment — share paying to see incentive first is 13% at $0.01 vs 41% at $0.15 vs 44% at $0.30 (concave in the incentive).
6. **Third parties are sophisticated**: Information Architects are more likely to have advisors assess quality first when IA incentives are aligned with the client (Section 8).

---

## 2. Methodology Notes

### Translation Choices
- `reg ..., vce(hc3)` → `sm.OLS(...).fit(cov_type="HC3")`.
- `prtest` → two-proportion Z-test implemented manually (pooled variance).
- `test good=bad` after a no-constant OLS → manual `t = (β_good − β_bad) / sqrt(Var[β_good] + Var[β_bad] − 2 Cov)`; squared to get the F statistic reported in the paper.
- `margins i.treatment, atmeans` → evaluate fitted regression at covariate means with treatment dummies toggled on one-at-a-time; matches the paper's Figure 4 bar heights to 2 decimal places.

### Estimator Equivalence
`statsmodels` HC3 standard errors match Stata 17 `vce(hc3)` to at least 4 significant figures across all regressions I re-ran. Coefficients match to 4+ figures.

### Sample Selection
The Stata code drops `alphavaluefinal==.` except for `study==1` (professionals, who did not do the MPL task). I reproduce this filter throughout. I do not apply any additional attention filter.

---

## 3. Replication Results

### Table 1 Sample Sizes

| Arm | Paper Table 1 | Stata `tab` (code) | My Python |
|---|---|---|---|
| NoChoice-SeeIncentiveFirst | 152 | 152 | 152 |
| NoChoice-AssessQualityFirst | 147 | 147 | 147 |
| ChoiceFree | 2,377 | 2,574 | 2,574 |
| ChoiceFree-Professionals | 712 | 712 | 712 |
| IncentiveFirst Costly | 1,358 | 1,562 | 1,562 |
| QualityFirst Costly | 1,067 | 1,067 | 1,067 |
| HighStakes 10-fold | 275 | 275 | 275 |
| HighStakes 100-fold | 110 | 110 | 110 |
| Stakes (Low/Med/High) | 484 / 511 / 478 | 483 / 511 / 478 | 483 / 511 / 478 |
| IA-Advisor / IA-Client | 245 / 253 | 245 / 253 | 245 / 253 |

**Discrepancy note:** In ChoiceFree and IncentiveFirst-Costly, the printed Table 1 reports 2,377 and 1,358 respectively, ~200 fewer than what the released code's `tab` yields. The code comments (lines 33, 888-897 of `analysis.do`) actually document the 2,574 / 1,562 numbers. The printed table presumably applies an additional attention screen that is not in the distributed Stata file. All subsequent regressions in the paper are reproduced using the *looser* filter embedded in the code, and my coefficients still match the paper's reported coefficients exactly, so any additional screen evidently has no material effect on point estimates.

### NoChoice Experiment (Section 5)

| Statistic | Published | Replicated | Match |
|---|---|---|---|
| P(recommend incentivized \| conflict, see incentive first) | 0.79 | 0.787 (108) | ✓ |
| P(recommend incentivized \| conflict, quality first) | 0.62 | 0.619 (105) | ✓ |
| Z-stat, p-value, N (conflict) | 2.69, 0.007, 213 | 2.686, 0.0072, 213 | ✓ |
| P(recommend incentivized \| no conflict, either arm) | 0.89 / 0.86 | 0.886 / 0.857 | ✓ |
| No-conflict Z-stat, p, N | -0.41, 0.685, 86 | -0.406, 0.6851, 86 | ✓ |
| P(not recommend B \| incentive=B, signal=$2) | 0.16 | 0.158 (57) | ✓ |

### Table 2 — Preferences for Information Order (Col 1)

OLS / HC3 dependent variable: prefer-to-see-incentive-first (1/0). N=5,908.

| Regressor | Published β (t) | Replicated β (t) | Match |
|---|---|---|---|
| Choice Free — Professionals | (not reported) | −0.0952 (−3.66) | — |
| See Incentive First Costly | −0.140 (−7.84) | −0.1393 (−7.84) | ✓ |
| Assess Quality First Costly | +0.152 (5.17) | +0.1517 (5.17) | ✓ |

Adjusted margins (Figure 4 bar heights):

| Arm | Paper | Replicated |
|---|---|---|
| ChoiceFree — Professionals | 0.45 / 0.55 | 0.451 / 0.549 |
| ChoiceFree | 0.55 / 0.45 | 0.546 / 0.454 |
| Incentive First Costly | 0.41 / 0.59 | 0.407 / 0.593 |
| Quality First Costly | 0.70 / 0.30 | 0.698 / 0.302 |

Columns 2 (AMT only, adds standardised selfishness) and 3 (selfishness × cost interactions) also replicate exactly: `seeincentivecostly` = −0.1395 / −0.1397 (t ≈ −7.86 / −7.88), `seequalitycostly` = +0.1523 / +0.1520 (t ≈ 5.19 / 5.18).

### Table 3 — Recommendations by Assignment × Preference

OLS / HC3 on `recommendincentive`.

| Coefficient | Paper | Replicated | N |
|---|---|---|---|
| Col 1 — `choicebefore` (Assigned pref, main effect) | +0.195 (t=12.17) | +0.1955 (t=12.17) | 4,448 |
| Col 1 — `choicebefore × no conflict` | negative, significant | −0.1371 (t=−5.43) | |
| Col 2 — `choicebefore` (Not assigned pref) | ≈0, NS | +0.0030 (t=0.10) | 1,460 |
| Col 3 — `choicebefore × not-assigned-pref` | negative | −0.1402 (t=−5.47) | 5,908 |

Text statistics:

| Claim | Paper | Replicated |
|---|---|---|
| Assigned-pref 19.5 pp more likely (conflict) | t=12.17, p<0.001 | 0.1955, t=12.17 |
| Prefer-incentive × assigned: 9.8 pp more | t=3.66, p<0.001 | 0.0982, t=3.66 |
| Prefer-quality × assigned: 9 pp less (moral commitment) | t=3.05, p=0.002 | 0.0906, t=3.05 |
| ChoiceFree-only gap (p. 33) | 23.5 pp, 95% CI [18.7, 28.4] | 0.2352, 95% CI [0.1868, 0.2836] |

### Table 4 — Beliefs by Preference and Assignment

No-constant OLS on log-odds belief, regressors are log-likelihood-ratio dummies.

| Sample | Coefficient | Paper | Replicated |
|---|---|---|---|
| Assigned, f=q (quality first) | βC | 0.346 | 0.346 |
| Assigned, f=q | βNC | 0.444 | 0.444 |
| Assigned, f=i (incentive first) | βC | 0.267 | 0.267 |
| Assigned, f=i | βNC | 0.324 | 0.324 |
| Assigned, f=q vs f=i βC | t = 2.45, p = 0.014 | t = 2.45, p = 0.014 | ✓ |
| Assigned, f=q vs f=i βNC | t = 2.19, p = 0.029 | t = 2.19, p = 0.029 | ✓ |
| Pooled assigned: βC = βNC | F = 5.57, p = 0.018 | F = 5.57 | ✓ |
| Excl. wrong direction: βC = βNC | F = 12.06, p < 0.001 | F = 12.06 | ✓ |
| Excl. wrong: f=q vs f=i βNC | t = −0.81, p = 0.417 | t = −0.81, p = 0.417 | ✓ |
| Excl. wrong: f=q vs f=i βC | t = −1.76, p = 0.078 | t = −1.76, p = 0.078 | ✓ |

### Stakes Experiment (Section 8)

| Condition | Paper share prefer-incentive-first | Replicated |
|---|---|---|
| Low ($0.01) | 13% (Z=9.79) | 13.0% (N=483) |
| Intermediate ($0.15) | 41% | 40.7% (N=511) |
| High ($0.30) | 44% (Δ ≈ +3 pp, NS) | 43.7% (N=478) |

All main-text tables and figures that I can replicate without additional Stata-specific machinery match the published numbers to three decimal places.

---

## 4. Data Audit Findings

### Coverage
- **NoChoice:** 327 raw rows → 299 after `alphavaluefinal` filter (attention screen). 213 in conflict condition, 86 in no-conflict. Balanced 152/147 across the two arms.
- **Choice:** 6,976 raw → 6,300 after attention screen → 5,915 in main (low-stakes) sample. Split 2,574 / 712 / 1,562 / 1,067 across the four main arms.
- **Stakes:** 1,681 raw → 1,472 after filter (483 / 511 / 478 across Low / Intermediate / High).
- **IA:** 549 raw → 498 after filter (245 IA-Advisor, 253 IA-Client).

### Distributions and Plausibility
- Ages in [18, 92]; no out-of-range values in any dataset.
- `recommendincentive`, `choicebefore`, `conflict`, `incentiveB`, `female` are all binary and fully populated in analysis samples.
- `belief` ∈ [0, 100]; `logitbelief` capped at ±4.595 (because 1/99 and 99/1 are the extremes of the logit transform under the clipping used by the authors).
- `alphavaluefinal` ∈ [0, 5] (integer count of selfish decisions out of 5 in MPL).

### Consistency
- `noconflict + conflict == 1` holds for every row in every analysis sample.
- `getyourchoice == 1 ⇒ getbefore == choicebefore` (0 mismatches).
- Among Choice's 712 "duplicate `id`" values all are NaN: professionals were deidentified without an id column, so pandas counts them as duplicates. Not a real data issue.

### Missingness
- The only systematic missingness is `alphavaluefinal` for professionals (who do not perform the MPL task, by design). The `study != 1` guard in the code handles this.
- `logitbelief` is missing for ~100 Choice participants, spread across arms (40, 8, 26, 9) — small and not concentrated in one condition.

### Balance
- **NoChoice arms** are slightly imbalanced on age (quality-first arm 38.8y vs incentive-first 35.4y) and on MPL selfishness (−0.12 vs +0.12 stdalpha). Neither imbalance is large enough to overturn the 16.8 pp treatment effect and the published regressions control for gender and age.
- **Choice arms:** female share ~53% in three arms but 60% in the Quality-First-Costly arm. Age and MPL scores are comparable. The paper's regressions include wave and demographic controls.

---

## 5. Robustness Check Results

I ran 13 checks targeting the three headline claims.

### A. NoChoice information-order effect (Δ=+16.8 pp)

| Check | Result | Conclusion |
|---|---|---|
| A1: Include inattentive participants (drop filter) | +17.24 pp, Z=2.86, p=0.004 | Survives |
| A2: Trim 5% tails of `stdalpha` | +16.80 pp, Z=2.69, p=0.007 | Identical (no tail obs affected) |
| A3: Permutation test (B=2,000 shuffles of treatment) | p_perm = 0.010 | Confirms parametric p |
| A4: Logit on conflict sub-sample | β = +0.789, p = 0.018 | Sign and significance preserved |

### B. Table 2 Col 1 cost-shift (−14 pp / +15 pp)

| Check | Δ SeeIFC (t) | Δ SeeQFC (t) | Conclusion |
|---|---|---|---|
| B1: Professionals only | n/a (not in costly conditions) | n/a | Sanity check — collinear |
| B2: AMT only | −0.139 (−7.83) | +0.152 (+5.18) | Professionals' inclusion immaterial |
| B3: Drop wave 3 | −0.145 (−7.52) | collinear (QFC only in wave 3) | IFC survives; QFC is a wave-3 effect |
| B4: HC1 instead of HC3 | −0.139 (−7.84) | +0.152 (+5.18) | SE choice irrelevant |
| B5: Logit | −0.568 (−7.75) | +0.631 (+5.11) | Functional form immaterial |

### C. Table 3 Col 1 prefer-incentive gap (+19.5 pp)

| Check | choicebefore β (t) | N | Conclusion |
|---|---|---|---|
| C1: Drop professionals | +0.186 (+10.76) | 3,915 | Survives |
| C2: ChoiceFree treatment only | +0.235 (+9.53) | 1,931 | **Stronger** |
| C3: Costless only (0 + 1) | +0.240 (+11.26) | 2,464 | **Stronger** |
| C4: Permutation (1,000 shuffles) | Δ=+19.79 pp, p_perm=0.000 | | Confirms parametric p |
| C5: **Placebo — no-conflict subset** | +0.060 (+2.99) | 1,325 | **Non-zero but much smaller than 19.5 pp** — preference is ~30% as predictive in the condition where motive isn't active. Still directional. |
| C6: Female only | +0.194 (+8.91) | 2,430 | Same magnitude |
| C6: Male only | +0.199 (+8.28) | 2,018 | Same magnitude |

### D. Stakes experiment — incentive-size sensitivity

| Arm | Share choosing incentive first | N |
|---|---|---|
| Low ($0.01) | 13.0% | 483 |
| Intermediate ($0.15) | 40.7% | 511 |
| High ($0.30) | 43.7% | 478 |

The concavity (sharp jump from Low → Intermediate, tiny jump from Intermediate → High) directly replicates Section 8 of the paper.

### Summary of robustness
- **All three headline findings are highly robust.** The preference shifts, information-order effect, and recommendation gap all survive subgroup splits, alternative functional forms (logit), alternative SEs (HC1), permutation tests, and sample restrictions.
- **The only notable wrinkle** is the non-zero placebo gap of +6 pp in the Table 3 Col 1 no-conflict sub-sample (check C5). The authors already report this as `choicebeforenoconflict = −0.137`, which is the Table 3 interaction that mostly — but not entirely — nets out the main effect in no-conflict rows. The residual is small relative to the 19.5 pp main effect but does suggest that prefer-incentive-first advisors may be marginally more self-serving even in contexts where the official "conflict" dummy is off.

---

## 6. Summary Assessment

This paper is an exemplary replication target: the design is clean, the data are well-organized, and every published number I attempted to reproduce matches the paper to three or four decimal places. The authors' README accurately locates every output in the code, the Stata scripts are logically structured, and the .dta files are internally consistent.

**What replicates (everything I tested):**
- All NoChoice proportions, Z-statistics, and p-values (Section 5).
- Table 2 all three columns (cost-shift in preferences).
- Figure 4 adjusted margins.
- Table 3 all three columns and the three in-text statistics on pages 30-33 (including the 95% confidence interval for the ChoiceFree-only gap).
- Table 4 all four pooled-model coefficients and the four t-tests reported on pages 34-35.
- The Section 8 Stakes experiment shares.
- Table 1 *internal* sample sizes, with the caveat that the printed Table 1 numbers for ChoiceFree (2,377) and IncentiveFirst-Costly (1,358) do not match the 2,574 / 1,562 produced by the released `tab` code. This is a documentation discrepancy, not a bug — the paper's *coefficients* replicate using the larger samples.

**What I did not replicate:**
- Figures involving `twoway` bar plots (visual only — no new numbers).
- Several appendix tables (C.2–C.24) and the prediction/explanation coding analyses (Section 6.2). Given the exactness of the matches on every main-text result, I have no reason to expect appendix results to differ.
- The Information Architect analyses (Section 8.2) — the code and data are there, but I did not spend compute on them once the main findings were confirmed.

**Concerns:** Minor. The NoChoice arms have mild demographic imbalances and the Choice "Quality First Costly" arm is over-female, but published regressions control for this and the effects are too large to be driven by covariates. The Table 1 documentation discrepancy should probably be noted in an erratum but has no substantive impact. The placebo gap in check C5 is modest and consistent with the interaction term the paper already reports.

**Bug status:** No coding bugs found.

**Bottom line:** Saccardo & Serra-Garcia (2023) replicates exactly on every main-text number I re-derived, and the findings are robust to reasonable alternative specifications. The paper is a high-quality experimental contribution and the replication package is in very good shape.

---

## 7. File Manifest

```
replication_180741/
├── utils.py                      Data loaders and HC3 helpers
├── 01_sample_sizes.py            Table 1 — experimental design sample sizes
├── 02_nochoice.py                Section 5 — NoChoice proportion tests + reg
├── 03_preferences_table2.py      Section 6 / Figure 4 / Table 2 — preferences
├── 04_recommendations_table3.py  Section 7 / Table 3 — recommendations
├── 05_beliefs_table4.py          Section 7 / Table 4 — belief updating
├── 06_data_audit.py              Coverage, plausibility, balance, missingness
├── 07_robustness.py              13 robustness checks across 3 headline claims
├── writeup_180741.md             This file
└── outputs/                      Text logs from each script
    ├── table1_sample_sizes.txt
    ├── nochoice_results.txt
    ├── table2_results.txt
    ├── table3_results.txt
    ├── table4_results.txt
    ├── data_audit.txt
    └── robustness.txt
```

Every script runs under the shared repo venv (`source venv/bin/activate && python replication_180741/<script>.py`) in under 20 seconds on a 2021 MacBook. No script writes anywhere outside `replication_180741/outputs/`.
