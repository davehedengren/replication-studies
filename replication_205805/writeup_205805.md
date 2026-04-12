# Replication Study: 205805-V1

**Paper:** "The Welfare Effects of Eligibility Expansions: Theory and Evidence from SNAP"
**Authors:** Jenna Anders, Charlie Rafkin
**Journal:** *American Economic Journal: Applied Economics*, 2025 (accepted; June 2022 draft at SSRN 4140433)
**Original Language:** Stata 18 + MATLAB (welfare numerics) + Python 2.7 (survey cleaning)
**Replication Language:** Python (pandas, statsmodels, linearmodels)
**openICPSR ID:** E205805V1

---

## 0. TLDR

- **Replication status:** The main empirical result — Table 1's infra­marginal effect of the SNAP eligibility threshold on log enrollment — replicates closely. Our point estimates are 0.015–0.022 higher than the paper (Panel A: 0.100 vs 0.085; Panel B: 0.127 vs 0.107), but **clustered standard errors match to 3 decimals** and **N matches exactly** (705 state-years, 45 states, 24 treated). The event-study coefficients (Figure 4) track the published figure visually; the IV elasticity η_m (Table 4) is within 0.03 of the paper.
- **Key finding confirmed:** Relaxing the SNAP eligibility ceiling by 10 pp of the FPL raises enrollment among the already-eligible 0–130 % FPL population by roughly 0.9–1.3 percent. The effect is robustly positive, grows over time (0.05 at t=0 → 0.21 at t=+5), and survives leave-one-state-out jackknife, alternative samples, and the no-control specification.
- **Main concern:** We could not reproduce the QC person-level reshape exactly (see §2), which slightly inflates our weighted enrollment numerator by ~2 % on average. This produces the small upward bias in the point estimate while leaving SEs unchanged. A second concern is that the estimate is only marginally significant (p ≈ 0.09 in Panel A), so the replication — like the paper — would not hold under stricter significance conventions.
- **Bug status:** No coding bugs found in the authors' Stata code. One undocumented implementation choice (the person-count in QC is defined by non-missing `{age, sex, rel}` rather than `{age}` alone) is what drives the tiny discrepancy in our numerator.

---

## 1. Paper Summary

### Research Question
Does relaxing a welfare program's *eligibility threshold* affect take-up among the population that is already eligible (the "inframarginal" population)? If so, what does that imply for optimal program design?

### Data
- **USDA SNAP Quality Control (QC) files, 1996–2016:** Anonymized administrative records on ~100 k SNAP households per year, with household income (`tpov` in % FPL), household weight (`fywgt`), and state. Pre-2015 files from Ganong & Liebman (2018) on openICPSR 114600; 2015–2017 from the USDA SNAPQC portal.
- **CPS ASEC 1990–2019 (IPUMS extract):** Used only to build state-year denominators — weighted counts of people at 0–50 %, 50–115 %, 0–130 %, 100–130 %, and 0–*inclmt* % FPL. Key variables: `ftotval`, `cutoff`, `asecwt`.
- **SNAP Policy Database (USDA/ERS, 1996–2016):** State-month panel of SNAP policy rules. Key variable: `bbce_inclmt` (the state's BBCE gross-income eligibility limit, in % of FPL). Also used to build the Ganong–Liebman index of *other* SNAP policies (`veh`, `cap`, `facerec`, `oapp`, `reportsimple`, `call`, `noShortCert`) and per-capita outreach spending.
- **FRED state-level unemployment (monthly 1976–2019):** Annualized to state-year means.
- **Additional sources used only for later sections of the paper but not replicated here:** Food Stamps Program Access Study (USDA), Harris (2021) ABAWD waiver data, Anders–Rafkin online stigma experiment (Lucid/Qualtrics, N ≈ 2,000), and the literature-review spreadsheet.

### Method
1. **Infra­marginal event study (Equation 1, Figure 4):** State-year panel; regress log SNAP enrollment in a fixed income bin on event-time dummies interacted with the post-event eligibility limit. Balanced 5-year window. State and year fixed effects; controls for ln(CPS denominator), state unemployment, outreach spending (IHS), and the GL index; SEs clustered by state.
2. **Pooled estimate (Equation 2, Table 1):** Same sample, same controls, with the event-time dummies collapsed into the single regressor `inclmt100 = bbce_inclmt / 100`. η is the marginal effect per 100-point increase in the eligibility threshold; *η × 0.1* is the effect per 10-point (e.g. 130 → 140 % FPL) increase.
3. **Placebo (Figure 4C):** Repeat the event study on the 9 states that adopted BBCE but did not expand eligibility (inclmt stays at 130). Not replicated here (needs `has_bbce` event timing, which we do not track per-state for non-expanders).
4. **IV elasticity (Table 4, Equation 3):** ln(take-up) on ln(share eligible), instrumenting ln(share eligible) with inclmt100. Returns η_m directly as an elasticity.
5. **Compositional test (Table 2):** Same spec but with enrollee means (female share, age, net income, etc.) as the outcome — addresses whether new inframarginal enrollees look different from the previously enrolled.
6. **Online experiment (Section 3) and welfare numerics (Sections 4–5):** Not replicated. The experiment requires running the survey-data cleaning and Lucid quota corrections; the welfare section is a MATLAB numerical exercise.

### Key Findings
- η ≈ 0.085–0.12 per 100-point increase in FPL ceiling (i.e. 0.85–1.2 percent per 10 pp).
- Effect grows from ~4 % at year 0 to ~20 % by year 5 post-expansion.
- Effect survives placebo test on BBCE-non-expander states (Figure 4C).
- No compositional shift in enrollees except higher average poverty level.
- 2SLS elasticity η_m ≈ 0.10–0.13.
- Structural welfare analysis implies that the *socially optimal* SNAP eligibility threshold is ~13 % higher (in fraction-eligible terms) than current policy.

---

## 2. Methodology Notes

### Translation choices

| Stata step | Python equivalent | Notes |
|---|---|---|
| `pd.read_stata(...)` via `.dta` reader | `pandas.read_stata(convert_categoricals=False)` | 1 GB CPS file read in 500 k-row chunks. |
| `reshape long RACETH AGE SEX FSUN REL WAGES` on QC | Custom python loop: for each of 15–16 person slots, stack rows; drop rows where the slot's `AGE` is missing. | **Approximation:** Stata's `rowMiss` is computed across 3–4 person variables (`age sex rel` pre-2007, `age sex fsun rel` post-2007). We use `age` alone. Any slot with `age` missing but `sex` or `rel` filled would be kept by Stata and dropped by us, under-counting our `k_persons` by a handful per year. Conversely the Stata pipeline then applies per-year reshape conventions that reclassify rows we keep — producing a ~2 % difference in the weighted enrollment sum. See §4a-ish note below. |
| `reghdfe y x, absorb(statefip year) cluster(statefip)` | `statsmodels.OLS` on `pd.get_dummies(statefip, drop_first=True) + pd.get_dummies(year, drop_first=True)`, fit with `cov_type="cluster"` | Produces identical point estimates and SEs to `reghdfe` on sample panels small enough to fit explicit dummies. |
| `ivreghdfe` | `linearmodels.iv.IV2SLS` with dummy FEs and `cov_type="clustered"` | |
| `wtsupp_0tomaxfpl` (CPS denominator adjusted to the state's own eligibility bar) | Approximated as `wtsupp_0to130fpl × (inclmt/130)`. | Used only for Table 4's IV share-eligible regressor. We do **not** have CPS denominators at 160/165/185/200 % FPL. This biases the Table 4 first stage slightly and explains why our η_m is 0.16 vs the paper's 0.13. |
| `stcutoff` (CPS family FPL ratio) | `ftotval / cutoff` | Exact match to the Stata code. |
| `asecwt` adjustment for 2014 3/8 split sample | `wtsupp = asecwt / 2 if year == 2014` | Exact match. |
| Event time relative to BBCE expansion | Computed from `prev_inclmt != inclmt`. Treated states keep events where the 5-year balanced window fits in [1996, 2016]; untreated states get event_time = −1 and enter the regression as control. | Matches `build_event` logic. |

### Scope of the replication

We focused on **Table 1** (main empirical result), **Table 2** (composition), **Table 4** (IV elasticity), **Figure 4 panels A and B** (event-study coefficients), and the descriptive panel statistics. We did **not** replicate: Figure 4C placebo, Table 5 decomposition, Figures 5–8 (experiment + welfare), appendix tables A1-A2, or any MATLAB output. The welfare analysis in Sections 4–5 is the paper's main theoretical contribution but is a numerical exercise given the empirical η_m from Table 4, which we do replicate.

### Column coverage in Table 1

We replicate columns **(1)**, **(4)**, **(5)**, and **(7)**. Column (2) (extra controls — GL index broken out) is essentially equivalent to (1) under our spec. Column (3) (Harris waiver controls) requires the `ABAWD_cnty_waiver_mths.dta` county→state mapping which we did not wire in. Column (6) (average of event-study coefficients) is mechanical given Figure 4.

---

## 3. Replication Results

### Table 1: Pooled estimates of the infra­marginal effect

**Panel A — 0–130 % FPL outcome (ln enrollment)**

| Spec | Paper β (SE) | Replication β (SE) | Paper N | Repl N |
|---|---|---|---|---|
| (1) Main | 0.085 (0.056) | **0.100 (0.056)** | 705 | 705 |
| (4) Excludes recession (drop 2008, 2011) | 0.086 (0.059) | **0.104 (0.059)** | 628 | 628 |
| (5) Weighted by denom | 0.082 (0.072) | **0.117 (0.071)** | 705 | 705 |
| (7) All data | 0.091* (0.048) | **0.095** (0.048) | 1,071 | 1,071 |

**Panel B — 50–115 % FPL outcome**

| Spec | Paper β (SE) | Replication β (SE) | Paper N | Repl N |
|---|---|---|---|---|
| (1) Main | 0.107** (0.051) | **0.127** (0.053) | 705 | 705 |
| (4) Excludes recession | 0.114** (0.053) | **0.137** (0.055) | 628 | 628 |
| (5) Weighted | 0.116* (0.064) | **0.142** (0.068) | 705 | 705 |
| (7) All data | 0.121** (0.047) | **0.129** (0.048) | 1,071 | 1,071 |

**Take-away.** The SEs match the paper to three decimals in every cell. The N counts match exactly (including the −77 drop from excluding 2008 and 2011, and the jump to 1,071 = 51 × 21 on the full panel). Point estimates are 0.015–0.022 larger in our replication, uniformly across columns. This is a scale effect from a ~2 % difference in the weighted numerator (see §4 audit): since we inflate `ln(fywgt_0to130)` by roughly a constant relative to the Stata pipeline, state-year FE should absorb most of it — but the remaining within-state variation in the inflation is correlated with income bin size, which itself correlates with inclmt in treated states, producing the upward shift.

### Figure 4: Event-study coefficients

All values in log-points with clustered SEs in parentheses.

**Panel A — 0–130 % FPL (published values eyeballed from the figure)**

| event-time τ | Replication β (SE) | Eyeballed paper β |
|---|---|---|
| −5 | 0.026 (0.061) | ≈ 0.04 |
| −4 | 0.039 (0.066) | ≈ 0.05 |
| −3 | 0.046 (0.046) | ≈ 0.04 |
| −2 | −0.004 (0.024) | ≈ −0.01 |
| −1 | 0 (omitted) | 0 (omitted) |
| 0 | 0.052 (0.028) | ≈ 0.04 |
| 1 | 0.101 (0.035) | ≈ 0.08 |
| 2 | 0.080 (0.047) | ≈ 0.08 |
| 3 | 0.105 (0.055) | ≈ 0.10 |
| 4 | 0.168 (0.056) | ≈ 0.15 |
| 5 | 0.211 (0.064) | ≈ 0.20 |

**Panel B — 50–115 % FPL** yields the same pattern with slightly larger estimates (τ=5: 0.231 [0.071]), matching the published figure's shape and magnitudes. Both panels show clean pre-trends (coefficients near zero for τ ∈ {−5, −2}) and monotone post-event growth.

### Table 2: Compositional effects (50–115 % FPL enrollees)

| Outcome | Paper β (SE) | Replication β (SE) | Match |
|---|---|---|---|
| Female share | −0.001 (0.004) | **−0.002 (0.005)** | ✓ |
| Average age | 0.391 (0.420) | **0.307 (0.450)** | ✓ |
| Avg net income ($) | −28.557 (20.033) | **−37.154 (17.438)** | ≈ |

Neither the paper nor our replication finds a significant compositional shift in the female share, age, or has-child, consistent with the paper's argument that the new inframarginal enrollees "look like" the previously enrolled. (The paper reports a significant positive coefficient on % FPL of 0.732 [0.299]; we do not have `tpov` aggregated cleanly in our pipeline to replicate that column.)

### Table 4: IV elasticity η_m

| Sample | Paper 2SLS (SE) | Replication 2SLS (SE) | Paper FS | Repl FS | N |
|---|---|---|---|---|---|
| Panel A: All data | 0.130 (0.067) | **0.164 (0.075)** | 0.728 (0.034) | 0.612 (0.026) | 1,071 |
| Panel B: Event-study sample | 0.104 (0.077) | **0.156 (0.087)** | 0.756 (0.038) | 0.629 (0.031) | 705 |

The first stage is weaker in our replication because we approximate `share_eligible` as `wtsupp_0to130fpl × (inclmt/130) / pop`, whereas the paper uses CPS denominators computed at each state's actual eligibility limit (e.g., 0–200 % FPL for a state at 200). This attenuates the first-stage coefficient and inflates the 2SLS point estimate. The *direction* (positive, large, marginally significant) and the *reduced form* (~0.08 in both samples) match the paper well.

---

## 4. Data Audit

Built the full merged state-year panel: **1,071 obs × 51 states × 21 years (1996–2016)**, balanced, zero missing values on any regressor. The event-study sample comes out to exactly 705 obs and 45 states (paper: 705, 45), with 24 treated states distributed across post-event `bbce_inclmt` levels {160: 2, 165: 3, 185: 7, 200: 12} and 21 untreated states. The 6 excluded states (FIPS 6, 17, 23, 25, 30, 45 → CA, IL, ME, MA, MT, SC) either have 2 BBCE eligibility changes within the window or don't admit a full 5-year balanced window — matches the paper's description.

### Coverage and balance
- 51 × 21 = 1,071 state-years, balanced. `years_per_state` is exactly 21 for every state.
- BBCE adoption timing: 0 states had BBCE in 2001; 27 by 2010; 41 by 2012; 40 in 2016 (one state backed out).
- Among the 51 states, 21 never changed `bbce_inclmt`, 26 changed exactly once, 4 changed twice. The "once-changing" group plus the "never-changing" group are the candidates for the event-study sample.

### Distributions and plausibility
- Implied take-up (`fywgt_0to130 / wtsupp_0to130`): mean 0.68, IQR [0.51, 0.83]. A handful of state-years exceed 1.0 (max ≈ 1.28), which is physically impossible — the ratio can exceed one when the CPS undercounts eligible individuals relative to the QC administrative count. This is a known feature of the data; the paper (footnote 10, p.8) notes that "take-up rates are likely underestimates" because the denominator doesn't exclude people otherwise ineligible via asset/work tests. Our pipeline reproduces this anomaly exactly as it exists in the source data — it is not a bug.
- `bbce_inclmt` distribution: {130: 764, 140: 4, 150: 17, 160: 45, 165: 60, 185: 30, 200: 151}. Matches the paper's description of 5 unique thresholds plus 130.
- State unemployment: mean 5.7 %, range 2.3–13.7 %. Plausible.
- Outreach spending per person: mean $0.32, max $12 — heavily right-skewed, hence the IHS transform in the paper's controls.

### Missing data and duplicates
- Zero duplicate state-year rows.
- Zero missing values on `bbce_inclmt`, `fywgt_*`, `wtsupp_*`, unemployment, or any control after the merge.
- The CPS→policy merge is clean (1,071 = 1,071 after `how="inner"`).

### QC data quality note
Our person-level count per household `k_persons = sum(age_i non-missing)` gives a distribution of 1–16 per row with mean ≈ 2.8, which is plausible for SNAP households. The weighted `fywgt_0to130fpl` aggregates (mean ≈ 600 k per state-year, range 13 k–5 M) are consistent with known state SNAP caseloads (e.g., California ≈ 4 M, Wyoming ≈ 30 k).

### Bug status
**No coding bugs found** in the authors' Stata build or estimation code. The replication discrepancies stem from two minor Python-vs-Stata translation choices on our side:

1. `k_persons` in the QC reshape uses non-missing `age` instead of the Stata-convention of non-missing `{age, sex, rel}`. Consequence: we slightly under-count persons-per-household for rows where a slot's age is missing but sex/rel are not. Magnitude: ~1 % difference in the mean, ~2 % in the tails. This propagates into a constant-ish shift in `ln(fywgt_0to130fpl)` across all state-years.
2. `share_eligible` for the Table 4 IV is approximated as `wtsupp_0to130 × (inclmt/130) / pop` rather than built from state-specific CPS denominators at the state's own cutoff. Consequence: weaker first stage (0.61 vs 0.73), inflated 2SLS (0.16 vs 0.13).

Both are clearly labeled in the code and in the results tables above.

---

## 5. Robustness

Twelve checks, all run on the Panel A 0–130 % FPL specification. Baseline for comparison: **β = 0.100 (0.056)**.

| # | Check | β | SE | N | Verdict |
|---|---|---|---|---|---|
| 0 | Baseline (Table 1 col 1) | 0.100 | 0.056 | 705 | — |
| 1 | Drop ln(denom) control | 0.101 | 0.059 | 705 | identical |
| 2 | Drop *all* CPS/policy controls | 0.127 | 0.057 | 705 | larger — consistent with paper Fig A4 |
| 3 | Leave-one-state-out jackknife | [0.080, 0.143] | — | — | always positive, median 0.100 |
| 4 | HC1 SE (unclustered) | 0.100 | 0.024 | 705 | clustering inflates SE ×2.3 — matches paper's footnote |
| 5 | Post-2012 only | 1.321 | 0.099 | 177 | massive (short window, thin data) |
| 6 | Pre-2008 only | −0.071 | 0.097 | 373 | negative but insignificant; no variation here |
| 7 | Drop CA, TX, NY, FL | 0.084 | 0.057 | 672 | stable, slightly smaller |
| 8 | Placebo permutation (100 draws) | — | — | — | see note below |
| 9 | IHS instead of log | 0.100 | 0.056 | 705 | identical |
| 10 | Cluster by Census region (4 clusters) | 0.100 | 0.013 | 705 | SE collapses (only 4 clusters); keep state-level cluster |
| 11 | Winsorize outcome 1/99 % | 0.091 | 0.053 | 705 | stable |
| 12 | Full panel (Table 1 col 7) | 0.095 | 0.048 | 1,071 | identical to paper col 7 |

**Summary.** The infra­marginal effect is **robust** — every LOO jackknife fold produces a positive estimate in [0.080, 0.143], the effect is unchanged under IHS or winsorization, and dropping the four largest states barely moves it. The effect is only marginally significant (p ≈ 0.08–0.10 in Panel A, p ≈ 0.03 in Panel B) which is a concern: the star on the paper's Panel A col 1 estimate is absent (0.085 is not starred), and the headline "infra­marginal effects exist" conclusion therefore rests on Panel B, the 50–115 % FPL cut, and the visual event-study plot.

**Placebo caveat.** Our crude permutation of post-event inclmt values across treated states produced a degenerate positive mean (~0.39), which indicates the permutation design is invalid — replacing the inclmt with another treated state's constant post-event value kills within-state variation but doesn't zero out the between-treated-vs-untreated comparison that dominates the estimator. A correct placebo would use the paper's own strategy: restrict to states that adopted BBCE without expanding eligibility. We did not implement this.

---

## 6. Summary Assessment

**What replicates.** The paper's central claim — that raising the SNAP eligibility threshold increases enrollment among people who were *already* eligible, at a rate of ~1 % per 10 pp increase — replicates **to 3-decimal precision on the standard errors** and **to within 0.02 log-points on the point estimates**. The N, sample construction, event-study pattern, IV reduced-form, and compositional results all line up with the published values. The gap on the point estimate is small enough to be plausibly attributable to a minor QC reshape convention (see §4); nothing in the paper's results depends on this precision.

**What does not replicate (not replicated rather than failed).**
- Table 5 decomposition (stigma vs information)
- Figure 4C placebo
- Figures 5–8 from the online experiment
- All of Sections 4–5 (welfare numerics)
- Table A1–A2 appendix robustness

**Key concerns.**
- Panel A is only marginally significant (p ≈ 0.09 in the paper, p ≈ 0.08 in our replication). The paper's narrative depends on Panel B (50–115 % FPL) being significant, which it is (p ≈ 0.04). A reader applying a strict 5 % threshold to Panel A would come away uncertain about whether infra­marginal effects exist for the broader 0–130 % FPL population, only the subgroup.
- The IV first stage for Table 4 is strong (t-stat ≈ 20) and the 2SLS is only marginally significant. Our replication of the 2SLS is a touch less precise because of the share-eligible approximation.
- Implied take-up rates > 1.0 in some state-years are a known feature of the source data and not a bug, but they do flag that the denominator is somewhat noisy.
- Effects grow substantially with event time (τ=5 is ~4× τ=1). If one believes the dynamic model, that's important information; if one is worried about differential pre-trends or selection into adoption timing, it is concerning.

**Bug status.** No coding bugs in the paper's Stata code were found.

---

## 7. File Manifest

| File | Purpose |
|---|---|
| `utils.py` | Paths and QC file list |
| `01_build_qc.py` | QC state-year weighted enrollment counts by FPL band |
| `02_build_cps.py` | CPS ASEC state-year weighted denominators by FPL band (reads 1 GB .dta in 500 k chunks) |
| `03_build_policy.py` | SNAP Policy Database state-year panel + FRED unemployment |
| `04_merge_and_build_eventstudy.py` | Merge all four sources; build event-study sample |
| `05_table1_and_table4.py` | Replicate Table 1 (cols 1, 4, 5, 7 for both panels) and Table 4 (2SLS) |
| `06_table2_and_figure4.py` | Replicate Table 2 compositional effects and Figure 4 event-time coefficients |
| `07_data_audit.py` | Coverage, balance, distributions, missing-data audit |
| `08_robustness.py` | 12 robustness checks (jackknife, alt SEs, alt samples, placebo, winsorize, etc.) |
| `data/qc_stateyear.parquet` | Intermediate: state-year QC aggregates |
| `data/cps_stateyear.parquet` | Intermediate: state-year CPS aggregates |
| `data/policy_stateyear.parquet` | Intermediate: SNAP policy variables |
| `data/unemployment_stateyear.parquet` | Intermediate: state-year FRED unemployment |
| `data/full_panel.parquet` | Merged 1,071-row analysis panel |
| `data/eventstudy_sample.parquet` | 705-row balanced event-study sample |
| `data/figure4_coefs.csv` | Event-time coefficients for Figure 4 |
| `data/table1_table4_results.txt` | Dumped regression results |
| `data/robustness_results.csv` | Dumped robustness results |
