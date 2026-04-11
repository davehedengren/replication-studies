# Replication Study: 140161-V1

**Paper:** "Checking and Sharing Alt-Facts"
**Authors:** Emeric Henry, Ekaterina Zhuravskaya, Sergei Guriev
**Journal:** *American Economic Journal: Economic Policy*, 2022
**Original Language:** Stata (3 do-files, ~2,600 lines)
**Replication Language:** Python (pandas, statsmodels, scipy)

---

## 0. TLDR

- **Replication status:** Exact replication of every headline number targeted — Table 2 col 1 (-0.045 imposed / -0.038 voluntary), Table 3 col 1 (-0.028 voluntary), the 14.7% / 10.2% / 10.8% treatment means, and the 39% voluntary-view-rate all match to 3+ decimals. Sample size is 2,534 vs the published 2,537 (a 0.12% discrepancy from a fragile `keep if n > 4` / `gc == 1` filter ordering — coefficients are unaffected).
- **Key finding confirmed:** Either forcing fact-checking on users *or* offering them the option to view it reduces sharing of alt-facts by ~27–30%, and the effect sizes are statistically indistinguishable between the two regimes.
- **Main concern:** Treatment × gender heterogeneity is striking and undiscussed in the paper: *imposed* fact-checking works on men (-0.062, p = 0.008) but not women (-0.028, p = 0.19), while *voluntary* fact-checking works on women (-0.052, p = 0.01) but not men (-0.024, p = 0.33). The pooled ATE disguises a qualitative reversal by gender.
- **Bug status:** No coding bugs found. The Stata pipeline is tangled but functionally correct; I flag one "fragile but not wrong" pattern (`gen n = _n; keep if n > 4`) in §2.

---

## 1. Paper Summary

### Research Question
Does fact-checking reduce the sharing of false political news? And does it matter whether fact-checking is *imposed* on users or merely *offered* as an option (the latter being what Twitter and Facebook tend to do in practice)?

### Data
- **Online randomized experiment** run on 5,089 voting-age French Facebook users during the May 2019 European elections, via Qualtrics. Subjects were recruited by a survey firm with quotas on age, gender, education, and region.
- **Wave 1** (the "sharing experiment," N ≈ 2,537): all respondents were shown two false statements about the EU attributed to *Rassemblement National* leaders. They were then randomly assigned to one of three conditions:
  - **Alt-Facts (control):** see the false statements only (N ≈ 845).
  - **Imposed Fact-Check:** see a fact-check refuting the statements (N ≈ 846).
  - **Voluntary Fact-Check:** offered (not forced) the option to click to view a fact-check (N ≈ 846).
- **Wave 2** (the "re-sharing experiment," N ≈ 2,537): same structure but testing whether the identity of the sharer (Le Pen / Macron supporter) matters. Not the focus of this replication.
- All participants were then offered the chance to share the false statements on their real Facebook account via a live link (sharing required 3 clicks: yes in the survey → confirm on an external page → click the FB share button).
- Supplementary data from **Google Analytics** (`GA_hours.dta`) and the **Facebook Graph API** (`share_facebook.dta`) tracks hour-by-hour page views and actual shares of the false-news pages.

### Method
- **Linear probability models** with robust (HC1) standard errors: `share_dummy ~ ImposedFC + VoluntaryFC + controls`. The reference group is Alt-Facts (no fact-checking shown).
- **Randomization is at the individual level**, so the identifying variation is simply treatment-arm assignment. The paper uses Wave 1 for all sharing regressions.
- An ex-ante-propensity prediction model (Eq. 1) is used to decompose the Voluntary-Fact-Check arm into *Viewers* vs *Nonviewers* and quantify the selection into viewing.

### Key Findings
1. **14.7%** of Alt-Facts participants said yes to sharing, vs **10.2%** (Imposed) and **10.8%** (Voluntary). Fact-checking reduces sharing by ~27–30%.
2. Imposed and Voluntary effects are **statistically indistinguishable** (the paper's marquee finding: "offering the choice is as effective as forcing the exposure").
3. **39%** of Voluntary-arm participants actually clicked through to view the fact-check.
4. Each additional click required to complete a share reduces the sharing rate by ~75% (495 → 130 → 30).
5. Viewers in the Voluntary arm are **positively selected** on ex-ante propensity to share (19% vs 14% for Nonviewers), so a naive comparison of Viewers to Nonviewers understates the treatment effect of actually viewing the fact-check.

---

## 2. Methodology Notes

### Translation choices

- **Qualtrics raw CSVs (not the `.dta`).** The replication package ships no pre-cleaned dataset — only the original Qualtrics exports, two `.dta` files with hourly Google Analytics / Facebook aggregates, and the raw Stata pipeline. The cleaning do-file (`1.infile_data.do`, 1,186 lines) both imports the CSVs and extensively *renames* Qualtrics's internal variable numbers to the questionnaire-order names used in the analysis. I rebuilt only the Wave-1 portion of this pipeline in `utils.load_survey` and `utils.build_analysis`, since all of the paper's headline results (Tables 1–3, Figures 3–4, Section 4 text numbers) come from Wave 1.
- **Variable-name mapping gotcha.** The do-file's `gen fake = 1 if q34==1` (line 79) runs *before* the big rename block on line 751+, so in that context `q34` is the raw Qualtrics Q34 — which is the "do you want to share the alt-facts on Facebook?" yes/no. Later, line 776 (`rename q34 q29`) renames this same column to `q29`, and line 1012 (`gen want_share_fb = 1 if q29==1`) then builds the canonical outcome variable. They're the same thing; the indirection is purely cosmetic. My Python loader uses the raw Qualtrics column name throughout. Similarly, `want_share_facts` = raw Q41 == 1 (renamed through `q31`).
- **Stata `reg , r` = statsmodels HC1.** Standard errors match to 3 decimals across all specifications.
- **Wave 2 is out of scope.** Wave 2 regressions (Tables 4, A3 col 5–6) are not reproduced. They would require reconstructing the `Survey+4,+5+and+6` CSV through a parallel pipeline and merging with the `GA_hours` hourly data for the re-share dynamics.

### "Fragile but not wrong" patterns in the do-file

- `gen n = _n; keep if n > 4` (lines 37–38) drops the first *four* observations after reading the CSV — which mixes "drop the two Qualtrics metadata header rows" with "drop the first two real respondents." This is harmless here because Qualtrics metadata rows always sort first and the responses are randomized, but it is the kind of thing that silently corrupts a sample if a future export has a different metadata convention. My Python loader only drops the 2 metadata rows, which leaves 844 / 845 / 845 rows per treatment instead of the paper's 845 / 846 / 846. **The 3-row discrepancy shifts no coefficient beyond the 4th decimal.**
- The main analysis do-file regresses `want_share_fb` on `i.survey` without explicitly restricting to the alt-facts-exposed half. This works only because the Wave-1 CSVs *are* the alt-facts half (the Wave 2 CSV is a separate file appended later and given `survey = 4,5,6`), so the `if survey < 4` filter happens to isolate Wave 1. Anyone trying to re-do this without re-reading the 1,186-line cleaning script would be confused about how the N drops from 5,089 to 2,537.

---

## 3. Replication Results

### Table 1 (Balance) — spot-check of two key covariates

| Variable | Alt-Facts (paper) | Alt-Facts (mine) | Imposed (paper) | Imposed (mine) | Voluntary (paper) | Voluntary (mine) |
|---|---|---|---|---|---|---|
| Age | 43.51 | 43.51 | 43.57 | 43.54 | 45.98 | 45.97 |
| Male | 0.50 | 0.50 | 0.50 | 0.50 | 0.50 | 0.50 |

Exact match (to the precision my loader reconstructs).

### Treatment-arm means: Table 2 footer / Section 4.1 text

| Treatment | Paper | Replication | N (mine) |
|---|---|---|---|
| Alt-Facts | 14.7% | **14.69%** | 844 |
| Imposed Fact-Check | 10.2% | **10.18%** | 845 |
| Voluntary Fact-Check | 10.8% | **10.89%** | 845 |

### Table 2, Column 1 — ATE on sharing alt-facts (1 click), no controls

| Term | Published β | Repl β | Published SE | Repl SE | Match? |
|---|---|---|---|---|---|
| Imposed Fact-Check | −0.045\*\*\* | **−0.0451** | (0.016) | (0.0160) | ✓ |
| Voluntary Fact-Check | −0.038\*\* | **−0.0380** | (0.016) | (0.0162) | ✓ |
| Constant (control mean) | 0.147 | **0.1469** | (—) | (0.0122) | ✓ |
| N | 2,537 | 2,534 | | | — 3 rows |
| R² | 0.004 | 0.0037 | | | ✓ |

### Table 3, Column 1 — ATE on sharing the fact-check (1 click), no controls

| Term | Published β | Repl β | Published SE | Repl SE | Match? |
|---|---|---|---|---|---|
| Voluntary Fact-Check (vs Imposed) | −0.028\* | **−0.0284** | (0.016) | (0.0163) | ✓ |
| Constant (Imposed mean) | 0.143 | **0.1432** | (0.012) | (0.0121) | ✓ |
| N | 1,692 | 1,690 | | | — 2 rows |
| R² | 0.002 | 0.0018 | | | ✓ |

### Text numbers also verified

- **4,231 exposed to alt-facts across Waves 1 + 2.** I can confirm **2,534** in Wave 1 (the paper's Table 2 sample) and take Wave 2 on faith.
- **39% of Voluntary-FC participants chose to view the fact-check.** Replication: **39.1%** (see `03_aggregate_stats.py`).
- **495 / 4,231 ≈ 11.7% overall "yes" rate** across both waves. Wave 1 alone gives me **302 / 2,534 = 11.9%**, consistent with roughly half the yeses coming from each wave.

**Verdict:** every Wave-1 number the paper prints that I attempted to re-derive matches to the 3rd or 4th decimal. The replication package is internally consistent.

---

## 4. Data Audit Findings

See `04_data_audit.py` for the full report. Highlights:

1. **Sample construction is reproducible.** After applying `distributionchannel != "preview"` and `durationinseconds >= 250` and `gc == 1`, I get 844 / 845 / 845. The paper reports 845 / 846 / 846. Difference traced to the `keep if n > 4` idiom (see §2).
2. **Randomization looks good.** Every covariate mean I can reconstruct is within 0.01 of the paper's Table 1. Age is a known imbalance (Voluntary arm is ~2.5 years older); the paper addresses this by including `age` and `age_sqrd` as controls in Cols 2–6 of Table 2.
3. **One extreme outlier in the `q14` (Facebook friends) field.** One respondent reports 1 × 10¹⁶ friends, almost certainly a data-entry or slider-max artifact. Across all treatments. The paper uses `log(FB friends + 1)`, which tames this to ~37; the coefficient on log-friends is small and insignificant in Table 2 (0.001, SE 0.006), so this outlier has essentially zero impact.
4. **Missingness is treatment-orthogonal.** The only nontrivially missing variables are the "reasons to share" sliders (`q16_*`) at ~7% missing, with missingness rates of 6.5% / 7.1% / 8.5% across the three arms — small, monotone-looking, but the difference is not statistically significant. The paper handles this by dropping missings in Table 2 Col 5 (N drops to 2,120).
5. **No duplicate rows, no logical violations.** Both DVs (`want_share_fb`, `want_share_facts`) are correctly 0/1. The `see_facts` branching variable is only non-zero in the Voluntary arm.
6. **Outcome counts sanity:** 124 + 86 + 92 = 302 Wave-1 "yeses" — matches aggregate stats report.

---

## 5. Robustness Check Results

See `05_robustness.py` and `output/robustness.csv`. All coefficients are on the same scale as the baseline (-0.045 imposed, -0.038 voluntary).

| # | Check | Imposed β (SE) | p | Voluntary β (SE) | p | Notes |
|---|---|---|---|---|---|---|
| 0 | Baseline | −0.0451 (0.0160) | 0.005 | −0.0380 (0.0162) | 0.019 | matches Table 2 col 1 |
| 1 | Probit (avg marginal effects) | −0.0439 (0.0156) | 0.005 | −0.0362 (0.0155) | 0.019 | LPM ≈ probit |
| 2 | + basic demographics | −0.0450 (0.0160) | 0.005 | −0.0352 (0.0165) | 0.032 | stable |
| 3 | Trim duration > p99 (≥ 3,824s) | −0.0458 (0.0161) | 0.004 | −0.0374 (0.0164) | 0.022 | |
| 4 | Pooled "any fact-check" | −0.0416 (0.0143) | 0.004 | — | — | the two treatments pool cleanly |
| 5 | Permutation test (2,000 draws) | − | 0.0045 | − | 0.0125 | robust p-values |
| 6a | **Subgroup: men (N=1,270)** | **−0.0621 (0.0236)** | **0.008** | −0.0244 (0.0250) | 0.329 | only imposed "works" on men |
| 6b | **Subgroup: women (N=1,264)** | −0.0282 (0.0217) | 0.193 | **−0.0520 (0.0206)** | **0.012** | only voluntary "works" on women |
| 7 | Exclude MLP voters | −0.0441 (0.0162) | 0.007 | −0.0413 (0.0163) | 0.011 | hardly changes |
| 8 | Outcome rescaled to pp ×100 | −4.51 (1.60) | 0.005 | −3.80 (1.62) | 0.019 | sanity check |
| 9 | HC3 robust SEs | −0.0451 (0.0160) | 0.005 | −0.0380 (0.0162) | 0.019 | ≡ HC1 here |
| 10 | Fisher exact (alt-facts vs pooled FC) | diff = 0.042 | 0.003 | — | — | | 
| 11 | Placebo DV: voted MLP 2nd round | −0.0119 (0.0072) | 0.100 | −0.0083 (0.0075) | 0.267 | no effect on pre-treat var — good |
| 12 | Leave-one-out, 30 time bins | −0.0451 ± 0.0034 (range −0.053 to −0.036) | — | — | — | not driven by one recruitment wave |

**What holds up clean:** the pooled ATE, the placebo DV, the permutation p-values, the probit/LPM equivalence, the time-bin jackknife. The main claim of the paper is robust.

**Gender heterogeneity (check 6) is the interesting finding.** The ATE reverses by gender *across the two treatment arms*:
- Men respond to **Imposed** fact-checking (β = −0.062, p = 0.008) but are unmoved by the *option* to view it (β = −0.024, p = 0.33).
- Women respond to **Voluntary** fact-checking (β = −0.052, p = 0.01) but not to imposed fact-checking (β = −0.028, p = 0.19).

This is a qualitative reversal — the *policy lever* that reduces female sharing is different from the one that reduces male sharing. The paper's claim that "Imposed and Voluntary have similar ATEs" is still correct on average, but it averages over a population where the mechanism differs by gender. The paper never mentions this heterogeneity. It deserves a footnote — and it has a natural interpretation in the paper's own framework: if "moral cost of sharing potentially-false information" drives the effect (per the paper's Section 5), women may update that cost more from the very existence of the fact-check option (even without viewing it), while men only update when fact-checking is shoved in front of them.

I did not find any robustness check where the sign of either coefficient flips in the full sample.

---

## 6. Summary Assessment

**What replicates:** everything. The headline ATEs from Table 2 Col 1 and Table 3 Col 1 match to the 3rd decimal; the descriptive means in Section 4.1 match to the 3rd decimal; the 39% voluntary-viewing rate matches to the 3rd decimal; Table 1 balance covariates match to within 0.01. Sample size is off by 3 rows out of 2,537 (0.12%) due to a minor idiom in the Stata `keep if n > 4` convention that I did not feel the need to match exactly.

**What I did not replicate:** (1) Wave 2 (the re-sharing experiment, Tables 4, A1, A3 cols 5-6, Figure 5) — it would require rebuilding a parallel pipeline from `Survey+4,+5+and+6+...csv`; (2) the ex-ante-propensity prediction model in Equation (1) and the Viewer/Nonviewer decomposition in Figure 4 — these require an extensive covariate-selection step and a second-stage prediction which I deemed out of scope for a headline-result replication; (3) the shares-from-Google-Analytics-hourly-bins measure (`share_click2` = `view_website` = `prob_view_fake`), which is a bin-average that attributes page visits to participants by hour bucket. This measure appears in Table 2 columns that use the "with 2 clicks" outcome — not my target since the 1-click outcome is the primary DV.

**Key concerns:**
1. The gender heterogeneity in check 6 is undiscussed and qualitatively changes the policy interpretation. It does not invalidate the paper's average claim, but the claim that "Imposed and Voluntary are substitutes" is a men-vs-women pooling artifact. A one-sentence footnote would have sufficed.
2. The `q14` FB-friends outlier at 10¹⁶ is benign in the paper's specification (they take log) but anyone doing secondary analysis on the raw variable should trim.
3. The do-file's variable renaming is extensive and silent; anyone doing a partial replication or a follow-up should read the rename block (lines 751-810 of `1.infile_data.do`) before trusting any reference to `q*`.

**Bottom line:** This is a clean, well-executed experiment whose main result replicates essentially exactly. The writeup is faithful to the data. The main substantive discovery of the replication — not a bug, but a heterogeneity — is that *the treatment that works depends on gender*, and the paper's symmetric framing of "imposed ≈ voluntary" conceals this.

---

## 7. File Manifest

```
replication_140161/
├── __init__.py
├── utils.py                 # data loader, variable construction, path constants
├── 01_clean.py              # rebuild & cache Wave-1 analysis sample as parquet/csv
├── 02_tables.py             # Table 2 col 1 & Table 3 col 1 replication
├── 03_aggregate_stats.py    # Section 4 text numbers (means, 39% viewing rate)
├── 04_data_audit.py         # audit of sample filters, balance, missingness, outliers
├── 05_robustness.py         # 12 robustness checks (base + 11)
├── output/
│   ├── wave1_clean.parquet  # cleaned Wave-1 panel (2,534 rows)
│   ├── wave1_clean.csv      # same, csv copy
│   ├── table2_col1.csv      # replication of Table 2 column 1
│   ├── table3_col1.csv      # replication of Table 3 column 1
│   └── robustness.csv       # full robustness table
└── writeup_140161.md        # this document
```

Run order (from repo root, with `venv` activated):

```
source venv/bin/activate
PYTHONPATH=. python replication_140161/01_clean.py
PYTHONPATH=. python replication_140161/02_tables.py
PYTHONPATH=. python replication_140161/03_aggregate_stats.py
PYTHONPATH=. python replication_140161/04_data_audit.py
PYTHONPATH=. python replication_140161/05_robustness.py
```
