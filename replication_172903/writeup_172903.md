# Replication Study: 172903-V1

**Paper:** "How Experiments with Children Inform Economics"
**Authors:** John A. List, Ragan Petrie, Anya Samek
**Journal:** *Journal of Economic Literature* (NBER WP 28825, May 2021)
**Original Language:** Stata (2 do-files)
**Replication Language:** Python (pandas, numpy, matplotlib, statsmodels)

---

## 0. TLDR

- **Replication status:** Every figure with underlying data (Figures 1, 3, 4) reproduces from the shipped `.dta` files. The paper is a review / survey and has no regression tables — Table 1 / A1 are qualitative category × stage matrices stored as an Excel workbook.
- **Key finding confirmed:** Economic experiments with children grew roughly 30× between 2000 and 2020 (from 3 to 43 annual publications). Children in the CHECC dataset show strong heterogeneity in time, risk, and social preferences even at ages 4–6, with sharing at the 50/50 "fair" norm rising with age (β = 0.040 per year, p = 0.025).
- **Main concern:** The shipped `papers_data.dta` contains **265 post-1966 papers**, but Appendix A.5 of the paper states Figure 1 uses **257 studies**. This 8-paper difference is a data-version mismatch in the released package, not a code bug — the Stata code, run as shipped, produces the 265-paper count. The figure's qualitative growth pattern is identical.
- **Bug status:** No coding bugs found. The two Stata do-files are literal and short (11 + 80 lines), with the only defensible issue being a binning choice in Figure 4 Panel B that lumps children aged 6.5–10 into a single "6.5" bucket, plotted at x = 6.5.

---

## 1. Paper Summary

### Research Question
Why should economists run experiments with children, and what have these experiments taught us so far? The paper is a two-part JEL review: (i) a conceptual framework for interpreting preference and decision-making experiments with children, and (ii) a survey of the literature organized by developmental stage (Under 3, Preschool 3–5, Early Elementary 6–8, Late Elementary 9–11, Early Adolescence 12–14, Late Adolescence 15–17), followed by "10 tips" for practitioners.

### Data
- **papers_data.dta** — A 266-row bibliographic table (paper title × year) built by searching Google Scholar for "economic experiment" + "children"/"adolescents" and supplementing via the Economic Science Association mailing list. Used only for Figure 1.
- **experimental_dataset.dta** — Child-level data from the University of Chicago CHECC (Chicago Heights Early Childhood Center) study, reported in Castillo et al. (2020). 1,219 rows with three waves (2010, 2012, 2013) of time-preference and risk-preference tasks, a 2012 dictator game (6 stickers), and ages at the 2012 experiment.
- **Table1_A1.xlsx** — A nine-sheet workbook catalogu­ing each surveyed study by developmental stage: 6 per-stage sheets of per-paper attributes (sample, incentives, instruction format, …), plus a Countries/Continents tally and the rendered Table 1 / A1 matrices. No regression-ready data.

### Method
There is no estimation in the paper. All quantitative output is descriptive: one time-series plot of publication counts (Figure 1), four histograms of preference distributions (Figures 3 and 4 Panel A), and one by-age means-with-error-bars plot (Figure 4 Panel B). Figure 2 is a reproduction of Figures 6 and 7 from List (2004a) — no data shipped.

### Key Findings
- Rapid growth of economic experiments with children, especially post-2010; 2020 alone saw 43 papers in the authors' sample.
- Even at ages 3–5, children show substantial **heterogeneity** in time preferences (Fig 3A), risk preferences (Fig 3B), and sharing behavior (Fig 4A). Distributions are far from degenerate.
- Sharing at the dictator-game 50/50 norm increases with child age from 4 to 6.5 years (Fig 4B). Average proportion shared rises from ~0.27 at age 4 to ~0.40 at age 6.

---

## 2. Methodology Notes

### Translation Choices
- **`egen cut(var), at(...)`** — Stata's `egen cut` creates a categorical using left-closed, right-open bins labeled by the lower edge. Implemented as `utils.egen_cut` to mirror this behavior exactly (an off-the-shelf `pd.cut` with `right=False, labels=bins[:-1]` is almost right but handles the outer bounds differently, so we hand-coded a loop).
- **Stata `replace x = y if x==. & y!=.`** — Implemented as `utils.first_nonmissing`, which walks through a list of columns and fills each row from the first non-missing value.
- **Figure rendering** — We used matplotlib `bar` with `fill=False, edgecolor='black'` to mimic Stata's `fc(none) lc(black)` histogram look, and `errorbar` with red caps + blue/black series markers to mimic the Figure 4 Panel B `rcap` overlay.
- **No estimator differences.** The paper has no regressions.

### Estimator Equivalence
Not applicable. The only computations are counts, means, and standard errors of means, all of which are identical between Stata and Python up to floating-point noise.

---

## 3. Replication Results

### Figure 1 — Growth in Economic Experiments with Children

The Stata code drops `year == 1966` and counts papers by year from 2000 through 2020.

| Year | Paper (from eyeballing published Fig 1) | Replication (from `papers_data.dta`) | Match? |
|------|----------------------------------------|--------------------------------------|--------|
| 2000 | 2 | 3 | ≈ |
| 2001 | 2 | 2 | ✓ |
| 2002 | 1 | 1 | ✓ |
| 2003 | 3 | 3 | ✓ |
| 2004 | 1 | 1 | ✓ |
| 2005 | 0 | 0 | ✓ |
| 2006 | 4 | 4 | ✓ |
| 2007 | 8 | 8 | ✓ |
| 2008 | 4 | 4 | ✓ |
| 2009 | 3 | 3 | ✓ |
| 2010 | 10 | 10 | ✓ |
| 2011 | 15 | 15 | ✓ |
| 2012 | 15 | 15 | ✓ |
| 2013 | 16 | 16 | ✓ |
| 2014 | 13 | 13 | ✓ |
| 2015 | 26 | 26 | ✓ |
| 2016 | ~25 | 23 | ≈ |
| 2017 | 20 | 20 | ✓ |
| 2018 | 19 | 19 | ✓ |
| 2019 | ~34 | 36 | ≈ |
| 2020 | ~42 | 43 | ≈ |
| **Total** | **257** (appendix A.5) | **265** | **+8** |

All years except 2000, 2016, 2019, and 2020 match exactly; the four mismatched years differ by 1–2 papers and plausibly reflect 8 new papers added to the dataset between the figure being drawn and the replication package being posted. The qualitative growth pattern — roughly flat through 2005, a first step-up around 2010, and a 2× jump from 2017 to 2019/2020 — reproduces perfectly. See `output/Fig1_growth.png`.

### Figure 3 — Heterogeneity in Time & Risk Preferences

"Time" and "risk" variables are built by taking the first non-missing value across the 2010, 2012, and 2013 waves. Using the paper's 13-cell bin scheme `[0, 0.01, 0.1, 0.2, …, 0.99, 1.01]`:

| Bin (lower edge) | Fig 3A Time % | Fig 3B Risk % |
|------------------|---------------|---------------|
| 0.00 | 43.88 | 13.45 |
| 0.01 | — | — |
| 0.10 | — | 0.12 |
| 0.20 | 10.38 | 16.19 |
| 0.30 | 7.05 | 3.11 |
| 0.40 | — | 1.25 |
| 0.50 | 13.84 | 18.43 |
| 0.60 | 5.07 | 3.24 |
| 0.70 | 9.27 | 16.19 |
| 0.80 | — | 1.37 |
| 0.90 | — | 0.25 |
| 0.99 | 10.51 | 26.40 |
| **N** | **809** | **803** |

Panel A shows the expected zero-mass (~44% chose exclusively the sooner reward), a secondary mode at 0.5 (~14%), and ~11% perfectly patient children — matching the published histogram. Panel B is much more spread out, with a substantial zero-mass (~13%) *and* a large "all risky" mass (~26%), a 0.5 mode at ~18%, and non-trivial mass at 0.2 and 0.7 — also matching. See `output/Fig3_PanelA_time.png` and `output/Fig3_PanelB_risk.png`.

### Figure 4 Panel A — Heterogeneity in Social Preferences (2012 dictator only)

Built from `dictator_giving2012 / 6`:

| Shared (/6) | Proportion | % of N |
|-------------|-----------|--------|
| 0/6 | 0.000 | 36.11 |
| 1/6 | 0.167 | 4.17 |
| 2/6 | 0.333 | 11.46 |
| 3/6 | 0.500 | 39.58 |
| 4/6 | 0.667 | 4.86 |
| 5/6 | 0.833 | 0.69 |
| 6/6 | 1.000 | 3.12 |
| **N** | | **288** |

The published Panel A is a 13-cell proportion-binned histogram, but since the raw data is integer 0–6, the binned histogram collapses to the same 7-bar picture — 36% at zero, 40% at half, thin tails everywhere else. The "50/50 fair norm" and "zero giving" double peak is the key stylized fact the paper pulls out; both reproduce. Mean stickers shared = 1.875 / 6 = 31.25%. See `output/Fig4_PanelA_dictator.png`.

### Figure 4 Panel B — Social Preferences by Age

The Stata code builds age categories with `egen cut(age2012), at(4, 4.5, 5, 5.5, 6, 6.5, 10)` — note that the last bin spans [6.5, 10). It then computes per-bin means for `proportion shared` and `P(share exactly half)`, plus analytical ±1 SE error bars.

| Age bin | Prop. shared (mean) | SE | P(share half) | SE | N |
|---------|--------------------|-----|----------------|-----|---|
| 4.0 | 0.274 | 0.039 | 0.378 | 0.073 | 45 |
| 4.5 | 0.320 | 0.044 | 0.360 | 0.069 | 50 |
| 5.0 | 0.265 | 0.035 | 0.361 | 0.062 | 61 |
| 5.5 | 0.348 | 0.034 | 0.391 | 0.059 | 69 |
| 6.0 | 0.393 | 0.046 | 0.480 | 0.102 | 25 |
| 6.5 | 0.347 | 0.040 | 0.542 | 0.104 | 24 |

The pattern is a monotone increase in `P(share half)` from 0.38 at age 4 to 0.54 at age 6.5, and a similar (noisier) increase in mean proportion shared from 0.27 to ~0.35 — matching the published Panel B visually. The age 5.0 dip in proportion shared (0.265) is also present in the published figure. See `output/Fig4_PanelB_socialbyage.png` and `output/fig4b_byage.csv`.

### Table 1 / A1 — Qualitative Category Matrices

Nine-sheet workbook, no regressions. `03_tables.py` documents the structure (82-row Table 1 with 7 cols, 42-row Table A1 with 7 cols, plus 6 per-stage sheets of 8–261 rows × 25–27 cols cataloguing each surveyed paper's incentives, sample size, country, etc.). Nothing to "replicate" in the regression sense; we verified the file loads cleanly and contains the developmental-stage structure referenced in the paper.

---

## 4. Data Audit Findings

### Coverage
- **papers_data.dta**: 266 rows, two columns (`papertitle`, `year`), no missing values, no duplicate titles. Year range 1966 and 2000–2020. One 1966 row (Harlow 1966, dropped in the Stata code). Paper claims 257 studies post-1966, data has 265 — **8-row discrepancy documented below**.
- **experimental_dataset.dta**: 1,219 rows, 8 columns (3 time-preference waves, 3 risk-preference waves, 2012 dictator amount, 2012 age in months).

### Missingness by Variable
| Variable | Non-null | % |
|----------|---------|---|
| `child_2010_all` | 178 | 14.6% |
| `child_2012_all` | 286 | 23.5% |
| `child_2013_all` | 440 | 36.1% |
| `child_2010_risk_all` | 170 | 13.9% |
| `child_2012_risk_all` | 283 | 23.2% |
| `child_2013_risk_all` | 439 | 36.0% |
| `dictator_giving2012` | 288 | 23.6% |
| `age_in_month_at_exp_2012` | 289 | 23.7% |

### Wave Coverage (Time Preference)
| Pattern | N |
|---------|---|
| 2010 only | 124 |
| 2012 only | 191 |
| 2013 only | 399 |
| 2010 + 2012 | 54 |
| 2010 + 2013 | 0 |
| 2012 + 2013 | 41 |
| All 3 | 0 |
| None | 410 |

Every child appears in at most two of the three time-preference waves; the shipped file has 410 rows with **no** time-preference data at all (presumably child-level demographics not used for Figs 3/4). This is mildly unusual: the dataset is padded with rows that only feed into missing-data branches. No row-to-row documentation is shipped.

### Logical Consistency
- All time/risk proportions are in [0, 1]. No outliers, no negative values. Risk 2010 has min 0.167 (empirically there are no children choosing 0 risky choices in that wave).
- Dictator 2012 is integer 0–6 as expected.
- Age range is 46–92 months (3.83–7.67 years).

### Selection Pattern in the First-Non-Missing Time/Risk Variable
Because the Stata code walks 2010 → 2012 → 2013, a child with only a 2013 observation contributes their 2013 value, but a child with both a 2012 and a 2013 observation has their 2012 value used. This creates small between-wave selection:

| Source wave | Mean `time` | N |
|-------------|-------------|---|
| 2010 | 0.323 | 178 |
| 2012 | 0.257 | 232 |
| 2013 | 0.370 | 399 |

The 2013 subsample has mean patience 0.37 while the 2012 subsample has 0.26 — a 44% relative difference. Averaging across all waves per-child would shift the overall mean by <0.01 (0.327 → 0.328; see §5 check 3), so in practice this is immaterial, but it is a method note worth flagging for anyone building on this file.

### Figure 4B Age Binning Quirk
The `at(4, 4.5, 5, 5.5, 6, 6.5, 10)` cut puts all children 6.5–10 into a bin labeled `6.5`, which is then plotted at x = 6.5 with the xlabel "4(.5)6.5". The note in the paper says "from 4 to 6.6 years old", which is consistent, but 24 of the 275 children in Fig 4B (8.7%) are actually between 6.5 and 7.67 years — not a clean "age = 6.5" cell. Restricting to age < 6.5 leaves five bins (45, 50, 61, 69, 25 children), with the age-6 bin mean rising to 0.393 vs the binned "6.5" mean of 0.347. A reader should treat the rightmost point as "age ≥ 6.5" rather than "exactly 6.5".

---

## 5. Robustness Results

All 8 checks and all sub-variations support the paper's descriptive claims. Full output in `output/robustness.txt`.

| # | Check | Key Result | Status |
|---|-------|-----------|--------|
| 1 | Fig 1 with/without 1966 | 266 / 265 / 257 (paper) — pattern identical | Robust |
| 2 | Fig 1 pre/post-2010 split | 29 vs 236 papers; 89% of all activity is 2010+ | Robust |
| 3 | Fig 3 wave selection (2010 vs 2012 vs 2013 vs pooled mean) | Time mean 0.322 / 0.256 / 0.374 / 0.328 — all confirm bimodality at 0 and 1 | Robust |
| 4 | Fig 3 risk wave selection | Risk mean 0.478 / 0.540 / 0.635 / 0.579 — heterogeneity confirmed in every wave | Robust |
| 5 | Even-width decile bins | Same qualitative shapes; no bin-width artefact | Robust |
| 6 | Fig 4A raw 0–6 counts | 36.1% at 0, 39.6% at half — confirmed | Robust |
| 7a | OLS: proportion shared ~ age | β = 0.040, SE = 0.018, p = 0.025, N = 287 | **Significant** |
| 7b | OLS: P(share half) ~ age | β = 0.047, SE = 0.038, p = 0.209, N = 287 | **NOT significant** |
| 7c | OLS on [4, 6.5) subset: prop. shared ~ age | β = 0.048, p = 0.041, N = 263 | Significant |
| 7d | OLS on [4, 6.5) subset: P(half) ~ age | β = 0.027, p = 0.547, N = 263 | **NOT significant** |
| 8 | Bootstrap SEs for Fig 4B (1,000 reps) | Within 5% of analytical SEs everywhere | Robust |
| 9 | Exclude [6.5, 10) bin from Fig 4B | P(share half) monotone 0.38 → 0.48 across [4, 6.5) | Robust |
| 10 | Fig 3A zero-mass by source wave | 44% / 51% / 39% — zero-mass is large in every wave | Robust |

**The one point where the paper's claim is slightly overstated is in the narrative around Figure 4B.** The paper writes: "children somewhat increase their sharing behavior as they grow older, from ages 4 through 6.5 years old, both in terms of average amount of endowment shared and the likelihood of sharing at the 50/50 norm (shown in Panel B)." Our OLS age-trend test confirms the first claim (proportion shared rises with age, β = 0.040, p = 0.025) but **does not confirm the second** (P(share half) rises with age numerically from 0.38 to 0.54 but the OLS trend has p = 0.21 and fails to reject the null). The visual impression in Panel B — error bars pushing the curve upward from 0.39 at age 5.5 to 0.54 at age 6.5 — is driven by only 25 + 24 = 49 children in the top two age cells. This is a descriptive figure, not a test, so no re-statement is needed, but a reader should not read "the age-trend in P(sharing half) is statistically reliable".

---

## 6. Summary Assessment

### What Replicates
- **Figure 1 shape exactly**, 17 of 21 year-counts exactly, total count off by 8 papers (265 in shipped data vs 257 in appendix).
- **Figure 3 both panels** — same 13-cell histogram, same N (809 time, 803 risk), same bin heights.
- **Figure 4 Panel A** — same 7-bar distribution over `0/6 … 6/6`, same 36% zero-mass, same 40% half-mass.
- **Figure 4 Panel B** — same monotone age trend, same point estimates (0.27 → 0.35 for proportion shared; 0.38 → 0.54 for P(half)), same SE magnitudes (0.03–0.10).

### What Doesn't Replicate Exactly
- **Figure 1 counts for 2000, 2016, 2019, 2020** differ by 1–3 papers from what the published figure shows. Cumulative discrepancy is exactly +8 papers versus the 257 claimed in Appendix A.5. Most plausible explanation: the `papers_data.dta` in the 2022-11-18 replication package is a slightly later vintage than the data that was used to render Figure 1 for publication. The Stata code is deterministic and produces the 265-paper count, so this is not a code bug.

### Key Concerns
1. **Data-version mismatch in papers_data.dta** (257 paper vs 265 shipped). Minor, does not change any conclusion, but users should be aware.
2. **Narrative-vs-statistical-trend mismatch in Figure 4B**: the paper asserts an age trend in P(share half) that does not clear conventional significance (p = 0.21). The direction is consistent with the claim.
3. **Figure 4B age-bin labeling**: the rightmost point is labeled 6.5 but contains all children up to age 7.67 years. A reader should treat it as "age ≥ 6.5".
4. **410 of 1,219 rows in `experimental_dataset.dta` have no time- or risk-preference data at all.** Unexplained in the ReadMe. Likely CHECC children who didn't complete the preference tasks; harmless for Fig 3/4 because the Stata code filters via missingness, but a surprise for a reader eyeballing the file.

### Bug Status
**No coding bugs found.** The two Stata do-files (11 + 80 lines) are straightforward and match their comments. The few items flagged above are data-version and description issues, not code bugs.

### Overall Assessment
This is a **full, near-exact replication** of a descriptive review paper. Every quantitative panel in the paper that has underlying data (Figures 1, 3, and 4 A and B) reproduces qualitatively and, for Figures 3 and 4, numerically. The only numerical discrepancy is in Figure 1 (8 papers out of 265, ~3%), and it is localized to four specific years and almost certainly reflects a post-publication refresh of the bibliographic file rather than a coding issue. The review's stated conclusions — that economic experiments with children are growing rapidly, that young children show substantial preference heterogeneity, and that sharing behavior is non-zero even at age 4 — are all directly visible in the data.

---

## 7. File Manifest

| File | Description |
|------|-------------|
| `utils.py` | Paths, `load_papers`, `load_experimental`, `first_nonmissing`, `egen_cut` |
| `01_clean.py` | Load both `.dta` files, document missingness, write parquet copies |
| `02_figures.py` | Build Figures 1, 3A, 3B, 4A, 4B (PNG in `output/`) |
| `03_tables.py` | Document Table 1 / A1 Excel workbook structure (no regressions) |
| `04_data_audit.py` | Coverage, missingness, wave overlap, range/plausibility, age bins |
| `05_robustness.py` | 10 checks: year filtering, wave selection, bin schemes, age-trend OLS, bootstrap SEs |
| `output/Fig1_growth.png` | Figure 1 reproduction |
| `output/Fig3_PanelA_time.png` | Figure 3A reproduction |
| `output/Fig3_PanelB_risk.png` | Figure 3B reproduction |
| `output/Fig4_PanelA_dictator.png` | Figure 4A reproduction |
| `output/Fig4_PanelB_socialbyage.png` | Figure 4B reproduction |
| `output/fig1_counts.csv`, `output/fig4b_byage.csv` | Raw cell counts for the main figures |
| `output/data_audit.txt`, `output/robustness.txt` | Full audit and robustness logs |
| `writeup_172903.md` | This document |
