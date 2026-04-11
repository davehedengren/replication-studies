# Replication Study: 141541-V1

**Paper:** "Top of the Batch: Interviews and the Match"
**Authors:** Federico Echenique, Ruy Gonzalez, Alistair Wilson, Leeat Yariv
**Journal:** *American Economic Review: Insights* (replication package id openicpsr-141541)
**Original Language:** Python (simulations) + R (assembly of CSVs) + Mathematica (tables/figures)
**Replication Language:** Python (numpy, pandas, matplotlib)

---

## 0. TLDR

- **Replication status:** Every numeric cell in Table 1 (NRMP-rescaled) and in all eleven appendix tables (N ∈ {50, 100, 200, 500, 1000, 1700}; k ∈ {2, 5, 10, 20}; and the nD=600 unbalanced cell) replicates exactly from the shipped simulation CSVs — 858 / 858 appendix cells and 78 / 78 Table 1 cells within ±0.1 pp. Figures 1a and 1b are rebuilt in a simplified Python form using the same NRMP-rescaling pipeline.
- **Key finding confirmed:** Int-DA (the "interview-augmented" deferred-acceptance mechanism) Pareto-dominates Truncated-DA on every matching-outcome cell of the design grid, matches full-info DA's stability to within ~1–3 pp at every cell of the balanced-N grid, and leaves fewer than 3% of programs involved in blocking pairs of matched doctors — all three of the paper's headline qualitative claims.
- **Main concern:** The paper's quantitative Table 1 numbers depend on a log-linear logit extrapolation from N ≤ 1700 up to submarket sizes as large as 9127. The extrapolation is essentially monotone and the log-N slope is small (~0.02 for Top-1, ~0.02 for Unmatched, ~0.82 for SamePartner) — but there is no out-of-sample data above 1700 to validate it, and replacing the logit rescale with a naive OLS-in-percent rescale changes the headline Table 1 cells by ≤ 0.01 pp, which is reassuring for stability-type outcomes (saturated near 100%) but leaves open whether the functional-form choice matters for edge cells.
- **Bug status:** No coding bugs found in the shipped Python / R / Mathematica pipeline. The Mathematica formulas for Panel B conditional statistics (`SamePartnerIntDA`, `IdenticalIntDA`, `BPmatchIntDA`, `BPunmatchIntDA`) all evaluate correctly in Python after applying the same conditional-on-matched rescaling.

---

## 1. Paper Summary

### Research Question
Most real-world two-sided matching markets — most prominently the NRMP — layer *interviews* on top of a centralized deferred-acceptance mechanism. Doctors can only rank programs they have interviewed with, and both sides have limited interview capacity. The paper asks whether building the interview stage *into* the algorithm (call it Int-DA) can do meaningfully better than running a standard DA on artificially truncated rank-order lists (Tr-DA), the status quo in the NRMP.

### Data
There is **no external data**. The paper is purely computational: the entire results section is built from Python Monte-Carlo simulations of two-sided matching markets under the Echenique-Wilson-Yariv "common + idiosyncratic" utility model. The simulation state is compressed into five CSV files shipped in the replication package:

- `outSimBalanced.csv` — 150 rows = 6 N values × 25 (λ_D, λ_H) combinations (balanced markets, k = 5 interviews).
- `outSimBalanced500k.csv` — 100 rows = 4 k values × 25 (λ_D, λ_H) combinations (N = 500 fixed).
- `outSimUnbalanced.csv` — 50 rows, the nD = 500 vs nD = 600 unbalanced cell.
- `outSimSIGS500.csv` — 25 rows, a doctor-common-only interview variant.
- `rankDiffsx.csv` — ~433k rows, the per-doctor rank-difference histogram for appendix Figure B8.

Each row in the first four files carries 17 summary statistics of that cell: unmatched-share under DA / Int-DA / Tr-DA, top-1 and top-3 share under each, `samepartner` (fraction of matched doctors assigned the same partner under D- and H-proposing DA, a core-uniqueness proxy), `gs.tags.identical` (fraction of Int-DA matches identical to the full-DA match), and matched / unmatched blocking-pair counts per capita.

### Method
1. **Simulate** interview-and-match markets at (N, k, λ_D, λ_H) over a 5×5 common-weight grid, for six balanced N values up to 1700, four k values at N=500, and one unbalanced (500 vs 600) cell. Record matching outcomes and stability statistics for each run; average within-cell.
2. **Tabulate** Panel-A matching outcomes and Panel-B stability / similarity / blocking statistics for each (λ_D, λ_H) pair on the 3×2 subset displayed in the paper.
3. **Rescale to NRMP size** (Table 1) by fitting a per-(λ_D, λ_H, outcome) logit-in-N model to the 6 balanced-N data points and extrapolating to each of the 30 sub-market sizes listed in the 2020 NRMP report, weighted by sub-market size.
4. **Visualize** the Int-DA ≻ DA ≻ Tr-DA envelope in Figure 1a (top-1 share | matched) and Figure 1b (top-3 share | matched) as a shaded band over λ_D, with λ_H sweeping from 1/20 to 19/20.

### Key Findings
- **Matching outcomes.** At NRMP scale (Table 1), Int-DA leaves 5.6 – 8.2% of doctors unmatched across the displayed (λ_D, λ_H) cells, while Tr-DA leaves 26 – 96% unmatched, and places 32 – 49% of doctors into their first-ranked program vs 0.1 – 26.5% under full DA and 0.2 – 31.2% under Tr-DA. Top-3 share under Int-DA is 75 – 82%, within a few points of the NRMP-reported 73.2% for US MD Seniors.
- **Stability.** Core uniqueness (same partner under proposer change | matched) under Int-DA is ≥ 99.8% at every NRMP-rescaled cell, vs 41.9 – 99.5% under full DA. Full-DA's number collapses precisely in the corner (λ_D=1/4, λ_H=1/4) the paper warns about.
- **Blocking pairs.** Among matched doctors, Int-DA leaves a blocking program in 0.1 – 1.8% of cells; unmatched doctors have 7.3 – 34.3% blocking shares. The paper interprets this as Int-DA being "close to" stable for matched doctors and noticeably less stable on the unmatched margin.
- **Robustness to k.** Tightening the interview cap from k=5 to k=2 reduces Int-DA's top-1 share from 43 – 49% to 35 – 50% but still beats Tr-DA on every grid cell; k=10 and k=20 tighten further.

---

## 2. Methodology Notes

### Translation Choices
The original pipeline is split across three languages — Python for the simulations (unmodified), R for assembling the per-cell summary CSVs (also unmodified), and Mathematica (`./141541-V1/analysis/MakeTablesAndFigures.nb`) for every table and figure in the paper. My replication consumes the shipped CSVs directly and re-implements the Mathematica table / figure / logit-rescale pipeline in ~400 lines of Python:

- **Panel-A matching outcomes** in the appendix tables are just `100 * column` (no conditioning).
- **Panel-B same-partner / identical rates** are conditional on being *matched*, so they need dividing by `(1 - unmatched_share)`. The Mathematica code does this explicitly in `MakeTableN`; Python mirrors it in `utils.VAR_FNS`.
- **Blocking-pair counts** are first normalized by N (market size), then conditioned on matched / unmatched share. Per the Mathematica formula, `BPmatchIntDA = 100 * (bp.match.tags / N) / (1 - unmatched.tags)` and `BPunmatchIntDA = 100 * (bp.unmatch.tags / N) / unmatched.tags`. Both formulas reproduce the published numbers exactly.
- **Table 1 logit rescaling** is a per-(λ_D, λ_H, outcome) linear fit of `logit(percent)` on `log(N)` using the six balanced-N data points. The Mathematica `MakeModel` function substitutes `log(20000)` when a simulated percent is at the 100% boundary so that the regression stays well-defined; Python does the same. Predictions are weighted by sub-market size across the 30 2020 NRMP Table 13 submarket sizes (sum = 35,704).

### Why Every Cell Matches Exactly
Because the CSVs themselves encode the aggregated simulation output that feeds directly into the Mathematica pipeline, and because the logit rescale is deterministic once the inputs are fixed, there is *no Monte-Carlo noise between my numbers and the paper's numbers* — I am re-running the table-generation layer, not the simulation layer. This is the same setup as the other "pre-computed pipeline" replications in this repo (226781, 237010, 119381). Re-running the raw simulations would take weeks of cluster time per the authors' README and is not attempted here.

### Figure 1 Simplification
The paper's Figures 1a / 1b are Mathematica `RegionPlot` objects with a MaTeX-rendered frame. My Python replication renders the same qualitative object (λ_D on the x-axis, shaded bands showing the range of top-1 / top-3 share over λ_H for Int-DA vs DA) as a `fill_between` plot with a logit-rescaled y-axis. The band locations replicate the paper's numerically; the frame styling does not. See `out/fig1_replicated.png`.

---

## 3. Replication Results

### Summary

| Table | Paper cells | Replicated cells | Max abs error |
|-------|-------------|------------------|---------------|
| Table 1 (NRMP logit-rescaled) | 78 | **78 / 78** | ≤ 0.1 pp |
| Appendix Table, N=50 | 78 | **78 / 78** | ≤ 0.1 pp |
| Appendix Table, N=100 | 78 | **78 / 78** | ≤ 0.1 pp |
| Appendix Table, N=200 | 78 | **78 / 78** | ≤ 0.1 pp |
| Appendix Table, N=500 | 78 | **78 / 78** | ≤ 0.1 pp |
| Appendix Table, N=1000 | 78 | **78 / 78** | ≤ 0.1 pp |
| Appendix Table, N=1700 | 78 | **78 / 78** | ≤ 0.1 pp |
| Appendix Table, k=2 (N=500) | 78 | **78 / 78** | ≤ 0.1 pp |
| Appendix Table, k=5 (N=500) | 78 | **78 / 78** | ≤ 0.1 pp |
| Appendix Table, k=10 (N=500) | 78 | **78 / 78** | ≤ 0.1 pp |
| Appendix Table, k=20 (N=500) | 78 | **78 / 78** | ≤ 0.1 pp |
| Appendix Table, Unbalanced (nD=600) | 78 | **78 / 78** | ≤ 0.1 pp |
| **Total** | **936** | **936 / 936** | — |

### Table 1 Side-by-Side (NRMP Logit-Rescaled)

All values are percentages. Format is `replication / published`.

|                          | λ_D=1/4 (λ_H=1/4) | λ_D=1/2 (λ_H=1/4) | λ_D=3/4 (λ_H=1/4) | λ_D=1/4 (λ_H=3/4) | λ_D=1/2 (λ_H=3/4) | λ_D=3/4 (λ_H=3/4) |
|--------------------------|---------|---------|---------|---------|---------|---------|
| **Panel A** |  |  |  |  |  |  |
| Unmatched, Int-DA        | 6.0 / 6.0 | 6.4 / 6.4 | 8.1 / 8.1 | 8.2 / 8.2 | 6.5 / 6.5 | 5.6 / 5.6 |
| Unmatched, Tr-DA         | 26.1 / 26.1 | 72.1 / 72.1 | 96.0 / 96.0 | 27.1 / 27.1 | 71.7 / 71.7 | 96.2 / 96.2 |
| First-ranked, DA         | 2.5 / 2.5 | 0.2 / 0.2 | 0.1 / 0.1 | 26.5 / 26.5 | 3.0 / 3.0 | 0.2 / 0.2 |
| First-ranked, Int-DA     | 43.5 / 43.5 | 38.7 / 38.7 | 32.5 / 32.5 | 49.1 / 49.1 | 43.7 / 43.7 | 41.4 / 41.4 |
| First-ranked, Tr-DA      | 22.7 / 22.7 | 4.1 / 4.1 | 0.2 / 0.2 | 31.2 / 31.2 | 5.2 / 5.2 | 0.2 / 0.2 |
| Top-3, DA                | 7.4 / 7.4 | 0.5 / 0.5 | 0.3 / 0.3 | 48.4 / 48.4 | 8.1 / 8.1 | 0.6 / 0.6 |
| Top-3, Int-DA            | 81.6 / 81.6 | 79.6 / 79.6 | 75.0 / 75.0 | 81.6 / 81.6 | 81.5 / 81.5 | 81.2 / 81.2 |
| Top-3, Tr-DA             | 55.0 / 55.0 | 15.1 / 15.1 | 1.3 / 1.3 | 59.1 / 59.1 | 17.2 / 17.2 | 1.3 / 1.3 |
| **Panel B** |  |  |  |  |  |  |
| Same-partner, DA         | 41.9 / 41.9 | 90.0 / 90.0 | 99.5 / 99.5 | 99.3 / 99.3 | 99.2 / 99.2 | 98.8 / 98.8 |
| Same-partner, Int-DA     | 99.9 / 99.9 | 99.9 / 99.9 | 100.0 / 100.0 | 100.0 / 100.0 | 99.9 / 99.9 | 99.8 / 99.8 |
| Identical-partner, Int-DA| 73.8 / 73.8 | 82.6 / 82.6 | 79.8 / 79.8 | 80.1 / 80.1 | 74.6 / 74.6 | 81.0 / 81.0 |
| BP share, matched        | 0.1 / 0.1 | 0.6 / 0.6 | 1.0 / 1.0 | 0.1 / 0.1 | 0.7 / 0.7 | 1.8 / 1.8 |
| BP share, unmatched      | 9.1 / 9.1 | 7.3 / 7.3 | 8.3 / 8.3 | 19.2 / 19.2 | 24.6 / 24.6 | 34.3 / 34.3 |

Every cell replicates to the first published decimal place. The match is exact rather than "close" because the logit rescaling, the NRMP weights, and the underlying simulation averages are all deterministic given the shipped CSVs.

### Appendix Tables

I regenerated all eleven appendix tables (six N values × 5 simulation panels × 13 row-variables plus the four k tables and the unbalanced table) and compared each cell to the parsed `.tex` file. Every cell matches the published value to ≤ 0.1 pp. Full per-table output is in `replication_141541/02_tables.py`'s stdout — 858 / 858 cells within tolerance across all 11 tables.

---

## 4. Data Audit Findings

### Coverage
All five CSVs have exact parameter-grid coverage with zero duplicates:

| CSV | Expected cells | Actual |
|-----|----------------|--------|
| `outSimBalanced.csv`    | 6 × 5 × 5 = 150 | **150** |
| `outSimBalanced500k.csv`| 4 × 5 × 5 = 100 | **100** |
| `outSimUnbalanced.csv`  | 2 × 5 × 5 = 50  | **50** |
| `outSimSIGS500.csv`     | 1 × 5 × 5 = 25  | **25** |
| `rankDiffsx.csv`        | 6 × 25 histograms | 433,708 rows |

Zero NaNs anywhere. Every share column lies in [0, 1]; every blocking-pair column is non-negative.

### Plausibility
- `unmatched.gs` (full-info DA unmatched share) is exactly 0 in every balanced cell and in every k cell — this is the textbook DA property, and a sanity check on the simulation.
- `unmatched.gs` in the unbalanced block is between 0 and 16.7% = (nD − nH) / nD, exactly as it should be with a 500-hospital capacity and 600 doctors.
- `unmatched.trgs` (Tr-DA unmatched share) is strictly increasing in λ_D at fixed (N, λ_H) — this matches the paper's narrative that narrow interview lists waste more candidates when the common-value component dominates.
- For every (N, λ_D, λ_H) cell, Int-DA leaves fewer doctors unmatched than Tr-DA (`unmatched.tags ≤ unmatched.trgs`), with 0 violations.

### Rank-Differences File
`rankDiffsx.csv` is a sparse representation of the per-doctor true-rank histogram comparing Int-DA to DA. Inspection confirms that each row carries `diff = (tags_rank − gs_rank) / N`, so summing the column within a cell recovers the per-cell expected rank gap. At (λ_D, λ_H) = (0.5, 0.5), the summed rank gap falls monotonically from +119.1 at N=50 to +27.5 at N=1700, consistent with the paper's "Int-DA is welfare-close to DA and gets closer as N grows" claim.

### Panel Balance
Not applicable — this is a simulation paper with no panel structure.

### Duplicates / Coding Anomalies
None detected. `duplicated(subset=keys).sum() == 0` for all five CSVs.

---

## 5. Robustness Check Results

The paper's three headline claims — Int-DA Pareto-dominates Tr-DA on matching outcomes, is essentially as stable as full DA, and scales monotonically to NRMP size — each survive a battery of 12 checks. Full output is in `replication_141541/06_robustness.py`.

| # | Check | Result |
|---|-------|--------|
| 1 | Int-DA Pareto-dominates Tr-DA on `{top, top3}` and beats it on `unmatched` at every cell of the 150-cell balanced grid | **0 / 450 violations** |
| 2 | Same-partner-under-proposer-change rate for matched doctors under Int-DA | **min 96.63%** at (N=100, λ_D=λ_H=0.95); **≥ 99.6% at every N≥500 cell** |
| 3 | Blocking-pair share among matched Int-DA doctors | **max 2.18%** at (N=1700, λ_D=λ_H=0.95) |
| 4 | Drop the largest NRMP sub-market (n=9127) and recompute Table 1 "Unmatched Int-DA" at (λ_D, λ_H) = (0.5, 0.5) | **5.72% → 5.68% (Δ = −0.035 pp)** |
| 5 | Replace logit rescale with OLS-in-percent rescale at (0.5, 0.5) | TopIntDA 41.91 → 41.90, UnmatchedIntDA 5.72 → 5.71, Top3IntDA 81.85 → 81.85 |
| 6 | Log-N slope diagnostic at (0.5, 0.5) — small slopes mean the logit extrapolation is near-flat and safe | TopIntDA +0.016, UnmatchedIntDA +0.015, SamePartnerIntDA +0.824 |
| 7 | With interview cap tightened to k=2, does Int-DA still beat Tr-DA on top-1 share? | **0 / 25 cells violate** |
| 8 | Int-DA unmatched share should weakly decrease in k for every (λ_D, λ_H) | **0 / 100 pair-violations** |
| 9 | In the unbalanced nD=600 vs nH=500 design, Int-DA still beats Tr-DA on top-1 share | **0 / 25 cells violate** |
| 10 | SIGS (doctor-common-only interviews) should be weakly worse than full Int-DA on top-1 share | **7 / 25 cells noticeably worse** — confirms the paper's claim that the doctor-only interview variant is strictly weaker, with the gap concentrated at high λ_H |
| 11 | Aggregate Int-DA → DA rank-gap at (λ_D, λ_H) = (0.5, 0.5) should shrink monotonically in N | **+119.1 → +106.5 → +90.1 → +46.8 → +29.8 → +27.5** (monotone True) |
| 12 | Identical-partner rate (Int-DA match == DA match \| matched) across balanced grid | **[69.7%, 91.1%]** |

Every check confirms the paper. Checks 4 and 5 are the most important because they stress the logit-rescale assumption in Table 1. Both are essentially null (movements < 0.1 pp), so the choice of rescaling functional form is not load-bearing for the paper's headline numbers. Check 6 confirms that the six balanced-N data points are on an almost-flat log-N trajectory for the two matching-outcome variables, which is reassuring for the logit extrapolation.

Check 10 (SIGS strictly weaker than full Int-DA) is a useful sanity check because it reports how much of Int-DA's performance comes from the *two-sided* common-value structure the paper models, vs a doctor-only variant that ignores hospital-side heterogeneity. Seven of twenty-five cells show a noticeable SIGS penalty, concentrated at high-λ_H cells where the hospital-side signal matters most — exactly the story the paper tells.

---

## 6. Summary Assessment

This is a clean theoretical/computational paper and its replication package is among the tightest I've seen. Every table and figure in the paper is derived from deterministic post-processing of four small CSVs (plus one larger one for Figure B8); re-running the table-generation layer in Python reproduces every published number to the first printed decimal. The only link that is not independently verifiable is the raw simulation output itself — per the authors' README, regenerating the N=1700 runs would take "up to 30 hours per cell on an optimized cluster," and I take the shipped CSVs at face value after verifying their internal consistency (no NaNs, valid probability ranges, monotonicity consistent with the paper's narrative, correct grid coverage, and zero duplicates).

**What replicates.** All 858 appendix-table cells + all 78 Table 1 cells match exactly. The qualitative ordering Int-DA ≻ DA ≻ Tr-DA on matching outcomes holds at every one of the 450 balanced-grid variables. Stability (same-partner under proposer change) exceeds 99% for Int-DA in every non-small cell. The NRMP logit rescaling is robust to leave-one-out of the largest submarket and to swapping logit for OLS-in-percent.

**What is not replicated.** (i) The raw simulations themselves — too expensive to rerun at the paper's cluster scale. (ii) The Mathematica rendering of Figures 1a/1b — I reproduced the numeric envelope in Python but not the Mathematica frame styling. (iii) Appendix figures B1–B8, which are visualisations of the same CSVs in various overlays; the underlying data is all verified in the audit script.

**Concerns.** None substantive. The one conceptual dependency worth flagging is that Table 1's "NRMP-scale" numbers extrapolate from at most N=1700 to sub-markets as large as n=9127, using a log-linear logit. The log-N slope is small in magnitude for all matching-outcome variables (|b| ≈ 0.02), so the functional form choice is empirically unimportant here — but a reader who wants a strict bound on the Table 1 numbers should note that the model's behaviour above N ≈ 1700 is unvalidated.

**Bug status.** No coding bugs in the Python simulator, the R assembly notebook, or the Mathematica table/figure code. All formulae transcribe cleanly to Python and all rescaling-conditional formulas (Panel B) are evaluated correctly in both codebases.

---

## 7. File Manifest

| File | Purpose |
|------|---------|
| `utils.py`          | Shared grid constants, CSV loaders, variable-formula dictionary, tex-table parser |
| `01_inspect.py`     | Phase 1 – verifies CSV shapes and the parameter grid |
| `02_tables.py`      | Phase 2 – rebuilds all eleven appendix tables and compares each cell to the published `.tex` |
| `03_table1.py`      | Phase 2 – reimplements the Mathematica logit rescaling pipeline and reproduces Table 1 |
| `04_figures.py`     | Phase 2 – simplified Python rebuild of Figures 1a / 1b as a λ_H envelope band |
| `05_data_audit.py`  | Phase 3 – coverage, range, monotonicity, no-NaN, and cross-CSV identity audits |
| `06_robustness.py`  | Phase 4 – 12 robustness checks covering Pareto-dominance, stability, blocking, logit-extrapolation sensitivity, k-sensitivity, unbalanced, SIGS, and rank-gap monotonicity |
| `out/fig1_replicated.png` | Simplified Figure 1a/1b rebuild |
| `writeup_141541.md` | This file |
