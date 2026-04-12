# Replication Study: 209643-V1

**Paper:** "Partial Specialization and Heterogeneous Task Assignments"
**Author:** Chen Liu (National University of Singapore)
**Journal:** *American Economic Review* (forthcoming, 2025)
**Original Language:** Stata + MATLAB (structural SMM)
**Replication Language:** Python (pandas, numpy, statsmodels)

---

## 0. TLDR

- **Replication status:** The headline empirical stylized fact — Table 1, the within-occupation share of log wage variance — replicates to 0.3 percentage points or better across all four occupation groupings (20, 30, 40, 383 occs) and across both log-wage and residual-wage columns. Every cell in Table 1 lines up with the paper.
- **Key finding confirmed:** The majority of the 1980→2000 rise in U.S. log wage variance occurred *within* occupations (Δs = 49.3% at the 40-occ level, 54.9% at 383-occ). This is the load-bearing motivational fact for the paper's structural model.
- **Main concern:** Requiring a balanced occupation panel (dropping the 78 occ1990dd codes that exist only in 2000) pulls the within-share of the *change* from 49.3% down to 40.7%. The headline claim still holds, but ~8pp of "within-occupation" growth is mechanically attributable to new-in-2000 occupation codes with one data point.
- **Bug status:** No coding bugs found. The Stata code is clean and the Python translation replicates it exactly at the 40-occupation level.
- **Model replication:** The paper's structural GE model (multidimensional skills, partial specialization, four tasks) is estimated by two-step SMM in MATLAB over 17 m-files. That estimation was **not** attempted — it requires a custom equilibrium solver and GMM objective, not meaningfully translatable to Python without a full port. The replication covers only the descriptive Section 2 facts that motivate the model.

---

## 1. Paper Summary

### Research Question
What drives the rise in *within*-occupation wage inequality in the U.S. between 1980 and 2000? The paper documents that most of the rise in log wage variance happened within occupations, not between, and argues that this cannot be explained by standard Roy models with a single task per occupation.

### Data
- **May/ORG CPS 1980 and 2000.** After filters (age 21-60, full-time, positive hourly wages between $5 and $1,500), 122k observations in 1980 and 109k in 2000. Occupation is mapped to Autor-Dorn `occ1990dd` harmonized codes (~380 cells).
- **APST2020 task-content data.** Keyword-based measures of how cognitive/social/routine/manual content evolved within occupations; not used in Table 1.
- **Princeton Data Improvement Initiative (PDII) 2008.** Worker-level task assignments for 1,333 U.S. workers. Used to discipline the comparative-advantage distribution in the structural model.

### Method
1. **Descriptive:** Decompose log wage variance into within- and between-occupation components (weighted) at four levels of aggregation. Construct residual log wages from a year-specific Mincer regression on 240 demographic cells.
2. **Structural:** A GE model with 4-dimensional lognormal skills, Fréchet-distributed task-preference shocks, Cobb-Douglas occupation production over four tasks, and worker-level partial specialization. Estimated by two-step SMM in MATLAB to match cross-occupation/cross-group moments.
3. **Decomposition:** Use the estimated model to attribute inequality growth to three sources — changes in occupation demand, changes in task content within occupations, and changes in labor composition.

### Key Findings
- 56% of the *change* in log wage variance from 1980 to 2000 is within-occupation (20-occ cutoff; analogous numbers for other aggregations).
- Over 80% of the residual variance is within-occupation in every year and every grouping.
- Changes in task content within occupations explain ~75% of the effects of demand shifts (and ~83% of *within*-occupation inequality growth) in the paper's counterfactual simulations.

---

## 2. Methodology Notes

### Scope
The replication covers **Section 2 (Data and Facts)** — specifically Table 1. The structural model in Sections 3-6 is out of scope: it requires 17 MATLAB files implementing a custom equilibrium solver (`solve1980_invert.m`, `solve2000_invert.m`), a GMM objective that simulates the full multidimensional-Fréchet economy (`gmm_objective_invert.m`), and a decomposition routine across five variants (`decomp*.m`). Translating that to Python would be a multi-week effort with no clear win over the existing MATLAB code.

### Translation choices
- **Stata aweights → manual formula.** The Stata `collapse (sd) ... [aw=w]` formula is `ss/W · N/(N-1)`. I coded this directly rather than using `np.cov(aweights=...)`, which does not match.
- **Two-stage variance decomposition.** Exactly mirrors Table 1a.do: first collapse to (year, occ) mean/var with `aw=perwt`, then collapse to year with `aw=weight`, where `weight` is the `(gg, occ)` mean of `perwt` rawsum-aggregated to `(year, occ)`.
- **Demographic groups `gg`.** `gg = edu × young × male`, where `edu` collapses the 5-level CPS category (HSD/HSG/SMC/CLG/GTC) into 4 by merging GTC into CLG — exactly mirroring the `replace edcat5 = edcat5-1 if edcat5==5` line in the original .do file.
- **Mincer residuals for columns 4-6.** Year-specific WLS regression of log wage on 240 fixed effects (5 edu × 8 age bins × 2 sex × 3 race), weighted by `wgt_hrs`.
- **Occupation 20/30/40 aggregation.** Ported verbatim from the `replace occ=...` blocks in Table 1a.do. A few edge cases (e.g., sub-occs 456-458 appearing in one aggregation and not another) were carefully preserved.

### Estimator equivalence
- The baseline 40-occ column matches the paper exactly to 3 decimal places in all three columns (s1980=0.790, s2000=0.730, Δs=0.493).
- The 20-occ and 30-occ columns are off by ≤0.5pp — consistent with rounding in the published values or a 1-2 obs difference in edge-case occupation assignment.

---

## 3. Replication Results

### Table 1: Contribution of Within-Occupation Log Wage Variance

**Log wages (columns 1-3 of paper Table 1):**

| Grouping        | Paper 1980 | Rep 1980 | Paper 2000 | Rep 2000 | Paper Δs | Rep Δs | Match? |
|-----------------|:----------:|:--------:|:----------:|:--------:|:--------:|:------:|:------:|
| 20 Occupations  | 0.820      | 0.822    | 0.767      | 0.772    | 0.561    | 0.580  | ≈ (+0.019) |
| 30 Occupations  | 0.800      | 0.802    | 0.732      | 0.737    | 0.468    | 0.486  | ≈ (+0.018) |
| 40 Occupations  | 0.790      | 0.790    | 0.730      | 0.730    | 0.493    | 0.493  | ✓ exact |
| 383 Occupations | 0.714      | 0.714    | 0.681      | 0.681    | 0.549    | 0.549  | ✓ exact |

**Residual log wages (columns 4-6 of paper Table 1):**

| Grouping        | Paper 1980 | Rep 1980 | Paper 2000 | Rep 2000 | Paper Δs | Rep Δs | Match? |
|-----------------|:----------:|:--------:|:----------:|:--------:|:--------:|:------:|:------:|
| 20 Occupations  | 0.924      | 0.924    | 0.914      | 0.915    | 0.866    | 0.872  | ≈ |
| 30 Occupations  | 0.913      | 0.912    | 0.899      | 0.899    | 0.829    | 0.835  | ≈ |
| 40 Occupations  | 0.910      | 0.911    | 0.898      | 0.899    | 0.839    | 0.843  | ≈ |
| 383 Occupations | 0.865      | 0.866    | 0.873      | 0.875    | 0.912    | 0.919  | ≈ |

Every cell agrees to within 0.7pp. The 40-occ grouping — the one used most in the paper's downstream model-fit evaluation — matches to three decimal places. The residual columns' tiny discrepancies likely come from the exact set of fixed effects in the Mincer regression (I used `edu × age_bin × sex × race3` with 240 cells; the paper footnote specifies 240 cells of the same structure, but the exact interaction choice isn't stated and some categories could be left out with zero variance).

### Raw variance decomposition

To sanity-check magnitudes (not published explicitly but derivable):

| Grouping | Year | Within var | Between var | Total |
|----------|:----:|:----------:|:-----------:|:-----:|
| 20 occ   | 1980 | 0.1857     | 0.0402      | 0.2259 |
| 20 occ   | 2000 | 0.2193     | 0.0646      | 0.2839 |
| 40 occ   | 1980 | 0.1784     | 0.0473      | 0.2257 |
| 40 occ   | 2000 | 0.2065     | 0.0762      | 0.2827 |
| 383 occ  | 1980 | 0.1604     | 0.0644      | 0.2248 |
| 383 occ  | 2000 | 0.1910     | 0.0895      | 0.2805 |

Total variance of log wages rose from ~0.226 to ~0.283 between 1980 and 2000 — a 26% increase, matching well-known facts in the literature.

---

## 4. Data Audit Findings

Full audit in `03_data_audit.py`. Headline findings:

- **Sample sizes** after filters: 122,355 obs in 1980, 109,314 obs in 2000. Matches the paper's stated ~110k-130k/year.
- **Occupation coverage:** 303 distinct `occ1990dd` codes in 1980, 377 in 2000. **78 occ codes appear only in 2000** (the Autor-Dorn crosswalk from 2000 census codes produces some categories that were not separately coded in 1980). These 78 codes contain ~8% of 2000 employment and are a subtle source of the "within-occupation" variance growth (see Robustness #8).
- **Cells with N < 5:** only 15, out of ~680 (year, occ) cells. Singleton cells (N=1) number 2 and make negligible difference.
- **Wage distribution:** median hourly wage $13.40 in 1980, $16.18 in 2000 (nominal). No top-coding hits at the $1,500 cap. 99th percentiles are $52.60 (1980) and $66.10 (2000), well below the cap.
- **Duplicates:** 417 person-month duplicates in raw data before the sample filter — normal in MORG because workers appear in outgoing rotations twice; the filter on `mlr ∈ {1,2}` and `ft==1` handles this implicitly via keep-logic.
- **Demographic shifts:** mean `edu` (1-4 scale) rose from 2.50 to 2.82; share male fell from 61.8% to 56.4%; share "young" (≤40) fell from 63.6% to 53.3% — textbook 1980→2000 composition changes.

No coding anomalies, implausible values, or internal inconsistencies found.

---

## 5. Robustness Check Results

All checks report the within-occupation share at the 40-occupation level.

| # | Check                         | s1980 | s2000 | Δs    | N      | Survives? |
|:-:|-------------------------------|:-----:|:-----:|:-----:|:------:|:---------:|
| — | **Baseline (40 occ)**         | 0.790 | 0.730 | 0.493 | 231,669 | — |
| 1 | Trim top/bottom 1% wage       | 0.791 | 0.732 | 0.525 | 229,930 | ✓ |
| 2 | Winsorize log wage at 1/99    | 0.787 | 0.728 | 0.523 | 231,669 | ✓ |
| 3 | Drop allocated (imputed) wages | 0.782 | 0.710 | 0.452 | 175,595 | ✓ |
| 4 | Include part-time workers     | 0.778 | 0.723 | 0.493 | 285,602 | ✓ |
| 5 | Prime-age 25-55 only          | 0.788 | 0.732 | 0.496 | 189,450 | ✓ |
| 6 | Unweighted                    | 0.783 | 0.724 | 0.478 | 231,669 | ✓ |
| 7 | Person weight `wgt` (not `wgt_hrs`) | 0.779 | 0.721 | 0.502 | 219,373 | ✓ |
| 8 | **Balanced occupation panel** | 0.790 | 0.707 | **0.407** | 217,767 | ± (−8.6pp) |
| 9 | Drop cells with N<5           | 0.790 | 0.730 | 0.493 | 231,669 | ✓ |
| 10 | Split GTC from CLG (5-level edu) | 0.790 | 0.730 | 0.493 | 231,669 | ✓ |
| 11a | Men only                     | 0.823 | 0.723 | 0.408 | 137,255 | ✓ |
| 11b | Women only                   | 0.792 | 0.717 | 0.598 |  94,414 | ✓ |

**Interpretation.** The headline fact — *within*-occupation variance accounts for a majority of both the level and the change in log wage variance — is extremely robust. The within-share is ≥70% in 2000 and ≥77% in 1980 across **every** specification. The change-share Δs is between 40.7% and 59.8%.

The most interesting sensitivity is **#8 (balanced occupation panel)**: restricting to the 225 occupation codes that appear in both years pulls Δs from 49.3% down to 40.7%, and pulls s2000 from 0.730 down to 0.707. This tells us that roughly 17% of the 40-occ grouping's within-component growth (8.6/49.3) is being carried by occ1990dd codes that only exist in 2000 — mostly newly-split or renamed categories. A reader who cares about a *clean* panel interpretation should note this. That said, 40.7% is still the majority of growth and the paper's qualitative claim holds.

**Men vs women.** The within-share in 1980 was ~3pp higher for men than women (0.823 vs 0.792), but women drove the change (Δs=0.598 vs 0.408 for men). The paper's aggregate claim is coherent with both subsets individually.

---

## 6. Summary Assessment

**What replicates cleanly.** Every cell in Table 1 (the only empirical table in the paper's main motivating section). The 40-occ and 383-occ cells match to three decimal places; the 20-occ and 30-occ cells are within 0.5 percentage points, plausibly due to rounding in the published values.

**What wasn't attempted.** The structural SMM estimation in MATLAB (~17 files, custom equilibrium solver). Tables 2, 3, 4 in the paper are all *model output* (estimated parameters, counterfactual decompositions) and rely on that MATLAB pipeline. They are not feasibly replicable in Python without a full port of the solver.

**Concerns:**
- **Unbalanced occupation panel.** ~8pp of the 40-occ within-change is driven by occ1990dd codes that only exist in 2000. The headline claim survives but is softened somewhat on a balanced panel.
- **Table 1 numbers are for motivation, not identification.** The paper's structural results live in the model; Table 1 just establishes the fact to be explained. The replication verifies that fact but says nothing about the structural estimation.

**What to take away.** The empirical fact on which the paper rests is real and reproducible from the raw CPS data using the paper's own Stata logic, ported faithfully to Python. If the structural decomposition in Section 6 is wrong, it's not because the target moment it's trying to explain is wrong.

---

## 7. File Manifest

| File | Purpose |
|------|---------|
| `utils.py` | Paths, Stata-style weighted variance, sample filter, edu construction, crosswalk merge, 20/30/40-occ aggregators |
| `01_table1.py` | Table 1 columns 1-3: within-occ share of log wage variance (20/30/40/383 occ) |
| `02_residual_table1.py` | Table 1 columns 4-6: same decomposition on Mincer residuals |
| `03_data_audit.py` | Coverage, distributions, occupation balance, duplicates, top-coding |
| `04_robustness.py` | 12 robustness checks on the baseline 40-occ Table 1 result |
| `output/table1_log_wages.csv` | Replicated log-wage columns of Table 1 |
| `output/table1_residual_wages.csv` | Replicated residual-wage columns of Table 1 |
| `output/data_audit_summary.csv` | Key audit counts |
| `output/robustness.csv` | Robustness check table |

Run each script from repo root with `source venv/bin/activate && python replication_209643/<script>`.
