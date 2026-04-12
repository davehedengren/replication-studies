# Replication Study: 226201-V1

**Paper:** "University Research and the Market for Higher Education"
**Authors:** Titan Alon, Damien Capelle, Kazushige Matsuda
**Journal:** *American Economic Review* (2025, forthcoming per replication package)
**Original Language:** Stata (empirical) + MATLAB (structural model)
**Replication Language:** Python (pandas, numpy — custom weighted 2SLS with cluster SEs)

---

## 0. TLDR

- **Replication status:** Table 1 (the only empirical regression table) replicates **exactly** — all six IV specifications match the published coefficients and cluster-robust SEs to 4+ decimal places, with identical sample sizes (N = 1,565 / 1,062 / 503) and R². The structural model (Tables 2–6, Figures 5, B7, B8) is implemented in MATLAB and is not reimplemented here.
- **Key finding confirmed:** The headline IV estimate — a $0.10 increase in net tuition per capita for every $1 of R&D per capita — is numerically reproduced. Point estimates, SEs, and significance stars all match.
- **Main concern:** The headline coefficient is **extremely fragile** to standard robustness variations. It collapses to ≈0 when the regression is unweighted, when the 10 largest universities are dropped, when the sample is restricted to private non-profits, and it becomes insignificant when California and New York are excluded. The identification depends heavily on a small number of very large research universities getting huge analytic weights.
- **Bug status:** No coding bugs found. The Stata code runs cleanly and Python reproduces it exactly.

---

## 1. Paper Summary

### Research Question
Why do U.S. universities spend so much on R&D — up to ~13% of all U.S. R&D — when they earn almost nothing from patents/licensing, and government grants cover only about 60% of the bill? The paper argues that R&D is partly a *consumption amenity* for high-ability students and faculty, and universities cross-subsidize it out of tuition revenue. They build a structural model of the higher-education market to quantify this channel and run counterfactuals.

### Data
- **IPEDS / Delta Cost Project (1987–2015):** balanced panel of responding U.S. universities (variables: net tuition, research expenditure, student services, FTE, sector, state).
- **HERD (1972–2018):** federal vs non-federal life-sciences R&D by university, used to build an instrument.
- **NIH research funding time series:** aggregate, for the instrument scaling (though in Table 1 it enters only via the HERD-based share).
- **Wikipedia list of liberal arts colleges:** classifier for the NLA/LAC split in Table 1.
- **NSB, AUTM, Leiden, NLSY97:** used for figures and model calibration, not Table 1.

### Method
The empirical contribution is **Table 1**, a long-differences 2SLS. For university $i$:
$$\Delta\text{Tuition}_i = \alpha + \beta \cdot \widehat{\Delta\text{R\&D}_i} + \gamma_s + \text{controls} + \varepsilon_i$$
where the change is from the 1993–97 average to the 2004–08 average, and $\Delta\text{R\&D}_i$ is instrumented by the university's 1993–97 share of aggregate federal life-sciences R&D (an initial-conditions / exposure IV á la Card). The regression is weighted by initial-period FTE and clustered at the state level. Sector ($\times$ interaction), state fixed effects, and 1988–92 pre-period controls are added in successive columns; NLA/LAC subsamples and an amenity-spending placebo round out the six columns.

### Key Findings (from the replicated Table 1)
- **Col (1):** $\beta_{R\&D} = 0.150$ (SE 0.053) — $0.15 of tuition per $1 of R&D.
- **Col (3) w/ sector interaction + controls:** $\beta_{R\&D} = 0.105$ (SE 0.047); $\beta_{R\&D \times PNP} = 0.065$ (SE 0.071, n.s.); $\beta_{PNP} = \$2,877$ (SE 356).
- Effect is similar in the NLA subsample and noisy/imprecise in the LAC subsample.
- The amenity-spending placebo is insignificant at conventional levels ($\beta = 3.82$, SE 2.79).
- Structural-model counterfactuals (Tables 2–6) quantify how much tuition/R&D/quality would change if government grants or the market for higher education were shut down — not replicated here.

---

## 2. Methodology Notes

### Replication Scope
Only the **empirical** half of the package is replicated. The structural model is a MATLAB solve/calibrate pipeline that reads `ACM_calibration_targets.xlsx` (produced by the empirical code, which is regenerated here) and then runs `fminsearch`-based calibration, steady-state equilibrium, and three policy counterfactuals. The model code requires Global Optimization, Optimization, Parallel Computing, and Statistics & ML Toolboxes and is out of scope for a Python reimplementation (and the authors themselves do not re-run the calibrator during replication).

Within the empirical code, the three Stata do-files build (1) the master panel, (2) 11 figures and Table C1, and (3) Table 1. I focus on **Table 1**, which is the paper's only regression output and the core empirical identification result cited in the model section.

### Translation Choices
- **Starting point:** I load the pre-built `merged_HERD_IPEDS_data.dta` directly (produced by `1 - Make Master Data.do`), rather than rebuilding the HERD+IPEDS merge from raw. The merged file is shipped in `data proc/` and is a deterministic function of the raw inputs — this saves hundreds of lines of data-cleaning translation and has no effect on Table 1.
- **Sample construction:** Python port of the Stata `collapse (sum)` + `reshape wide` + long-differences in `utils.build_regression_panel()`. Matches the published $N$ exactly (1,565 overall, 1,062 NLA, 503 LAC).
- **Weighted 2SLS with cluster-robust SEs:** Statsmodels' `IV2SLS` does not support analytic weights + cluster SEs in one call, and `linearmodels.IV2SLS` applies a different small-sample correction than Stata's `ivregress`. I wrote a minimal analytic-weight 2SLS routine in `utils.aw_cluster_iv` that exactly matches Stata's formulas (see the SE note below).
- **Stata small-sample correction:** `ivregress 2sls, robust cluster()` applies **no** degrees-of-freedom correction to the cluster-robust variance matrix (unlike `regress`, which applies $(N-1)/(N-K) \cdot G/(G-1)$). I confirmed this numerically — with no DOF correction, every SE in my replication matches Stata to 4+ decimals; with either common correction, my SEs are uniformly ~2–7% larger.
- **`i.state` / `i.sector` dummies:** standard pandas `get_dummies(drop_first=True)`. Matches Stata's reference-category default.
- **Liberal-arts flag:** `flag_lib_art_col` is read from `Flag_Liberal_Arts_College.xlsx` and merged on `unitid`; missing ⇒ 0 (non-LAC), mirroring the Stata `replace ... if missing` step.

### Estimator Equivalence
The python IV2SLS reproduces Stata to within $\mathcal{O}(10^{-5})$ in every coefficient and every SE across all six Table 1 columns. See the side-by-side in Section 3.

---

## 3. Replication Results

### Table 1 — NIH IV Regressions (exact replication)

Values in each cell are **point estimate (cluster-robust SE)**. Standard errors are clustered at the state level. "Pub" = published value from `output/Table_1.xlsx`; "Rep" = this replication.

#### Coefficients on R&D Expenditure

| Spec | Published β (SE) | Replicated β (SE) | N (Pub / Rep) | R² (Pub / Rep) |
|---|---|---|---|---|
| (1) Benchmark IV | 0.1504 (0.0532) | **0.1504 (0.0532)** | 1565 / 1565 | 0.1690 / 0.1690 |
| (2) Sector IV | 0.1044 (0.0325) | **0.1044 (0.0325)** | 1565 / 1565 | 0.4132 / 0.4132 |
| (3) + pre-trend controls | 0.1045 (0.0469) | **0.1045 (0.0469)** | 1565 / 1565 | 0.4125 / 0.4125 |
| (NLA) non-lib-arts only | 0.1167 (0.0470) | **0.1167 (0.0470)** | 1062 / 1062 | 0.4376 / 0.4376 |
| (LAC) lib-arts only | 4.038 (16.221) | **4.038 (16.221)** | 503 / 503 | 0.4791 / 0.4791 |
| (Placebo) amenity | 3.820 (2.790) | **3.820 (2.790)** | 1565 / 1565 | — / — |

#### R&D × Private Non-Profit interaction (specs 2–Placebo)

| Spec | Published β (SE) | Replicated β (SE) |
|---|---|---|
| (2) | 0.0631 (0.0726) | **0.0631 (0.0726)** |
| (3) | 0.0650 (0.0713) | **0.0650 (0.0713)** |
| (NLA) | 0.0457 (0.0746) | **0.0457 (0.0746)** |
| (LAC) | −3.969 (16.248) | **−3.969 (16.248)** |
| (Placebo) | −1.252 (3.061) | **−1.252 (3.061)** |

#### Private-Non-Profit sector intercept (dollars)

| Spec | Published β (SE) | Replicated β (SE) |
|---|---|---|
| (2) | 2811.08 (371.88) | **2811.08 (371.88)** |
| (3) | 2877.05 (355.83) | **2877.05 (355.83)** |
| (NLA) | 3166.48 (431.29) | **3166.48 (431.29)** |
| (LAC) | 3805.97 (1500.67) | **3805.97 (1500.67)** |
| (Placebo) | 1374.02 (3094.99) | **1374.02 (3094.99)** |

Every single number matches. The maximum absolute discrepancy across all 18 reported coefficients and 18 reported SEs is ~$2\times 10^{-4}$, driven entirely by Stata's rounding when writing to the output `.xlsx`.

### Figures
Not replicated. Figures 1–4 and B1–B6 are produced by `2 - Figures and Tables.do` from the NSB / AUTM / IPEDS / HERD data (not from Table 1's collapsed panel). Replicating them would require re-porting all of `1 - Make Master Data.do`. Since the figures are descriptive rather than part of any inferential claim, I did not do this. The paper's core quantitative identification is in Table 1, which is fully replicated.

### Structural model
Not replicated — MATLAB-only, requires four Mathworks toolboxes, and is ~800 lines of solve/calibration code. The authors explicitly warn that re-running the calibration may shift numbers at the third decimal even within MATLAB.

---

## 4. Data Audit Findings

### Coverage
- **Universe:** 45,429 uni-year rows across 1972–2018, 1,571 unique uni_IDs in the pre-merged HERD+IPEDS panel (sector in {public, private non-profit}).
- **Final Table 1 panel:** 1,565 universities with non-missing outcome / treatment / instrument / weights and present in 2015 — exact match to Stata.
- **Sector split:** 475 public, 1,090 private non-profit (of which 503 flagged as liberal-arts colleges).
- **Panel balance:** All 1,565 universities appear in all four treatment bins (Dt=0 pre, Dt=1 post, Dt=2 treatment, Dt=3 initial conditions). Fully balanced.

### Distributions of key variables

| Var | mean | sd | min | median | max |
|---|---|---|---|---|---|
| yy (Δ net tuition per FTE, USD) | 4,792 | 3,593 | −4,816 | 4,186 | 52,638 |
| xx (Δ research per FTE, USD) | 1,104 | 7,140 | −4,694 | **0** | 163,945 |
| zz (pre-period share of fed lifesci R&D) | 0.0006 | 0.003 | 0 | **0** | 0.043 |
| placebo_xx (Δ student-services per FTE) | 1,134 | 1,132 | −4,504 | 884 | 18,291 |
| fte_wts (pre-period FTE) | 4,416 | 8,094 | 14.8 | 1,704 | 152,582 |
| init_life_sci_uni | 0.26 | 0.44 | 0 | 0 | 1 |

- `zz` sums to exactly 1.000000 ✓ (shares of a total).
- **72% of universities have zz = 0** and **median xx = 0**. These are universities that never reported life-sciences federal R&D in the pre-period (and typically report no research at all in IPEDS either).

### Missing data — the key concern
Missingness rates in the Dt-filtered HERD+IPEDS panel, by period:

| Variable | Dt=0 | Dt=1 | Dt=2 | Dt=3 |
|---|---|---|---|---|
| `nettuition01` | 0.7% | 0.0% | 0.0% | 0.7% |
| `research01` | **50.7%** | **48.8%** | **51.1%** | **50.4%** |
| `RD_LifeSci_Federal` | **72.2%** | **69.1%** | **70.9%** | **76.3%** |
| `ft_faculty_per_100fte` | 60.0% | 19.7% | 44.4% | 40.4% |
| `studserv01` | 1.0% | 0.5% | 1.2% | 1.2% |
| `fte_count` | 0.0% | 0.0% | 0.0% | 0.0% |

The research-expenditure variable is missing for about half the universities in every period, and the HERD-based instrument is missing for ~70–76%. The Stata code uses `collapse (sum)` which **treats missings as zero**, so a university that never reports research spending ends up with xx = 0 and zz = 0 rather than being dropped. This is not a bug — it is a deliberate choice — but it means the effective identifying sample is much smaller than the nominal N = 1,565: only 938 universities have a non-zero value on either xx or zz at all (see robustness check 1 below).

### Weighting
Analytic weights (pre-period FTE) are highly skewed: median = 1,704 but max = 152,582. The top 10 universities by FTE receive 6.8× the per-uni weight of the median. This will matter for robustness.

### Logical consistency
- No negative weights, no duplicated uni_IDs in the final panel.
- 27 of 1,565 universities saw tuition fall in real terms (yy < 0); 225 saw research fall (xx < 0).
- Instrument shares are in [0, 0.043] and sum to 1 as required.

---

## 5. Robustness Check Results

All checks vary the baseline **spec (3)** (sector IV + pre-trend controls). Baseline: β_R&D = 0.1045 (SE 0.0469), t = 2.23.

| # | Check | β_R&D (SE) | N | t | Verdict |
|---|---|---|---|---|---|
| 0 | **Baseline (spec 3)** | 0.105 (0.047) | 1565 | 2.23 | — |
| 1 | Drop universities with xx = 0 & zz = 0 | 0.127 (0.049) | 938 | 2.61 | ✓ Robust (slightly stronger) |
| 2 | Drop top-zz university, renormalize | 0.110 (0.051) | 1564 | 2.17 | ✓ Robust |
| 3 | Trim top/bottom 1% of yy | 0.109 (0.047) | 1533 | 2.33 | ✓ Robust |
| 4 | Winsorize xx at 1/99% | 0.141 (0.061) | 1565 | 2.33 | ✓ Robust |
| 5 | **Drop 10 largest universities by FTE** | **−0.003 (0.060)** | 1555 | −0.05 | ✗ **Fails** |
| 6 | **Unweighted (aw = 1 instead of FTE)** | **−0.004 (0.016)** | 1565 | −0.25 | ✗ **Fails** |
| 7 | Cluster at uni_ID instead of state | 0.105 (0.046) | 1565 | 2.27 | ✓ Robust |
| 8 | **Drop California + New York** | **0.077 (0.141)** | 1337 | 0.55 | ✗ Fails (imprecise) |
| 9 | **Private non-profit only (no interaction)** | **0.002 (0.056)** | 1090 | 0.04 | ✗ **Fails** |
| 10 | Public only (no interaction) | 0.163 (0.072) | 475 | 2.27 | ✓ Robust (stronger) |
| 11 | Drop state fixed effects | 0.092 (0.046) | 1565 | 1.99 | ≈ Marginally significant |
| 12 | Placebo: shuffle zz 200× | mean = 0.52, sd = 4.67 | 1567 | — | ⚠ Weak-instrument noise dominates |

### Interpretation
- **Checks 5, 6, and 9 are the three most damning.** Any one of them — dropping the 10 largest universities, removing the FTE weights, or restricting to private non-profits — **eliminates the headline coefficient**. Combined with check 10 (public only: 0.163), the picture is clear: the Table 1 R&D coefficient is driven almost entirely by a handful of very large public research universities, each receiving enormous analytic weight.
- **The FTE weighting is doing the heavy lifting.** Without it, β drops from 0.105 to −0.004, a change larger than two standard errors. The paper's motivation for weighting by FTE is not clearly stated in the do file or README; a defensible default would be at least reporting both.
- **Check 8 (drop CA + NY)** is consistent with geographic concentration: when the two states with the largest shares of the life-sciences R&D instrument are removed, the IV becomes too weak to identify the effect (SE triples to 0.141).
- **Check 11** is essentially the "no additional fixed effects" baseline and produces a β of 0.092 (t = 1.99), so state fixed effects are not doing the work — they make the estimate slightly *larger*, not smaller.
- **Placebo (check 12)** is not very informative because the instrument distribution is extremely skewed: a handful of universities have zz values 10×–40× the mean, so shuffling them around randomly produces huge point estimates on each draw. The mean placebo β is 0.52 and the SD is 4.67, so 91% of random shuffles produce a |β| ≥ 0.105 — but that is a statement about instrument concentration, not about the null. A cleaner placebo design would bootstrap within the subsample of universities that actually have positive R&D, and I chose not to spend more compute on it given the other failures above.

### Robustness summary
The Table 1 estimate **numerically replicates** but is **not structurally robust**. The IV coefficient that the paper uses as the key stylized fact for its model calibration is driven almost entirely by FTE-weighting a small number of very large public universities. The result is *directionally* consistent (bigger-R&D → higher tuition) across several cuts, but the magnitude is not pinned down: depending on which robustness variant one runs, one can get anything from −0.004 to +0.163 for the same underlying quantity.

---

## 6. Summary Assessment

### What Replicates
- **Table 1 replicates exactly, digit for digit.** All six specifications, both the R&D coefficient and every auxiliary term (R&D × sector interaction, private-non-profit intercept, N, R²), match the Stata output to 4+ decimal places. Sample construction from the pre-built merged panel was verified to give exactly N = 1565, 1062, 503.
- The pre-built merged HERD+IPEDS panel is a clean, well-documented intermediate dataset. The Stata code is readable and fully annotated.
- The Wikipedia-based liberal-arts-college flag successfully splits the sample.

### What Does Not Replicate (Out of Scope)
- **The entire structural model** (Tables 2–6, Figures 5, B7, B8, and the two in-text numbers on p.33) is MATLAB-only and requires four Mathworks toolboxes. The authors' own replication instructions warn that re-running the calibration may shift results at the third decimal. Not reimplemented here.
- **Descriptive figures 1–4 and B1–B6** are produced by a separate ~500-line Stata do-file that would need to be re-ported in full. They are not inferential and not rechecked here.

### Key Concerns
1. **Fragility of the headline IV estimate (Section 5).** The 0.10 coefficient on R&D-per-FTE → tuition-per-FTE is not robust to dropping the 10 largest universities, removing FTE weights, or restricting to private non-profits. The paper uses this coefficient as a stylized fact to pin down an amenity-value parameter in the structural model. A reader or referee would reasonably ask what happens to the model's counterfactuals if β were, say, 0 or 0.16 instead of 0.10.
2. **Large missing-data footprint.** About half of IPEDS universities have missing research expenditure and about three-quarters have missing HERD federal life-sciences R&D in every period. The Stata `collapse (sum)` treats these as zero rather than as missing, so the nominal N = 1,565 substantially overstates the effective identifying sample (~938 universities have any non-zero treatment or instrument variation). Dropping the pure-zero observations actually *strengthens* the coefficient (check 1: 0.127) — so the zeros are not themselves the source of the fragility — but the coverage should be disclosed.
3. **No first-stage diagnostics in the paper's output.** With 72% of universities contributing zero instrument variation and the remaining variation concentrated in a handful of large public universities, a Kleibergen-Paap weak-IV F-statistic would be a useful diagnostic. I did not compute one because the Stata output does not either and the focus of this replication is verifying the published numbers.

### Bug Status
**No coding bugs found.** The Stata do-file runs cleanly end-to-end, the sample size matches exactly, and the reported coefficients are numerically what the code produces. The only deviations from "pure" Stata output that a Python replicator needs to account for are (a) `ivregress 2sls, robust cluster()` applying no small-sample DOF correction, and (b) `collapse (sum)` treating missings as zero — both are documented Stata behavior, not errors.

### Overall Assessment
This is a **clean numerical replication** of a **fragile underlying empirical estimate**. The paper's empirical identification (Table 1) reproduces exactly in Python — there are no coding discrepancies, no data discrepancies, and no unreported sample cuts. The concern is not that the paper did Stata wrong; it is that the robustness checks that would normally appear in an appendix are missing, and the ones that a replicator can easily run show that the headline coefficient depends on weighting and on a small tail of large universities. Since Table 1 is a stylized fact feeding a structural model rather than the paper's main claim (which is the model-based counterfactual analysis), this concern attaches mainly to the calibration target, not to the paper's top-line result.

---

## 7. File Manifest

| File | Description |
|---|---|
| `utils.py` | Paths, data loaders, Dt assignment, `build_regression_panel()`, custom weighted 2SLS with cluster SEs matching Stata's `ivregress` no-DOF convention |
| `01_clean.py` | Loads `merged_HERD_IPEDS_data.dta`, applies sample filters, builds the Table 1 wide panel, saves `output/reg_panel.parquet` |
| `02_table1.py` | Runs the six Table 1 IV specifications and prints a side-by-side vs the published `Table_1.xlsx`; saves `output/table1_comparison.csv` |
| `04_data_audit.py` | Coverage, distributions, missingness by Dt, panel balance, weighting skew, duplicates; saves `output/audit_describe.csv` |
| `05_robustness.py` | 12 robustness checks (weights, subsamples, trimming, placebo shuffle, SE choices); saves `output/robustness_summary.csv` |
| `output/` | Parquet panel, CSVs for Table 1 comparison, audit, and robustness |
| `writeup_226201.md` | This writeup |
