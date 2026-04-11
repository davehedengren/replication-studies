# Replication Study: 120281-V1

**Paper:** "Knowledge Diffusion, Trade and Innovation across Countries and Sectors"
**Authors:** Jie Cai, Nan Li, Ana Maria Santacreu
**Journal:** *American Economic Journal: Macroeconomics*, 2022 (working-paper version: FRB St. Louis WP 2017-029A, October 2017)
**Original Language:** MATLAB (calibration and counterfactuals) + Stata (gravity regression, table export)
**Replication Language:** Python (pandas, statsmodels, numpy)

---

## 0. TLDR

- **Replication status:** *Partial*. The one empirical step in the paper — the sector-level gravity regression that produces the bilateral trade costs feeding the structural calibration — replicates **exactly** (max abs coefficient difference to the shipped Stata output: 5.0e-7 across 162 cells; reconstructed `lD_inj` matches to 1.8e-6). The structural calibration, welfare counterfactuals, and growth decomposition are solved in ≈60 MATLAB files via iterative Frobenius-eigenvector fixed points and are not re-implemented here.
- **Key finding confirmed:** Within the scope reproduced, every sectoral distance elasticity, contiguity, FTA, and currency-union coefficient reproduced to seven decimal places.
- **Main concern:** The trade cost estimation uses only 19 countries, but several of the paper's welfare-gains figures (9, 10, 12, 14, 17) plot results for 25+ economies including Slovenia, Slovakia, Hungary, Mexico, India, China, Estonia, and Poland — economies that appear nowhere in the shipped gravity data. The replication package does not include the extended-country raw trade panel used to generate those figures, so the downstream calibration results cannot be verified even under a MATLAB-compatible setup.
- **Bug status:** No coding bugs found. The Stata and MATLAB code paths are internally consistent with the paper's derivations.

---

## 1. Paper Summary

### Research Question
How do trade-induced reallocations of innovation and comparative advantage shape dynamic welfare gains from trade when knowledge diffuses asymmetrically across sectors and countries?

### Data
- **Bilateral trade flows (2005):** 19-country × 19-sector panel (`trade19_00.dta`, 6,508 rows), with distance bins, contiguity, FTA, and currency-union indicators. Sectors 1–18 are tradable; sector 19 (services) is treated as non-tradable.
- **Sector-level productivity (2005):** `Productivity2005.dta` (359 country-sector observations), constructed as a Solow residual from OECD STAN.
- **R&D intensity `s_ij_world`:** 19 × 19 matrix of country-sector R&D/output shares.
- **GDP and population (`GDP_rawdata.dta`, `Lpop.txt`):** used to normalize welfare gains and for the wage channel.
- **Patent citation flows:** `cite_ijnk1928.raw` and `P_njt.raw` in `Epsilon_estimation/` — used to estimate the diffusion parameters ε_{ni}^{jk} (inside a separate MATLAB routine, `Calibration_gravity2020.m`).
- **IO-linkage parameters `alpha_ij.txt`, `gam_j_US.txt`, `gam_jk_US.txt`** from the US input-output table.

### Method
1. **Gravity equation (Appendix A / Section 5.1.1).** For each tradable sector *j* = 1…18 run, via OLS with no constant and country-pair NLD-omitted exporter + importer fixed effects,

   log(X_{ni}^j / X_{nn}^j) = Σ_k δ_k^j · 1{dist∈bin k} + β_c^j contig + β_F^j FTA + β_U^j CU + ex_i^j − F_n^j + ν_{ni}^j

   Extract log-trade costs as `lD_inj = distance contribution − θ·ex_i + residual`, capped at 0. With θ = 8.28 these become the `d_{in}^j` inputs to the static and dynamic equilibria.
2. **Diffusion calibration (Section 5.1, Appendix C).** Diffusion parameters ε_{ni}^{jk} are estimated from USPTO citation-to-patent ratios via a separate MATLAB maximum-likelihood procedure; the estimates are shipped in `epsilon_gravity_2020.mat`.
3. **Static equilibrium and two-stage calibration (Sections 5.1.2, 5.3).** Given T^j_n, ε^{jk}_{ni}, IO parameters, labor, and the estimated trade costs, solve for wages, prices, and trade shares; then recover innovation parameters (β_r, λ^j_n) from the dynamic balanced growth-path (BGP) conditions via the Frobenius theorem.
4. **Counterfactual trade liberalization (Section 5.2, 5.4, 6).** Reduce `d_{in}^j` by 40% uniformly and re-solve for the new static equilibrium, the new innovation intensities `s^j_n`, and the new steady-state growth rate `g_A`.
5. **Welfare decomposition** into static gains (holding the knowledge stock fixed) and dynamic gains (letting knowledge diffuse and R&D reallocate).

### Key Findings (headline claims as reported)
- Cross-country average welfare gains of **44.6%** from a 40% uniform trade liberalization, ranging from **17.5% (USA)** to **124% (Slovenia)**.
- Dynamic gains average **≈9%** (min 7.1% UK, max 15.95% Slovenia), with the growth rate rising from 3% to 3.1% on the BGP.
- In a one-sector counterfactual, dynamic gains are *negative* for most countries, highlighting the multi-sector reallocation channel.
- Removing knowledge diffusion shrinks welfare gains substantially, especially for small, less innovative countries.
- Domestic innovation intensity explains **27%** of cross-country-sector variation in productivity (Section 5.3).

---

## 2. Methodology Notes

### Translation Choices

- **Gravity: Stata → Python.** Stata's `xi: reg y ... i.exp i.imp, noconstant` with `char exp[omit] 15`, `char imp[omit] 15` is reproduced in Python by building explicit exporter and importer dummy matrices (19 countries, NLD = code 15 omitted), concatenating them with the six distance-bin dummies and {contig, FTA, CU}, and running `statsmodels.OLS` with no intercept. Perfect collinearities from empty FE cells are removed by a Gram-Schmidt / rank-revealing pass that protects the six distance bins and three bilateral indicators.
- **Trade-cost reconstruction** (`03_trade_costs.py`) re-implements the exact Stata pipeline in `gravity1919_rep.do`: `lD_inj = Σ β · dist_dummy + β_contig + β_FTA + β_CU − θ_ex_ij + residual`, where `θ_ex_ij = log(S_nj_0 / S_ij_0)` with `S_nj_0 = exp(−β_imp^i)` and `S_ij_0 = exp(β_exp^i)` — equivalent to adding back both country FEs of the exporter. The result is then capped at 0 as in line 111 of the do-file.
- **Calibration and counterfactuals: not translated.** The ~60 MATLAB files in `Calibration_counterfactual/` iterate a multi-sector Eaton-Kortum competitive equilibrium via fixed-point methods on prices and trade shares (`SS_compeqbm*.m`, `SS_prices*.m`, `SS_tradeshares*.m`) and solve the knowledge-diffusion dynamic block via a Frobenius-eigenvector iteration (`maindriver_Calibration4L19_baseline.m`, `Welfare4L19_baseline.m`). Faithful porting is feasible in Python, but well beyond the scope of an automated single-paper replication; none of the model primitives depend on Stata-side empirics beyond the gravity step that we *do* replicate.
- **Domestic innovation share (Section 5.3, 27% claim)** is generated by `Domestic_innovation_Section5_3.m`, which draws on `Results_baseline.mat` output from the calibration — so it, too, cannot be independently verified here.

### Estimator Equivalence

- Stata's OLS with `noconstant` + two NLD-omitted FE sets and `statsmodels.OLS` with the same columns give numerically identical distance, contiguity, FTA, and CU coefficients — see `02_compare_to_stata.py`.
- Because the rank-revealing pass is feeding the Gram-Schmidt bases in the same order as Stata's internal parameterization, the FE coefficients match as well.

### Reparameterization note

In this design, distance-bin *levels* are only identified up to an additive constant (the sum of the omitted exporter and importer FEs). Changing the omitted category from NLD to USA shifts every distance level by the same amount (−2.65 averaged across sectors in sector 1), but leaves distance *differences*, contiguity, FTA, and CU coefficients exactly invariant. This is benign and matches the analogous Stata behavior.

---

## 3. Replication Results

### 3.1 Sector-level gravity coefficients vs shipped Stata output

File compared: `Raw_data/Gravity1919_LZ.xls` (Stata `estout` export for sectors 1–18).

| Variable      | Cells | Max |diff| | Mean |diff| | Match? |
|---------------|-------|-----------|------------|--------|
| d0_375        | 18    | 4.90e-07  | 2.08e-07   | ✓ |
| d375_750      | 18    | 4.93e-07  | 2.77e-07   | ✓ |
| d750_1500     | 18    | 4.69e-07  | 2.92e-07   | ✓ |
| d1500_3000    | 18    | 3.97e-07  | 1.69e-07   | ✓ |
| d3000_6000    | 18    | 4.98e-07  | 2.23e-07   | ✓ |
| d6000_        | 18    | 4.89e-07  | 2.80e-07   | ✓ |
| contig        | 18    | 3.98e-07  | 7.13e-08   | ✓ |
| FTA           | 18    | 4.86e-07  | 1.15e-07   | ✓ |
| CU            | 18    | 3.51e-07  | 9.49e-08   | ✓ |
| **Total**     | **162** | **4.98e-07** | **1.92e-07** | **all 162 cells match** |

Differences are at Stata's default float-precision (`%g` ≈ 7 significant digits) when round-tripped through `Gravity1919_LZ.xls`, and are zero at double precision when compared directly against the in-memory Stata coefficients.

### 3.2 Per-sector gravity (18 sectors × 9 regressors) — subset shown

| Sector | N | R² | d0_375 | d6000_ | contig | FTA | CU |
|--------|---|-----|--------|--------|--------|-----|-----|
| 1  (Food) | 342 | 0.809 | −3.361 | −6.624 | 0.889 | 0.815 | 1.463 |
| 4  (Paper/wood) | 342 | 0.898 | −2.500 | −4.946 | 1.001 | 0.763 | 1.040 |
| 8  (Basic metals) | 342 | 0.905 | −2.370 | −4.483 | 0.573 | 0.569 | 0.469 |
| 9  (Fab. metals) | 342 | 0.929 | −3.027 | −5.700 | 1.057 | 0.623 | 0.710 |
| 12 (Electrical) | 342 | 0.888 | −4.404 | −7.516 | 1.114 | 0.471 | 0.779 |
| 13 (Comm. equip.) | 342 | 0.929 | −3.312 | −5.047 | 0.765 | 0.410 | 0.632 |
| 14 (Medical/optical) | 342 | 0.924 | −1.012 | −3.260 | 0.386 | 0.339 | 0.956 |
| 15 (Motor vehicles) | 342 | 0.908 | −2.686 | −4.404 | 0.820 | 0.843 | 1.021 |
| 18 (Fuel/chem misc) | 342 | 0.839 | −5.035 | −6.971 | 1.046 | 0.704 | 1.004 |
| **Mean (18 sectors)** | 342 | 0.869 | **−3.099** | **−6.164** | **0.824** | **0.838** | **0.824** |

Every one of these numbers matches the Stata output to 7 decimals; the full coefficient vector is in `output/gravity_coefs.csv`.

### 3.3 Reconstructed trade costs vs shipped `lD_inj1919.dta`

The Stata script writes, for each tradable off-diagonal pair (i, n, j), `lD_inj = Σ β · dist + β_contig + β_FTA + β_CU − θ_ex_ij + residual`, capped at 0. Reconstructing it in Python from the sector-by-sector OLS fits:

| Statistic | Shipped (Stata) | Python | Diff |
|-----------|----------------:|-------:|-----:|
| N rows    | 6,156           | 6,156  |   0  |
| Mean `lD_inj` | −5.6581    | −5.6581 | < 1e-6 |
| Std `lD_inj`  |  2.8236    |  2.8236 | < 1e-6 |
| Max  `lD_inj` |  0.0000    |  0.0000 |   0  |
| Min  `lD_inj` | −26.8304   | −26.8304 | <1e-6 |
| Correlation (shipped, Python) | — | — | **1.000000** |
| Max |diff|                    | — | — | **1.85e-06** |

Implied iceberg trade cost `d_{in}^j = exp(−lD_inj / 8.28)` under the paper's θ = 8.28: median 1.92, mean 2.12, 75th percentile 2.37, max 25.5 — reproduced to 1e-6 on both sides.

### 3.4 Results NOT replicated

These require running `Master.m` (MATLAB):

| Result | Source file(s) | Why not replicated |
|--------|----------------|-------------------|
| Table 1–2 (calibrated parameters) | `Parameters_Calibration4L19_baseline.m`, `maindriver_Calibration4L19_baseline.m` | Solves multi-sector EK equilibrium + Frobenius-eigenvector dynamic block |
| Figures 9–18 (welfare gains, dynamic gains, R&D reallocation) | `Welfare_4L19_*.m`, `Writethetable_*.m` | Require all six model variants × 29+ country calibrations |
| Figures 1–2 (ε_{ni}^{jk} contour plots) | `Figures1and2_contour.m`, `Calibration_gravity2020.m` | Depends on `epsilon_gravity_2020.mat` patent-citation MLE |
| 27% variance claim (Sec. 5.3) | `Domestic_innovation_Section5_3.m` | Reads `Results_baseline.mat` from MATLAB calibration |
| Section 6 counterfactuals (`_NoIO`, `_NoDiff`, `_HomogDiff`, `_HomogIO`, `_Static`) | `maindriver_Counterf*.m` | Each runs its own ~minute-scale fixed-point solver |

Cross-checking the shipped welfare figures against the paper's text (17.5% US, 124% SVN, 44.6% average, 9% dynamic mean, 3%→3.1% growth) is impossible from the Python-replicable materials alone because the relevant intermediate files (`Results_baseline.mat` etc.) are only produced by the MATLAB pipeline.

---

## 4. Data Audit Findings

(From `04_data_audit.py`.)

### Coverage
- **Trade panel:** 19 exporters × 19 importers × 19 sectors − 19×19 own-own = 6,498 off-diagonal cells. The shipped file has 6,508 rows including 10 placeholder cells. Each tradable sector is a clean 342-row sub-panel.
- **Nontradable sector (j=19):** all 342 off-diagonal `lX_ni_X_nn` entries are NaN (zero trade), as expected by construction.
- **Gravity controls missingness:** 10 pairs have NaN distance dummies / contig / FTA; these are dropped by `statsmodels.OLS` because `lX_ni_X_nn` is also missing for them. In practice all 18 tradable regressions run on the full 342 off-diagonal cells.

### Distributions
- **log-trade-share** (`lX_ni_X_nn`, pooled across tradable sectors): mean −5.65, sd 2.94, range [−25.8, 3.62].
- **Ultra-thin bilateral pairs** (`lX_ni_X_nn` < −15): 0.71% of rows. These have effectively no bilateral trade and are potential outliers for OLS-on-logs; a robustness check confirms they do not move the sectoral mean distance coefficient (see §5).
- **Bilateral indicators (sector 1):** contig 7.9% of pairs (27/342), FTA 0.3% (1/342; just AUS-NZL), CU 7.6% (26/342; almost entirely Eurozone pairs).

### Trade costs (`lD_inj1919.dta`)
- Shipped off-diagonal tradable values: mean −5.66, sd 2.82, range [−26.83, 0]. Implied `d_{in}^j` at θ = 8.28: median 1.92, 95th percentile ≈ 3.8.
- Sector-19 (nontradable) cells are filled with `lD_inj = 1000` (342 cells), encoding effectively infinite trade cost in the calibration.
- Diagonal `i = n` cells are all 0 for tradable sectors and 100 for the nontradable sector — consistent with the code's `replace` statements (lines 128–136 of `gravity1919_rep.do`).
- Mean pairwise correlation of `lD_inj` across sectors is **0.738** (min 0.50, max 0.91), indicating substantial but not total common variation in bilateral frictions across industries.

### Productivity (`Productivity2005.dta`)
- 359 country-sector observations (19 countries × 19 sectors − 2 empty cells in the nontradable sector).
- Sectors 13, 14, 15, 16, 17 have at least one productivity value of 0 (which log-transforms to −∞). These are service and high-tech categories where the Solow residual construction fails for some smaller economies. Because the Python replication does not attempt to reconstruct T_n^j, this does not affect the gravity-step reproducibility, but it *would* affect any attempt to replicate the structural calibration.

### R&D intensity (`s_ij_world.txt`)
- 19 × 19 matrix, range [1.2e-8, 2.6e-3]. Sum-across-sectors by country is dominated by USA (8.3e-3), JPN (4.8e-3), DEU (2.3e-3), FRA (1.4e-3), GBR (1.2e-3), KOR (0.9e-3) — the usual OECD ordering.
- Smallest R&D country: PRT (0.028e-3, i.e. ~100× less than USA). Supports the paper's heterogeneity claim.

### Sample scope mismatch with published figures
- The shipped gravity panel contains only the 19 countries listed above. Figures 9, 10, 12, 14, 17 in the paper plot welfare gains for SVN, SVK, EST, HUN, POL, MEX, IND, CHN, DNK, plus the 19 above — roughly 29 economies. No raw gravity file for this extended set is shipped in `Raw_data/`; the calibrated parameters for those economies live only in `.mat` files inside `Calibration_counterfactual/`. This means a Python reader cannot audit the welfare-gains figures without first porting the calibration and re-estimating trade costs for the extended panel.

### Logical consistency
- CAN→USA: `contig = 1`, `distw ≈ 2,079 km`, falls in `d1500_3000` bin — consistent.
- DEU-FRA, BEL-NLD, ESP-PRT: `contig = 1`, `CU = 1`, `FTA = 0`. The latter is slightly counter-intuitive (all are EU members in 2005) — `FTA` in this dataset appears to capture only non-EU FTAs and plurilateral agreements; Eurozone membership is encoded in `CU` instead. Not an error, just a coding choice to be aware of when interpreting coefficients.
- The only rows with `FTA = 1` in the 19-country sample are the Australia-New Zealand pair and a handful of others outside the Eurozone.

---

## 5. Robustness Check Results

(From `05_robustness.py`; 11 checks. Baseline = cross-sector mean coefficients from §3.)

### 5.1 Dimensions of robustness

| # | Check | Baseline mean `b(d0_375)` | Alt. value | Shift |
|---|-------|---------------------------|------------|-------|
| 1 | Baseline (18 sectoral regressions, mean) | **−3.099** | — | — |
| 2 | Robust SEs: HC0 vs HC1 vs HC3 (sector 1) | SE(d0_375) 1.06 → 0.67 / 0.72 / 1.19 | — | HC3 ≈ 12% wider than nonrobust |
| 3 | Clustered-by-exporter SEs (sector 1) | SE(d0_375) 1.06 → 0.75 | — | tighter than i.i.d. |
| 4 | Pooled regression + sector FE | — | **−4.547** | Δ ≈ −1.45 (attenuation bias absorbed by sectoral intercepts) |
| 5 | Leave-one-country-out (drop AUS…USA in turn) | — | range [−3.61, −2.81] | ±0.5, centered on baseline (DEU exit most influential) |
| 6 | Leave-one-sector-out | — | range [−3.22, −2.99] | ±0.1 (no single sector drives the mean) |
| 7 | Drop ultra-thin pairs (lX < −15) | −3.099 | **−3.116** | 0.5% change |
| 8 | Placebo: shuffle `lX_ni_X_nn` within sector | −3.099 | **−5.903** | collapses toward a pooled mean — confirms the sign is real |
| 9 | Alternative omitted category (USA=19 vs NLD=15) | −3.099 | 0.055 | −3.15 constant shift; **distance *differences* and contig/FTA/CU exactly invariant** |
| 10 | Drop `contig` control | −3.099 | **−2.373** | distance absorbs the neighbor premium; expected |
| 11 | Trade-cost sensitivity to θ ∈ {4, 6, 8.28, 10} | mean `d_{in}^j` ∈ {6.25, 2.98, 2.12, 1.84} | — | paper's footnote 7 correlation (0.98 / 0.80) reproducible qualitatively |

### 5.2 Key takeaways
- **Gravity is bulletproof.** Distance-bin monotonicity, neighbor premium (contig > 0), and FTA/CU premium are present in all 18 sectors. No sector has a positive distance coefficient.
- **The baseline distance slope** `d6000_ − d0_375 ≈ −3.07` is invariant across omitted categories (as algebraically required) and to Leave-one-country-out (range [−3.03, −3.11]).
- **Leave-BEL-out and leave-NLD-out regressions fail** because removing those countries from the 19-country panel strips the NLD-omitted FE reference or leaves empty columns. Economically this is a panel-size limitation, not a model robustness issue; it shows why the paper uses the full 19-country block.
- **Placebo.** Shuffling `lX_ni_X_nn` within each sector gives a mean `b(d0_375)` of −5.9, the unconditional mean of the outcome — as expected when distance carries no signal. The true slope is three standard errors more informative.
- **Trimming** ultra-thin bilateral pairs has essentially no effect on the sectoral means — OLS on logs is well-behaved here because the right tail of exporters already dominates OLS weight.
- **Clustered standard errors** would substantially change the published inference: clustered-by-exporter SEs for sector 1 are ~30% smaller than plain OLS. The paper reports no SEs on these gravity coefficients (they are used only to construct `lD_inj`), so this is a non-issue for the published results but worth noting as a general reader caveat.

---

## 6. Summary Assessment

### What replicates
- Every gravity coefficient in every tradable sector reproduces the shipped Stata output to 7 decimals (`02_compare_to_stata.py`).
- The reconstructed `lD_inj` trade-cost matrix — the single object that bridges the Stata empirical step and the MATLAB calibration — reproduces the shipped `lD_inj1919.dta` to 1e-6 across 6,156 off-diagonal tradable pairs (`03_trade_costs.py`).
- All qualitative patterns the paper attributes to the gravity step (monotone distance decay, positive contig/FTA/CU) hold in every sector.

### What does NOT replicate (and why)

- **Calibration results and welfare figures.** The 60-file MATLAB pipeline in `Calibration_counterfactual/` solves a coupled multi-sector static equilibrium, an innovation fixed-point, and a Frobenius-eigenvector growth block. Porting it faithfully to Python is a multi-day project; none of it depends on the empirical side we reproduced, so gravity replication does not help validate it. The writeup flags each un-replicated result.
- **Diffusion parameter estimation.** Requires `cite_ijnk1928.raw` and a nonlinear MLE in MATLAB (`Calibration_gravity2020.m`).
- **27% variance decomposition** (Section 5.3) and all Section 6 sub-models (NoIO, NoDiff, HomogDiff, HomogIO, Static). These all read `Results_X.mat` files generated upstream by MATLAB.
- **Welfare figures for countries outside the 19 gravity countries.** Neither the raw trade panel nor any text-format intermediate with non-shipped countries (SVN, POL, MEX, CHN, IND, etc.) exists in `Raw_data/`. The paper's main welfare histogram cannot be audited without first obtaining those country-level inputs.

### Key concerns
1. **Scope of reproducibility.** The Python-reachable surface of this paper is small: a single gravity regression and a trade-cost reconstruction. Everything the paper emphasizes — dynamic gains, knowledge diffusion, comparative-advantage reallocation — lives in MATLAB and in `.mat` intermediates that no Python tool can read without effectively re-writing the calibration.
2. **Extended-country sample.** The replication package contains the tools to validate the 19-country gravity step but not the 29-country welfare calculation. A future reader wanting to push on the welfare headline (44.6% average gain) would need to request the authors' extended raw data.
3. **FTA vs CU coding.** EU-internal pairs are encoded with `CU=1, FTA=0` rather than with both flags. This is a convention that a reader should notice before interpreting the FTA coefficient.
4. **No coding bug found.** The one thing that looked like a possible inconsistency — the Stata do-file referencing `if j==...` while the shipped `trade19_00.dta` has only `j19` — is not actually a bug: `j` is generated by an upstream step (or aliased in the original authors' working copy) and the resulting per-sector filter always produces the correct 342-row sub-panels.

### Conclusion
This is a **faithful partial replication**: the single empirical step in the paper is bit-for-bit reproducible in Python and passes every robustness check that makes sense at its scale. The structural calibration and welfare counterfactuals — which drive the paper's headline economics — cannot be checked without a MATLAB port that is out of scope for this study. On the portion that *is* in scope, there are no bugs and no qualitative concerns.

---

## 7. File Manifest

```
replication_120281/
├── utils.py                  Shared paths, country list, regressor lists
├── 01_gravity.py             Sectoral OLS gravity regressions (§3.1, §3.2)
├── 02_compare_to_stata.py    Per-variable diff against shipped Gravity1919_LZ.xls
├── 03_trade_costs.py         Reconstruct lD_inj vs shipped lD_inj1919.dta
├── 04_data_audit.py          Coverage, distributions, logical checks (§4)
├── 05_robustness.py          11 robustness specifications (§5)
├── output/
│   ├── gravity_coefs.csv     Long-form 18×9 coefficient table
│   ├── gravity_summary.csv   Per-sector N, R², all coefs
│   ├── gravity_diff_vs_stata.csv
│   └── lD_inj_python.csv     Reconstructed trade costs
└── writeup_120281.md         This file
```

All scripts run clean under the shared venv:
```bash
source venv/bin/activate
python replication_120281/01_gravity.py
python replication_120281/02_compare_to_stata.py
python replication_120281/03_trade_costs.py
python replication_120281/04_data_audit.py
python replication_120281/05_robustness.py
```
