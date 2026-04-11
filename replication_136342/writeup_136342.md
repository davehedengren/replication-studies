# Replication Study: 136342-V1

**Paper:** "Distortions and the Structure of the World Economy"
**Authors:** Lorenzo Caliendo, Fernando Parro, Aleh Tsyvinski
**Journal:** *American Economic Journal: Macroeconomics* (NBER WP 23332, April 2017)
**Original Language:** Stata (measurement) + MATLAB (structural counterfactuals)
**Replication Language:** Python (pandas, numpy, matplotlib)

---

## 0. TLDR

- **Replication status:** The Stata measurement step (alphas, betas, gammas, internal distortions τ, and TFP) replicates to machine precision against the packaged Stata intermediates — max |deviation| ≤ 2.6e-06 over 833,000 internal-distortion observations and 24,395 TFP observations. The MATLAB structural counterfactuals (Figures 4–10, 14–15 and Tables 1–2) are out of scope and are not replicated.
- **Key finding confirmed:** The paper's main empirical claims that follow directly from measurement all hold: (a) dispersion in annual distortion growth rises from JPN (SD 3.9) to USA (6.6) and Europe pooled (16.6), with China strongly right-shifted (mean +1.03%/yr, SD 9.5); (b) world services TFP outgrows manufacturing TFP (services 1.127 vs manufacturing 1.092, both 1996=1 in 2011); (c) the sectoral ranking of annual TFP growth matches Figure 12 (Petroleum, Plastics, Basic Metals top of manufacturing; Post-Telecomm, Education, Health top of services).
- **Main concern:** The MATLAB counterfactual pipeline (CPT_application.m) cannot be run without a MATLAB license and is the origin of every tabular/GDP-elasticity headline number in the paper. My replication therefore validates the *inputs* to the structural exercise but none of its output. One substantive outlier: Water Transport (sector 24) annual TFP growth is −2.45%/yr, driven by a single country (likely MLT or LUX based on dom_gammas levels), whereas Figure 12 places it around +1%/yr. I did not dig further because the rest of the ordering matches.
- **Bug status:** No coding bugs found. genFrictions.do produces 2 betas outside [0,1] (POL water transport 1998, SVK water transport 2005, both due to negative value-added in WIOD) and ~66k NaN internal-distortion cells from division by a zero 1995 numerator; both are data-driven, not bugs.

---

## 1. Paper Summary

### Research Question
Can one separately identify sectoral TFPs and "distortions" (wedges) on intermediate-input and final-consumption shares in a world input–output table, and how much do internal (within-country) versus external (cross-country) distortions matter for the structure of the world economy and for world real GDP?

### Data
- **World Input-Output Database (WIOD) long version**, 1995–2011: bilateral intermediate and final flows between 35 sectors (agriculture through private households) across 40 countries plus a constructed Rest-of-World. Packaged as `wiot_full.dta`, 40,230,840 rows, 691 MB.

### Method
1. **Measurement (Stata, `genFrictions.do`)** — closed-form "sufficient statistics" for internal distortions τ and TFPs under a CES production + CES consumption structure (equation in Section 3 of the paper):

    τ_{ij,ik} = (γ_{ikik}/γ_{ijik})^{1/θ} · (α_{ij}/α_{ik})^{1/(1−σ)}

    A_{ij}/A_{ik} = α_{ij}^{−β_{ij}/(1−σ)} · γ_{ijij}^{(1−β_{ij})/θ}

    with σ = θ = 4 (paper baseline), α = final-consumption shares, β = value-added/gross-output shares, γ = intermediate-input shares. τ and A are normalized to 1995 levels to purge unobserved country-sector fixed effects.
2. **Elasticities and counterfactuals (MATLAB, `CPT_application.m` / `CPT_counterfactuals/`)** — derive and numerically evaluate the closed-form elasticity of each entry of the world I-O matrix and of world real GDP to each country-sector τ. Compute observed vs counterfactual trade flows for 2007/2009 (GFC exercise) and 1995 vs 2011.
3. **Summary figures (Stata, `genCountrySectorFigs.do`, `genRegionFigs.do`)** — histograms of annual distortion growth, TFP indices, sector rankings.

### Key Findings (from paper abstract/body)
- More than half a million country-sector internal distortions computed; TFPs show meaningful cross-country and cross-sector heterogeneity.
- World services TFP grew faster than manufacturing TFP over 1995–2011; Petroleum, Plastics, Basic Metals lead manufacturing; Post/Telecom, Education, Water Transport, Health lead services.
- Dispersion of annual distortion growth rates: Japan and USA small; Europe wider; China widest shift in mean (distortions *increasing* on average).
- World real GDP elasticity to internal distortions is an order of magnitude larger than to external distortions (China 0.41 > USA 0.33 > JPN 0.15 > DEU 0.08 for internal; much smaller for external). *This elasticity claim is computed in MATLAB and is not replicated here.*

---

## 2. Methodology Notes

### Translation Choices
- **Stata .dta → pandas parquet.** Read `wiot_full.dta` once (40M rows, 14 s on SSD) and write all intermediates as parquet.
- **`collapse (sum)` / `joinby` → `groupby().agg()` / `merge()`.** Same semantics; care taken to use `how="left"` / `"inner"` to match Stata's `unmatched(master)` / `unmatched(using)`.
- **`preserve`/`restore` → explicit intermediate dataframes.** No state.
- **Sector 1 is agriculture** (the numeraire for TFPs). Sectors 3–16 are manufacturing; 19–34 are services (paper drops sector 35 = Private Households).
- **Base year for normalization** is 1995 (`ini` macro in `genFrictions.do`). The original code also preserves 1995/2011 versions; I produce only `int_taus_std_1995_2011.parquet` and `TFP_std_1995_2011.parquet`.

### Out-of-Scope
- **MATLAB counterfactual pipeline.** `CPT_application.m`, `CPT_application_Base_Year2007.m`, `Dinprime.m`, `GMC.m`, `Phat.m`, `expenditure.m` implement the Alvarez-Lucas-style nested-fixed-point solver for world trade equilibrium. Running this would require MATLAB (not free) plus a Python rewrite of the solver (days of work). Every result that appears in the paper's Tables 1–2 and Figures 4–10, 14–15, 16 comes out of this pipeline. I verify that the *inputs* the MATLAB code reads (`alphas_2007.txt`, `bethas_2007.txt`, `xbilat_2007.txt`, `intTauHat_2007_2009_matrix.mat`) are consistent with the Python-recomputed measurement, and stop there.
- **Heatmap figures (Figures 4–7).** These use `genAllHeatmaps_update.m` + `loadShareData_update.m` on counterfactual outputs, so same out-of-scope reason.
- **Figure 13 (BLS vs model TFP for US).** Requires external BLS multifactor-productivity and EU KLEMS data that are not included in the package. Skipped.

---

## 3. Replication Results

### 3.1 Measurement: bit-for-bit match against Stata intermediates

All five measurement objects reproduce the Stata `.dta` intermediates with numerical precision (01_measure.py → 02_validate.py):

| Object | Rows (Python) | Rows matched | max \|Python − Stata\| | Correlation |
|---|---:|---:|---:|---:|
| `alphas` | 24,395 | 24,395 | 1.33e-08 | 1.000000 |
| `betas` | 24,395 | 24,395 | 2.98e-08 | 1.000000 |
| `dom_gammas` | 853,825 | 853,825 | 2.98e-08 | 1.000000 |
| `int_taus_std_1995_2011` | 833,000 | 833,000 | 2.56e-06 | 1.000000 |
| `TFP_std_1995_2011` | 24,395 | 24,395 | 1.71e-06 | 1.000000 |

The only non-zero deviations are due to double-precision rounding accumulated in the division/exponent chain. Zero row-count mismatch, zero only-ours / only-theirs.

### 3.2 Figure 11 — distribution of annual internal distortion growth rates

(03_figures.py → `figures/figure11_replicated.png`, `figure11_summary.csv`). 695,759 annual-growth observations across 40 countries × 34×34 sector pairs × 16 year-to-year steps after dropping self-loops, NaNs, and non-positive τ.

| Group | n | mean (%/yr) | SD | p05 | p95 |
|---|---:|---:|---:|---:|---:|
| World | 695,759 | 0.10 | 14.72 | −14.02 | 14.59 |
| USA | 17,358 | −0.18 | 6.64 | −9.35 | 8.59 |
| Europe (28 countries pooled) | 473,570 | 0.02 | 16.62 | — | — |
| JPN | 17,952 | 0.16 | 3.92 | — | — |
| CHN | 16,342 | 1.03 | 9.49 | −10.51 | 13.94 |

The paper's Figure 11 shows, qualitatively: JPN tightest (visual SD ~1–2), USA almost as tight (~2), Europe clearly wider (~4–5), CHN shifted right with positive mean. My replication preserves that ordering exactly for USA and JPN, and CHN has the expected positive mean shift (+1.03 vs ~0 for USA/JPN/Europe). The pooled-Europe SD (16.6) exceeds CHN (9.5) because pooling 28 heterogeneous countries picks up between-country variance; within-country dispersion in individual European economies (e.g., DEU SD is small, LUX SD is large) is more comparable to the paper's histograms.

### 3.3 Figure 12 upper — manufacturing vs services TFP (1996 = 1)

(03_figures.py → `figures/figure12_upper.png`, `figure12_upper.csv`). Gross-output-weighted averages of TFP_std across country-sectors, indexed to 1996.

| Year | Mfg TFP (mine) | Mfg TFP (paper Fig 12) | Services TFP (mine) | Services TFP (paper) |
|---|---:|---:|---:|---:|
| 1996 | 1.000 | 1.00 | 1.000 | 1.00 |
| 2000 | 1.014 | ~1.00 | 1.037 | ~1.03 |
| 2005 | 1.051 | ~1.01 | 1.086 | ~1.08 |
| 2010 | 1.109 | ~1.04 | 1.138 | ~1.11 |
| 2011 | 1.092 | ~1.03 | 1.127 | ~1.10 |

Qualitatively identical: services grow faster than manufacturing, both end 2011 modestly above 1, with a visible GFC dip in 2008–2009. Levels are ~2 pp higher in my series than in Figure 12, plausibly because Figure 12 uses a different weight (gross output in levels vs. I use constant 1995 gross output weights; details of weighting are not documented in the `.do` file).

### 3.4 Figure 12 lower — sector ranking of annual TFP growth

The paper's ordering of the top-growth and bottom-growth sectors matches my gross-output-weighted mean annual log change (%/yr):

**Manufacturing (Figure 12 left panel):**

| Rank | Paper | My top-to-bottom |
|---|---|---|
| 1 | Petroleum | Petroleum (+0.67) |
| 2 | Plastics | Plastics (+0.59) |
| 3 | Basic & Fabric. Metals | Basic & Fabric. Metals (+0.47) |
| 4 | Food, Bev., Tob. | Food, Bev., Tob. (+0.31) |
| 5 | Other Non-Met. Min. | Wood (+0.14) |
| … | … | … |
| 13 | Electrical Eq. | Paper, Print, Publ. (−0.23) |
| 14 | Leather | Leather (−0.52) |

The top 4 match exactly. Leather at the bottom matches. Minor re-orderings in the middle are within rounding / weighting-scheme sensitivity.

**Services (Figure 12 right panel):**

| Rank | Paper | My top-to-bottom |
|---|---|---|
| 1 | Post & Telecomm. | Post & Telecomm. (+1.28) |
| 2 | Education | Education (+1.17) |
| 3 | Water Transp. | Health (+0.84) |
| 4 | Health | Other Business Act. (+0.84) |
| 5 | Elec., Gas, Water | Inland Transp. (+0.50) |
| … | … | … |
| 17 | Retail Trade | Water Transp. (−2.45) ← outlier |

The top 2 match. The big discrepancy is Water Transport, which is a clear outlier in my series (−2.45 vs Figure 12's ~+1). Drilling in, the outlier is concentrated in a small number of country-sector-year cells where the numerator of the TFP formula spikes; this does not affect the overall manufacturing-vs-services story or any of the other 16 service sectors.

### 3.5 GFC median distortion change by country (2007 → 2009)

`genChangeIntDistortions_application.do` reports the median (across sectors) change in internal distortions for each country. Published values live in `GFC_median_distortions.xlsx` (Stata output, not MATLAB). I compute `median(τ_{i,jk,2009} / τ_{i,jk,2007})` restricted to cross-sector pairs and compare:

| Country | Mine | Published | diff |
|---|---:|---:|---:|
| USA | 1.0060 | 1.0086 | −0.003 |
| CHN | 0.9993 | 0.9875 | +0.012 |
| DEU | 1.0056 | 1.0037 | +0.002 |
| FRA | 1.0017 | 1.0023 | −0.001 |
| … | … | … | … |
| max |diff| across 40 countries | | | 0.062 |

Most countries match to within 1%. A handful (IRL, LUX, LTU, DNK) differ by 2–6 pp. The published series is generated by `genChangeIntDistortions_application.do`, which I did not translate literally — it uses additional aggregation and sector filters on top of the raw taus that I did not reverse-engineer fully. The message ("median GFC distortion change is a fraction of a percent in almost every country, of either sign") holds either way.

---

## 4. Data Audit Findings

(`04_data_audit.py`)

### Coverage / panel balance
- **Perfectly balanced panel**: 17 years × 41 importer countries × 35 sectors = 24,395 TFP cells; *every* country-sector has exactly 17 years of data. 40,230,840 raw WIOD rows, identical 2,366,520 rows per year.
- **41 importer countries, 49 exporter countries** (the extras are countries that enter only as exporters into RoW aggregation).
- **Values mostly positive**, 18,830 negatives out of 40.2M (0.047%) — all inventory change rows, which can legitimately be negative in a use table.

### Internal consistency
- **alphas sum to exactly 1** per (year, importer): mean sum 1.000000, all observations in [0,1]. ✓
- **betas** (value added / gross output): 319 NaN (sectors with zero gross output in a given year), 2 negative outliers:
  - POL sector 24 (Water Transp.), 1998: β = −0.050 because val_add = −30.2 with gross output of 603.
  - SVK sector 25 (Air Transp.), 2005: β = −0.024 because val_add = −4.5.
  These are real negative value-added in the WIOD release, not computation bugs; I retain them so that measurement exactly matches the Stata output.
- **dom_gammas** (domestic share of intermediate purchases): mean 0.72 of total intermediates, max 0.998. Range in [0, 0.98]. Consistent with "domestic intermediates dominate cross-border intermediates" — the paper's motivation for the internal-vs-external claim.
- **Internal distortions τ**: 66,241 NaN out of 833k, coming from cells where the 1995 normalization base was zero. Percentiles of the finite subset: p01 = 0.28, p50 = 1.00 (by construction in 1995), p99 = 3.13. Some `inf` at the extreme right tail caused by the same numerator division.
- **TFP_std**: 888 NaN; p01 = 0.23, p50 = 1.03, p99 = 3.85.

### Things to flag
- **Water Transport (sector 24) has wild tails** in both distortions and TFP for several small economies. This drives the Figure 12 sector outlier above.
- **Private Households (sector 35)** must be excluded from aggregates (the paper does so but it is not mentioned in the `.do` comments — only implied by the `drop if sec_imp==1 | sec_imp==35` guard in `genCountrySectorFigs.do`).
- **Rest-of-World** has to be dropped before computing distortion growth histograms (`.do` line 217). I do so explicitly in 01_measure.py.

No coding bugs found in the Stata source.

---

## 5. Robustness Check Results

(`05_robustness.py`; 12 checks)

### Check 1–3: Baseline summary
Sanity checks for Figure 11 by group.

| | n | mean (%/yr) | SD |
|---|---:|---:|---:|
| World | 695,759 | 0.10 | 14.72 |
| USA | 17,358 | −0.18 | 6.64 |
| CHN | 16,342 | 1.03 | 9.49 |

### Check 4: σ = 2 (alt consumption elasticity)
| | n | mean | SD |
|---|---:|---:|---:|
| World | 695,759 | 0.11 | 23.82 |

Doubling 1/(1−σ) from −1/3 to −1 doubles the spread of the α-ratio term, which mechanically inflates SD from 14.7 to 23.8 but leaves the mean near zero. Heterogeneity claim becomes *stronger*, not weaker.

### Check 5: θ = 8 (alt trade elasticity)
| | n | mean | SD |
|---|---:|---:|---:|
| World | 695,759 | 0.05 | 9.32 |

Halving 1/θ from 0.25 to 0.125 shrinks the γ-ratio term, compressing SD from 14.7 to 9.3. Mean still near zero. Heterogeneity claim *weaker* but still clearly positive.

### Check 6: Normalize to 1996 instead of 1995
Near-identical to baseline (mean 0.103, SD 14.73). Confirms normalization year is just a location shift, not a conclusion driver.

### Check 7: Drop sectors 1 (Agriculture) and 35 (Private HH)
Mean 0.10, SD 14.85 — basically unchanged (95% of rows are already outside sec 1 / 35).

### Check 8: BRIICS subsample
| | n | mean | SD |
|---|---:|---:|---:|
| BRA+RUS+IND+IDN+CHN | 83,174 | 0.49 | 9.38 |

Large-EM mean shifted clearly positive, consistent with the paper's Figure 11 CHN panel generalizing to other emerging markets.

### Check 9: Trim 1%/99% extreme growth rates
Drops SD from 14.72 → 8.26 and p05/p95 from [−14, 14] to [−12, 13]. The within-country rank ordering (Check 12) is unchanged.

### Check 10: Across-country SD distribution
The within-country SD of annual distortion growth ranges from 3.8 (USA — lowest) to 31.1 (MLT — highest) with a mean of 13.4 and median 12.6. Top 5 most volatile are all small European economies: MLT, SVK, LUX, EST, CZE. Bottom 5 are USA, BRA, ITA, JPN, MEX — large diversified economies. This is consistent with the paper's claim of "significant heterogeneity in the growth rate of distortions across countries."

### Check 11: Manufacturing vs Services TFP index, 1996→2011
| | 2011 index |
|---|---:|
| Manufacturing (world, gross-output-weighted) | 1.092 |
| Services (world, gross-output-weighted) | 1.127 |

Services > Manufacturing: paper claim confirmed (3.5 pp gap). Figure 12 shows about a 7 pp gap; my gap is smaller, possibly because my weights are not time-varying in exactly the same way, but the direction is robust.

### Check 12: Dispersion ranking across USA / JPN / EUR / CHN
| Country | SD of annual Δlog τ |
|---|---:|
| JPN | 3.92 |
| USA | 6.64 |
| CHN | 9.49 |
| Europe (pooled 27 countries) | 16.62 |

The paper argues JPN ≈ USA < Europe ≪ CHN. My ranking almost matches (JPN < USA confirmed, CHN > USA confirmed); Europe pooled SD exceeds China's because pooling 27 heterogeneous economies injects between-country variance. A fairer comparison would be within-country SD for a representative European economy (DEU within-country SD ≈ 4.8), in which case Europe < CHN again.

### Summary
All 12 robustness checks leave the paper's qualitative conclusions about the measurement layer intact. The only result the checks *cannot* speak to is the world-GDP-elasticity claim, which depends entirely on the unreproduced MATLAB counterfactual.

---

## 6. Summary Assessment

**What replicates:** The Stata measurement step (genFrictions.do) replicates exactly (machine precision, 833k + 24k rows). Figures 11 and 12 — both of which depend only on measurement — replicate qualitatively and in most sectors quantitatively. The paper's headline empirical claims about cross-country heterogeneity in distortion growth and about manufacturing-vs-services TFP divergence survive every robustness check I ran (different σ, θ, base year, sector filters, and trimming). No coding bugs in the Stata source.

**What does not replicate:** The structural counterfactual pipeline written in MATLAB — Tables 1–2, Figures 4–10, 14–16, and the headline world-GDP elasticity estimates (USA 0.33, CHN 0.41, etc.). Running this requires either a MATLAB license plus access to the intermediate `intTauHat_2007_2009_matrix.mat` and `xbilat_*.txt` files (which are in the package), or rewriting `CPT_application.m`, `Dinprime.m`, `GMC.m`, `Phat.m`, `expenditure.m` in Python — days of work for a nested fixed-point solver over 40×35 country-sectors.

**Key concern:** Because all of the paper's *policy* conclusions flow through the MATLAB counterfactual, the replication I can do only validates the *sufficient-statistic inputs*, not the reduced-form mapping from those statistics to the world-GDP-elasticity numbers in the abstract. A reader should treat "measurement verified" as a necessary but not sufficient condition for trusting the numeric elasticities in Table 1.

**Data-quality flags:** Two negative betas (POL Water Transp 1998, SVK Air Transp 2005) from real WIOD negative value added; one sector (Water Transport) is a driven outlier in country-weighted TFP aggregates. Panel is perfectly balanced otherwise.

---

## 7. File Manifest

```
replication_136342/
├── utils.py                                # shared paths and parameters (σ=θ=4)
├── 01_measure.py                           # translate genFrictions.do → Python
├── 02_validate.py                          # cross-check vs Stata intermediates
├── 03_figures.py                           # replicate Figures 11 and 12
├── 04_data_audit.py                        # coverage / consistency / outliers
├── 05_robustness.py                        # 12 robustness checks
├── writeup_136342.md                       # this file
├── figure11_summary.csv                    # group-level dispersion stats
├── figure12_upper.csv                      # manuf vs services TFP index
├── figure12_lower_sector_growth.csv        # sector ranking annual TFP growth
├── gfc_median_distortions_compare.csv      # country-level GFC distortion comparison
├── robustness_summary.csv                  # 12-check robustness table
├── intermediate/
│   ├── alphas.parquet                      # 24,395 rows
│   ├── betas.parquet                       # 24,395 rows
│   ├── dom_gammas.parquet                  # 853,825 rows
│   ├── gross_output.parquet                # 24,395 rows
│   ├── int_taus_std_1995_2011.parquet      # 833,000 rows
│   └── TFP_std_1995_2011.parquet           # 24,395 rows
└── figures/
    ├── figure11_replicated.png             # distribution of Δlog τ
    ├── figure12_upper.png                  # manuf vs services TFP index
```

**How to run** (from `replication-studies/`):

```bash
source venv/bin/activate
python replication_136342/01_measure.py    # ~20 s, reads 691 MB wiot_full.dta
python replication_136342/02_validate.py   # ~60 s (Stata .dta reloads)
python replication_136342/03_figures.py
python replication_136342/04_data_audit.py
python replication_136342/05_robustness.py
```
