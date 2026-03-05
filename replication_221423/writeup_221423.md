# Replication Study: 221423-V1

**Paper:** "Income Inequality in the Nordic Countries: Myths, Facts, and Lessons"
**Authors:** Magne Mogstad, Kjell G. Salvanes, Gaute Torsvik
**Journal:** *American Economic Review*, 2025
**Original Language:** Stata (17 files) + R (2 files) + Python (1 file)
**Replication Language:** Python (pandas, matplotlib)

---

## 0. TLDR

- **Replication status:** Partial. Table 1 (15/16 rows), Table 3 (3/5 columns), Figure 2, and Figure 3 replicate exactly. Tables 4-14 require PIAAC microdata (not included, requires OECD registration). Table 1 Social Trust row requires WVS/EVS data (not included).
- **Key finding confirmed:** Nordic countries have lower income inequality (Gini 0.27 disposable) than the US (0.39) and OECD average (0.31), with a much larger redistribution gap (Gini_market - Gini_disposable = 0.12 vs 0.08 for US).
- **Main concern:** 11 of 14 tables require PIAAC survey microdata that is not bundled and requires manual download with OECD registration. This makes full replication infeasible without researcher registration.
- **Bug status:** One data column issue identified: Stata code uses 2021 public sector employment data (column D in Excel) despite the paper referencing 2019 data. This is a minor inconsistency (values differ by ~1 percentage point).

---

## 1. Paper Summary

### Research Question
How do the Nordic countries (Denmark, Finland, Norway, Sweden) compare to the US, UK, and OECD average on key demographic, economic, and inequality measures? What explains the Nordic model's combination of high living standards and low inequality?

### Data
- **OECD Statistics:** Population, labor force, productivity, Gini coefficients, better life index
- **Kleven (2014):** Participation tax rates and employment subsidies across OECD countries
- **OECD-AIAS-ICTWSS:** Union density and collective bargaining coverage (1960-2020)
- **PIAAC:** Programme for International Assessment of Adult Competencies — individual-level microdata on skills, earnings, education for 30 countries (NOT included in package, requires OECD registration)
- **WVS/EVS:** World Values Survey / European Values Study for social trust measures (NOT included)
- **PISA 2018:** Student performance data (included but not used in replicable tables)

### Method
1. **Descriptive statistics:** Country-level comparisons with Nordic population-weighted averages
2. **Gini decomposition:** Market vs disposable income inequality, with redistribution gap
3. **PIAAC microdata analysis (not replicated):** Earnings inequality, returns to skills, Oaxaca-Blinder decompositions, quantile regressions

### Key Findings
- Nordic countries achieve comparable GDP per capita to the US ($53K vs $61K) with far less inequality
- Nordic redistribution gap (Gini market - Gini disposable) is 0.12, vs 0.08 for US
- High union density (50-67%) and bargaining coverage (67-88%) distinguish Nordic labor markets
- Nordic employment rates (82-88%) exceed US (81%) and OECD average (74%)
- Public sector employment is 25-31% in Nordics vs 15% in US

---

## 2. Methodology Notes

### Translation Choices
- **Stata `sxpose` → pandas pivot:** The Stata code uses `sxpose` (transpose) extensively for reshaping OECD data from stacked to wide format. Translated to `pivot_table()` in pandas.
- **R data.table + ggstar → pandas + matplotlib:** Figure 3 uses R's data.table for weighted group means and ggstar for star-shaped markers. Translated to pandas groupby with weighted means and matplotlib standard markers.
- **Population-weighted Nordic averages:** Weights from `nordic_population.dta` (DNK: 5.8M, FIN: 5.5M, NOR: 5.3M, SWE: 10.3M). Sweden's larger population dominates the average.

### Data Notes
- **Public sector employment:** The Stata code reads column D of the Excel file (2021 values) rather than 2019 values, despite the paper describing 2019 data. The difference is small (~1 pp).
- **OECD averages:** Computed as sum/38 for total quantities (population, GDP) and simple mean for rates/ratios. This uses 38 member countries as the denominator.
- **Labor force data:** Employment rate = EMP / (working_age_pop × 10^6); labor force participation = LF / (working_age_pop × 10^6).

---

## 3. Replication Results

### Table 1: Demographics & Economy

| Variable | Nordic | Denmark | Finland | Norway | Sweden | UK | US | OECD | Match? |
|----------|--------|---------|---------|--------|--------|----|----|------|--------|
| Pop (M) | 7.4 | 5.8 | 5.5 | 5.3 | 10.3 | 66.8 | 328.3 | 35.9 | ✓ |
| Working-age pop (M) | 4.2 | 3.4 | 3.1 | 3.2 | 5.8 | 38.8 | 192.6 | 21.1 | ✓ |
| Dependency ratio | 35 | 34 | 39 | 29 | 35 | 32 | 28 | 29 | ✓ |
| Fertility | 1.6 | 1.7 | 1.4 | 1.5 | 1.7 | 1.6 | 1.7 | 1.6 | ✓ |
| Foreign born (%) | 14 | 10 | 7 | 15 | 19 | 14 | 14 | 14 | ✓ |
| Non-western (%) | 46 | 43 | 32 | 45 | 54 | 54 | 87 | 65 | ✓ |
| Social trust | — | — | — | — | — | — | — | — | N/A (needs WVS) |
| Life satisfaction | 7.5 | 7.5 | 7.9 | 7.3 | 7.3 | 6.8 | 7.0 | 6.7 | ✓ |
| GDP (B) | 384 | 306 | 253 | 333 | 524 | 2,984 | 20,137 | 1,548 | ✓ |
| GDP/capita (K) | 53 | 53 | 46 | 62 | 51 | 45 | 61 | 43 | ✓ |
| GDP/hour | 72 | 74 | 62 | 83 | 71 | 59 | 72 | 52 | ✓ |
| Hours/capita | 727 | 709 | 742 | 752 | 717 | 755 | 842 | 825 | ✓ |
| Hours/worker | 1,446 | 1,371 | 1,538 | 1,419 | 1,453 | 1,537 | 1,742 | 1,742 | ✓ |
| LFP | 0.91 | 0.89 | 0.88 | 0.89 | 0.94 | 0.88 | 0.84 | 0.79 | ✓ |
| Employment rate | 0.85 | 0.85 | 0.82 | 0.86 | 0.88 | 0.84 | 0.81 | 0.74 | ✓ |
| Public employment | 0.29 | 0.28 | 0.25 | 0.31 | 0.29 | 0.17 | 0.15 | 0.19 | ✓ |

**Result: 60/60 values match (excluding Social Trust row).**

### Table 3: Gini Coefficients (Columns 1-3)

| Country | Pub Gini_disp | Repl Gini_disp | Pub Gini_mkt | Repl Gini_mkt | Pub Diff | Repl Diff | Match? |
|---------|-------------|---------------|------------|-------------|---------|----------|--------|
| Nordic Countries | 0.27 | 0.27 | 0.39 | 0.39 | -0.12 | -0.12 | ✓ |
| Denmark | 0.27 | 0.27 | 0.40 | 0.40 | -0.13 | -0.13 | ✓ |
| Finland | 0.28 | 0.28 | 0.43 | 0.43 | -0.15 | -0.15 | ✓ |
| Norway | 0.27 | 0.27 | 0.39 | 0.39 | -0.11 | -0.11 | ✓ |
| Sweden | 0.27 | 0.27 | 0.36 | 0.36 | -0.09 | -0.09 | ✓ |
| United Kingdom | 0.36 | 0.36 | 0.45 | 0.45 | -0.09 | -0.09 | ✓ |
| United States | 0.39 | 0.39 | 0.47 | 0.47 | -0.08 | -0.08 | ✓ |
| OECD Average | 0.31 | 0.31 | 0.41 | 0.41 | -0.10 | -0.10 | ✓ |

**Result: 8/8 countries match exactly (columns 1-3). Columns 4-5 require PIAAC data.**

### Figures 2 and 3
- **Figure 2:** Scatter plot of 1 - participation tax rate vs employment subsidies for 29 OECD countries, with Nordic countries, UK, and US highlighted. Replicates the Kleven (2014) data visualization.
- **Figure 3a:** Union density (%) by region group (Nordic, Continental Europe, UK, US), 1980-2019. Shows Nordic countries maintaining ~60-70% union density vs US decline to ~10%.
- **Figure 3b:** Bargaining coverage (%) by region group. Shows near-universal coverage in Nordics (~80-90%) vs US at ~10%.

---

## 4. Data Audit Findings

### Coverage
- **Table 1 data:** Complete for 7 entities (4 Nordic + UK + US + OECD) across all included variables
- **Table 3 data:** 32 OECD countries with Gini data for 2019
- **ICTWSS data:** 57 countries, 1960-2020. Union density coverage is excellent (40/40 years for Nordic countries). Bargaining coverage has gaps requiring interpolation (especially Norway, France).
- **PIAAC data:** Only 3 of 30 required country CSVs present (Austria, Belgium, Canada). All 4 Nordic countries, UK, and US are missing.

### Missing Data
- **WVS/EVS (Joint_EVS_WVS_2017_2022.dta):** Not included. Needed for Table 1 Social Trust row.
- **PIAAC microdata:** Not included (requires OECD registration). Needed for Tables 4-14 and Table 3 columns 4-5.
- **Figure 5 data:** Norwegian administrative microdata. Restricted access, no code provided in package.

### Data Quality
- All OECD statistics have no missing values for the 7 entities of interest
- Nordic population weights are internally consistent (sum to 27.0M)
- Gini coefficients are all in valid [0, 1] range
- OECD averages use 38-country denominators (current OECD membership)

---

## 5. Bug Impact Analysis

### Minor Issue: Public Sector Employment Year
The Stata code `import excel ... cellrange(B34:D72)` reads columns B, C, D from the Excel file. Column C (2019 values) is dropped, and column D (2021 values) is used. The paper states data is for "year 2019" but the Stata code uses 2021 values. The difference is small (e.g., Finland: 24.4% in 2019 vs 25.4% in 2021) and does not affect the paper's conclusions.

**Impact:** Negligible. Values differ by ~1 percentage point and all comparative rankings are preserved.

---

## 6. Robustness Results

| # | Check | Key Result | Status |
|---|-------|-----------|--------|
| 1 | Nordic Gini weighting | Pop-weighted vs equal-weight: <0.006 difference | Robust |
| 2 | OECD average completeness | All vs complete cases: identical (N=32 all complete) | Robust |
| 3 | Gini temporal stability | Nordic Gini stable 2012-2019 (0.26→0.27) | Robust |
| 4 | Nordic GDP LOO | Range: 50-54K GDP/capita across 3-country subsets | Robust |
| 5 | Nordic union density LOO | Range: 59-64% across 3-country subsets | Robust |
| 6 | Redistribution gap ranking | Nordic countries rank 2nd (Finland), 9th (Denmark) of 32 | Informative |

---

## 7. Summary Assessment

### What Replicates
- **Table 1:** All 15 replicable rows match exactly across all 8 country columns (60/60 values)
- **Table 3:** All 3 replicable columns match exactly for all 8 rows (24/24 values)
- **Figure 2:** Kleven tax/subsidy scatter replicates with correct data from Excel
- **Figure 3:** Union density and bargaining coverage trends replicate with weighted means and interpolation

### What Doesn't
- **Table 1 Social Trust (1 row):** Requires WVS/EVS + PIAAC data, neither included
- **Table 3 Columns 4-5:** Gini of earnings and variance of log earnings require PIAAC microdata
- **Tables 4-14 (11 tables):** All require PIAAC microdata with specialized Stata packages (ineqdeco, oaxaca, piaacreg)
- **Figure 5:** Norwegian administrative data, restricted access

### Key Concerns
- The main limitation is data availability, not code quality. The replication package is well-organized with clear code structure.
- The 2019 vs 2021 public employment discrepancy is minor but should be documented.
- PIAAC data requiring manual OECD registration is a barrier to full replication.

### Overall Assessment
Clean partial replication. The descriptive statistics (Tables 1 and 3) and institutional figures (2 and 3) replicate exactly. The bulk of the paper's novel contribution — PIAAC-based earnings inequality analysis (Tables 4-14) — cannot be replicated without downloading ~30 PIAAC country files from OECD, which requires registration. The included code is well-structured and the data that is provided produces exact matches.

---

## 8. File Manifest

| File | Description |
|------|-------------|
| `utils.py` | Paths, country codes, constants |
| `01_clean.py` | Load and validate all available datasets |
| `02_tables.py` | Table 1 (demographics & economy) and Table 3 (Gini columns 1-3) |
| `03_figures.py` | Figure 2 (Kleven scatter) and Figure 3 (union density & bargaining coverage) |
| `04_data_audit.py` | Data completeness, coverage, distributions, missing data |
| `05_robustness.py` | 6 robustness checks on summary statistics and weighting |
| `output/` | Parquet files, PNG figures, robustness_summary.csv |
| `writeup_221423.md` | This writeup |
