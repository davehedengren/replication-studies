# Replication Study: 239496-V1

**Paper**: Geruso, M. and Spears, D. (2026). "The Likelihood of Persistently Low Global Fertility." *Journal of Economic Perspectives*, 40(1): 3-26.

**Replication package**: openICPSR 239496-V1

---

## 0. TLDR

- **Replication status**: All replicable figures and tables reproduce correctly. Key claims verified: 0-for-24 CCF rebound statistic confirmed exactly, 67.1% of world population in TFR<2 countries (paper: "two-thirds"), Kitagawa decomposition yields ~39% childlessness / ~61% parous fertility (paper: ~38%/~62%). Figures 5b and Table A1 (India DHS) not replicated due to unavailable survey data.
- **Key finding confirmed**: Where cohort fertility has fallen below 1.9, no subsequent cohort has rebounded above 2.1 (0-for-24 countries; 0-for-26 including sub-national).
- **Main concern**: None. The analysis is descriptive and transparent. All claims check out.
- **Bug status**: No bugs found.

---

## 1. Paper Summary

**Research question**: How likely is persistently low global fertility? The authors distinguish period TFR (snapshot) from completed cohort fertility (CCF) and argue that cohort measures are more informative for long-run population dynamics.

**Data**: UN World Population Prospects 2024 (237 countries, 1950-2023), Human Fertility Database (34 countries, CCF and parity data for cohorts 1935-1980), Romania CCF from Ghetau (1997), historical/projected births from Spears et al. (2024), India DHS (NFHS-5).

**Method**: Descriptive analysis using:
1. Time series of global TFR and population fraction below replacement
2. Country-by-country comparison of period TFR vs completed cohort fertility
3. Scatter of lowest vs latest CCF across countries (no-rebound test)
4. Kitagawa decomposition of CCF decline into childlessness and parous fertility components
5. Romania vs Bulgaria comparison to test whether pronatalist policy (Decree 770) raised CCF

**Key findings**:
- Two-thirds of world population lives in countries with TFR below 2
- In 24 countries where CCF fell below 1.9, no subsequent cohort rebounded to 2.1 ("0-for-24")
- About 38% of CCF decline is due to rising childlessness; 62% is due to smaller families among parents
- Romania's Decree 770 caused a spike in period TFR but only a modest temporary increase in CCF

---

## 2. Methodology Notes

**Translation**: Stata → Python (numpy, pandas, matplotlib, statsmodels).

**Key translation decisions**:
- **WPP2024 loading**: Stata uses `import excel ... cellrange(A17:BM22000)` with header row 17. Python equivalent: `pd.read_excel(header=16)`.
- **Cohort alignment**: CCF data is aligned to calendar year by adding 30 to the birth cohort (matching age ~30 when cohort completes most fertility).
- **UN member state identification**: Stata manually flags 44 territories for exclusion. Python uses the same list to identify 193 UN member states.
- **Population fraction sub-2**: For each year, sum population of countries with TFR<2, divide by total population. Stata uses `egen total()` with `by(year)`; Python uses `groupby().sum()`.
- **HFD sub-national**: Five HFD entities (East/West Germany, England & Wales, Northern Ireland, Scotland) excluded from main analysis but included in footnote variant.
- **Kitagawa decomposition**: Following Stata exactly — `(total CCF change) - (parous CEB change * (1 - mean childlessness))` gives the childlessness component.

**Missing data**: India DHS (IAIR7EFL.DTA) requires DHS registration. Figure 5b, Table A1, and the India portion of Figure 4 could not be replicated.

---

## 3. Replication Results

### Figure 1: World births spike (1700-2300)
Births data from three sources (Spears et al. history, WPP 1950-2023, WPP medium projection 2024-2100, Spears et al. TFR→1.5 scenario) merged correctly. Pattern matches published figure: peak ~140 million births/year around 2015-2020, declining under all scenarios.

### Figure 2: World TFR and fraction below replacement

| Statistic | Published | Replication | Status |
|-----------|-----------|-------------|--------|
| World TFR 1950 | ~5 | 4.85 | Match |
| World TFR 2023 | ~2 | 2.25 | Match |
| Frac pop TFR<2 (2023) | "two-thirds" | 0.671 | Match |

### Figure 3: CCF vs TFR by country
31 country panels generated. Period TFR (solid) and CCF (dashed, lagged 30 years) plotted for 1960-2020. Visual patterns match: CCF tracks below period TFR during fertility decline (tempo effects).

### Figure 5a: CEB among parous (HFD)
R² from regressing CEB-among-parous on CCF: 0.80. Pattern confirms that even among parents, fertility falls with overall CCF.

### Figure 6: Latest vs Lowest CCF

| Statistic | Published | Replication | Status |
|-----------|-----------|-------------|--------|
| Countries ever below 1.9 | 24 | 24 | EXACT |
| Rebounds to >2.1 | 0 | 0 | EXACT |
| Footnote: ever below 1.9 (sub-national) | 26 | 26 | EXACT |
| Footnote: rebounds | 0 | 0 | EXACT |
| Countries where latest = lowest | — | 13/31 | — |

### Figure 7: Romania vs Bulgaria
Period TFR spike during Decree 770 (1966-1989) visible in Romania but not Bulgaria. CCF shows only modest elevation for Romania cohorts affected by the decree.

### Table 1: Kitagawa decomposition

| Statistic | Published | Replication | Status |
|-----------|-----------|-------------|--------|
| Countries with 20-year pairs | — | 19 | — |
| Mean later cohort year | — | 1976.9 | — |
| Mean earlier cohort year | — | 1956.9 | — |
| % due to childlessness | ~38% | 39.4% | Close |
| % due to parous fertility | ~62% | 60.6% | Close |

Note: The small difference (39.4% vs ~38%) may reflect rounding in the published text or minor differences in which country-cohort pairs are included.

### Table A2: Country-cohort details
19 countries with valid 20-year cohort pairs reproduced. All values match the Stata code's computation logic.

### Not replicated
- **Figure 4** (childlessness vs CCF scatter): India portion requires IAIR7EFL.DTA. HFD portion produced.
- **Figure 5b** (India district analysis): Requires IAIR7EFL.DTA.
- **Table A1** (India Kitagawa): Requires IAIR7EFL.DTA.

---

## 4. Data Audit Findings

### Coverage
- WPP: 237 countries/territories, 193 UN members, 1950-2023 (74 years). Zero missing TFR or population values.
- HFD CCF: 34 countries (excl. sub-national), 1,131 country-cohort observations, cohorts 1935-1980.
- HFD Parity: 707 country-cohort observations with childlessness data.
- Romania CCF: 41 observations, cohorts 1930-1970.

### Distributions
- TFR 2023: mean 2.34, median 1.95, range [0.66, 6.13].
- CCF: mean 1.960, median 1.933, range [1.353, 3.214].
- Childlessness: mean 11.4%, range [0.3%, 28.3%].

### Logical consistency
- Zero CCF values ≤ 0; zero childlessness outside [0, 100].
- CEB-among-parous always ≥ CCF (mathematically required).
- Zero duplicate observations in any dataset.
- Romania CCF is NOT monotone decreasing (cohorts affected by Decree 770 show higher CCF).

### Missing data patterns
- CCF series length varies: Sweden, Japan, Denmark, Finland have 46 cohorts; Slovenia has only 8.
- Parity data available for fewer cohorts than CCF.

---

## 5. Robustness Check Results

| # | Check | Finding | Status |
|---|-------|---------|--------|
| 1 | Alt TFR thresholds (1.8-2.3) | 43-76% of world pop sub-replacement at any threshold | Robust |
| 2 | Alt rebound thresholds | Zero rebounds for CCF<1.9 → >2.1; marginal at CCF<2.0 → >2.0 (6 cases) | Robust |
| 3 | Sub-national entities | 0 rebounds with 28 (all) or 26 (footnote) entities ever below 1.9 | Robust |
| 4 | Kitagawa with 15/20/25-year windows | Childlessness share: 36-41% | Robust |
| 5 | Leave-one-country-out | Zero rebounds in all 31 iterations | Robust |
| 6 | Cohort alignment age (25-35) | Correlation TFR-CCF: 0.73-0.86 | Robust |
| 7 | UN members vs all territories | Fractions differ by <0.002 | Robust |
| 8 | Regional TFR decline | All 9 regions declined; SSA least (-2.09) | Confirms paper |
| 9 | Alt TFR projections | Births in 2200: 36-99M depending on scenario | Confirms paper |
| 10 | Romania Decree 770 CCF | Cohort CCF barely elevated vs Bulgaria | Confirms paper |

### Key takeaways:
- The 0-for-24 result is rock-solid: zero rebounds under any reasonable threshold combination, leave-one-out, or entity definition.
- The Kitagawa decomposition is stable across cohort window choices (36-41% childlessness share).
- The paper's claims about population fractions are insensitive to territorial definitions.

---

## 6. Summary Assessment

**What replicates**: All key claims verified. The 0-for-24 statistic, the "two-thirds" population fraction, the Kitagawa decomposition, the Romania comparison, and all figure patterns reproduce correctly.

**What doesn't replicate**: Nothing failed. Three outputs (Figure 5b, Table A1, India portion of Figure 4) were not attempted due to India DHS data requiring registration.

**Key concerns**: None. This is a clean descriptive paper using publicly available demographic data. The analysis is straightforward and the conclusions follow directly from the data.

**Assessment**: This is a strong replication package. The data sources are well-documented, the Stata code is clear, and all results reproduce in Python. The paper's main claim — that cohort fertility has not rebounded where it has fallen — is a simple empirical fact confirmed exactly by the data.

---

## 7. File Manifest

| File | Purpose |
|------|---------|
| `replication_239496/utils.py` | Shared paths, data loaders, country codes |
| `replication_239496/01_clean.py` | Load and validate all data sources |
| `replication_239496/02_tables.py` | Reproduce Table 1 (Kitagawa) and Table A2 |
| `replication_239496/03_figures.py` | Reproduce Figures 1-3, 5a, 6, 7 |
| `replication_239496/04_data_audit.py` | Coverage, distributions, consistency, missing patterns |
| `replication_239496/05_robustness.py` | 10 robustness checks |
| `replication_239496/output/Figure1_births_spike.png` | World births 1700-2300 |
| `replication_239496/output/Figure2_TFR_sub2.png` | World TFR and fraction below 2 |
| `replication_239496/output/Figure3_CCF_vs_TFR.png` | CCF vs TFR by country (31 panels) |
| `replication_239496/output/Figure5a_CEB_parous.png` | CEB among parous vs CCF |
| `replication_239496/output/Figure6_latest_vs_lowest.png` | Latest vs lowest CCF scatter |
| `replication_239496/output/Figure7_Romania_Bulgaria.png` | Romania vs Bulgaria TFR and CCF |
