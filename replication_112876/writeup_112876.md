# Replication Study: Paper 112876

**Paper**: "The Anatomy of a Credit Crisis: The Boom and Bust in Farm Land Prices in the United States in the 1920s"
**Authors**: Raghuram G. Rajan and Rodney Ramcharan
**Source**: American Economic Review, 2015
**Replication by**: Claude (automated replication pipeline)

---

## 0. TLDR

- **Replication status**: All regression coefficients across Tables 5A, 5B, 6, 9, 10, 11, 12, and Figure 5 replicate exactly (within rounding tolerance).
- **Key finding confirmed**: More banks in 1920 led to higher land prices, which subsequently reversed in the 1930s bust. Bank suspensions in the 1920s had lasting effects on local banking structure through at least 1972.
- **Main concern**: The coefficient on log banks drops 61% (0.60 to 0.23) when controls are added, and the panel specification becomes fragile when dropping the 1900 year (t falls from 2.48 to 0.58). The IV estimate exceeds OLS by 39%, which is substantial.
- **Bug status**: No coding bugs found.

---

## 1. Paper Summary

**Research question**: Did the expansion of the U.S. banking system in the early 1920s inflate farmland prices, and did the subsequent credit contraction cause a persistent collapse in land values and banking structure?

**Data**:
- County-level cross-sectional data (~3,300 counties) from the 1920 Census of Agriculture
- Panel data (1900, 1910, 1920) for county-level land prices and banking
- Bank suspension data 1921-1929 from the Federal Reserve
- Long-run land prices (1930-1960) and banking structure (1972)

**Method**: OLS cross-sectional regressions with state fixed effects and state-clustered standard errors. Panel specifications with county FE (absorbed) and year FE. IV using 1910 bank counts as instruments for 1920 banks. Spatial analysis using distance bands from state borders.

**Key findings**:
1. More banks in a county in 1920 predicted higher land prices (Table 5A)
2. The relationship is causal: IV using 1910 banks gives even larger estimates (Table 5B)
3. Credit expansion operated through extensive margin (more acres) and intensive margin (more investment) (Table 9)
4. The boom reversed: counties with more banks in 1920 saw larger price declines in 1930s (Table 11)
5. Bank suspensions had persistent effects on banking structure through 1972 (Table 12)

---

## 2. Methodology Notes

**Translation choices**:
- Stata `xi: reg Y X i.state, cluster(state)` translated using `pd.get_dummies(drop_first=True)` for state FE + `statsmodels` `cov_type='cluster'`
- Stata `areg Y X, absorb(fips) cluster(statename)` translated using Frisch-Waugh-Lovell demeaning with manual cluster-robust variance
- Stata `ivreg2` translated using `statsmodels` IV2SLS with manual cluster-robust SEs
- DOF correction for cluster SEs: `(G/(G-1)) * ((N-1)/(N-K))`

**Variable mapping**:
- `state` variable is numeric in table_5A, table_11 but `statename` (string) in table_9, table_10, table_12
- Control variable globals match across do files: `$win_geo_log`, `$win_dem_log`, `$win_oth2`, `$win_oth`

---

## 3. Replication Results

### Table 5A: Land Prices and Credit Availability (Cross-section)

| Column | Key Variable | Published | Replicated | Match? |
|--------|-------------|:---------:|:----------:|:------:|
| 1 | win_banks_l (no controls) | 0.603 (17.2) | 0.6026 (17.17) | Yes |
| 2 | win_banks_l (full controls) | 0.234 (6.04) | 0.2338 (6.04) | Yes |
| 3 | win_banks_l (drop mfg outliers) | 0.216 (5.56) | 0.2155 (5.56) | Yes |
| 4 | a (banks/area) | 18.6 (4.23) | 18.6330 (4.23) | Yes |
| 5 | p (banks/capita) | 332.2 (5.73) | 332.2237 (5.73) | Yes |

### Table 5B: Panel and IV Estimates

| Column | Key Variable | Published | Replicated | Match? |
|--------|-------------|:---------:|:----------:|:------:|
| 1 | l (panel FE) | 0.0925 (2.48) | 0.0925 (2.48) | Yes |
| 2 | l_1910 (OLS) | 0.277 (7.30) | 0.2765 (7.30) | Yes |
| 3 | l_1920 (IV) | 0.587 (7.40) | 0.5866 (7.40) | Yes |

### Table 9: Channels (Selected)

| Column | l coef | l_sq coef | Match? |
|--------|:------:|:---------:|:------:|
| 1 (Price change) | 0.2152 (4.36) | -0.0411 (-3.84) | Yes |
| 4 (Investment) | 0.3980 (8.07) | -0.0624 (-5.61) | Yes |
| 5 (Debt/acre) | 0.2592 (3.13) | -0.0361 (-1.68) | Yes |

### Table 11: Long-run Persistence (Selected)

| Column | l coef | Match? |
|--------|:------:|:------:|
| 1 (1930-1920) | -0.2349 (-4.60) | Yes |
| 3 (1950-1940) | -0.5316 (-4.28) | Yes |
| 5 (1960 level) | -0.4640 (-7.28) | Yes |

### Table 12: Long-run Outcomes

| Column | win_banks_susp2129 | Match? |
|--------|:------------------:|:------:|
| 1 (Banks 1972) | -1.2833 (-3.10) | Yes |
| 2 (HHI 1972) | 0.3280 (1.75) | Yes |
| 3 (Price 1960, OLS) | -1.4853 (-4.31) | Yes |
| 4 (Price 1960, IV) | -8.1263 (-2.09) | Yes |

**Verdict**: All coefficients and t-statistics match within rounding precision across all replicated tables. **EXACT REPLICATION**.

---

## 4. Data Audit Findings

**Coverage**:
- Table 5A: 3,337 obs (3,336 unique counties, 49 states), matching paper's county-level cross-section
- Table 5B: 9,945 obs (3,336 counties x 3 years: 1900, 1910, 1920), 50 states
- Table 9-12: ~3,336 counties (1920 cross-section)
- Figure 5: 2,582 obs (48 states)

**Panel balance (Table 5B)**:
- Balanced: 3,304 counties appear in all 3 years (1900, 1910, 1920)
- 32 counties have fewer than 3 observations

**Missing data**:
- Moderate and variable: 7-15% missing across geographic and demographic controls in Table 5A
- Debt variables have higher missingness (~27% for win_debt_acre_2010 in Table 9)
- Effective sample drops from 3,337 to ~2,744 in main specification due to listwise deletion

**Logical consistency**:
- l_sq = l^2 verified exactly (max diff = 0.000000)
- Winsorized variables confirmed: win_banks_l has 0 obs outside [p1, p99]
- win_landval_update_ppa_log has 32 obs above p99, suggesting asymmetric or partial winsorization

**No duplicates** in any dataset.

---

## 5. Robustness Check Results

### Results that survive all checks

**Banks-to-land-price relationship (Table 5A Col 2)**:
- Leave-one-state-out: coef range [0.205, 0.244] with t-range [5.50, 6.77] — no sign changes, always significant
- Cook's D outlier removal: coef increases to 0.254 (t=7.89) — strengthens
- HC1 robust SEs: t-stat increases from 6.04 to 10.90 (cluster SEs are 1.8x HC1)
- Manufacturing crop cutoffs (p90 to no restriction): coef range [0.214, 0.234] — stable
- Dropping small states: coef=0.251 (t=6.43) — stable
- Geographic subsamples: significant in both high and low rainfall areas

**Long-run reversal (Table 11)**:
- 1930-1920 decline: l coef = -0.235 (t=-4.60) — highly significant
- 1950-1940 decline: l coef = -0.532 (t=-4.28) — very large
- Cumulative through 1960: l coef = -0.464 (t=-7.28)

**Bank suspension persistence (Table 12)**:
- Effect on 1972 banking: coef=-1.283 (t=-3.10) — robust
- IV magnifies substantially: -8.13 (t=-2.09), IV/OLS ratio ~5.5

### Results that are fragile

**Panel specification (Table 5B Col 1)**:
- Baseline: l = 0.0925 (t=2.48), barely significant at 5%
- Drop year 1900: l = 0.036 (t=0.58) — **insignificant**
- Drop year 1910: l = 0.080 (t=1.54) — **insignificant**
- The panel result depends on having all three census years; with only two periods, within-county variation is insufficient

**Coefficient stability**:
- No controls: 0.603
- Full controls: 0.234
- 61% reduction — the raw correlation between banks and land prices is largely explained by observable county characteristics
- This doesn't invalidate causality (the IV confirms it), but the OLS coefficient is heavily confounded

**IV magnitude**:
- IV/OLS ratio = 1.39 (0.587 vs 0.421 for OLS on 1920 banks)
- A ratio substantially above 1 could indicate: (a) attenuation bias from measurement error in OLS, (b) LATE > ATE if compliers have larger effects, or (c) weak instrument concerns

---

## 6. Summary Assessment

### What replicates
All published regression results across 8 tables replicate exactly from the provided code and data. The replication package is well-organized with separate datasets per table and clean Stata do files.

### Key concerns

1. **Coefficient instability with controls**: The bank coefficient drops 61% when controls are added (0.60 → 0.23), suggesting substantial confounding between bank presence and county characteristics that independently predict land values.

2. **Fragile panel estimates**: The within-county panel estimate (Table 5B Col 1) becomes insignificant when dropping either the 1900 or 1910 census year. With only 3 time periods and a balanced panel, within-county variation is limited.

3. **Large IV premium**: The IV estimate exceeds OLS by 39%. While this could reflect attenuation bias correction, a large IV premium in a cross-sectional setting with ~45 clusters warrants scrutiny of instrument strength and exclusion restriction.

4. **Cluster SEs are conservative**: State-clustered SEs are 1.8x HC1 SEs, reflecting substantial within-state correlation. With only 45-49 state clusters, cluster SE asymptotics may be strained.

5. **Missing data**: 15-27% missingness in some variables reduces effective sample by ~18% (3,337 to 2,744), though results are stable across winsorization thresholds.

### Overall assessment
The cross-sectional relationship between banking density and land values in 1920 is robust and well-documented. The long-run reversal (1930s price decline and persistent banking structure effects through 1972) is strongly supported. The identification strategy through IV (1910 banks → 1920 banks → land values) is reasonable but the large IV premium deserves careful interpretation. The panel specification, while conceptually important for within-county identification, is fragile to sample composition. The paper's contribution as a historical analysis of credit-fueled asset price booms is well-supported by the data.

---

## 7. File Manifest

| File | Description |
|------|-------------|
| `utils.py` | Shared paths, data loading, `run_reg_cluster()`, `run_areg_cluster()`, `run_ivreg2_cluster()` |
| `02_tables.py` | Replication of Tables 5A, 5B, 6, 9, 10, 11, 12 and Figure 5 |
| `04_data_audit.py` | Comprehensive data audit across all 8 datasets |
| `05_robustness.py` | 12 robustness checks including LOO, Cook's D, HC1, panel sensitivity |
| `writeup_112876.md` | This writeup |
