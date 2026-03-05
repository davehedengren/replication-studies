# Replication Study: Paper 112786

**Paper**: "Rainfall Forecasts, Weather and Wages over the Agricultural Production Cycle"
**Authors**: Mark R. Rosenzweig and Christopher Udry
**Source**: NBER Working Paper 19808, January 2014
**Replication by**: Claude (automated replication pipeline)

---

## 0. TLDR

- **Replication status**: All regression coefficients in Tables 2 and 3 replicate exactly (within rounding tolerance). Figure 1 cannot be precisely replicated due to missing year variable.
- **Key finding confirmed**: Rainfall forecasts reduce planting-stage migration and wages; the forecast-bad rainfall interaction on harvest wages replicates in coefficient and t-statistic.
- **Main concern**: Several harvest-stage results (forecast x bad rainfall shock, NREGA effects) lose statistical significance with robust standard errors or when restricting to districts closer to weather stations. The migration result is driven by a single village.
- **Bug status**: No coding bugs found.

---

## 1. Paper Summary

**Research question**: How do IMD long-range monsoon rainfall forecasts and realized rainfall affect equilibrium agricultural wages at different stages (planting vs harvest) of the Indian crop cycle?

**Data**:
- ICRISAT Village Level Survey (VLS): Household panel from 6 villages (2005-2011, excluding 2007-2008), tracking planting-stage migration decisions
- District-level male agricultural wages by month/activity from the Directorate of Economics and Statistics (2005-2010), 106 districts for planting wages, 95 for harvest
- District-level monthly rainfall from Terrestrial Air Temperature and Precipitation (Wilmott-Matsuura), matched to districts

**Method**: Fixed-effects OLS regressions (`areg` in Stata). Village fixed effects for migration equations, district fixed effects for wage equations. Conventional (non-robust) standard errors.

**Key findings**:
1. A good rainfall forecast reduces planting-stage out-migration, especially for men (Table 2)
2. Higher forecast reduces planting-stage wages (Table 3, planting column)
3. A good forecast exacerbates the negative effect of bad realized rainfall on harvest wages (Table 3, harvest column)
4. NREGA implementation raises planting wages and moderates harvest-wage volatility

---

## 2. Methodology Notes

**Translation choices**:
- Stata `areg y X, a(group)` translated using Frisch-Waugh-Lovell demeaning with DOF correction: DOF = N - K - n_groups
- Conventional (non-robust) standard errors, matching Stata's default for `areg` without `,r`
- The Stata variable `ed` maps to `edu` in the dataset (Stata allows abbreviations)

**Reparameterization in Table 3 harvest column**:
The data variable `foregood = fore * I(jsdev >= 1)` (forecast when good rainfall), but the paper reports the equivalent reparameterization using `fore` and `fore * I(bad rainfall)`. The algebra:
- Paper's "Forecast" coef = code's `fore` coef + code's `foregood` coef = -0.000194 + 0.001092 = **0.000898** (matches)
- Paper's "Forecast x bad rainfall shock" coef = -code's `foregood` coef = **-0.00109** (matches)

This is mathematically equivalent; the regression is identical.

---

## 3. Replication Results

### Table 2: Migration Regressions (Village FE)

| Variable | Published (All) | Replicated (All) | Published (Males) | Replicated (Males) | Published (Females) | Replicated (Females) |
|----------|:---:|:---:|:---:|:---:|:---:|:---:|
| Forecast (sp) | -.00293 (2.69) | -.00293 (2.69) | -.00485 (2.68) | -.00485 (2.68) | -.000332 (0.33) | -.000332 (0.33) |
| Age | .000853 (1.32) | .000854 (1.32) | .00141 (1.32) | .00141 (1.32) | -.000195 (0.32) | -.000195 (0.32) |
| Age² | -.0000194 (2.38) | -.0000194 (2.38) | -.0000314 (2.28) | -.0000314 (2.28) | -.0000683 (0.91) | -.0000068 (0.91) |
| Education | .00711 (10.0) | .00711 (10.0) | .0118 (10.9) | .01182 (10.9) | -.00184 (2.47) | -.00184 (2.47) |
| Male | .0557 (9.58) | .05567 (9.58) | - | - | - | - |
| N | 6,501 | 6,501 | 3,507 | 3,507 | 2,994 | 2,994 |

**Verdict**: All coefficients and t-statistics match within rounding precision. **EXACT REPLICATION**.

### Table 3: Wage Regressions (District FE)

| Variable | Published (Planting) | Replicated (Planting) | Published (Harvest) | Replicated (Harvest) |
|----------|:---:|:---:|:---:|:---:|
| Rainfall shock | .192 (3.09) | .1915 (3.09) | .056 (0.73) | .0560 (0.73) |
| Rainfall shock x distance | -.00196 (2.56) | -.00196 (2.56) | -.00152 (1.72) | -.00152 (1.72) |
| Forecast | -.00622 (2.71) | -.00622 (2.71) | .000898 (0.30) | .000898 (0.30)* |
| Forecast x bad rainfall shock | - | - | -.00109 (2.09) | -.00109 (2.09)* |
| NREGA in place | .096 (4.50) | .096 (4.50) | .0623 (1.76) | .0623 (1.76) |
| NREGA x bad rainfall shock | - | - | .0929 (1.78) | .0929 (1.78) |
| N | 387 | 387 | 337 | 337 |

*Harvest "Forecast" and "Forecast x bad" are recovered via reparameterization (see Section 2).

**Verdict**: All coefficients and t-statistics match within rounding precision. **EXACT REPLICATION**.

### Figure 1: Ratio of Harvest to Planting Wages

The datasets do not include an explicit year variable. Since IMD forecasts are area-specific (not national), multiple forecast values correspond to each year, making exact year recovery impossible from the provided data alone. The figure cannot be precisely replicated from the replication package.

---

## 4. Data Audit Findings

**Coverage**:
- Sample sizes exactly match paper: 337 harvest obs (95 districts), 387 planting obs (106 districts), 6,501 migration obs (6 villages)
- No missing values in any dataset

**Panel balance**:
- Highly unbalanced wage panel: districts have 1-6 observations, with many having just 1-2 (18 harvest districts, 19 planting districts have only 1 obs)
- 92 districts appear in both wage datasets; 3 harvest-only, 14 planting-only

**Variable consistency**:
- All constructed variables verified: agesq = age², foregood = fore * I(jsdev >= 1), postbad = post * I(jsdev < 1)
- Distance to weather station is constant within districts (as expected)

**Migration heterogeneity across villages**:
- Migration rates vary enormously: Village 1 at 11.2% vs Village 6 at 0.1%
- Village size varies 3x (552 to 1,682 obs)

**Rainfall shocks**:
- 6 observations with jsdev < 0.5 (less than half the 60-year mean) - quite extreme
- 47.8% of observations classified as "bad" rainfall shock (jsdev < 1)

**No year variable**: None of the three datasets includes an explicit year identifier, preventing temporal analysis and making Figure 1 non-replicable.

---

## 5. Robustness Check Results

### Results that survive all checks

**Planting-stage wage**: The rainfall shock effect (jsdev coef ~0.19, |t|~3.1) and forecast effect (fore coef ~-0.006, |t|~2.7) are highly robust. They survive:
- HC1 robust SEs (|t| actually increases slightly)
- Dropping extreme rainfall (|t|=2.98)
- Winsorizing wages (<3% change)
- Leave-one-district-out (no sign changes, max 12% change)
- Permutation test (p=0.005)
- Level wage specification (same signs, similar significance)
- Distance restriction to 100 miles (|t|=2.01 for jsdev, |t|=2.62 for fore)

**NREGA on planting wages**: Very robust, |t|=4.5 baseline, 4.1 with HC1.

### Results that are fragile

**Harvest: Forecast x bad rainfall shock** (foregood, the paper's key interaction):
- Baseline |t|=2.09 (barely significant at 5%)
- HC1 robust SEs: |t|=1.76 (loses significance at 5%)
- Distance restriction to 100 miles: |t|=1.27 (insignificant)
- Alternative bad threshold (jsdev < 0.9): |t|=0.56 (insignificant)
- This result is fragile to the specific definition of "bad" rainfall and to SE correction

**Harvest: Other borderline results**:
- NREGA (post): |t|=1.76 baseline, drops to 1.31 with HC1
- NREGA x bad (postbad): |t|=1.78 baseline, drops to 1.53 with HC1
- Rainfall shock x distance (jsdevmiles): |t|=1.72 baseline, drops to 1.42 with HC1

**Migration: village sensitivity**:
- Dropping village 2 makes forecast effect insignificant (|t|=0.66 vs 2.69 baseline)
- The migration-forecast result is driven substantially by one village
- Concentrated in large villages (|t|=2.61) vs small (|t|=0.59)
- Permutation test still significant (p=0.003), but this doesn't address the village-concentration issue

---

## 6. Summary Assessment

### What replicates
All published regression results (Tables 2 and 3) replicate exactly from the provided code and data. The replication package is clean, minimal, and functional.

### Key concerns

1. **No robust standard errors**: The paper uses conventional (homoskedastic) SEs throughout. With HC1 robust SEs, the key harvest-stage interaction (forecast x bad rainfall) drops below the 5% significance threshold (|t| from 2.09 to 1.76). Several other harvest-stage results also lose significance.

2. **Migration driven by one village**: The forecast-migration relationship (Table 2) is heavily driven by a single village. Dropping village 2 reduces |t| from 2.69 to 0.66. With only 6 villages providing village-level variation in the forecast variable, this is a substantial concentration of influence.

3. **Sensitive to bad rainfall definition**: The harvest-stage interaction relies on a binary "bad" indicator at jsdev < 1.0. Moving this threshold to 0.9 eliminates the significance of both the forecast x bad and NREGA x bad interactions.

4. **Missing year variable**: The replication package does not include a year identifier, preventing Figure 1 replication and limiting the ability to conduct temporal robustness checks.

5. **Unbalanced panel**: Many districts appear only 1-2 times (out of 6 possible years), limiting within-district variation for the fixed-effects estimator.

### Overall assessment
The planting-stage results are robust and well-identified. The harvest-stage results, which involve the theoretically more novel forecast-rainfall interaction, are fragile - they depend on the use of conventional SEs, the specific bad-rainfall cutoff, and the full sample of districts. The migration result is driven by a single village. These concerns do not invalidate the paper's theoretical contribution but suggest the empirical evidence for the harvest-stage channel is weaker than the paper implies.

---

## 7. File Manifest

| File | Description |
|------|-------------|
| `utils.py` | Shared paths, data loading, `run_areg()` FE regression function |
| `01_clean.py` | Data loading and Table 1 descriptive statistics verification |
| `02_tables.py` | Replication of Tables 2 and 3 with comparison to published values |
| `03_figures.py` | Attempted replication of Figure 1 (limited by missing year variable) |
| `04_data_audit.py` | Comprehensive data audit across all three datasets |
| `05_robustness.py` | 11 robustness checks including HC1 SEs, LOO, permutation tests |
| `writeup_112786.md` | This writeup |
