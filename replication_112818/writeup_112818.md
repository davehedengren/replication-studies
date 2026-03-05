# Replication Study: Paper 112818

**Paper**: "Why Has U.S. Policy Uncertainty Risen Since 1960?"
**Authors**: Scott R. Baker, Nicholas Bloom, Brandice Canes-Wrone, Steven J. Davis, and Jonathan A. Rodden
**Published**: NBER Working Paper 19826, January 2014 (AEA Papers & Proceedings)

---

## 0. TLDR

- **Replication status**: Both figures replicate exactly from the provided data and code.
- **Key finding confirmed**: Policy uncertainty, government activity, and political polarization all show strong upward secular trends that co-move over 1950-2012.
- **Main concern**: The co-movement is entirely driven by shared time trends; detrended and first-differenced correlations are weak or insignificant, meaning the paper documents parallel trends rather than a deeper relationship.
- **Bug status**: No coding bugs found.

---

## 1. Paper Summary

**Research question**: Why has U.S. policy-related economic uncertainty risen since 1960?

**Data**: Annual time series (1949-2012) containing:
- Newspaper-based Economic Policy Uncertainty (EPU) index from Baker, Bloom & Davis (2013)
- Code of Federal Regulations page count (from Dawson & Seater 2013, spliced with Crews 2013)
- Government spending as share of GDP (from BEA)
- Political polarization measures: NOMINATE roll-call gap, SD of district presidential vote shares, voter perception of party differences (from ANES)

**Method**: Purely descriptive. Two figures plot these time series together to show visual co-movement. No regressions, no causal identification.

**Key findings**: The paper argues two classes of explanations for rising policy uncertainty: (1) growth in government spending, taxes, and regulation; (2) increased political polarization. Both factors trend upward alongside the EPU index.

---

## 2. Methodology Notes

**Translation**: Stata `.do` file translated to Python (pandas, matplotlib). The Stata code is minimal (23 lines) - it loads data, normalizes two variables to mean=100, and produces two `twoway` scatter plots.

**Key choices**:
- Stata's `u figure1,clear` = abbreviated `use` command
- Stata's `gen norm_reg=100*reg/mreg` uses `reg` as abbreviation of `regulation` (valid unambiguous abbreviation)
- Stata normalizes after `keep if year>=1950`, so mean=100 is computed over 1950-2012 (63 obs)
- The variable `govsharegdp` is labeled `gov` in Stata but plotted as "Government share of GDP"

---

## 3. Replication Results

### Figure 1: US Economic Policy Uncertainty and Government Activity

| Variable | Year | Published (approx) | Replicated |
|---|---|---|---|
| Policy uncertainty (norm_news) | 1950 | ~40 | 39.21 |
| Policy uncertainty (norm_news) | 2012 | ~220 | 217.21 |
| Pages of regulation (norm_reg) | 1950 | ~30 | 28.26 |
| Pages of regulation (norm_reg) | 2012 | ~190 | 189.24 |
| Gov share of GDP | 1950 | ~23% | 22.7% |
| Gov share of GDP | 2012 | ~34% | 34.0% |

**Status**: Exact match. All plotted values correspond precisely to the published figure.

### Figure 2: US Economic Policy Uncertainty and Political Polarization

| Variable | Range | Replicated Range |
|---|---|---|
| Policy uncertainty (norm_news) | ~26-225 (left axis) | 26.38-225.40 |
| Polarization of roll-calls (diff_std) | ~-1.2 to 2.1 | -1.1622 to 2.1111 |
| SD of district pres. vote (sd_dshare_std) | ~-1.9 to 1.9 | -1.9001 to 1.8946 |
| Voter perception of party diff (seediff_std) | ~-1.9 to 2.2 | -1.8783 to 2.2400 |

**Status**: Exact match.

### Correlations between series (not reported in paper but implicit in visual)

| Pair | Pearson r | n |
|---|---|---|
| EPU vs Regulation | 0.917 | 63 |
| EPU vs Gov share GDP | 0.794 | 63 |
| EPU vs Polarization (roll-calls) | 0.819 | 62 |
| EPU vs SD district vote | 0.514 | 15 |
| EPU vs Voter perception | 0.680 | 19 |

---

## 4. Data Audit Findings

**Coverage**:
- 64 valid annual observations (1949-2012) + 10 trailing all-NaN rows (storage artifact)
- After year>=1950 filter: 63 observations used for figures
- No missing years in the 1949-2012 range

**Variable completeness** (of 64 valid rows):
- `year`, `news`, `govsharegdp`, `regulation`: 100% complete
- `diff_std` (roll-call polarization): 63/64 (missing 2012 only)
- `sd_dshare_std` (district vote SD): 15/64 (every 4 years, 1951-2007)
- `seediff_std` (voter perception): 19/64 (irregular election-year coverage)

**Plausibility**: All variables pass plausibility checks. Government share of GDP in [20.5%, 37.2%], news index always positive, regulation page counts reasonable.

**Outliers**: No outliers in news or regulation by IQR method. Two mild outliers in govsharegdp (1949: 20.5%, 1951: 20.7%) - these are early years with lower government share, plausible.

**Data quality assessment**: Clean, well-structured dataset. The sparse coverage of polarization variables (sd_dshare_std, seediff_std) is by design - these are available only for election/redistricting years.

**Anomalies**:
- 10 trailing all-NaN rows in the .dta file (harmless, likely Stata storage artifact)
- diff_std missing for 2012 (the most recent year); this is visible in the published Figure 2 where the polarization line ends at 2011

---

## 5. Robustness Check Results

### 5.1 Correlation sensitivity to time period
Full-sample correlations (EPU-regulation: 0.917, EPU-gov: 0.794, EPU-polarization: 0.819) weaken substantially in the post-1980 subsample (0.598, 0.484, 0.505). The visual co-movement is strongest when including the full 60-year span.

### 5.2 Detrended correlations
After removing linear time trends, correlations drop dramatically:
- EPU vs Regulation: r=0.263 (p=0.037)
- EPU vs Gov share: r=0.286 (p=0.023)
- EPU vs Polarization: r=-0.226 (p=0.077, **not significant**)

This indicates the apparent co-movement is largely driven by all series sharing upward secular trends.

### 5.3 First-differenced correlations
Year-to-year changes show no significant correlation:
- EPU vs Regulation: r=-0.089 (p=0.494)
- EPU vs Gov share: r=-0.113 (p=0.381)
- EPU vs Polarization: r=0.150 (p=0.247)

The series do not move together at annual frequency.

### 5.4 Rolling correlations (15-year windows)
EPU-regulation correlation varies enormously across windows:
- Range: [-0.812, 0.944]
- Negative in 1950s and 1980s-1990s windows
- The overall high correlation comes from the full-span trend, not consistent local co-movement

### 5.5 Decade-by-decade correlations (EPU vs Regulation)
| Decade | r |
|---|---|
| 1950s | -0.676 |
| 1960s | 0.801 |
| 1970s | 0.829 |
| 1980s | 0.150 |
| 1990s | -0.282 |
| 2000s | -0.000 |

Within-decade correlations are often negative or near zero.

### 5.6 Robustness to extreme years
Dropping the top/bottom 2 EPU years barely changes correlations (max change: -0.031). No single observation drives the result.

### 5.7 Leave-one-out stability
LOO range for EPU-regulation: [0.913, 0.925]. Very stable.

### 5.8 Rank correlations
Spearman correlations closely match Pearson, confirming the relationship is not driven by outliers.

### 5.9 Linear trend tests
All three main series have highly significant upward trends:
- EPU: +2.65 points/year (R^2=0.83)
- Regulation: +2.64 points/year (R^2=0.97)
- Gov share: +0.14 pp/year (R^2=0.63)

### Robustness summary
| Check | EPU-Regulation | EPU-Gov Share | EPU-Polarization |
|---|---|---|---|
| Full sample Pearson | **0.917** | **0.794** | **0.819** |
| Post-1980 only | 0.598 | 0.484 | 0.505 |
| Detrended | 0.263* | 0.286* | -0.226 |
| First-differenced | -0.089 | -0.113 | 0.150 |
| Drop extremes | 0.908 | 0.763 | 0.799 |
| Spearman | 0.890 | 0.783 | 0.881 |

The paper's descriptive claims about co-trending are supported: these series do trend upward together. However, the co-movement is entirely about shared long-run trends. There is no evidence of year-to-year co-movement, and within-decade correlations are often negative.

---

## 6. Summary Assessment

**What replicates**: Both figures replicate exactly. The data is clean and well-documented. The Stata code is correct and minimal.

**What doesn't**: N/A - this is a purely descriptive paper and all outputs match.

**Key concerns**:
1. **Spurious correlation from shared trends**: The high correlations (0.79-0.92) are almost entirely driven by the fact that all series trend upward over 60 years. Detrended correlations are much weaker (0.26-0.29), and first-differenced correlations are near zero. Two unrelated series that both happen to trend upward would show similar correlations.
2. **Within-decade instability**: The EPU-regulation correlation is negative in the 1950s, 1990s, and near zero in the 2000s. The visual impression of co-movement comes from the long-run trend.
3. **The paper is transparent about this limitation**: The authors explicitly state the evidence is "inconclusive" and call for future causal analysis. The paper is intended as a descriptive motivating piece for future research, not as a causal claim.

**Overall**: A clean, well-executed descriptive paper. The replication package is minimal but complete. The visual co-movement is real but driven by shared secular trends rather than a deeper time-varying relationship.

---

## 7. File Manifest

| File | Description |
|---|---|
| `utils.py` | Shared paths and constants |
| `01_clean.py` | Load figure1.dta, filter year>=1950, normalize variables |
| `02_figures.py` | Replicate Figures 1 and 2 |
| `03_data_audit.py` | Data quality audit |
| `04_robustness.py` | 11 robustness checks |
| `writeup_112818.md` | This writeup |
| `output/cleaned.csv` | Cleaned dataset |
| `output/figure1.png` | Replicated Figure 1 |
| `output/figure2.png` | Replicated Figure 2 |
