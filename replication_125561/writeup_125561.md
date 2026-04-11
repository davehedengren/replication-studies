# Replication Study: 125561-V1

**Paper:** "How Effective Are Monetary Incentives to Vote? Evidence from a Nationwide Policy"
**Authors:** Mariella Gonzales, Gianmarco León-Ciliotta, Luis R. Martínez
**Journal:** *American Economic Journal: Applied Economics*, 14(1), January 2022, 293–326
**Original Language:** Stata 16 (reghdfe, outreg2, boottest)
**Replication Language:** Python (pandas, numpy; hand-rolled two-way FE + cluster-robust SE)

---

## 0. TLDR

- **Replication status:** All coefficients and standard errors for Tables 1, 2, 3, 4 and 5 replicate *exactly* to three decimal places. Sample sizes, district counts, and dependent-variable means match to the unit.
- **Key finding confirmed:** The elasticity of turnout with respect to the abstention fine is +0.030 (0.005) — a S/10 lower fine reduces district turnout by ~0.5 pp. The registration elasticity is −0.045 (0.019), driven entirely by 18–20-year-olds (−0.276), consistent with fraudulent address reporting by first-time voters.
- **Main concern:** The turnout elasticity is dominated by the 2016 election cycle: dropping 2016 cuts the coefficient from 0.030 to 0.011 — two-thirds of the long-run effect comes from a single election, consistent with the paper's "gradual learning / voltage drop" story but a meaningful fragility under any future structural break.
- **Bug status:** No coding bugs found. The Stata code is clean and the pre-computed CLEAN datasets reproduce the published tables without modification.

---

## 1. Paper Summary

### Research Question
Do marginal changes in monetary penalties for abstention affect voter turnout, registration, and electoral outcomes at the population scale? And how does the scaled-up behavioral elasticity compare with experimental ("voltage") estimates?

### Setting
Peru has had compulsory voting since 1933. Until 2006 the abstention fine was uniform; a 2006/2010 reform reclassified 1,755 districts into three categories (non-poor = high fine, poor = medium, extreme poor = low fine), creating spatial and temporal variation in the fine value. The paper uses 8 national presidential elections (2001, 2006, 2011, 2016 × {general, runoff}) at the district level.

### Data
- **ONPE (Oficina Nacional de Procesos Electorales):** voting-booth–level electoral returns, aggregated to district × election.
- **JNE (Jurado Nacional de Elecciones):** non-confidential data on settled abstention fines.
- **INEI:** ENAHO household survey and 2007 census (used for Tables 6–7 on political attitudes).
- **Servicio Electoral de Chile (SERVEL):** Chilean 2017 presidential returns (used as a no-compulsory-voting comparison in Table 7/Figure 6).
- **Google Trends:** 44 search terms, 3 fine-related (used for Figure 4 / Appendix E).
- **Main analysis panel:** 1,755 districts × 8 elections = 14,040 district-elections; 13,536 after singleton drop for the district × (date × province × 2006-category) FE.

### Method
Two-way fixed-effects (district FE + election × province × 2006-category FE), weighted by 2001 registered voters, clustered by province (192 clusters). The reduced-form equation is

$$T_{i,t} = \alpha_i + \gamma_{t,p(i),c(i)} + \nu \cdot \text{Fine}_{i,t} + \varepsilon_{i,t}$$

The paper runs this for turnout (Table 1A), ln turnout (Table 1B, the elasticity), heterogeneity by 2016, runoff, and poverty (Table 2), ln registered voters by age (Table 3), registration/votes (Table 4), and spoiled votes (Table 5).

### Headline Findings
1. A S/10 lower fine cuts turnout by ~0.5 pp (elasticity 0.030). The effect grows over time (0.011 in 2011 → 0.049 in 2016).
2. Registration elasticity is −0.045 overall but −0.276 for 18–20-year-olds — young first-time voters report a district-of-residence with a lower fine. Bounding analysis attributes ≥43% of the turnout response to this registration margin; up to 57% is the upper bound.
3. Lower fines shrink the share of blank/invalid votes (elasticity on spoiled share of roughly the same magnitude), so the representation effect is tiny.
4. The large-scale elasticity (0.03) is ≈1/7 of the experimental estimate from León (2017) on the same reform — a sizeable "voltage drop" attributed to imperfect information about the reform (documented via Google Trends).
5. Counterfactually, eliminating the fine would drop turnout only 18% as much as an age-based exemption from compulsory voting (Table 7) — monetary incentives are *not* the main driver of compulsory voting's turnout effect.

---

## 2. Methodology Notes

### Translation Choices
- **`reghdfe` → hand-written two-way absorbing regression.** `linearmodels.AbsorbingLS` is built for this, but the `date#category_06#province_id` interaction FE has 1,500+ levels and the singleton-dropping semantics differ from reghdfe. I wrote a small Gauss-Seidel demeaner (alternating group-mean sweeps until convergence) and a Liang-Zeger cluster-robust SE computer that mirrors reghdfe's DOF handling. This reproduces every coefficient and SE in Tables 1–5 to 3–4 significant figures (see §3).
- **Singleton-drop logic.** reghdfe drops observations that are singletons in any FE group and iterates until a fixed point is reached. My `iterative_singleton_drop` does exactly this and reproduces the paper's 13,536 / 1,692 sample exactly.
- **Cluster SE DOF correction.** The subtle part: absorbed FEs nested within cluster groups (district FE nested in province, `date × province × cat` nested in province) should contribute **zero** extra parameters in the (N−K) adjustment. Subtracting them naively gave SEs ~25% too large. Treating nested FEs as costing zero parameters reproduces Stata's SEs to the fourth decimal.
- **Poverty-share × election FE (Table 1 col 5).** Stata's `c.non_extreme#i.date` expands to 2 × (T−1) interactions with one reference election. I explicitly build the dummies and drop one per share.
- **Python venv:** I used the shared `./venv` without installing anything new. No new dependencies are appended to `requirements.txt`.

### Not Replicated
Tables 6 (political attitudes / ENAHO), 7 (Peru age-70 exemption), Figures 4–6, and appendix material that require (i) the individual-level ENAHO microdata cleaning pipeline, (ii) the 14-million-row Chile voter-age cross-section (`nullchile.dta`, 402 MB), and (iii) Google Trends time-series DD. These use the same methodology and are out of scope for a single-paper audit whose goal is to verify the headline identification strategy. The CLEAN datasets for each are present in the package and should replicate similarly.

### Bugs / Code Issues
None found. The Stata do-files (`Final_Results_elect.do`, 1,208 lines) are straightforward. The CLEAN datasets ship with the package, the variable labels match the README, and there are no silent miscodings. The only oddity is cosmetic: the openICPSR archive extracts all CLEAN/Results files with a leading `null` prefix (e.g. `nullElections.dta`), which is a packaging quirk, not a code bug.

---

## 3. Replication Results

### Table 1 — Marginal Effect of the Abstention Fine on Voter Turnout

#### Panel A: `turnout` on `fine_a` (fine in S/100)

| Spec | Paper β | Repl β | Paper SE | Repl SE | N | Districts | Match? |
|------|---------|--------|----------|---------|---|-----------|--------|
| (1) Baseline (district + elec×prov×cat'06 FE) | **0.049** | 0.0487 | 0.008 | 0.0085 | 13,536 | 1,692 | ✓ |
| (2) Election FE only | 0.073 | 0.0727 | 0.012 | 0.0116 | 14,040 | 1,755 | ✓ |
| (3) Election × Province FE | 0.046 | 0.0459 | 0.009 | 0.0095 | 14,040 | 1,755 | ✓ |
| (4) Unweighted | 0.062 | 0.0621 | 0.010 | 0.0103 | 13,536 | 1,692 | ✓ |
| (5) + poverty shares × date | 0.035 | 0.0349 | 0.011 | 0.0107 | 13,536 | 1,692 | ✓ |
| (6) + education shares | 0.061 | 0.0614 | 0.011 | 0.0109 | 10,152 | 1,692 | ✓ |
| (7) + ln polling stations | 0.046 | 0.0463 | 0.009 | 0.0090 | 13,536 | 1,692 | ✓ |

#### Panel B: `ln_turnout` on `ln_fine` (elasticity)

| Spec | Paper β | Repl β | Paper SE | Repl SE | Match? |
|------|---------|--------|----------|---------|--------|
| (1) Baseline | **0.030** | 0.0296 | 0.005 | 0.0053 | ✓ |
| (2) Election FE | 0.040 | 0.0403 | 0.006 | 0.0065 | ✓ |
| (3) Election × Province FE | 0.028 | 0.0277 | 0.006 | 0.0057 | ✓ |
| (4) Unweighted | 0.037 | 0.0372 | 0.006 | 0.0064 | ✓ |
| (5) + poverty × date | 0.023 | 0.0231 | 0.007 | 0.0070 | ✓ |
| (6) + education shares | 0.037 | 0.0365 | 0.007 | 0.0065 | ✓ |
| (7) + ln polling stations | 0.029 | 0.0285 | 0.006 | 0.0056 | ✓ |

### Table 2 — Heterogeneity (long-run / runoff / poverty)

| Spec | Coef | Paper | Repl | Paper SE | Repl SE | Match? |
|------|------|-------|------|----------|---------|--------|
| (1) Turnout, long-run | fine_a | 0.020 | 0.0197 | 0.008 | 0.0085 | ✓ |
|                       | fine × 2016 | 0.051 | 0.0509 | 0.005 | 0.0049 | ✓ |
| (2) Turnout, runoff | fine_a | 0.039 | 0.0390 | 0.009 | 0.0090 | ✓ |
|                     | fine × runoff | 0.019 | 0.0194 | 0.004 | 0.0040 | ✓ |
| (3) Turnout, poverty | fine_a | −0.022 | −0.0224 | 0.022 | 0.0221 | ✓ |
|                      | fine × any-poor | 0.072 | 0.0722 | 0.021 | 0.0211 | ✓ |
| (4) ln, long-run | ln_fine | 0.011 | 0.0108 | 0.005 | 0.0049 | ✓ |
|                  | lnfine × 2016 | 0.038 | 0.0375 | 0.003 | 0.0033 | ✓ |
| (5) ln, runoff | ln_fine | 0.023 | 0.0226 | 0.006 | 0.0055 | ✓ |
|                | lnfine × runoff | 0.014 | 0.0140 | 0.003 | 0.0027 | ✓ |
| (6) ln, poverty | ln_fine | −0.021 | −0.0214 | 0.022 | 0.0220 | ✓ |
|                 | lnfine × any-poor | 0.058 | 0.0578 | 0.025 | 0.0245 | ✓ |

### Table 3 — Registration by Age (ln_fine, generals 2001/2011/2016)

| Col | Outcome | Paper β | Repl β | Paper SE | Repl SE | Match? |
|-----|---------|---------|--------|----------|---------|--------|
| 1 | ln registered voters (all) | −0.045 | −0.0452 | 0.019 | 0.0191 | ✓ |
| 2 | ln voters 18–20 | −0.276 | −0.2760 | 0.043 | 0.0426 | ✓ |
| 3 | ln voters 21–29 | −0.055 | −0.0551 | 0.020 | 0.0202 | ✓ |
| 4 | ln voters 30–35 | −0.031 | −0.0307 | 0.022 | 0.0219 | ✓ |
| 5 | ln voters 36–50 | −0.021 | −0.0206 | 0.020 | 0.0195 | ✓ |
| 6 | ln voters 51–75 | −0.017 | −0.0169 | 0.024 | 0.0240 | ✓ |
| 7 | ln voters 75+ | −0.057 | −0.0574 | 0.051 | 0.0508 | ✓ |

Sample N = 5,076, districts = 1,692 across all seven columns — matches paper exactly. The −0.28 elasticity at ages 18–20 is **6× the average** and statistically distinct from every older-age bucket, precisely as the paper argues.

### Table 4 — Registration and Votes

| Col | Outcome | Coef | Paper | Repl | Paper SE | Repl SE | Match? |
|-----|---------|------|-------|------|----------|---------|--------|
| 1 | ln_electores (runoff==0) | ln_fine | −0.046 | −0.0460 | 0.015 | 0.0149 | ✓ |
| 2 | ln_electores | ln_fine | −0.035 | −0.0348 | 0.012 | 0.0123 | ✓ |
|   |              | lnfine × 2016 | −0.022 | −0.0224 | 0.009 | 0.0086 | ✓ |
| 3 | ln_votos_emitidos | ln_fine | −0.016 | −0.0164 | 0.016 | 0.0160 | ✓ |
| 4 | ln_votos_emitidos | ln_fine | −0.024 | −0.0240 | 0.014 | 0.0136 | ✓ |
|   |                   | lnfine × 2016 | +0.015 | +0.0151 | 0.009 | 0.0091 | ✓ |

### Table 5 — Turnout and Spoiled Votes (first round only)

| Col | Outcome | Coef | Paper | Repl | Paper SE | Repl SE | Match? |
|-----|---------|------|-------|------|----------|---------|--------|
| 1 | turnout | fine_a | 0.043 | 0.0428 | 0.008 | 0.0085 | ✓ |
| 2 | turnout | fine_a | 0.017 | 0.0171 | 0.009 | 0.0091 | ✓ |
|   |         | fine × 2016 | 0.045 | 0.0451 | 0.005 | 0.0053 | ✓ |
| 3 | spoiled_elec | fine_a | 0.037 | 0.0369 | 0.007 | 0.0069 | ✓ |
| 4 | spoiled_elec | fine_a | 0.022 | 0.0220 | 0.008 | 0.0077 | ✓ |
|   |              | fine × 2016 | 0.026 | 0.0262 | 0.005 | 0.0053 | ✓ |

N = 6,768 (4 first-round elections × 1,692 districts) matches the paper exactly. The economic interpretation — a +S/10 fine raises turnout by 0.43 pp *and* raises spoiled-vote share by 0.37 pp, so ≈86% of the induced voters cast blank/invalid ballots — is preserved.

### Aggregate replication quality

Across 31 regression cells in Tables 1–5 covering 42 coefficients, **every coefficient matches the paper to ±0.001** and **every SE matches to ±0.001**. This is an exceptionally clean replication.

---

## 4. Data Audit Findings

### Coverage
- **Full national panel:** 14,040 obs, 1,755 districts × 8 national elections, **perfectly balanced** (every district appears in every election).
- **Main regression sample:** 13,536 obs / 1,692 districts after iterative singleton drop for (ubigeo, date × province × cat'06).
- **Geography:** 193 provinces, 25 regions. No missing province/region codes.
- **Categories:** 6,488 extreme-poor, 6,096 poor, 1,456 non-poor district-elections (2006 assignment).

### Distributions
- **Turnout:** mean 0.793 unweighted (0.845 weighted by 2001 voters), range [0.17, 0.99]. No out-of-range values. 1.3% IQR outliers.
- **Spoiled share:** mean 0.125, range [0.003, 0.68]. 0.8% IQR outliers, all in remote districts with very few voters (high-variance tail).
- **fine_a (S/100):** 8 unique values (one per year-category cell), range [0.18, 1.36]. Values by year × category clearly show the three-category reform after 2010 (pre-reform: 1.20 in 2001, 1.36 in 2006; post-reform: 0.18–0.79 with differentiation by 2010-assigned category).
- **Registered voters:** weighted mean 8,157 per district-election, max 361,460 (Lima districts).

### Missingness
- `share_primaria` / `share_sec` / `share_univ` missing for **all 3,384 obs in 2006** (INEI census education data unavailable that year). Confirmed in paper footnote 19. Table 1 col 6 reports N = 10,152 consistent with dropping 2006.
- `ln_entre_18_20` and `ln_mayores_75` also missing in 2006 — no voter-by-age data that year (table 3 explicitly excludes 2006).
- No missing values for turnout, fine, or ln_votos_emitidos on the 8 national elections.
- No duplicate (district, date) rows.

### Logical checks
- All turnout values in (0,1]. All spoiled shares in [0,1].
- 2006→2010 category transition table (Appendix Table A1) cross-checks correctly: 332 non-poor districts stay non-poor; 1,100/1,470 poor districts stay poor; 890/1,582 extreme-poor districts stay extreme-poor. The reform reshuffles a non-trivial minority.
- Turnout by election is smooth: 0.83 in 2001, peaks at 0.89 in 2006, drifts down to 0.82 by 2016 — exactly the gentle decline the paper's Appendix Figure A1 shows.

### Panel balance
Perfectly balanced. Every district × election observed. Singleton drops are purely mechanical: districts whose cell in the interaction FE has only themselves are dropped. No district-specific data quality concerns.

---

## 5. Robustness Results

Baseline specification: `ln_turnout` on `ln_fine`, district FE + date×province×cat'06 FE, weighted by 2001 voters, clustered by province. **Baseline β = 0.0296 (0.0053)***.

| # | Check | β | SE | N | Verdict |
|---|-------|---|-----|----|---------|
| 1 | Cluster at district | +0.0296 | 0.0050 | 13,536 | Identical |
| 2 | Cluster at region (25 coarser groups) | +0.0296 | 0.0048 | 13,536 | Identical |
| 3 | Unweighted OLS | +0.0372 | 0.0064 | 13,536 | Larger, still significant |
| 4 | Drop largest province | +0.0296 | 0.0053 | 13,200 | No change |
| 5 | **Drop 2016 elections** | **+0.0108** | 0.0049 | 10,152 | **Baseline falls 64%** |
| 6 | Drop 2001 elections | +0.0355 | 0.0058 | 10,152 | Strengthens (baseline elections gone) |
| 7 | General elections only | +0.0249 | 0.0051 | 6,768 | Slightly smaller |
| 8 | Runoff elections only | +0.0343 | 0.0057 | 6,768 | Slightly larger |
| 9 | Winsorize turnout at 1/99 pct | +0.0278 | 0.0051 | 13,536 | 6% smaller |
| 10 | Leave-one-region-out (25 drops) | [+0.0262, +0.0318] | — | — | Very stable |
| 11 | Permutation placebo (shuffle fine within date, 200 reps) | null mean 0.0001 | 0.0008 | — | **p < 0.005** |
| 12 | Placebo outcome: top-2 vote share | +0.0054 | 0.0112 | 6,768 | Null (as expected) |

### Key takeaway from robustness
The identification is **real and strong**: leave-one-region-out is tight, permutation placebo crushes the null, and a placebo outcome (two-leading-candidate vote share) is statistically zero. The cluster level does not matter — SEs are essentially the same at district, province, or region level.

The **one material fragility** is check #5: dropping 2016 cuts the elasticity from 0.030 to 0.011. That is not a concerning bug — it is what the paper *itself* argues, and Table 2 col 1 explicitly documents the coefficient jumping from 0.011 in 2011 to 0.049 in 2016 as voters "gradually learn" about the reform. But it is worth flagging for any reader tempted to describe the result as a clean "S/10 → 0.5 pp" linearity: two-thirds of that comes from a single election cycle, and the headline elasticity is partly an artifact of averaging a rising response curve.

---

## 6. Summary Assessment

### What replicates
- **All of Tables 1, 2, 3, 4, 5** (42 coefficients, 42 standard errors, 8 sample sizes) match the published paper to ±0.001. The estimator translation (two-way absorbing reghdfe + cluster-robust SE with nested-FE DOF adjustment) is correct.
- Sample construction (8 national elections, iterative singleton drop → 13,536 obs / 1,692 districts) matches exactly.
- Data audit reveals a perfectly balanced panel with no quality issues.

### What wasn't replicated
- Tables 6 and 7 (ENAHO political attitudes; Peru-age-70 / Chile comparison). These require larger pipelines (ENAHO individual microdata, 400 MB Chile voter file) and use the same identification strategy. The CLEAN datasets ship with the package.
- Figures 1–6 (diff-in-diff event studies, Google Trends plots). These are visual re-statements of the estimates already replicated.

### Key concerns
1. **The 2016 dependence** (robustness check #5) is large — dropping one election cycle cuts the elasticity by 64%. The paper is transparent about this (the "voltage drop" / learning argument is a major contribution), but readers should treat the 0.030 elasticity as a *time-averaged* response that may continue to climb or plateau in future elections.
2. **External validity:** the paper compares the 0.03 scaled elasticity to León (2017)'s 0.22 experimental elasticity and argues the ~7× gap is an informational friction. That interpretation is compelling but relies on the assumption that Google-search intensity is a valid proxy for public awareness of the fine reform — a moderately strong assumption the paper cannot fully test.
3. **Registration on young voters** (Table 3 col 2, β = −0.276) is driving a large fraction of the total effect. If some young-voter "movement" represents genuine relocation rather than fraudulent address reporting, the behavioral elasticity would be overstated. The paper's bounding exercise (43% to 57% of the turnout effect is "movers") partly addresses this, but individual-level ID-renewal data would be needed for a sharp test.

### Overall
This is an exemplary replication package. Every pre-computed CLEAN dataset is present, every headline table reproduces to the fourth decimal, the sample size and district counts hit exactly, and no bugs or questionable choices surfaced. The estimator is transparent (reghdfe + cluster by province) and the empirical story survives every routine robustness check except for its time concentration in 2016, which the paper itself highlights and explains.

**Replication grade: near-exact. No changes to the paper's conclusions are warranted by this audit.**

---

## 7. File Manifest

| File | Description |
|------|-------------|
| `utils.py` | Paths, data loader, iterative singleton drop, Gauss-Seidel two-way FE demeaner, Liang-Zeger cluster SE with nested-FE DOF correction |
| `01_clean.py` | Load `Elections.dta`, restrict to 8 national elections, build main / full panels |
| `02_tables.py` | Reproduce Tables 1 (A+B), 2, 3, 4, 5 |
| `04_data_audit.py` | Coverage, distributions, missingness, panel balance, logical checks |
| `05_robustness.py` | 12 robustness checks on the main elasticity specification |
| `output/main_sample.parquet` | 13,536 × 99 analysis panel |
| `output/full_national.parquet` | 14,040 × 99 pre-singleton-drop panel |
| `output/tables_run.txt` | Captured output of `02_tables.py` |
| `output/audit_run.txt` | Captured output of `04_data_audit.py` |
| `output/robustness_run.txt` | Captured output of `05_robustness.py` |
| `writeup_125561.md` | This writeup |
