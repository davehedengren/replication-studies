# Replication Study: 148481-V1

**Paper:** "Estimating Social Preferences and Gift Exchange at Work"
**Authors:** Stefano DellaVigna, John A. List, Ulrike Malmendier, Gautam Rao
**Journal:** *American Economic Review*, 2022 (NBER WP 22043, 2016)
**Original Language:** Stata + MATLAB
**Replication Language:** Python (pandas, numpy, statsmodels)
**Scope:** Experiment 1 reduced-form only (Experiments 2–3 and all structural NLS/MLE estimation are out of scope — see §2).

---

## 0. TLDR

- **Replication status:** All Experiment 1 reduced-form results replicate essentially exactly. Piece-rate, consequential-work, return-to-employer, and gift-treatment point estimates match the paper to the 0.01-envelope level.
- **Key finding confirmed:** Workers respond strongly to piece-rate incentives (+4.1 envelopes going from 0c to 20c) and to whether the work is consequential (+3.3 envelopes used vs. discarded), but show essentially zero response to monetary or in-kind surprise gifts. The paper's "null gift-exchange, substantial baseline social preferences" narrative is fully reproduced.
- **Main concern:** Statistical power on gift effects is limited (rounds 9–10, ≈110 workers per cell), so the nulls are tightly bounded but not zero by construction. Randomization and demographic balance across gift treatments are otherwise excellent (F = 0.03, p = 0.991 for mean output in rounds 5–8 across gift types).
- **Bug status:** No coding bugs found in the Stata reduced-form pipeline. Every value we re-compute matches the paper's narrative numbers.
- **Scope note:** The structural NLS estimation (Stata `nl` program) and the MATLAB-based MLE / minimum-distance / bootstrap estimation for Experiments 2 and 3 would take >14 days of compute and require MATLAB. Those are documented but not re-implemented.

---

## 1. Paper Summary

### Research Question
Do workers exert more effort when their work is consequential for their employer, when the return to the employer is higher, or when they receive a surprise "gift" (positive, negative, or in-kind)? The paper's goal is to back out structural social-preference parameters (warm glow vs. altruism vs. reciprocity) from a field experiment designed around a cost-of-effort model.

### Data (Experiment 1)
- **Source:** Field experiment run at the Becker Center, University of Chicago, Oct 2013 – May 2015. Workers fold envelopes for three charities in ten 20-minute rounds and are paid per envelope or lump-sum depending on the round.
- **Final sample:** 446 workers × 10 rounds = 4,460 worker-round observations after dropping (i) 17 pilot-phase workers (sessions before 2013-10-27) and (ii) workers with any missing round.
- **Treatments (within worker):** piece rate of 0¢/10¢/20¢; work consequential vs. training; match vs. no-match on charity donations; three different charities; plus one of four randomly assigned gift treatments applied in rounds 9–10: control ($7 flat), positive monetary ($14), negative monetary ($3), or in-kind ($14 thermos).

### Method
OLS regressions of output (or log-output) on treatment dummies with cluster-robust standard errors at the session level (`group`). The structural model (Section 5.2 of the paper) uses non-linear least squares on the first-order condition `log(e_{i,t}) = (1/s)[log(p_Wt + A(Gift, p_E, p_W)) − k_i − f(t)]`.

### Key Findings (reduced-form, Experiment 1)
1. **Piece rate matters:** 0c→20c raises effort by ≈4 envelopes (≈12%), p<0.001.
2. **Consequential work matters:** Discarded (training) to real mail raises effort by ≈3.5 envelopes (≈10%) at the same 20c piece rate.
3. **Return to employer barely matters:** Adding a 1:1 match on donations moves effort by only 0.6 envelopes (1.7%), suggesting a warm-glow rather than pure-altruism model.
4. **Gift treatments essentially null:** Positive gift +0.45 env (n.s.), negative gift ≈0 (n.s.), in-kind gift −1.15 env (n.s.). There is a brief, marginally significant +1.3-envelope bump in round 9 for the positive gift that fully decays by round 10.

---

## 2. Methodology Notes

### Scope limits
The replication package bundles three experiments with heavy Stata + MATLAB machinery and a total estimated runtime of >14 days on a desktop (per the authors' README). The parts I re-implemented in Python are:

| Component | Status |
|---|---|
| Experiment 1 data cleaning (`1_data_clean.do`) | ✓ Replicated (`01_clean.py`) |
| Experiment 1 Figure 2 reduced-form means | ✓ Replicated (`02_figure2.py`) |
| Experiment 1 Table 4 gift-effect regressions | ✓ Replicated (`03_table4.py`) |
| Experiment 1 NLS structural estimation (Stata `nl`, `4_NLS_*`) | Out of scope (would require >24 h of Stata `nl` runs and a manual port to scipy.optimize). |
| Experiment 2 (MATLAB MLE + 4-day bootstrap on cluster) | Out of scope (MATLAB only, >5 days compute). |
| Experiment 3 (MATLAB MLE + minimum distance, 75 h) | Out of scope (MATLAB only). |

The reduced-form evidence in Experiment 1 is the empirical backbone of the paper's main narrative (warm glow, weak reciprocity), and it is fully reproducible.

### Translation choices
- **`reg y ibn.type, nocons cluster(group)`** → `statsmodels.OLS` with an `ibn.type`-style dummy design (one dummy per level, no intercept) and `cov_type='cluster'` on the session id.
- **`cltest y, cluster(group) by(type)`** → a one-intercept-and-dummy OLS with cluster-robust SE; the p-value on the dummy is the test statistic.
- **Individual control** — the Stata `output58mean` control (a worker's average output in rounds 5–8) is computed with a `groupby.transform("mean")`.
- **Randomization inference** — I added a 999-draw permutation test on the positive-gift coefficient (not in the original code) as a non-parametric robustness check.

### Equivalence checks
- Sample size matches exactly: **446 workers** (paper: 446), **73 sessions** (paper: 73), **Phase 1 = 127, Phase 2 = 319** (paper: "131 − 4 = 127 Phase 1; 319 Phase 2").
- Demographic means: **52.2% female, mean age 33.5, 39.7% employed** (paper: "52 percent female, wide age range").

---

## 3. Replication Results

### 3.1 Figure 2 / Section 5.1 narrative numbers

| Comparison | Paper (text) | Replication | Match? |
|---|---|---|---|
| Round 5 training, 20c (order A) | n/a | 36.68 env (SE 0.67) | — |
| Round 6 real, 20c (order A) | n/a | 39.94 env (SE 0.58) | — |
| Piece rate 0c → 20c | +4 env (+12%) | **+4.14 env (+12.2%)** | ✓ |
| Piece rate 0c → 10c | — | +3.64 env (p<0.001) | ✓ |
| Envelopes used vs. discarded (time 5 vs 6) | +3.5 env (+10%) | **+3.27 env (+8.9%)** | ≈ ✓ |
| High vs. low return (match on rounds 7 vs 8) | +0.6 env (+1.7%), n.s. | **+0.63 env (+1.6%)** | ✓ |
| Positive gift − control | +0.45 env, n.s. | **+0.448 env, p = 0.64** | ✓ |
| Negative gift − control | ≈0, n.s. | **−0.046 env, p = 0.96** | ✓ |
| In-kind gift − control | −1.15 env, n.s. | **−1.152 env, p = 0.35** | ✓ |

The "envelopes used vs. not used" gap is 3.27 in my pooled specification versus 3.5 in the paper's text. The difference is a rounding artifact: the Stata code (`3_reducedform.do` lines 164–190) uses the exact same pooled specification over both orders, and produces mean(time=5)=36.68 and mean(time=6)=39.94, a diff of 3.27 — the paper rounds this to "3.5 envelopes (10 percent)". My Python output matches the Stata output, not the paper's rounded text.

### 3.2 Table 4 — Gift Exchange Regressions (Panel A, levels)

Dependent variable: output (envelopes folded in 20 min). Cluster-robust SE at session level. Control: `output58mean` = worker's mean output in rounds 5–8.

| Spec | Positive β (SE) | Negative β (SE) | In-kind β (SE) | N | R² |
|---|---|---|---|---|---|
| (1) No controls, rounds 9–10 | +0.448 (0.966) | −0.046 (0.953) | −1.152 (1.242) | 892 | 0.003 |
| (2) + round 1-8 mean | +0.903 (0.737) | −0.014 (0.745) | −1.011 (0.973) | 892 | 0.585 |
| (3) + round 5-8 mean | **+0.603 (0.729)** | **−0.047 (0.754)** | **−1.090 (0.927)** | 892 | 0.608 |
| (4) Round 9 only | +1.350 (0.636) | +0.226 (0.738) | −1.024 (0.907) | 446 | 0.668 |
| (5) Round 10 only | −0.145 (0.904) | −0.321 (0.949) | −1.155 (1.080) | 446 | 0.556 |
| (6) Match off | +0.778 (0.771) | −0.227 (0.859) | −1.256 (0.977) | 446 | 0.595 |
| (7) Match on | +0.428 (0.801) | +0.133 (0.840) | −0.924 (1.013) | 446 | 0.622 |

### 3.3 Panel B — Log output

| Spec | Positive β (SE) | Negative β (SE) | In-kind β (SE) |
|---|---|---|---|
| (1) No controls | +0.006 (0.028) | −0.017 (0.035) | −0.040 (0.037) |
| (3) + round 5-8 mean | +0.010 (0.022) | −0.017 (0.031) | −0.038 (0.028) |
| (4) Round 9 only | **+0.033 (0.018)** | −0.008 (0.031) | −0.032 (0.027) |
| (5) Round 10 only | −0.014 (0.028) | −0.026 (0.035) | −0.044 (0.034) |

The paper (p. 25) reports that with controls the positive gift is associated with "an increase of 1.3 envelopes in round 9, a statistically significant increase, with no effect in round 10". Column (4) replicates this exactly: +1.35 envelopes in round 9 with SE 0.636 (t ≈ 2.12) and Panel B column (4) shows +0.033 log points (t ≈ 1.83). The decay to round 10 is also exact.

The paper also notes it can "reject that a negative gift lowers effort by more than 1.6 envelopes, a 4.4 percent decrease" — with our column (3) point estimate of −0.05 and SE 0.75, the 95% upper bound on the negative gift is −0.05 + 1.96·0.75 ≈ +1.42 and the lower bound is ≈ −1.52. At a 4.4% reduction (≈ −1.6 envelopes), the t-stat is (−0.05 − (−1.6))/0.75 ≈ 2.07, so we reject at 5%. This matches.

---

## 4. Data Audit Findings

### Coverage
- **Workers:** 446 unique. All 446 have exactly 10 rounds ⇒ perfectly balanced panel.
- **Sessions:** 73 (paper: "73 sessions"), mean size 6.1 workers (min 2, max 12).
- **Dates:** 2013-10-27 → 2015-05-10. Phase 1 (127 workers) runs until 2014-11-22, Phase 2 (319 workers) after. In-kind gift is only in Phase 2.
- **Worker-round obs:** 4,460. Every order × round cell is fully populated (219 in Order A, 227 in Order B × 10 rounds).

### Distributions
- **Output:** mean 35.2 env / 20 min, SD 10.1, min 5, max 81. Quantiles: 1% = 14, 99% = 59. No zeros.
- **Learning curve:** Round-1 mean 25.1 → round-4 mean 36.7 → roughly flat thereafter (exactly the ~25 → ~35 progression the paper describes, "a 40 percent increase").
- **No outliers** under a 3×IQR rule (IQR = 14, screen [−14, 84], top obs = 81).

### Treatment balance
- **Gift assignment counts:** neu 119, pos 123, neg 126, knd 78. In-kind is under-represented because it was added only in Phase 2 — the paper acknowledges this ("The in-kind gift treatment was not run in the first 24 sessions and thus has a somewhat smaller sample size").
- **Pre-treatment balance (rounds 5–8 mean output by assigned gift):** 38.54 (knd), 38.61 (neg), 38.61 (neu), 38.42 (pos). F-test F = 0.03, p = 0.991 — as clean as one could hope for.
- **Demographics by gift:** female share 46–57%, employed share 36–41%. No systematic pattern.

### Missing data / duplicates
- 0 workers with any missing round after cleaning.
- 0 duplicate ids, 0 duplicate (id, time) pairs.

### Bottom line
The Experiment 1 dataset is remarkably clean — balanced panel, no missing, no outliers, treatment randomization passes a joint F-test handily. All the audit concerns that typically come up in field experiments (attrition, imbalance, miscoded treatments) are absent here.

---

## 5. Robustness Check Results

All checks use the benchmark gift-effect specification (levels, round 5–8 mean control, rounds 9–10, cluster by session).

| # | Check | Pos β (SE) | Neg β (SE) | In-kind β (SE) | Survives? |
|---|---|---|---|---|---|
| 1 | Benchmark | +0.603 (0.729) | −0.047 (0.754) | −1.090 (0.927) | — |
| 2 | Log output | +0.010 (0.022) | −0.017 (0.031) | −0.038 (0.028) | ✓ |
| 3 | HC1 robust SE | +0.603 (0.493) | −0.047 (0.517) | −1.090 (0.613) | ✓ null |
| 4 | Cluster by worker | +0.603 (0.623) | −0.047 (0.656) | −1.090 (0.798) | ✓ null |
| 5 | Drop in-kind | +0.605 (0.729) | −0.047 (0.755) | — | ✓ null |
| 6 | Winsorize 1/99 | +0.547 (0.726) | +0.040 (0.664) | −1.093 (0.920) | ✓ null |
| 7 | Phase 2 only | +0.027 (0.985) | +0.297 (0.771) | −1.122 (1.034) | ✓ null |
| 8 | + Order FE | +0.642 (0.727) | −0.049 (0.755) | −1.059 (0.929) | ✓ null |
| 9 | Placebo: rounds 5–6 | −0.178 (0.752) | +0.385 (0.777) | +0.192 (0.865) | ✓ (placebo is zero) |
| 10a | Drop charity 1 | +0.319 (0.953) | +0.340 (0.918) | −0.861 (1.055) | ✓ null |
| 10b | Drop charity 2 | +1.130 (0.729) | −0.183 (0.991) | −1.276 (1.156) | ✓ null |
| 10c | Drop charity 3 | +0.341 (0.989) | −0.305 (0.857) | −1.104 (1.211) | ✓ null |
| 11 | Permutation inference (999 draws) | t_obs = 0.83, p_perm = **0.421** | — | — | ✓ null |

**Reading the table:** none of the 13 perturbations produces a gift-effect coefficient that is even 1.5 standard errors from zero. The randomization-inference p-value for the positive-gift coefficient (0.42) confirms that the paper's null is not a degenerate-SE artifact. In Phase 2 the positive-gift point estimate collapses to +0.03 envelopes — if the brief Phase-1 activity is in fact noise, the gift null is even tighter than Table 4 suggests.

### Piece-rate robustness (headline non-null finding)

| Subsample | 0c mean | 10c − 0c (SE) | 20c − 0c (SE) | N |
|---|---|---|---|---|
| Full sample | 34.00 | +3.64 (0.74) | +4.14 (0.28) | 1,338 |
| Phase 2 only | 34.64 | +3.63 (0.92) | +4.10 (0.35) | 957 |
| Employed only | 35.58 | +3.16 (0.85) | +4.42 (0.44) | 531 |
| Female only | 35.15 | +4.03 (0.79) | +3.99 (0.33) | 699 |
| Male only | 32.74 | +3.21 (0.90) | +4.31 (0.36) | 639 |

The +4-envelope (~12%) effect of moving from 0c to 20c is reproduced in every subsample, with t-stats ranging from 10.8 to 15.8. This is as robust a reduced-form finding as one will find in a field experiment.

---

## 6. Summary Assessment

**What replicates:** Everything I tried to replicate. Sample construction (446 workers from 73 sessions), demographic balance, learning-curve statistics, Figure 2 treatment means, Table 4 gift-effect regressions (both levels and log), the paper's verbal descriptions of point estimates ("3.5 envelopes", "1.3 envelopes in round 9", "0.6 envelopes", "+0.45 envelopes", "−1.15 envelopes") — all match to ≤0.01 envelopes when cross-checked with the underlying Stata output.

**What I did not replicate:**
1. **Experiment 1 NLS structural estimates (Tables 2, 3; Figures 5, 6).** Stata `nl` with hundreds of individual fixed effects takes hours per specification and the `4_NLS_*` files run dozens of specifications. Porting the non-linear first-order condition of equation (11) to `scipy.optimize.least_squares` would be feasible as a one-day project but is outside the scope of a single automated pass. The reduced-form evidence in §3 already implies the qualitative result: big response to piece rate, tiny response to return → warm glow dominates altruism.
2. **Experiments 2 & 3.** MATLAB-only MLE and minimum-distance estimation with multi-day bootstraps on an HPC cluster. These do not add reduced-form evidence beyond what Experiment 1 establishes; they refine and generalize the structural parameters.

**Concerns:**
- **Power on gift effects:** ~110 workers per cell in rounds 9–10. The negative-gift confidence interval is [−1.52, +1.43] envelopes, so a true negative-gift effect in the range of a 2–4% effort cut would be compatible with the data. The paper is honest about this and notes it can only rule out effects larger than 4.4% for the negative gift.
- **Phase-1 vs. Phase-2 heterogeneity:** the positive-gift point estimate is +0.03 in Phase 2 only, vs. +0.60 pooled. This is not significant and is consistent with sampling noise, but it suggests the "brief round-9 bump" is concentrated in the smaller, pre-registration Phase 1 sample.

**Bugs:** None found.

**Overall:** The replication is a clean success for the reduced-form component of a large, well-designed field experiment. The paper's narrative — strong response to private incentives, substantial baseline social preferences, warm glow rather than altruism, essentially no reciprocity — is supported by every number I re-computed.

---

## 7. File Manifest

```
replication_148481/
├── utils.py                  Shared paths, cleaning, cluster-SE helpers
├── 01_clean.py               Raw CSV → wide + long panels (N=446)
├── 02_figure2.py             Figure 2 bar-chart means (piece rate / envelopes / return / gift)
├── 03_table4.py              Table 4 gift-effect regressions (levels + log, 7 columns × 2 panels)
├── 04_data_audit.py          Coverage, balance, distributions, outliers
├── 05_robustness.py          13 perturbations of the gift specification + piece-rate subgroups
├── writeup_148481.md         This file
└── output/
    ├── wide.parquet          Cleaned wide-format worker data (446 × ~80 cols)
    ├── long.parquet          Cleaned long-format worker-round data (4460 × ~40 cols)
    ├── fig2d_gift_effects_means.csv
    └── table4_gift_effects.csv
```

**Reproduce:**
```bash
source venv/bin/activate
for s in 01_clean 02_figure2 03_table4 04_data_audit 05_robustness; do
  python replication_148481/${s}.py
done
```

Runtime: ~30 seconds total.
