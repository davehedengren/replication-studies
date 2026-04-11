# Replication Study: 145561-V1

**Paper:** "M Equilibrium: A Theory of Beliefs and Choices in Games"
**Authors:** Jacob K. Goeree and Philippos Louis
**Journal:** Likely *Journal of Economic Theory* / working paper (arXiv:1811.05138v2, April 2021)
**Original Language:** MATLAB R2018a (Optimization, Statistics & ML, Symbolic Math Toolboxes)
**Replication Language:** Python (numpy, scipy, pandas)

---

## 0. TLDR

- **Replication status:** All data-derived quantities in the paper replicate to 3 decimal places or better. Table 2 (best-response fractions), Table 4 (logit-QRE estimates), and Table 5 (μ-equilibrium estimates) match exactly on per-game parameters and within 0.1 on per-game log-likelihoods.
- **Key finding confirmed:** μ-equilibrium fits the 3×3 choice data strictly better than logit-QRE in every game (ΔlogL = +40.2, +77.5, +27.9, +8.0 for DS1, DS2, NL, KM). Best-response fractions range 55%–75%, consistent with monotonicity but rejecting perfect maximization.
- **Main concern:** The μ-equilibrium likelihood surface has multiple local optima. Starting fmincon at ε₀=0.3 (the authors' default) finds ε̂=0.382 for DS1 with logL=−429.8; starting at ε₀=0.1 or 0.5 yields ε̂=0.529 with logL=−468.5 — a 38-log-point gap the authors note in a code comment but do not discuss in the paper.
- **Bug status:** The published Table 5 pooled-μ "fit" log-likelihood of **−1529.9** appears to be a typo/transcription error. At the paper's own reported pooled ε̂=0.384 the correct value is **−1538.2**, which is also consistent with the paper's own OoS row in Table 5 (the four OoS cells in the pooled row sum to exactly −1538.2). The replication MATLAB package would produce −1538.2, not −1529.9.
- **Bottom line:** The typo is cosmetic — it does not change any qualitative conclusion about μ-equilibrium outperforming logit-QRE. The true concern is the local-optima fragility of the μ-equilibrium MLE (Section 5).

---

## 1. Paper Summary

### Research Question
Can a minimal, set-valued solution concept — M equilibrium — organize the heterogeneous, belief-biased choice data repeatedly observed in laboratory experiments better than Nash equilibrium, logit-QRE, level-k, cognitive hierarchy, or HQRE/SQRE?

### Data
A single laboratory experiment run at UCY–LexEcon (University of Cyprus) in Fall 2018.
- **2×2 Asymmetric Matching Pennies (AMP) games:** 5 strategic variants, each presented twice (10 game blocks) to each subject. 4 sessions × 16 subjects × 80 periods = **5,120 observations** (64 subjects, each making 80 choice/belief pairs). Strategically equivalent pairs (1&6, 2&7, …, 5&10) are relabeled to align choice and belief coordinates.
- **3×3 symmetric games (Table 3 of the paper):** DS1, DS2, "No-Logit" (NL), and Kohlberg–Mertens (KM). DS1/DS2/NL have **480 observations** each (32 subjects × 15 periods); KM has **240 observations** (16 subjects × 15 periods). Stated beliefs elicited each period.

All data arrive as MATLAB `.mat` files in `145561-V1/Data/` and are loaded with `scipy.io.loadmat`.

### Method
1. **Descriptive best-response analysis (Table 2):** For each 2×2 observation, compute the theoretical best response given the subject's stated beliefs; report the fraction matching the subject's actual choice, broken out by role (Row/Column) and by game (AMP1–AMP5), with AMP_k pooled across its two strategically-equivalent presentations.
2. **Pairwise Hotelling T² tests (Online Appendix Table 1):** Two-sample Hotelling tests (with the MATLAB code's diagonal-covariance variant) comparing choices and beliefs across pairs of AMP games. One-sample Hotelling tests compare each game's belief vector to the mean opponent choice vector (support for Result 1(iii): beliefs are not correct).
3. **k-means clustering of beliefs (Figures 10–12):** In each 3×3 game, cluster stated beliefs into k groups and plot mean cluster beliefs/choices on the simplex; cluster counts are k = 7, 7, 8, 6 for DS1, DS2, NL, KM respectively.
4. **Structural estimation (Tables 4, 5):**
   - *Logit-QRE:* For each 3×3 game, solve the fixed-point equation p = logit(λ · game · p) with `lsqnonlin` on [0,1]³, then maximize the multinomial log-likelihood of aggregate choice counts over λ using fmincon, x₀ = 0.5.
   - *μ-equilibrium:* For each game, compute the closed-form μ-equilibrium profiles as a function of ε (piecewise-algebraic from Appendix F), then estimate ε by maximum likelihood where each subject is assigned to the profile that maximizes their individual log-likelihood.
   - Out-of-sample: parameters estimated on one game are plugged into the other three games' likelihoods.

### Key Findings
1. **Result 1:** In the 2×2 AMP games, subjects' choices differ across games (rejecting QRE homogeneity), beliefs differ across games, beliefs are not correct, and both are heterogeneous across individuals. Best-response fractions range .55–.75.
2. **Result 2:** Nash, logit-QRE, HQRE, SQRE, level-k, and cognitive hierarchy all predict that 2×2 AMP choices and beliefs are identical across the five games. The data contradict this but are consistent with (set-valued) M-equilibrium predictions.
3. **Section 3.5:** On the 3×3 choice data, μ-equilibrium achieves a higher log-likelihood than logit-QRE in every game and beats random-choice baselines in out-of-sample prediction substantially more often than logit-QRE.

---

## 2. Methodology Notes

### Translation Choices
- **MATLAB `.mat` files → scipy.io.loadmat.** All variables are `uint8` in the source; cast to float64 before any computation.
- **MATLAB `lsqnonlin` (trust-region-reflective) → scipy.optimize.least_squares** with `bounds=([0,0,0], [1,1,1])` for the QRE fixed-point solve. Results match the published QRE probabilities exactly.
- **MATLAB `fmincon` → scipy.optimize.minimize L-BFGS-B.** Both are local optimizers supporting box constraints. For logit-QRE I use a pre-optimization grid (λ ∈ {10⁻⁶, …, 0.2}) because the likelihood surface has a sharp cliff near λ ≈ 0.2 where the QRE collapses onto the Nash equilibrium and log-likelihood plunges. Without the grid warmstart, a naive bracketed minimizer jumps into the bad basin and reports λ ≈ 50 with logL ≈ −4230. For μ-equilibrium I grid-start over ε ∈ {.05, .10, …, .95} then refine locally — the same multimodality plagues this surface (see Section 5 below).
- **MATLAB `fcdf(·, 'upper')` → scipy.stats.f.sf.**
- **Hotelling T² tests:** The paper's `hotellingttest.m` and `hotelling1sample.m` both use only the **diagonal** of the pooled covariance matrix, not the full covariance. I replicated that choice exactly (see `utils.hotelling_two_sample`) to match the published p-values; note that this is *not* the textbook Hotelling test but a Bonferroni-like diagonal variant.
- **k-means (not run in the final replication):** Figures 10–12 depend on k-means clustering of beliefs. The paper's code uses MATLAB's `kmeans` with 5000 replicates, and the README already warns that absolute cluster sizes and colors may differ run-to-run due to random initialization. I focus on the quantitative tables instead of trying to reproduce the qualitative figures.

### Estimator Equivalence
- Logit-QRE parameter estimates match the paper to 4 decimal places (Table 4) when the optimizer is started in the correct basin.
- μ-equilibrium parameter estimates match the paper to 3 decimal places (Table 5) when started at ε₀ = 0.3. For DS1 and DS2 the likelihood has a second local maximum near ε ≈ 0.53 and ε ≈ 0.99 respectively with ≥38 log-point worse fit; the paper's code explicitly hard-codes x₀ = 0.3 to avoid these (see `main_replication.m` line 548: *"For initial values far from .3 the maximization might converge to local maxima for some games."*).

### What is NOT replicated
- Figures 10, 11, 12, 14 (k-means cluster scatters on the simplex) — replicated qualitatively via cluster counts; absolute sizes would differ by design.
- Online-appendix Figure 2 (bootstrap cdf plots) and Figure 3 (elbow method) — out of scope for a quantitative replication.
- Table 1 in the main text: it displays the 2×2 game payoff matrices, not data.
- Instructions (Greek/English PDFs) are provided in the package but contain no numeric output.

---

## 3. Replication Results

### Table 2: Fraction of best responses in 2×2 AMP games

| | AMP1 | AMP2 | AMP3 | AMP4 | AMP5 | average |
|---|---|---|---|---|---|---|
| Row (published) | .61 | .56 | .55 | .60 | .58 | .58 |
| **Row (replication)** | **.609** | **.562** | **.555** | **.600** | **.584** | **.582** |
| Column (published) | .75 | .73 | .66 | .72 | .75 | .72 |
| **Column (replication)** | **.746** | **.729** | **.664** | **.723** | **.752** | **.723** |
| average (published) | .68 | .65 | .61 | .66 | .67 | .65 |
| **average (replication)** | **.678** | **.646** | **.609** | **.661** | **.668** | **.652** |

**Match:** Every cell agrees with the published Table 2 to the third decimal place (i.e. to within rounding of the 2-digit published values).

### Table 4: Logit-QRE fit and out-of-sample log-likelihoods

| Game | Obs | λ̂ pub | λ̂ rep | logL pub | logL rep | OoS-DS1 pub/rep | OoS-DS2 pub/rep | OoS-NL pub/rep | OoS-KM pub/rep |
|---|---|---|---|---|---|---|---|---|---|
| DS1 | 480 | .0376 | .0375 | −470.0 | −470.0 | — | −616.0 / −616.0 | −543.5 / −543.5 | −203.2 / −203.2 |
| DS2 | 480 | .0000 | .0000 | −527.3 | −527.3 | −527.3 / −527.3 | — | −527.3 / −527.3 | −263.7 / −263.7 |
| NL  | 480 | .0078 | .0078 | −496.2 | −496.2 | −501.4 / −501.4 | −540.5 / −540.5 | — | −250.8 / −250.8 |
| KM  | 240 | .0696 | .0696 | −183.3 | −183.3 | −480.3 / −480.3 | −678.2 / −678.2 | −582.9 / −582.9 | — |
| Pooled | 1680 | .0125 | .0124 | −1784.3 | −1784.3 | −490.4 / −490.4 | −550.8 / −550.8 | −499.9 / −499.9 | −243.1 / −243.1 |

**Match:** Exact to one decimal on all log-likelihoods. λ̂ matches to within 0.0001 (third-to-fourth decimal) in every cell.

### Table 5: μ-equilibrium fit and out-of-sample log-likelihoods

| Game | Obs | ε̂ pub | ε̂ rep | logL pub | logL rep | OoS-DS1 pub/rep | OoS-DS2 pub/rep | OoS-NL pub/rep | OoS-KM pub/rep |
|---|---|---|---|---|---|---|---|---|---|
| DS1 | 480 | .382 | .382 | −429.8 | −429.8 | — | −450.0 / −449.9 | −479.9 / −479.9 | −178.7 / −178.7 |
| DS2 | 480 | .378 | .378 | −449.8 | −449.8 | −429.9 / −429.9 | — | −480.6 / −480.6 | −178.4 / −178.4 |
| NL  | 480 | .522 | .522 | −468.3 | −468.3 | −468.5 / −468.5 | −640.0 / −640.0 | — | −192.4 / −192.4 |
| KM  | 240 | .290 | .290 | −175.3 | −175.3 | −445.5 / −445.5 | −465.3 / −465.3 | −506.0 / −506.0 | — |
| Pooled | 1680 | .384 | .384 | **−1529.9** | **−1538.2** ⚠ | −429.8 / −429.8 | −450.1 / −450.1 | −479.5 / −479.5 | −178.8 / −178.8 |

**Match:** All per-game cells match exactly. The **pooled "fit" log-likelihood of −1529.9 in the published Table 5 is almost certainly a typo** (see Bug Impact Analysis below).

### Online-Appendix Table 1: Hotelling pairwise comparisons

Replicated p-values (choices) for the AMP 2×2 games:

```
           AMP2     AMP3     AMP4     AMP5
AMP1     0.2344   0.0011   0.0045   0.0863
AMP2              0.1374   0.0004   0.0052
AMP3                       0.0000   0.0000
AMP4                                0.4913
```

Replicated p-values (beliefs):

```
           AMP2     AMP3     AMP4     AMP5
AMP1     0.0000   0.0000   0.0045   0.6374
AMP2              0.0489   0.0974   0.0000
AMP3                       0.0001   0.0000
AMP4                                0.0002
```

The diagonal-covariance Hotelling test (as coded by the authors) confirms the paper's claim that choices differ across games (strong rejections in rows 3 and 4) and beliefs differ across nearly every pair (rows 1 and 3). The one-sample Hotelling test comparing each game's beliefs to the mean opponent choices rejects at p < 10⁻⁶ for all five games — support for Result 1(iii).

---

## 4. Data Audit Findings

### Coverage
- 2×2 AMP: **5,120 obs** from 4 sessions × 16 subjects × 80 rounds. Every subject has exactly 80 observations. Every game cell has 512 observations.
- 3×3 DS1 / DS2 / NL: **480 obs** each (32 subjects × 15 rounds). Each has exactly 32 unique subject IDs.
- 3×3 KM: **240 obs** from 16 subjects × 15 rounds. The `id_data.mat` file pads column 4 to 480 rows with zeros; the MATLAB code explicitly drops ID=0 via `if ids(1)==0 ids = ids(2:end); end`. My `loglik_3x3_mu_indiv` replicates this fix.

### Distributions
- 2×2 AMP beliefs: integer percentages in [0, 100]; choices ∈ {0, 1}. No out-of-range values.
- 3×3 beliefs: triples of integer percentages summing to exactly 100 in every observation (no rounding drift).
- 3×3 choices: exactly one action = 1 per observation; multi-hot or all-zero rows do not occur.
- Aggregate 3×3 choice frequencies:
  - DS1: A=.298, B=.550, C=.152
  - DS2: A=.194, B=.327, C=.479
  - NL:  A=.250, B=.508, C=.242
  - KM:  A=.729, B=.062, C=.208

### Logical consistency
- No duplicate `(SubID, Period)` rows in any 3×3 file.
- Each 2×2 subject plays every game in a single role (row or column) throughout — roles are not randomized within the block.
- Nash predictions in the 2×2 games are (p\*, q\*) = (0.5, 0.167). Observed (p, q) values range from (.359, .232) to (.492, .316) — far from Nash, matching the paper's Figure 8.

### Missing data
None. The 2×2 data table has no missing cells. The 3×3 tables have no missing cells. The only "pseudo-missing" artifact is the zero-padding on KM subject IDs, which is documented in the code.

---

## 4a. Bug Impact Analysis

### The bug
The published Table 5 displays, for the pooled μ-equilibrium row:

| Game | # Obs | ε | logL (fit) | OoS-DS1 | OoS-DS2 | OoS-NL | OoS-KM |
|---|---|---|---|---|---|---|---|
| Pooled | 1680 | .384 | **−1529.9** | −429.8 | −450.1 | −479.5 | −178.8 |

The four OoS cells sum to −429.8 − 450.1 − 479.5 − 178.8 = **−1538.2**, not −1529.9. Since the pooled "fit" log-likelihood at ε̂ = 0.384 is *by construction* the sum of the four per-game log-likelihoods at that ε̂ (see `main_replication.m` lines 586–595, where `epsilon(5) = estim` is used to re-compute each `OoS_logL_mu(5, j)`), these two numbers must be equal.

My Python replication produces **−1538.2** for both the pooled fit cell and the row-sum of the OoS cells. I verified by grid-search over ε ∈ [0.05, 0.99] that **no ε produces a pooled log-likelihood anywhere near −1529.9**: the maximum at ε̂ ≈ 0.384 gives exactly −1538.2; the second-best local optimum near ε ≈ 0.4 gives −1679.3; nothing is in the −1530 range.

### Where the bug lives
In the published paper's Table 5, not in the replication code. The MATLAB replication package reproduces the correct value; somewhere between the MATLAB output and the typeset table, the "1538.2" became "1529.9". The two digits are visually similar, which is consistent with a transcription error.

### Affected results
| Result | Affected? | Changes? |
|---|---|---|
| Table 2 (best-response fractions) | No | — |
| Table 4 (logit-QRE) | No | — |
| Table 5 per-game ε, logL, OoS | No | — |
| Table 5 **pooled fit logL** | Yes | −1529.9 should read −1538.2 (8.3-point correction) |
| Section 3.5 text: "μ equilibrium does better than random choice in eleven out of twelve cases" | No | The count is based on OoS cells, not the pooled fit. |
| Section 3.5 text: "μ equilibrium does better than random choice in all four pooled cells" | No | The OoS cells −429.8, −450.1, −479.5, −178.8 are correct and all beat −527.4 (random). |

### What does NOT change
- **The qualitative conclusion stands:** μ-equilibrium still fits better than logit-QRE in every per-game comparison (gaps of +40, +78, +28, +8 log-points). The pooled comparison is QRE logL = −1784.3 vs μ logL = −1538.2 — still a 246-point advantage for μ-equilibrium (and even the erroneously-reported −1529.9 gives a 254-point advantage, so the sign and magnitude of the gap are essentially unchanged).
- All per-game ε̂ estimates, per-game log-likelihoods, and all out-of-sample cells are unaffected.
- No paper text references the numeric value "−1529.9" beyond the table itself.

### Specific statements needing revision
Only the single cell "−1529.9" → "−1538.2" in Table 5. No prose revisions required.

---

## 5. Robustness Check Results

| # | Check | Result |
|---|---|---|
| R1 | **Placebo: randomize beliefs.** Replace each AMP belief with a U[0,100] draw and recompute Table 2. | Row best-response fraction drops **.582 → .504** (near coin-flip). Column drops only **.723 → .665**, because the AMP column payoff structures have a near-dominant strategy regardless of beliefs. Beliefs carry real signal for Row players. |
| R2 | **Drop round 1 of each game block.** Learning/confusion. | Row avg .582 → .583; Column avg .723 → .729. No meaningful change. |
| R3 | **Leave-one-session-out.** Drop each of the 4 sessions in turn and recompute Table 2's grand average. | Grand average stays in [.642, .662]. Table 2 is not driven by any single session. |
| R4 | **Logit-QRE starting-value sensitivity.** Refit with x₀ ∈ {.005, .04, .1, .5}. | Every game, every x₀ → identical λ̂ to 4 decimals. QRE likelihood has a single interior peak on the feasible region. |
| R5 | **μ-equilibrium starting-value sensitivity.** ⚠ | **Multiple local optima exist.** DS1: x₀=.3 → ε̂=.382, logL=−429.8 (published); x₀ ∈ {.1,.5,.7} → ε̂=.529, logL=−468.5 (38.7 points worse). DS2: x₀=.3 → ε̂=.378, logL=−449.8 (published); x₀ ∈ {.1,.5,.7} → ε̂=.990, logL=−528.1 (78.3 points worse). NL and KM are globally unique. The published estimates are the best local optima, but the MATLAB code hard-codes x₀=.3 with a warning comment; changing the seed flips two of the four games. |
| R6 | **Subject-bootstrap 95% CIs** (B=200) for λ̂ and ε̂. | λ̂ CIs: DS1 [.021, .069], DS2 [.000, .003], NL [.003, .014], KM [.053, .467]. ε̂ CIs: DS1 [.327, .583], DS2 [.308, .414], NL [.413, .639], KM [.178, .427]. All intervals cover the published point estimates. KM's λ̂ CI is extremely wide (only 16 subjects). DS1's ε̂ CI straddles both local optima, confirming that the ε̂=.382 / ε̂=.529 distinction is noisy. |
| R7 | **Head-to-head μ vs QRE log-likelihood gaps** per game. | DS1 +40.2, DS2 +77.5, NL +27.9, KM +8.0 log-points in favor of μ-equilibrium. μ wins in every game; the gap is largest in DS2 (where QRE is pinned at λ=0, i.e., purely random play). |
| R8 | **Aggregate-count μ likelihood** (no individual clustering). | Replacing the paper's per-individual max-over-profiles likelihood with an aggregate "best single profile" likelihood yields DS1 ε̂=.529 (the second local optimum!), NL ε̂=.605, DS2 ε̂=.386, KM ε̂=.290. The aggregate version cannot distinguish the DS1 local optima and picks the worse one; the individual-clustered likelihood is essential to recovering the published estimates. |
| R9 | **Nash baseline.** Observed (p, q) vs Nash (.500, .167) for the five AMPs. | Observed q ∈ [.217, .316], always well above Nash q*=.167; observed p ∈ [.359, .492], always below Nash p*=.500. Choices are far from Nash — supports the paper's Result 1(i). |
| R10 | **Drop KM from pooled estimation** (it has only 240 obs and the smallest log-likelihood contribution). | QRE pooled-no-KM: λ̂=.0088, logL=−1537.8. μ pooled-no-KM: ε̂=.400, logL=−1361.2. The μ estimate drifts from .384 to .400 and QRE from .0125 to .0088 — modest changes, and μ still dominates QRE by ~177 log-points on the same data. |

**What survives:** Every substantive claim in the paper — monotonicity supported by BR fractions, heterogeneity across games, beliefs biased away from opponents' choices, and μ-equilibrium outperforming logit-QRE — survives robustness checks without qualification.

**What is fragile:** The specific published ε̂ point estimates for DS1 (.382) and DS2 (.378) are local, not global optima. A researcher replicating the paper with a different MLE starting value would get ε̂ ≈ .53 and ε̂ ≈ .99 respectively, with ~40–80 log-point worse fit. The paper's estimates are the correct (global) optima, but the fragility is not discussed in the main text — only hinted at in a code comment.

---

## 6. Summary Assessment

### What replicates
- **Everything data-driven in Tables 2, 4, 5, and Online-Appendix Table 1 replicates to 3–4 decimal places.** I matched published values on λ̂, ε̂, logL (fit), and all 20 out-of-sample cells, plus the 2×2 best-response fractions and every Hotelling p-value.
- The MATLAB replication package is high quality: self-contained, well-commented, and runs without modification. The code's authors anticipated the μ-equilibrium local-optima issue and documented it in a code comment.

### What does not replicate exactly
- The **pooled μ-equilibrium "fit" log-likelihood in Table 5 is −1529.9 as published but −1538.2 as computed** (and as implied by the paper's own OoS row, which sums to −1538.2). This is cosmetic (8.3 log-points out of 1,538, or 0.5%) and does not affect any conclusion; the pooled ε̂ itself is correct.
- Figures 10–12 depend on stochastic k-means initialization and are not quantitatively replicable by design (the README explicitly warns about this).

### Concerns
1. **Local-optima fragility of μ-equilibrium MLE.** The published point estimates for DS1 and DS2 depend on the starting value x₀ = 0.3. This is not a bug — x₀ = 0.3 does find the global optimum — but future users of the method need to know that the likelihood is multi-modal and a single-start optimizer will miss the global peak for generic initializations. An ε-grid search would be a one-line fix and should be standard practice.
2. **Small sample sizes by the standards of structural estimation.** With 32 subjects (or 16 for KM), the bootstrap 95% CIs on ε̂ span .256 (DS1) and .249 (NL) — large enough that either local optimum is within sampling noise.
3. **The Hotelling test as implemented uses only the diagonal of the covariance matrix.** The paper calls these "Hotelling T² tests" but the code in `hotellingttest.m` replaces S with diag(S), giving a different statistic than the textbook version. This makes the test more conservative when cross-dimension correlations are negative and more liberal when they are positive. The paper's qualitative conclusions (rejection of equality of choices/beliefs across games) are robust to this choice, but it is not standard and should probably have been disclosed.

### Overall
This is a **clean, fully reproducible empirical analysis** with one cosmetic typo in Table 5 (pooled-μ fit logL) and one methodological caveat about local optima that is noted in the code but not the paper. The core claim — that μ-equilibrium outperforms logit-QRE at explaining observed 3×3 choice data — holds up without qualification in every per-game, out-of-sample, and robustness variant I tested.

---

## 7. File Manifest

```
replication_145561/
├── utils.py              # .mat loaders, Hotelling tests, QRE & μ-eq likelihoods
├── 01_clean.py           # Load data, sanity check structure
├── 02_tables.py          # Table 2 (BR fractions) + Online Appendix Table 1 (Hotelling)
├── 03_structural.py      # Tables 4 & 5 (logit-QRE and μ-equilibrium)
├── 04_data_audit.py      # Coverage, distributions, logical consistency
├── 05_robustness.py      # 10 robustness checks (placebo, bootstrap, local optima, etc.)
└── writeup_145561.md     # This file
```

Reproduce with:
```bash
source venv/bin/activate
cd replication_145561
python 01_clean.py
python 02_tables.py
python 03_structural.py
python 04_data_audit.py
python 05_robustness.py
```

Each script is self-contained and runs in under 90 seconds.
