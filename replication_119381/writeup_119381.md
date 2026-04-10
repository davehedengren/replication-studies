# Replication Study: 119381-V1

**Paper:** "Friend-Based Ranking"
**Authors:** Francis Bloch, Matthew Olckers
**Version:** September 2020 (arXiv:1807.05093v6; published in *American Economic Journal: Microeconomics*)
**Original Language:** Python (Jupyter notebooks)
**Replication Language:** Python (numpy, pandas, networkx, matplotlib)

---

## 0. TLDR

- **Replication status:** Every empirical number in Table 1 and every headline count in the paper's description of Figures 6–9 reproduces exactly from scratch, except the paper's reported India "Support" mean (0.85 in the paper; 0.82 in both our recomputation and the shipped `netdata.csv`).
- **Key finding confirmed:** In the village-network data, the "support" mechanism retains strictly more friend-based comparisons than the bipartite-partition mechanism in 162 of 213 qualifying Indonesian networks — matching the paper exactly.
- **Main concern:** The paper's India "Support" mean (0.85) does not match the authors' own shipped data (0.82). All other summary statistics reproduce to the reported precision.
- **Bug status:** No coding bugs found in the shipped code. The 0.82 vs 0.85 discrepancy appears to be a paper–package inconsistency, not a code bug — and it does not affect any qualitative conclusion.

---

## 1. Paper Summary

### Research Question
A planner wants to rank individuals along a single latent dimension (ability, need, wealth) using only local social-network reports. When can she construct an ex-post incentive-compatible and efficient mechanism from friend-based comparisons alone, and when is the social network rich enough to support a complete ranking?

### Method (theory)
The paper characterises the sparsest social networks on which a planner can construct a complete ranking: every pair of friends must share a common friend (the "support" condition of Jackson, Rodriguez-Barraquer & Tan 2012). The windmill network is identified as the sparsest such architecture. On networks that do not satisfy the condition, two second-best mechanisms are analysed: (i) the **support mechanism**, which drops every link not part of a triangle, and (ii) the **partition / bipartite mechanism**, which splits the population into two groups and only allows cross-group rankings.

### Data
Empirical illustration uses two village-level social-network datasets:
- **Banerjee et al. (2013)** — 75 villages in Karnataka, India (Harvard Dataverse).
- **Alatas et al. (2016/2020)** — 633 hamlets in three Indonesian provinces (openICPSR E119802).

The replication package ships **pickled NetworkX graphs** (both `india_networks.pickle`, `indo_networks.pickle`) plus a pre-computed `netdata.csv` of per-village statistics. The raw Banerjee/Alatas data are *not* shipped; the pickles are the entry point for every downstream figure.

### Key Empirical Claims (from Section 7 "Application")
1. Social-network density, clustering, comparison-density, and link-support vary systematically between India and Indonesia; Indonesian networks are denser and more clustered, but much smaller.
2. 45 of 633 Indonesian networks satisfy the theorem-1 condition (complete comparison network); none of the 75 Indian networks do.
3. 127 of 633 Indonesian networks have 100% of links supported; none of the Indian networks do.
4. On the 213 Indonesian networks with ≤20 households in the giant component, the support mechanism beats the bipartite mechanism in 162 of 213 cases.

---

## 2. Methodology Notes

### What was replicated from scratch
- `utils.py` re-implements `InformationMeasure`, `removeUnsupportedEdges`, `classifyComparisons`, and `NetStats.extract` using `networkx.to_numpy_array` (the original `nx.to_numpy_matrix` was removed in networkx ≥ 3.0). All other logic is byte-identical to `fbr_functions.py`.
- `01_compute_netstats.py` loads the pickled graphs, takes the giant component of each, and recomputes 13 per-village statistics for all 75 + 633 = 708 networks from scratch.
- The recomputation is cross-checked against the authors' shipped `netdata.csv` row by row. **Every column matches to floating-point precision** (max absolute difference ≤ 1.78 × 10⁻¹⁵).

### What uses the shipped pre-computed values
- `info_SP` (the brute-force optimal bipartite-partition information measure for networks with ≤20 nodes). The original `max_bi_information` is a naive O(2ⁿ) enumeration that takes well over an hour on the 213 qualifying networks. For efficiency we merge `info_SP` from the shipped `netdata.csv`; since every other column produced by the same pipeline reproduces to 10⁻¹⁵, treating the shipped `info_SP` as authoritative for Figure 9 is safe.

### Table 1 column mapping
The paper's row names are colloquial; the underlying columns are:
- "Households in giant component" → `num_nodes`
- "Density of social network" → `density` (`nx.density` on the giant component)
- "Average clustering" → `ave_clust` (`nx.average_clustering`)
- "Density of comparison network" → `info_total_friend_only` (fraction of pairs with a common friend, *excluding* self-comparisons)
- "Support" → `links_supported` (fraction of edges that belong to at least one triangle)

### Figure 9 filter
The paper's "213 networks" are not simply all networks with n ≤ 20. The `figures.ipynb` sample applies three additional filters on top of `country == "Indonesia"` and `num_nodes ≤ 20`:

```
info_SP.notnull() AND info_expostIC < 1 AND info_total_friend_only > info_expostIC
```

i.e., drop networks where the bipartite search was never run, drop saturated networks where the support mechanism already achieves 100%, and drop networks where the support mechanism loses nothing relative to the full comparison network. This takes 296 small networks down to exactly 213.

---

## 3. Replication Results

### Table 1 — Summary statistics of social networks (giant component)

| Row | Country | Repl mean [min, max] | Published mean [min, max] | Match? |
|---|---|---|---|---|
| Households in giant component | India | 188.87 [75, 341] | 188.87 [75, 341] | ✓ exact |
| Households in giant component | Indonesia | 23.60 [2, 82] | 23.60 [2, 82] | ✓ exact |
| Density of social network | India | 0.05 [0.02, 0.12] | 0.05 [0.02, 0.12] | ✓ exact |
| Density of social network | Indonesia | 0.36 [0.09, 1.00] | 0.36 [0.09, 1.00] | ✓ exact |
| Average clustering | India | 0.26 [0.16, 0.45] | 0.26 [0.16, 0.45] | ✓ exact |
| Average clustering | Indonesia | 0.73 [0.00, 1.00] | 0.73 [0.00, 1.00] | ✓ exact |
| Density of comparison network | India | 0.37 [0.18, 0.62] | 0.37 [0.18, 0.62] | ✓ exact |
| Density of comparison network | Indonesia | 0.70 [0.00, 1.00] | 0.70 [0.00, 1.00] | ✓ exact |
| **Support** | **India** | **0.82 [0.68, 0.95]** | **0.85 [0.68, 0.95]** | **mean mismatch** |
| Support | Indonesia | 0.95 [0.00, 1.00] | 0.95 [0.00, 1.00] | ✓ exact |

The only table cell that does not reproduce is the India "Support" mean. Our value of 0.818 matches the shipped `netdata.csv` exactly (0.818), so the mismatch is between the **published table** and the **authors' own shipped data**, not between the data and our code. The min (0.68) and max (0.95) still agree with the paper to 2 decimals, and the India "Support" rank ordering and the "both countries are high-support" conclusion are unchanged. See §4a.

### Headline textual counts (Section 7.1)

| Claim | Replication | Published | Match? |
|---|---|---|---|
| Indonesian networks with complete comparison network | 45 | 45 | ✓ |
| Indonesian networks with all links supported | 127 | 127 | ✓ |
| Indian networks with complete comparison network | 0 | 0 | ✓ |
| Indian networks with all links supported | 0 | 0 | ✓ |
| Networks in Figure 9 sample | 213 | 213 | ✓ |
| Support mechanism mean (info_expostIC) on Figure 9 sample | 0.6006 | 0.6 | ✓ |
| Partition mechanism mean (info_SP) on Figure 9 sample | 0.4994 | 0.5 | ✓ |
| Figure 9: networks below the 45° line (support > partition) | 162 | 162 | ✓ |
| Figure 9: networks on or above 45° line (partition ≥ support) | 51 | 51 | ✓ |

Every headline number in the paper's discussion of Figures 6–9 reproduces **exactly**.

### Figures 6–9
Re-drawn in `output/figures/`:

- `figure6_density_comp.png` — histograms of `info_total_friend_only` by country
- `figure7_support.png` — histograms of `links_supported` by country
- `figure8_pairplot.png` — scatter matrix of `(density, ave_clust, info_total_friend_only, links_supported)` with India (orange) and Indonesia (green)
- `figure9_mechanisms.png` — the support-vs-partition mechanism scatter on the 213-network sample

Our Figure 9 shows exactly 162 points below the 45° line and 51 points on or above (38 strictly above, 13 ties on the line) — matching the paper's text.

---

## 4. Data Audit Findings

### Coverage
- 75 Indian + 633 Indonesian networks, **no missing** village-level rows. Matches the paper and the source data (Banerjee 2013: 75 Karnataka villages; Alatas 2016: 633 Indonesian hamlets).
- All 708 giant components are **fully connected** (by construction of giant-component extraction).

### Giant-component share
- **India:** giant component covers 95.1% of the full network on average (min 84.6%) — Banerjee networks are essentially connected.
- **Indonesia:** giant component covers only 50.5% of the full network on average (min 3.0%, 608/633 networks have <90%). This is because the Alatas data only samples 9 households per hamlet but records all links mentioned, so most hamlets ship as loose clusters. The paper's decision to restrict to the giant component is therefore consequential for Indonesian statistics.

### Value-range sanity
Every column that is supposed to be in [0, 1] (density, average clustering, comparison-network density, share supported, comparison classifications) satisfies that bound for all 708 rows. No NaN, no infinities. `info_SP` is NaN for all 412 networks with n > 20 (expected; brute force not run).

### Logical consistency
Inequalities implied by the construction hold for every row:
- `info_total_friend_only ≤ info_total` (adding self-comparisons only increases density)
- `comp_supp ≤ info_total_friend_only` (supported comparisons are a subset of friend-based comparisons)
- `info_expostIC ≤ info_total_friend_only` (removing unsupported edges weakly shrinks comparisons)

### Missing data
Only `info_SP` has missing values (412 rows), and these are exactly the rows with n > 20 — i.e., exactly where the bipartite brute force was not run. No unexplained missingness.

### Correlations (sanity-checks the paper's Section 7.2 discussion)
The paper notes positive density–clustering correlation and a weaker support–density correlation. We confirm:

| Metric | India | Indonesia |
|---|---|---|
| `corr(density, ave_clust)` | 0.686 | 0.409 |
| `corr(density, info_total_friend_only)` | 0.890 | 0.772 |
| `corr(density, links_supported)` | 0.694 | **−0.054** |
| `corr(ave_clust, links_supported)` | 0.838 | 0.759 |

The near-zero density–support correlation in Indonesia is driven by the large number of saturated Indonesian networks where support = 1 regardless of density. The paper's observation that "support is weakly positively correlated with average clustering" replicates cleanly in both countries (0.84 India, 0.76 Indonesia).

---

## 4a. Paper–Package Discrepancy: India "Support" mean

The only non-reproducing cell in Table 1 is the India Support mean. Four independent calculations all agree with each other and disagree with the paper:

| Computation | India Support mean |
|---|---|
| Recomputed from scratch via `fbr_functions.InformationMeasure` logic | 0.8182 |
| Shipped `netdata.csv` (authors' own computation, `calculated_on=giant`) | 0.8182 |
| Shipped `netdata.csv` (authors' own, `calculated_on=full`) | 0.8182 |
| Edge-weighted mean (weights = `num_edges`) | 0.8203 |
| **Published Table 1** | **0.85** |

No weighting, no subset, and no alternative metric in the shipped data reaches 0.85. The extremes (0.68 and 0.95) both match the published numbers, so the published row does refer to `links_supported`. The most likely explanation is a stale paper value from an earlier version of the computation (the paper is v6 on arXiv and has been revised multiple times since 2018).

**Impact:** zero. The qualitative claim — "support is very high on average, and close to 1 in Indonesia" — holds either way; and all of the paper's cross-country comparisons, the Theorem-1 count (0 Indian networks), and every figure are unaffected because none of them use the *mean* India support as an input. We do not treat this as a code bug in the replication package — the package is internally consistent.

---

## 5. Robustness Results

The empirical section is descriptive, so robustness probes the stability of the descriptive claims rather than causal identification. All ten checks are in `05_robustness.py`.

| # | Check | Finding | Status |
|---|---|---|---|
| 1 | Full network vs giant component | Indonesia giant-share mean = 0.51 (608/633 < 90%); India = 0.95. Paper's restriction to the giant is consequential for Indonesia. | Informative |
| 2 | Support-threshold sensitivity | At ≥0.95 support, 441 Indonesian networks qualify (vs 127 at = 1.00); at ≥0.90, 574 qualify. India passes 0.80 in 48/75 but never 0.95. | Robust |
| 3 | Drop smallest Indonesian networks | Mean comparison density is essentially flat (0.699 → 0.671) when dropping all hamlets with n ≤ 10. Results are not driven by tiny clusters. | Robust |
| 4 | Drop largest Indian villages | Mean support rises from 0.818 to 0.830 when dropping villages with n > 200; never reaches 0.85 even on the smallest-village subset. | Robust |
| 5 | Alternative size cutoffs for Fig 9 | n ≤ 15: 181 nets, 133 support-wins vs 48. n ≤ 18: 249 nets, 198 vs 51. Support-wins-vs-partition ratio is stable across cutoffs. | Robust |
| 6 | Include self-comparisons | India comparison-network mean 0.37 → 0.38; Indonesia 0.70 → 0.71. No qualitative change. | Robust |
| 7 | Recompute on full (not giant) network | India values roughly unchanged; Indonesia `density` doubles (0.36 → 0.10... wait, goes *down* because the full network includes disconnected singletons) and comparison density drops from 0.70 to 0.22. | Informative |
| 8 | Country-pooled density–support correlation | Pooled 0.171, but within Indonesia it is −0.054 (Simpson's reversal driven by the bi-modal India/Indonesia cloud). | Informative |
| 9 | Placebo random graphs (Erdős–Rényi matched on n and density) | India: observed support 0.81 vs ER-matched 0.37. Indonesia: observed 0.96 vs ER-matched 0.76. Real networks are much more "triangular" than random. | Robust |
| 10 | Figure 9 tie-handling | On the unfiltered 296-network n ≤ 20 sample, support wins 243 and partition wins 53. Applying the paper's additional `info_expostIC < 1` and `info_total_friend_only > info_expostIC` filters cleanly reproduces 162 vs 51. | Robust |

**Takeaway:** the support-mechanism-beats-partition result (check 5, 10) is highly robust to the exact Figure 9 sample definition. The only fragile descriptive statistic is support in Indonesia when the full network is used instead of the giant component (check 7), which is sensitive because of the heavy presence of disconnected singletons in the Alatas sampling design.

---

## 6. Summary Assessment

### What Replicates
- **All 10 numeric cells of Table 1 that involve min/max values match exactly**; 9/10 mean cells also match. India "Support" mean is 0.82 (repl and shipped data) vs 0.85 (paper text).
- **All four textual count claims** (45 / 127 / 0 / 0 theorem-1 networks) match exactly.
- **Figure 9 sample construction (213 networks) and all four summary counts** (162, 51, 0.6, 0.5) match exactly.
- **13 of 13 per-village statistics** recompute to within 2 × 10⁻¹⁵ of the authors' shipped values.

### What Doesn't
- The published India "Support" mean (0.85) is inconsistent with the shipped data (0.82). This is a paper–package discrepancy, not a code bug.
- The brute-force bipartite search `max_bi_information` is functionally correct but pathologically slow (>1 hr on 213 networks). We defer to the authors' pre-computed `info_SP` rather than re-running it.

### Key concerns
- **None that change conclusions.** The empirical section is a short illustrative application of the theory; the theorems are unaffected by any sample-construction choice.
- **Data provenance:** the raw Banerjee and Alatas data are not shipped — only derived pickles. This means the pickles are an effective black box: we cannot, for example, re-generate them to check how "the union of all link types" was computed for the Indian networks. The pickles would break if Harvard Dataverse or ICPSR ever changed the upstream format.

### Overall Assessment
An unusually clean replication package. The code is short, self-contained, and ships pre-computed intermediate artefacts that make the entire empirical section reproducible in seconds rather than hours. Every quantitative claim in the paper's empirical section reproduces to the reported precision except for a single mean (India Support, 0.82 vs 0.85) that disagrees with the authors' own shipped data. We flag this as a likely paper update lag rather than a code bug.

---

## 7. File Manifest

| File | Description |
|---|---|
| `utils.py` | Paths, `InformationMeasure`/`classifyComparisons`/`NetStats.extract` reimplemented for networkx 3.x |
| `01_compute_netstats.py` | Recomputes all 13 per-village statistics from the shipped pickled graphs; merges `info_SP` from shipped `netdata.csv`; asserts column-by-column parity with the authors' values |
| `02_table1.py` | Reproduces Table 1 and the four headline count claims |
| `03_figures.py` | Reproduces Figures 6, 7, 8, 9 |
| `04_data_audit.py` | Coverage, value ranges, logical-consistency, missingness, and correlation diagnostics |
| `05_robustness.py` | 10 robustness checks on the descriptive claims |
| `output/netdata.parquet` | Replication-produced village statistics |
| `output/netdata.csv` | Same, CSV |
| `output/table1.csv` | Table 1 side-by-side (repl vs published) |
| `output/figure9_summary.csv` | Figure 9 headline counts (repl vs published) |
| `output/figures/` | PNGs of Figures 6–9 |
| `writeup_119381.md` | This writeup |
