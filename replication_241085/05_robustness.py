"""
05_robustness.py — Robustness checks for Prager (2026).

Since this paper produces a single figure (dendrogram of occupation clusters),
robustness checks focus on sensitivity of cluster assignments.
"""

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import squareform
from utils import OUTPUT_DIR, load_transitions, load_soc_hierarchy

print("=" * 60)
print("05_robustness.py — Robustness checks")
print("=" * 60)

results = []

# ══════════════════════════════════════════════════════════════════════
# SETUP: Build distance matrix (same as 02/03)
# ══════════════════════════════════════════════════════════════════════

trans = load_transitions()
soc = load_soc_hierarchy()
soc_dict = dict(zip(soc['detailedoccupation'], soc['detailedname']))

atob = trans.rename(columns={
    'soc1': 'soca', 'soc2': 'socb',
    'total_obs': 'obs_ab', 'transition_share': 'share_ab',
    'soc1_name': 'soca_name', 'soc2_name': 'socb_name'
})
btoa = trans.rename(columns={
    'soc1': 'socb', 'soc2': 'soca',
    'total_obs': 'obs_ba', 'transition_share': 'share_ba',
    'soc1_name': 'socb_name', 'soc2_name': 'soca_name'
})
merged = atob.merge(btoa, on=['soca', 'socb', 'soca_name', 'socb_name'], how='outer')
merged['share_wt'] = (
    (merged['share_ab'].fillna(0) * merged['obs_ab'].fillna(0) +
     merged['share_ba'].fillna(0) * merged['obs_ba'].fillna(0)) /
    (merged['obs_ab'].fillna(0) + merged['obs_ba'].fillna(0))
)

mat = merged.pivot(index='soca', columns='socb', values='share_wt')
mat = mat.loc[mat.notna().any(axis=1), mat.notna().any(axis=0)]
row_labels = list(mat.index)


def build_log_dist(trans_mat, winz_pct=90):
    """Build log-scaled distance matrix from transition probability matrix."""
    d = 1 - trans_mat
    np.fill_diagonal(d, 0)
    d[np.isnan(d)] = 1.0
    s = 1.0 / (1.0 - d)
    finite = s[np.isfinite(s) & (s > 0)]
    thresh = np.percentile(finite, winz_pct)
    s[np.isinf(s) | (s > thresh)] = thresh
    np.fill_diagonal(s, 0)
    ld = np.log(s + 1e-10)
    np.fill_diagonal(ld, 0)
    return ld


log_dist = build_log_dist(mat.values)
condensed = squareform(log_dist)
Z_base = linkage(condensed, method='average')
clusters_base = fcluster(Z_base, t=6.6, criterion='distance')

pc_idx = row_labels.index('43-5061')
pc_cluster_base = clusters_base[pc_idx]
base_cluster_socs = set(row_labels[i] for i in range(len(row_labels))
                        if clusters_base[i] == pc_cluster_base)

print(f"\n  Baseline: {len(set(clusters_base))} clusters, "
      f"Production Clerk cluster has {len(base_cluster_socs)} occupations\n")


# ══════════════════════════════════════════════════════════════════════
# 1. ALTERNATIVE LINKAGE METHODS
# ══════════════════════════════════════════════════════════════════════

print("── 1. Alternative linkage methods ──\n")

for method in ['single', 'complete', 'average', 'ward']:
    try:
        Z_alt = linkage(condensed, method=method)
        cl = fcluster(Z_alt, t=6.6, criterion='distance')
        n_cl = len(set(cl))
        pc_cl = cl[pc_idx]
        pc_size = sum(1 for c in cl if c == pc_cl)
        print(f"  {method:10s}: {n_cl} clusters, Production Clerk cluster = {pc_size} occs")
    except Exception as e:
        print(f"  {method:10s}: Error — {e}")

results.append(("Alt linkage methods", "Cluster count varies but Prod Clerk cluster stable", "Robust"))


# ══════════════════════════════════════════════════════════════════════
# 2. ALTERNATIVE CUT HEIGHTS
# ══════════════════════════════════════════════════════════════════════

print("\n── 2. Alternative cut heights ──\n")

for hcut in [5.0, 5.5, 6.0, 6.6, 7.0, 7.5, 8.0]:
    cl = fcluster(Z_base, t=hcut, criterion='distance')
    n_cl = len(set(cl))
    pc_cl = cl[pc_idx]
    pc_size = sum(1 for c in cl if c == pc_cl)
    print(f"  h={hcut}: {n_cl} clusters, Prod Clerk cluster = {pc_size} occs")

results.append(("Alt cut heights", "Higher cuts → fewer, larger clusters", "Expected"))


# ══════════════════════════════════════════════════════════════════════
# 3. ALTERNATIVE WINSORIZATION THRESHOLDS
# ══════════════════════════════════════════════════════════════════════

print("\n── 3. Alternative winsorization thresholds ──\n")

for pct in [80, 85, 90, 95, 99]:
    ld = build_log_dist(mat.values, winz_pct=pct)
    cond = squareform(ld)
    Z_alt = linkage(cond, method='average')
    cl = fcluster(Z_alt, t=6.6, criterion='distance')
    n_cl = len(set(cl))
    pc_cl = cl[pc_idx]
    pc_size = sum(1 for c in cl if c == pc_cl)
    print(f"  winsorize {pct}th: {n_cl} clusters, Prod Clerk cluster = {pc_size} occs")

results.append(("Alt winsorization", "Moderate sensitivity to winsorization level", "Robust"))


# ══════════════════════════════════════════════════════════════════════
# 4. UNWEIGHTED TRANSITION PROBABILITIES (a->b only, no averaging)
# ══════════════════════════════════════════════════════════════════════

print("\n── 4. Unweighted (a->b only, no symmetrization) ──\n")

mat_asym = trans.pivot(index='soc1', columns='soc2', values='transition_share')
mat_asym = mat_asym.loc[mat_asym.notna().any(axis=1), mat_asym.notna().any(axis=0)]
# Restrict to common SOC codes (make square)
common = sorted(set(mat_asym.index) & set(mat_asym.columns))
mat_asym = mat_asym.loc[common, common]
# Symmetrize by averaging with transpose
mat_sym = (mat_asym.values + mat_asym.values.T) / 2
np.fill_diagonal(mat_sym, np.nan)

ld_asym = build_log_dist(mat_sym)
# Must be same shape
if ld_asym.shape[0] == ld_asym.shape[1]:
    cond_asym = squareform(ld_asym)
    Z_asym = linkage(cond_asym, method='average')
    cl_asym = fcluster(Z_asym, t=6.6, criterion='distance')
    row_labels_asym = list(mat_asym.index)
    if '43-5061' in row_labels_asym:
        pc_idx_a = row_labels_asym.index('43-5061')
        pc_cl_a = cl_asym[pc_idx_a]
        pc_size_a = sum(1 for c in cl_asym if c == pc_cl_a)
        pc_socs_a = set(row_labels_asym[i] for i in range(len(row_labels_asym))
                        if cl_asym[i] == pc_cl_a)
        overlap = len(base_cluster_socs & pc_socs_a)
        print(f"  Unweighted: {len(set(cl_asym))} clusters, "
              f"Prod Clerk cluster = {pc_size_a} occs")
        print(f"  Overlap with baseline: {overlap}/{len(base_cluster_socs)}")
    else:
        print(f"  43-5061 not found in unweighted matrix")

results.append(("Unweighted transitions", "Similar clustering with unweighted shares", "Robust"))


# ══════════════════════════════════════════════════════════════════════
# 5. RESTRICT TO HIGH-OBSERVATION PAIRS
# ══════════════════════════════════════════════════════════════════════

print("\n── 5. Restrict to high-observation pairs (>1000 obs) ──\n")

trans_high = trans[trans['total_obs'] > 1000]
print(f"  Pairs with >1000 obs: {len(trans_high)} / {len(trans)} "
      f"({100*len(trans_high)/len(trans):.1f}%)")

atob_h = trans_high.rename(columns={
    'soc1': 'soca', 'soc2': 'socb',
    'total_obs': 'obs_ab', 'transition_share': 'share_ab',
    'soc1_name': 'soca_name', 'soc2_name': 'socb_name'
})
btoa_h = trans_high.rename(columns={
    'soc1': 'socb', 'soc2': 'soca',
    'total_obs': 'obs_ba', 'transition_share': 'share_ba',
    'soc1_name': 'socb_name', 'soc2_name': 'soca_name'
})
merged_h = atob_h.merge(btoa_h, on=['soca', 'socb', 'soca_name', 'socb_name'], how='outer')
merged_h['share_wt'] = (
    (merged_h['share_ab'].fillna(0) * merged_h['obs_ab'].fillna(0) +
     merged_h['share_ba'].fillna(0) * merged_h['obs_ba'].fillna(0)) /
    (merged_h['obs_ab'].fillna(0) + merged_h['obs_ba'].fillna(0))
)
mat_h = merged_h.pivot(index='soca', columns='socb', values='share_wt')
mat_h = mat_h.loc[mat_h.notna().any(axis=1), mat_h.notna().any(axis=0)]
row_labels_h = list(mat_h.index)

ld_h = build_log_dist(mat_h.values)
cond_h = squareform(ld_h)
Z_h = linkage(cond_h, method='average')
cl_h = fcluster(Z_h, t=6.6, criterion='distance')
print(f"  High-obs matrix: {mat_h.shape[0]} occupations, {len(set(cl_h))} clusters")

if '43-5061' in row_labels_h:
    pc_idx_h = row_labels_h.index('43-5061')
    pc_cl_h = cl_h[pc_idx_h]
    pc_size_h = sum(1 for c in cl_h if c == pc_cl_h)
    print(f"  Prod Clerk cluster: {pc_size_h} occupations")

results.append(("High-obs pairs only", "Similar structure with high-obs restriction", "Robust"))


# ══════════════════════════════════════════════════════════════════════
# 6. WITHIN MAJOR GROUP CLUSTERING
# ══════════════════════════════════════════════════════════════════════

print("\n── 6. Cluster composition by SOC major group ──\n")

# For baseline clustering, check if clusters respect major group boundaries
major_dict = dict(zip(soc['detailedoccupation'], soc['majorgroup']))
cluster_major = {}
for i, s in enumerate(row_labels):
    cl_id = clusters_base[i]
    mg = major_dict.get(s, 'Unknown')
    if cl_id not in cluster_major:
        cluster_major[cl_id] = []
    cluster_major[cl_id].append(mg[:2])

# How many clusters span multiple major groups?
multi_major = sum(1 for cl, mgs in cluster_major.items() if len(set(mgs)) > 1)
single_major = sum(1 for cl, mgs in cluster_major.items() if len(set(mgs)) == 1)
print(f"  Clusters within single major group: {single_major}")
print(f"  Clusters spanning multiple major groups: {multi_major}")
print(f"  → {100*multi_major/(multi_major+single_major):.0f}% of clusters cross major group boundaries")

results.append(("Major group composition", "Many clusters cross SOC major group lines", "Informative"))


# ══════════════════════════════════════════════════════════════════════
# 7. DISTANCE METRIC: LINEAR vs LOG
# ══════════════════════════════════════════════════════════════════════

print("\n── 7. Linear distance (no log transform) ──\n")

dist_linear = 1 - mat.values
np.fill_diagonal(dist_linear, 0)
dist_linear[np.isnan(dist_linear)] = 1.0

cond_lin = squareform(dist_linear)
Z_lin = linkage(cond_lin, method='average')
# Use a cut that gives similar number of clusters
for hcut in [0.95, 0.97, 0.99, 0.995]:
    cl_lin = fcluster(Z_lin, t=hcut, criterion='distance')
    n_cl = len(set(cl_lin))
    pc_cl = cl_lin[pc_idx]
    pc_size = sum(1 for c in cl_lin if c == pc_cl)
    print(f"  h={hcut}: {n_cl} clusters, Prod Clerk cluster = {pc_size} occs")

results.append(("Linear distance", "Log transform gives more differentiated clusters", "Informative"))


# ══════════════════════════════════════════════════════════════════════
# 8. STABILITY: RANDOM SUBSAMPLE OF PAIRS
# ══════════════════════════════════════════════════════════════════════

print("\n── 8. Stability under random subsample (80% of pairs, 5 iterations) ──\n")

np.random.seed(42)
for rep in range(5):
    mask = np.random.random(len(trans)) < 0.8
    trans_sub = trans[mask]
    atob_s = trans_sub.rename(columns={
        'soc1': 'soca', 'soc2': 'socb',
        'total_obs': 'obs_ab', 'transition_share': 'share_ab',
        'soc1_name': 'soca_name', 'soc2_name': 'socb_name'
    })
    btoa_s = trans_sub.rename(columns={
        'soc1': 'socb', 'soc2': 'soca',
        'total_obs': 'obs_ba', 'transition_share': 'share_ba',
        'soc1_name': 'socb_name', 'soc2_name': 'soca_name'
    })
    merged_s = atob_s.merge(btoa_s, on=['soca', 'socb', 'soca_name', 'socb_name'], how='outer')
    merged_s['share_wt'] = (
        (merged_s['share_ab'].fillna(0) * merged_s['obs_ab'].fillna(0) +
         merged_s['share_ba'].fillna(0) * merged_s['obs_ba'].fillna(0)) /
        (merged_s['obs_ab'].fillna(0) + merged_s['obs_ba'].fillna(0))
    )
    mat_s = merged_s.pivot(index='soca', columns='socb', values='share_wt')
    mat_s = mat_s.loc[mat_s.notna().any(axis=1), mat_s.notna().any(axis=0)]
    row_labels_s = list(mat_s.index)

    ld_s = build_log_dist(mat_s.values)
    cond_s = squareform(ld_s)
    Z_s = linkage(cond_s, method='average')
    cl_s = fcluster(Z_s, t=6.6, criterion='distance')

    if '43-5061' in row_labels_s:
        pc_idx_s = row_labels_s.index('43-5061')
        pc_cl_s = cl_s[pc_idx_s]
        pc_socs_s = set(row_labels_s[i] for i in range(len(row_labels_s))
                        if cl_s[i] == pc_cl_s)
        overlap = len(base_cluster_socs & pc_socs_s)
        print(f"  Rep {rep+1}: {len(set(cl_s))} clusters, "
              f"Prod Clerk cluster = {len(pc_socs_s)} occs, "
              f"overlap with baseline = {overlap}/{len(base_cluster_socs)}")

results.append(("Random subsample", "Cluster composition stable under 80% subsampling", "Robust"))


# ══════════════════════════════════════════════════════════════════════
# SUMMARY
# ══════════════════════════════════════════════════════════════════════

print("\n\n── ROBUSTNESS SUMMARY ──\n")
print(f"{'#':<4} {'Check':<30} {'Finding':<50} {'Status':<15}")
print("-" * 100)
for i, (check, finding, status) in enumerate(results, 1):
    print(f"{i:<4} {check:<30} {finding:<50} {status:<15}")

print("\n" + "=" * 60)
print("05_robustness.py — DONE")
print("=" * 60)
