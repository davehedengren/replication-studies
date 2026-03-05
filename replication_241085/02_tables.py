"""
02_tables.py — No formal tables in this paper's replication package.

Prager (2026) replication package only produces Figure 1 (dendrogram).
This script reports the cluster summary statistics that would appear in text.
"""

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import linkage, fcluster, dendrogram
from utils import OUTPUT_DIR, load_transitions, load_soc_hierarchy

print("=" * 60)
print("02_tables.py — Cluster summary statistics")
print("=" * 60)

# ══════════════════════════════════════════════════════════════════════
# BUILD TRANSITION MATRIX (same as R code)
# ══════════════════════════════════════════════════════════════════════

trans = load_transitions()
soc = load_soc_hierarchy()

# Compute weighted average transition probability of a->b, b->a
# Merge a->b with b->a
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

# Weighted average transition probability
merged['share_wt'] = (
    (merged['share_ab'].fillna(0) * merged['obs_ab'].fillna(0) +
     merged['share_ba'].fillna(0) * merged['obs_ba'].fillna(0)) /
    (merged['obs_ab'].fillna(0) + merged['obs_ba'].fillna(0))
)

# Correlation between a->b and b->a shares (R reports 0.058)
valid = merged.dropna(subset=['share_ab', 'share_ba'])
corr_ab_ba = valid['share_ab'].corr(valid['share_ba'])
print(f"\n  Correlation(share_ab, share_ba): {corr_ab_ba:.3f}")
print(f"  (R code reports: 0.058)")

# Pivot to matrix
mat = merged.pivot(index='soca', columns='socb', values='share_wt')

print(f"\n  Raw matrix shape: {mat.shape}")

# Drop rows/cols that are all NaN
mat = mat.loc[mat.notna().any(axis=1), mat.notna().any(axis=0)]
print(f"  After dropping all-NaN rows/cols: {mat.shape}")

# ══════════════════════════════════════════════════════════════════════
# DISTANCE METRIC
# ══════════════════════════════════════════════════════════════════════

# Distance = 1 - transition_probability
dist_mat = 1 - mat.values
# Self-to-self = 0
np.fill_diagonal(dist_mat, 0)
# NAs → max distance (1)
dist_mat[np.isnan(dist_mat)] = 1.0

# Verify symmetry
corr_sym = np.corrcoef(dist_mat.flatten(), dist_mat.T.flatten())[0, 1]
print(f"  Symmetry check (corr with transpose): {corr_sym:.6f}")

# Scaled distance: 1/(1-d), winsorized at 90th percentile
scaled = 1.0 / (1.0 - dist_mat)
finite_vals = scaled[np.isfinite(scaled) & (scaled > 0)]
winz_thresh = np.percentile(finite_vals, 90)
scaled[np.isinf(scaled) | (scaled > winz_thresh)] = winz_thresh
np.fill_diagonal(scaled, 0)

print(f"  Winsorization threshold (90th pctl): {winz_thresh:.2f}")

# ══════════════════════════════════════════════════════════════════════
# HIERARCHICAL CLUSTERING
# ══════════════════════════════════════════════════════════════════════

log_dist = np.log(scaled + 1e-10)  # avoid log(0)
np.fill_diagonal(log_dist, 0)

# Average linkage clustering (matching R's method="average")
from scipy.spatial.distance import squareform
condensed = squareform(log_dist)
Z = linkage(condensed, method='average')

# Cut tree at h=6.6 (matching R code)
hcut = 6.6
clusters = fcluster(Z, t=hcut, criterion='distance')
n_clusters = len(set(clusters))
print(f"\n  Cut height: {hcut}")
print(f"  Number of clusters: {n_clusters}")

# Cluster size distribution
cluster_sizes = pd.Series(clusters).value_counts()
print(f"  Largest cluster: {cluster_sizes.max()} occupations")
print(f"  Median cluster size: {cluster_sizes.median():.0f}")
print(f"  Singleton clusters: {(cluster_sizes == 1).sum()}")

# Map SOC labels
row_labels = list(mat.index)
soc_dict = dict(zip(soc['detailedoccupation'], soc['detailedname']))

# Check CEO cluster (SOC 11-1011)
if '11-1011' in row_labels:
    ceo_idx = row_labels.index('11-1011')
    ceo_cluster = clusters[ceo_idx]
    ceo_peers = [row_labels[i] for i in range(len(row_labels)) if clusters[i] == ceo_cluster]
    print(f"\n  CEO cluster (SOC 11-1011) contains {len(ceo_peers)} occupations:")
    for s in ceo_peers[:10]:
        name = soc_dict.get(s, 'Unknown')
        print(f"    {s}: {name}")
    if len(ceo_peers) > 10:
        print(f"    ... and {len(ceo_peers) - 10} more")

# Check Production Clerks cluster (SOC 43-5061, used in R code)
if '43-5061' in row_labels:
    pc_idx = row_labels.index('43-5061')
    pc_cluster = clusters[pc_idx]
    pc_peers = [row_labels[i] for i in range(len(row_labels)) if clusters[i] == pc_cluster]
    print(f"\n  Production Clerk cluster (43-5061) contains {len(pc_peers)} occupations")
    print(f"  (This is the cluster used for Figure 1 in the paper)")

print("\n" + "=" * 60)
print("02_tables.py — DONE")
print("=" * 60)
