"""
03_figures.py — Reproduce Figure 1 (dendrogram) from Prager (2026).

Figure 1 shows hierarchical clustering of occupations by transition probabilities,
focused on the cluster containing Production Clerks (SOC 43-5061).
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import linkage, fcluster, dendrogram
from scipy.spatial.distance import squareform
from utils import OUTPUT_DIR, load_transitions, load_soc_hierarchy

print("=" * 60)
print("03_figures.py — Reproducing Figure 1")
print("=" * 60)

# ══════════════════════════════════════════════════════════════════════
# BUILD TRANSITION MATRIX (replicating R code exactly)
# ══════════════════════════════════════════════════════════════════════

trans = load_transitions()
soc = load_soc_hierarchy()
soc_dict = dict(zip(soc['detailedoccupation'], soc['detailedname']))

# Weighted average transition probability
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

# Distance metric
dist_mat = 1 - mat.values
np.fill_diagonal(dist_mat, 0)
dist_mat[np.isnan(dist_mat)] = 1.0

# Scaled distance with winsorization
scaled = 1.0 / (1.0 - dist_mat)
finite_vals = scaled[np.isfinite(scaled) & (scaled > 0)]
winz_thresh = np.percentile(finite_vals, 90)
scaled[np.isinf(scaled) | (scaled > winz_thresh)] = winz_thresh
np.fill_diagonal(scaled, 0)

# Log distance for clustering
log_dist = np.log(scaled + 1e-10)
np.fill_diagonal(log_dist, 0)

# ══════════════════════════════════════════════════════════════════════
# FULL CLUSTERING (average linkage, matching R)
# ══════════════════════════════════════════════════════════════════════

condensed = squareform(log_dist)
Z_full = linkage(condensed, method='average')

# Cut at h=6.6 to find cluster containing 43-5061
clusters = fcluster(Z_full, t=6.6, criterion='distance')
pc_idx = row_labels.index('43-5061')
pc_cluster = clusters[pc_idx]

# Get all SOC codes in this cluster
cluster_socs = [row_labels[i] for i in range(len(row_labels)) if clusters[i] == pc_cluster]
cluster_indices = [i for i in range(len(row_labels)) if clusters[i] == pc_cluster]

print(f"\n  Production Clerk cluster: {len(cluster_socs)} occupations")
for s in cluster_socs:
    name = soc_dict.get(s, 'Unknown')
    print(f"    {s}: {name}")

# ══════════════════════════════════════════════════════════════════════
# SUBSET CLUSTERING (complete linkage for subplot, matching R)
# ══════════════════════════════════════════════════════════════════════

# R code uses method="complete" for the subset dendrogram
sub_dist = log_dist[np.ix_(cluster_indices, cluster_indices)]
sub_condensed = squareform(sub_dist)
Z_sub = linkage(sub_condensed, method='complete')

# Labels with occupation titles
sub_labels = [f"{s}: {soc_dict.get(s, 'Unknown')}" for s in cluster_socs]

# ══════════════════════════════════════════════════════════════════════
# FIGURE 1: DENDROGRAM
# ══════════════════════════════════════════════════════════════════════

fig, ax = plt.subplots(figsize=(10, 8))

dendro = dendrogram(
    Z_sub,
    orientation='right',
    labels=sub_labels,
    leaf_font_size=9,
    ax=ax,
    color_threshold=0,
    above_threshold_color='steelblue'
)

ax.set_xlabel('Distance metric (smaller = closer)', fontsize=12)
ax.set_title('Figure 1: Occupation Clusters by Transition Probability\n'
             '(Cluster containing Production Clerks, SOC 43-5061)', fontsize=11)
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/Figure1_dendrogram.png", dpi=150, bbox_inches='tight')
plt.close()
print(f"\n  Saved Figure1_dendrogram.png")

# ══════════════════════════════════════════════════════════════════════
# FULL DENDROGRAM (all occupations)
# ══════════════════════════════════════════════════════════════════════

fig, ax = plt.subplots(figsize=(14, 6))
dendro_full = dendrogram(
    Z_full,
    orientation='top',
    no_labels=True,
    ax=ax,
    color_threshold=6.6,
)
ax.axhline(y=6.6, color='red', linestyle='--', alpha=0.5, label=f'Cut height = 6.6')
ax.set_ylabel('Distance', fontsize=12)
ax.set_title('Full Dendrogram: All Occupations (average linkage on log distance)', fontsize=11)
ax.legend()
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/Figure_full_dendrogram.png", dpi=150, bbox_inches='tight')
plt.close()
print(f"  Saved Figure_full_dendrogram.png")

print("\n" + "=" * 60)
print("03_figures.py — DONE")
print("=" * 60)
