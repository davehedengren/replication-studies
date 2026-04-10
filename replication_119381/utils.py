"""Shared helpers for replication 119381 (Bloch & Olckers, Friend-Based Ranking)."""
from pathlib import Path
import itertools
import numpy as np
import networkx as nx

REPO_ROOT = Path(__file__).resolve().parents[1]
PKG_ROOT = REPO_ROOT / "119381-V1" / "Friend-based_Ranking"
NB_ROOT = PKG_ROOT / "Notebooks"
PD_DF = NB_ROOT / "pd_df"
OUTPUT = REPO_ROOT / "replication_119381" / "output"
OUTPUT.mkdir(parents=True, exist_ok=True)


def _adj(g):
    """Dense 0/1 adjacency array for a simple undirected graph."""
    return nx.to_numpy_array(g, dtype=int, weight=None)


def information_measure(g, include_self_comparisons=True):
    """Density of the comparison network induced by g.

    Rebuilt from fbr_functions.InformationMeasure using nx.to_numpy_array
    (nx.to_numpy_matrix was removed in networkx 3.x).
    """
    G = _adj(g)
    H = G @ G
    if include_self_comparisons:
        H = G + H
    np.fill_diagonal(H, 0)
    H[H > 1] = 1
    n = nx.number_of_nodes(g)
    return H.sum() / (n * (n - 1))


def remove_unsupported_edges(g):
    """Drop any edge (i,j) not part of a triangle."""
    G = _adj(g)
    G2 = G @ G
    G2[G2 > 1] = 1
    G_supp = G * G2
    return nx.from_numpy_array(G_supp)


def classify_comparisons(g):
    """Classify pairs (i,j) into supported / transitive / compared-by-three.

    Counts are over i != j, undirected (each ordered pair counted once each way,
    then scaled by n*(n-1) downstream, matching the original code).
    """
    result = {}
    G = _adj(g)
    G2 = G @ G
    H = G2.copy()
    H[H > 1] = 1
    np.fill_diagonal(H, 0)
    result["Total"] = H.sum()
    G_supp = G * H
    result["Supported"] = G_supp.sum()
    G_supp_2 = G_supp @ G_supp
    G_supp_2[G_supp_2 > 1] = 1
    np.fill_diagonal(G_supp_2, 0)
    result["Transitive"] = (G_supp_2 * (1 - G_supp)).sum()
    G_comp3 = G2.copy()
    G_comp3[G_comp3 < 3] = 0
    G_comp3[G_comp3 >= 3] = 1
    np.fill_diagonal(G_comp3, 0)
    result["By three"] = (G_comp3 * (1 - G_supp_2)).sum()
    return result


def remove_within_group_edges(g, A):
    bipart = g.copy()
    x = nx.subgraph(g, nbunch=A)
    if len(x.edges()) > 0:
        bipart.remove_edges_from(list(x.edges()))
    B = set(g.nodes) - set(A)
    y = nx.subgraph(g, nbunch=B)
    if len(y.edges()) > 0:
        bipart.remove_edges_from(list(y.edges()))
    return bipart


def max_bi_information(g):
    """Brute-force optimal bipartite partition (for small networks, n<=20).

    Uses `range(1, n//2)` to match the original fbr_functions.max_bi_information
    exactly — this only matters for exact parity with the shipped notebook.
    """
    nodes = list(g.nodes)
    max_info = 0.0
    max_info_nodes = []
    for r in range(1, len(nodes) // 2):
        for s in itertools.combinations(nodes, r):
            b = remove_within_group_edges(g, s)
            info = information_measure(b, include_self_comparisons=False)
            if info > max_info:
                max_info = info
                max_info_nodes = list(s)
    return {"max_bi_info": max_info, "max_bi_info_nodes": max_info_nodes}


def giant_component(g):
    return g.subgraph(max(nx.connected_components(g), key=len)).copy()


def extract_network_stats(g, key, country, compute_bipartite=True, giant=True):
    """Replacement for NetStats.extract, returns a dict row."""
    if giant:
        g = giant_component(g)
    n = nx.number_of_nodes(g)
    m = nx.number_of_edges(g)
    row = {
        "key": key,
        "country": country,
        "num_nodes": n,
        "num_edges": m,
        "ave_deg": 2 * m / n,
        "density": nx.density(g),
        "ave_clust": nx.average_clustering(g),
    }
    if nx.is_connected(g):
        row["ave_dist"] = nx.average_shortest_path_length(g)
        row["diameter"] = nx.diameter(g)
    else:
        row["ave_dist"] = np.nan
        row["diameter"] = np.nan

    g_supp = remove_unsupported_edges(g)
    row["links_supported"] = (nx.number_of_edges(g_supp) / m) if m else np.nan
    row["info_total"] = information_measure(g)
    row["info_total_friend_only"] = information_measure(g, include_self_comparisons=False)
    row["info_expostIC"] = information_measure(g_supp, include_self_comparisons=False)

    if compute_bipartite and n <= 20:
        row["info_SP"] = max_bi_information(g)["max_bi_info"]
    else:
        row["info_SP"] = np.nan

    comp = classify_comparisons(g)
    row["comp_total"] = comp["Total"] / (n * (n - 1))
    row["comp_supp"] = comp["Supported"] / (n * (n - 1))
    row["comp_trans"] = comp["Transitive"] / (n * (n - 1))
    row["comp_by_three"] = comp["By three"] / (n * (n - 1))
    return row


def load_networks():
    """Load the pickled NetStats objects and return dicts key->graph (raw)."""
    import pickle, sys
    sys.path.insert(0, str(NB_ROOT))
    with open(PD_DF / "india_networks.pickle", "rb") as f:
        ind = pickle.load(f)
    with open(PD_DF / "indo_networks.pickle", "rb") as f:
        ino = pickle.load(f)
    india = {str(k): ind.networks[str(k)]["g"] for k in ind.key_list}
    indo = {str(k): ino.networks[str(k)]["g"] for k in ino.key_list}
    return india, indo
