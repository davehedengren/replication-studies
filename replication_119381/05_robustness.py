"""Robustness checks for Bloch & Olckers (2020).

The paper's empirical section is descriptive — it reports histograms,
summary statistics, and a support-vs-partition comparison. Robustness here
therefore probes the stability of those descriptive claims across:

  1) full network vs giant component
  2) alternative "support" thresholds (0.8, 0.9) for the text claim
  3) dropping the smallest networks (n<=5, which are fragile in Indonesia)
  4) dropping the largest networks (n>=300, India long tail)
  5) support vs partition comparison on n<=15 and n<=18 subsamples
  6) including self-comparisons in the comparison-network density
  7) alternative measure: compute comparison density on full graph (not giant)
  8) correlation between density and support — sensitivity to country mix
  9) placebo random graphs of matched size/density: do support rates differ?
 10) support-vs-partition: tie-breaking sensitivity (>= vs > 45-degree line)
"""
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import networkx as nx

sys.path.insert(0, str(Path(__file__).resolve().parent))
from utils import (OUTPUT, classify_comparisons, extract_network_stats,
                   giant_component, information_measure, load_networks,
                   remove_unsupported_edges)


def check1_full_vs_giant():
    """Summary stats on full network vs giant component."""
    india, indo = load_networks()
    rows = []
    for country, nets in [("India", india), ("Indonesia", indo)]:
        for key, g in nets.items():
            n_full = nx.number_of_nodes(g)
            n_giant = nx.number_of_nodes(giant_component(g))
            rows.append((country, key, n_full, n_giant, n_giant / n_full))
    df = pd.DataFrame(rows, columns=["country", "key", "n_full", "n_giant", "giant_share"])
    print("\n[1] Giant-component share of original network")
    for c in ["India", "Indonesia"]:
        s = df[df["country"] == c]["giant_share"]
        print(f"  {c:<10} mean={s.mean():.3f}  min={s.min():.3f}  "
              f"networks with giant_share<0.9: {(s < 0.9).sum()}")


def check2_support_thresholds(df):
    print("\n[2] Support-threshold sensitivity (text claim: Indonesia = 127 full, India = 0)")
    for country in ["India", "Indonesia"]:
        sub = df[df["country"] == country]["links_supported"]
        for thr in [0.80, 0.90, 0.95, 0.9999]:
            n = (sub >= thr).sum()
            print(f"  {country:<10} >= {thr:.4f}: {n} / {len(sub)}")


def check3_drop_small(df):
    print("\n[3] Drop smallest Indonesian networks")
    indo = df[df["country"] == "Indonesia"]
    for thr in [2, 5, 10]:
        sub = indo[indo["num_nodes"] > thr]
        print(f"  n > {thr:2d}: {len(sub):3d} nets  mean density={sub['density'].mean():.3f}  "
              f"mean support={sub['links_supported'].mean():.3f}  "
              f"mean comp_density={sub['info_total_friend_only'].mean():.3f}")


def check4_drop_large(df):
    print("\n[4] Drop largest Indian networks")
    india = df[df["country"] == "India"]
    for thr in [300, 250, 200]:
        sub = india[india["num_nodes"] <= thr]
        print(f"  n <= {thr:3d}: {len(sub):2d} nets  mean density={sub['density'].mean():.3f}  "
              f"mean support={sub['links_supported'].mean():.3f}  "
              f"mean comp_density={sub['info_total_friend_only'].mean():.3f}")


def check5_support_vs_partition_subsamples(df):
    print("\n[5] Support-vs-partition comparison under alternate size cutoffs")
    for cutoff in [15, 18, 20]:
        sub = df[(df["num_nodes"] <= cutoff) & df["info_SP"].notna()]
        x = sub["info_expostIC"].values
        y = sub["info_SP"].values
        below = (y < x).sum()
        above = (y >= x).sum()
        print(f"  n<={cutoff}: {len(sub):3d} nets  "
              f"support-mean={x.mean():.3f}  partition-mean={y.mean():.3f}  "
              f"support wins {below}, partition wins {above}")


def check6_with_self(df):
    print("\n[6] Include self-comparisons in comparison-network density")
    for country in ["India", "Indonesia"]:
        sub = df[df["country"] == country]
        print(f"  {country:<10} friend-only mean={sub['info_total_friend_only'].mean():.3f}  "
              f"with-self mean={sub['info_total'].mean():.3f}  "
              f"diff={sub['info_total'].mean() - sub['info_total_friend_only'].mean():.4f}")


def check7_full_network_stats():
    """Recompute summary stats on the full network (not just giant component)."""
    print("\n[7] Table 1 stats computed on full network (giant=False)")
    india, indo = load_networks()
    rows = []
    for country, nets in [("India", india), ("Indonesia", indo)]:
        for key, g in nets.items():
            r = extract_network_stats(g, key=key, country=country,
                                      compute_bipartite=False, giant=False)
            rows.append(r)
    full = pd.DataFrame(rows)
    cols = ["density", "ave_clust", "info_total_friend_only", "links_supported"]
    for country in ["India", "Indonesia"]:
        sub = full[full["country"] == country][cols]
        print(f"  {country}")
        for c in cols:
            print(f"    {c:<26} mean={sub[c].mean():.3f}")


def check8_density_support_corr(df):
    print("\n[8] Correlation between density and support (by country)")
    for country in ["India", "Indonesia"]:
        sub = df[df["country"] == country]
        r = np.corrcoef(sub["density"], sub["links_supported"])[0, 1]
        print(f"  {country:<10} corr(density, support) = {r:.3f}")
    r_all = np.corrcoef(df["density"], df["links_supported"])[0, 1]
    print(f"  Pooled:    corr(density, support) = {r_all:.3f}")


def check9_placebo_random_graphs(df):
    """Does the observed support share exceed what an Erdős-Rényi G(n, p) would give?"""
    print("\n[9] Placebo: Erdős-Rényi graphs matched on n and density")
    rng = np.random.default_rng(2026)
    # Sample 100 networks stratified by country
    for country in ["India", "Indonesia"]:
        sub = df[df["country"] == country].sample(min(50, (df["country"] == country).sum()),
                                                  random_state=0)
        obs_support = []
        er_support = []
        for _, row in sub.iterrows():
            n = int(row["num_nodes"])
            p = float(row["density"])
            obs_support.append(row["links_supported"])
            g_er = nx.fast_gnp_random_graph(n, p, seed=int(rng.integers(1e9)))
            if g_er.number_of_edges() == 0:
                er_support.append(np.nan)
                continue
            g_er_supp = remove_unsupported_edges(g_er)
            er_support.append(g_er_supp.number_of_edges() / g_er.number_of_edges())
        obs = np.array(obs_support)
        er = np.array(er_support, dtype=float)
        mask = ~np.isnan(er)
        print(f"  {country:<10} n={mask.sum():2d}  observed mean support={obs[mask].mean():.3f}  "
              f"ER-matched mean={er[mask].mean():.3f}")


def check10_tie_sensitivity(df):
    print("\n[10] Support-vs-partition: strict vs weak inequality")
    sub = df[df["num_nodes"] <= 20].dropna(subset=["info_SP"])
    x = sub["info_expostIC"].values
    y = sub["info_SP"].values
    print(f"  partition > support (strictly): {(y > x).sum()}")
    print(f"  partition == support:           {np.isclose(y, x).sum()}")
    print(f"  partition >= support:           {(y >= x).sum()}  (matches pub: 51)")
    print(f"  partition < support (strictly): {(y < x).sum()}  (matches pub: 162)")


def main():
    df = pd.read_parquet(OUTPUT / "netdata.parquet")
    t0 = time.time()
    check1_full_vs_giant()
    check2_support_thresholds(df)
    check3_drop_small(df)
    check4_drop_large(df)
    check5_support_vs_partition_subsamples(df)
    check6_with_self(df)
    check7_full_network_stats()
    check8_density_support_corr(df)
    check9_placebo_random_graphs(df)
    check10_tie_sensitivity(df)
    print(f"\nRobustness checks complete in {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
