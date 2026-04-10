"""Data audit for the Bloch-Olckers network dataset.

Coverage, value ranges, and logical consistency of the village network
statistics underlying Table 1.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from utils import OUTPUT, giant_component, load_networks
import networkx as nx


def main():
    df = pd.read_parquet(OUTPUT / "netdata.parquet")

    print("=" * 70)
    print("Data audit — Bloch & Olckers network dataset")
    print("=" * 70)

    print("\n[1] Coverage")
    print(f"  Total networks:           {len(df)}")
    print(f"  India:                    {(df['country']=='India').sum()}  (expected 75)")
    print(f"  Indonesia:                {(df['country']=='Indonesia').sum()}  (expected 633)")

    print("\n[2] Giant-component sizes")
    for country in ["India", "Indonesia"]:
        sub = df[df["country"] == country]["num_nodes"]
        print(f"  {country:<10} n: mean={sub.mean():.1f}  min={sub.min()}  max={sub.max()}  "
              f"median={sub.median():.0f}")

    print("\n[3] Value-range sanity (should all be in [0, 1])")
    for col in ["density", "ave_clust", "info_total_friend_only",
                "info_total", "info_expostIC", "links_supported",
                "comp_total", "comp_supp", "comp_trans", "comp_by_three"]:
        vals = df[col].dropna()
        bad = ((vals < 0) | (vals > 1)).sum()
        print(f"  {col:<26} min={vals.min():.4f}  max={vals.max():.4f}  out-of-[0,1]={bad}")

    print("\n[4] Logical consistency")
    # Comparison density = friend-only + self-comparison share; friend_only <= info_total
    assert (df["info_total_friend_only"] <= df["info_total"] + 1e-9).all()
    print("  info_total_friend_only <= info_total:  OK")
    # Supported density <= comparison density
    assert (df["comp_supp"] <= df["info_total_friend_only"] + 1e-9).all()
    print("  comp_supp <= info_total_friend_only:   OK")
    # Density of social <= 1
    assert (df["density"] <= 1 + 1e-9).all()
    print("  density <= 1:                          OK")
    # info_expostIC <= info_total_friend_only (removing links can only lose info)
    # This holds definitionally since g_supp is a subgraph of g, but confirm.
    assert (df["info_expostIC"] <= df["info_total_friend_only"] + 1e-9).all()
    print("  info_expostIC <= info_total_friend:    OK")

    print("\n[5] Missing values")
    for col in df.columns:
        nmiss = df[col].isna().sum()
        if nmiss > 0:
            print(f"  {col}: {nmiss} missing")
    print("  (info_SP is expected missing whenever n > 20)")
    big = df[df["num_nodes"] > 20]
    print(f"  info_SP missing for n>20 only: "
          f"{big['info_SP'].isna().all()} (all of {len(big)} such rows)")

    print("\n[6] Connectedness of giant components")
    india, indo = load_networks()
    n_india_conn = sum(nx.is_connected(giant_component(g)) for g in india.values())
    n_indo_conn = sum(nx.is_connected(giant_component(g)) for g in indo.values())
    print(f"  India giant components connected:     {n_india_conn}/75")
    print(f"  Indonesia giant components connected: {n_indo_conn}/633")

    print("\n[7] Pairwise correlations (India vs Indonesia separately)")
    cols = ["density", "ave_clust", "info_total_friend_only", "links_supported"]
    for country in ["India", "Indonesia"]:
        sub = df[df["country"] == country][cols]
        print(f"\n  {country}")
        print(sub.corr().round(3).to_string())

    print("\n[8] Density vs clustering — well-known positive correlation")
    for country in ["India", "Indonesia"]:
        sub = df[df["country"] == country]
        r = np.corrcoef(sub["density"], sub["ave_clust"])[0, 1]
        print(f"  {country:<10} corr(density, clustering) = {r:.3f}")

    print("\n[9] Networks with exact theorem 1 property (info_total_friend_only == 1)")
    for country in ["India", "Indonesia"]:
        sub = df[df["country"] == country]
        n = (sub["info_total_friend_only"] >= 0.9999).sum()
        print(f"  {country:<10} {n}")

    print("\n[10] Networks with all links supported (links_supported == 1)")
    for country in ["India", "Indonesia"]:
        sub = df[df["country"] == country]
        n = (sub["links_supported"] >= 0.9999).sum()
        print(f"  {country:<10} {n}")

    print("\nAudit complete.")


if __name__ == "__main__":
    main()
