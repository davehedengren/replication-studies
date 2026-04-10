"""Reproduce Table 1 (summary statistics of social networks) from netdata.parquet.

Published values (Bloch & Olckers 2020, Table 1):

                                    India                 Indonesia
                                    mean [min, max]       mean [min, max]
    Networks                        75                    633
    Households in giant comp.       188.87 [75, 341]      23.6 [2, 82]
    Density of social network       0.05 [0.02, 0.12]     0.36 [0.09, 1.00]
    Average clustering              0.26 [0.16, 0.45]     0.73 [0.00, 1.00]
    Density of comparison network   0.37 [0.18, 0.62]     0.70 [0.00, 1.00]
    Support                         0.85 [0.68, 0.95]     0.95 [0.00, 1.00]

"Support" in the paper is the share of supported links (`links_supported`
in netdata); "Density of comparison network" is `info_total_friend_only`.
"""
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from utils import OUTPUT

VARS = [
    ("num_nodes", "Households in giant component"),
    ("density", "Density of social network"),
    ("ave_clust", "Average clustering"),
    ("info_total_friend_only", "Density of comparison network"),
    ("links_supported", "Support"),
]

PUBLISHED = {
    "India": {
        "num_nodes": (188.87, 75, 341),
        "density": (0.05, 0.02, 0.12),
        "ave_clust": (0.26, 0.16, 0.45),
        "info_total_friend_only": (0.37, 0.18, 0.62),
        "links_supported": (0.85, 0.68, 0.95),
    },
    "Indonesia": {
        "num_nodes": (23.6, 2, 82),
        "density": (0.36, 0.09, 1.00),
        "ave_clust": (0.73, 0.00, 1.00),
        "info_total_friend_only": (0.70, 0.00, 1.00),
        "links_supported": (0.95, 0.00, 1.00),
    },
}


def fmt(x, decimals=2):
    return f"{x:.{decimals}f}" if isinstance(x, float) else str(x)


def main():
    df = pd.read_parquet(OUTPUT / "netdata.parquet")
    assert (df["country"] == "India").sum() == 75
    assert (df["country"] == "Indonesia").sum() == 633

    rows = []
    for var, label in VARS:
        for country in ["India", "Indonesia"]:
            sub = df[df["country"] == country][var]
            pub_mean, pub_min, pub_max = PUBLISHED[country][var]
            rows.append({
                "variable": label,
                "country": country,
                "repl_mean": sub.mean(),
                "repl_min": sub.min(),
                "repl_max": sub.max(),
                "pub_mean": pub_mean,
                "pub_min": pub_min,
                "pub_max": pub_max,
            })
    out = pd.DataFrame(rows)
    out.to_csv(OUTPUT / "table1.csv", index=False)

    # Print comparison
    print("\nTable 1 — Summary statistics of social networks")
    print("=" * 86)
    print(f"{'Variable':<32} {'Country':<10} {'Repl mean[min,max]':<25} {'Pub mean[min,max]':<25}")
    print("-" * 86)
    for _, r in out.iterrows():
        repl = f"{r.repl_mean:.2f} [{r.repl_min:.2f}, {r.repl_max:.2f}]"
        pub = f"{r.pub_mean:.2f} [{r.pub_min:.2f}, {r.pub_max:.2f}]"
        print(f"{r.variable:<32} {r.country:<10} {repl:<25} {pub:<25}")
    print("=" * 86)
    print(f"Networks: India={(df['country']=='India').sum()} Indonesia={(df['country']=='Indonesia').sum()}")

    # Headline counts from the text
    print()
    print("Published textual claims (page 24):")
    indo = df[df["country"] == "Indonesia"]
    india = df[df["country"] == "India"]

    n_indo_full_comp = (indo["info_total_friend_only"] >= 0.9999).sum()
    n_indo_full_supp = (indo["links_supported"] >= 0.9999).sum()
    n_india_full_comp = (india["info_total_friend_only"] >= 0.9999).sum()
    n_india_full_supp = (india["links_supported"] >= 0.9999).sum()

    print(f"  Indonesian networks with complete comparison network: {n_indo_full_comp} (published: 45)")
    print(f"  Indonesian networks with full support:                 {n_indo_full_supp} (published: 127)")
    print(f"  Indian networks with complete comparison network:      {n_india_full_comp} (published: 0)")
    print(f"  Indian networks with full support:                     {n_india_full_supp} (published: 0)")

    small = df[df["num_nodes"] <= 20]
    n_small = len(small)
    n_small_indo = (small["country"] == "Indonesia").sum()
    print(f"  Networks with <=20 nodes (bipartite sample):           {n_small} (all Indonesian: {n_small_indo}; published: 213)")


if __name__ == "__main__":
    main()
