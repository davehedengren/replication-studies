"""Reproduce Figures 6-9 from Bloch & Olckers (2020).

- Fig 6: histograms of density of comparison network (India, Indonesia)
- Fig 7: histograms of share of supported links
- Fig 8: scatter matrix of (density, avg clustering, comparison density, support)
- Fig 9: support-mechanism vs bipartite-mechanism share for n<=20 networks
"""
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from utils import OUTPUT

FIGS = OUTPUT / "figures"
FIGS.mkdir(exist_ok=True)


def hist_pair(df, column, xlabel, title, outfile):
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    bins = np.linspace(0, 1, 21)
    for ax, country, color in [(axes[0], "Indonesia", "#2ca02c"),
                               (axes[1], "India", "#ff7f0e")]:
        sub = df[df["country"] == country][column].dropna()
        ax.hist(sub, bins=bins, color=color, alpha=0.85, edgecolor="white")
        ax.set_xlabel(xlabel)
        ax.set_ylabel("Networks")
        ax.set_title(f"({'i' if country == 'Indonesia' else 'ii'}) {country}n networks"
                     if country == "Indonesia" else f"(ii) Indian networks")
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(FIGS / outfile, dpi=130)
    plt.close(fig)


def figure6(df):
    hist_pair(df, "info_total_friend_only",
              "Density of comparison network",
              "Figure 6: Histograms of density of the comparison network",
              "figure6_density_comp.png")


def figure7(df):
    hist_pair(df, "links_supported",
              "Share of supported links",
              "Figure 7: Histograms of share of supported links",
              "figure7_support.png")


def figure8(df):
    cols = [
        ("density", "Density\nsocial"),
        ("ave_clust", "Average\nclustering"),
        ("info_total_friend_only", "Density\ncomparison"),
        ("links_supported", "Support"),
    ]
    n = len(cols)
    fig, axes = plt.subplots(n, n, figsize=(10, 10))
    colors = {"India": "#ff7f0e", "Indonesia": "#2ca02c"}
    for i, (ci, li) in enumerate(cols):
        for j, (cj, lj) in enumerate(cols):
            ax = axes[i, j]
            if i == j:
                for country, col in colors.items():
                    sub = df[df["country"] == country][ci].dropna()
                    ax.hist(sub, bins=20, color=col, alpha=0.6)
            else:
                for country, col in colors.items():
                    sub = df[df["country"] == country][[cj, ci]].dropna()
                    ax.scatter(sub[cj], sub[ci], s=4, alpha=0.5, color=col)
            if j == 0:
                ax.set_ylabel(li, fontsize=9)
            if i == n - 1:
                ax.set_xlabel(lj, fontsize=9)
            ax.set_xlim(-0.05, 1.05)
            if i != j:
                ax.set_ylim(-0.05, 1.05)
            ax.tick_params(labelsize=7)
    fig.suptitle("Figure 8: Scatter plots of social-network measures\n(India=orange, Indonesia=green)")
    fig.tight_layout()
    fig.savefig(FIGS / "figure8_pairplot.png", dpi=130)
    plt.close(fig)


def figure9(df):
    """Figure 9 filter exactly matches the original figures.ipynb:
       Indonesia only, giant component, n<=20, info_SP not null,
       info_expostIC<1, and info_total_friend_only>info_expostIC
       (i.e. support mechanism discards at least one link).
    """
    small = df[
        (df["country"] == "Indonesia")
        & (df["num_nodes"] <= 20)
        & df["info_SP"].notna()
        & (df["info_expostIC"] < 1)
        & (df["info_total_friend_only"] > df["info_expostIC"])
    ].copy()
    x = small["info_expostIC"].values  # unnormalized "share retained by support"
    y = small["info_SP"].values        # unnormalized "share retained by partition"
    small["share_supp"] = x / small["info_total_friend_only"]
    small["share_partition"] = y / small["info_total_friend_only"]
    below = (y < x).sum()
    equal = (y == x).sum()
    above_strict = (y > x).sum()

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(small["share_supp"], small["share_partition"],
               s=40, alpha=0.6, color="#2ca02c")
    ax.plot([0, 1], [0, 1], "k--", lw=1)
    ax.set_xlabel("Support mechanism")
    ax.set_ylabel("Partition mechanism")
    ax.set_xlim(-0.02, 1.05)
    ax.set_ylim(-0.02, 1.05)
    ax.set_aspect("equal")
    ax.set_title(f"Figure 9: Share of comparisons retained (n = {len(small)})\n"
                 f"{below} below 45° (support > partition); "
                 f"{equal + above_strict} on/above (partition >= support)")
    fig.tight_layout()
    fig.savefig(FIGS / "figure9_mechanisms.png", dpi=130)
    plt.close(fig)

    print(f"Figure 9: {len(small)} networks (published 213)")
    print(f"  info_expostIC (support-mechanism density) mean = {x.mean():.4f} "
          f"(published 0.6)")
    print(f"  info_SP (partition-mechanism density) mean    = {y.mean():.4f} "
          f"(published 0.5)")
    print(f"  support wins (support > partition): {below} (published 162)")
    print(f"  ties (support = partition):         {equal}")
    print(f"  partition wins (partition > support): {above_strict}")
    print(f"  non-below (partition >= support):   {equal + above_strict} (published 51)")
    return {
        "n": int(len(small)),
        "info_expostIC_mean": float(x.mean()),
        "info_SP_mean": float(y.mean()),
        "support_wins": int(below),
        "ties": int(equal),
        "partition_wins_strict": int(above_strict),
        "non_below_45deg": int(equal + above_strict),
    }


def main():
    df = pd.read_parquet(OUTPUT / "netdata.parquet")
    figure6(df)
    figure7(df)
    figure8(df)
    summary = figure9(df)
    pd.Series(summary).to_csv(OUTPUT / "figure9_summary.csv")
    print(f"\nFigures written to {FIGS}")


if __name__ == "__main__":
    main()
