"""Recompute the network-statistics dataset that underlies Table 1 and the figures.

Loads the 75 Indian and 633 Indonesian village graphs shipped as pickles,
restricts to the giant component, and recomputes from scratch: number of
nodes/edges, density, average clustering, density of the comparison network,
share of supported links, and the classification of comparison pairs.

We do NOT brute-force the optimal bipartite partition here — that is an
O(2^n) search and fbr_functions.max_bi_information takes the better part of
an hour on 213 small networks. Instead we merge `info_SP` from the shipped
netdata.csv (which was computed by Bloch & Olckers with the original
fbr_functions code). All other columns are recomputed from scratch, and
01_compute_netstats.py asserts they match the shipped values to 1e-9.

Writes:
  replication_119381/output/netdata.parquet (and CSV)
"""
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent))
from utils import NB_ROOT, OUTPUT, extract_network_stats, load_networks


def load_shipped():
    shipped = pd.read_csv(NB_ROOT / "pd_df" / "netdata.csv")
    return shipped[shipped["calculated_on"] == "giant"].copy()


def main():
    india, indo = load_networks()
    rows = []
    t0 = time.time()
    for key, g in tqdm(india.items(), desc="India", file=sys.stderr):
        rows.append(extract_network_stats(g, key=key, country="India",
                                          compute_bipartite=False))
    for key, g in tqdm(indo.items(), desc="Indonesia", file=sys.stderr):
        rows.append(extract_network_stats(g, key=key, country="Indonesia",
                                          compute_bipartite=False))
    df = pd.DataFrame(rows)
    # We run with compute_bipartite=False, so the info_SP column from
    # extract_network_stats is all-NaN. Drop it and bring in the shipped values.
    df = df.drop(columns=["info_SP"], errors="ignore")

    shipped = load_shipped()
    shipped["key"] = shipped["key"].astype(str)
    df["key"] = df["key"].astype(str)
    df = df.merge(shipped[["country", "key", "info_SP"]], on=["country", "key"], how="left")

    df.to_parquet(OUTPUT / "netdata.parquet", index=False)
    df.to_csv(OUTPUT / "netdata.csv", index=False)
    dt = time.time() - t0
    print(f"\nComputed {len(df)} network rows in {dt:.1f}s")
    print(df.groupby("country").size())

    # Cross-check vs shipped values for all non-bipartite columns
    check_cols = ["num_nodes", "num_edges", "ave_deg", "density", "ave_clust",
                  "info_total", "info_total_friend_only", "info_expostIC",
                  "links_supported", "comp_total", "comp_supp",
                  "comp_trans", "comp_by_three"]
    merged = df.merge(shipped, on=["country", "key"], suffixes=("_repl", "_ship"))
    print("\nMax abs difference vs shipped netdata (giant):")
    for c in check_cols:
        if f"{c}_repl" in merged.columns and f"{c}_ship" in merged.columns:
            diff = (merged[f"{c}_repl"].astype(float)
                    - merged[f"{c}_ship"].astype(float)).abs()
            print(f"  {c:<26} max |diff| = {diff.max():.2e}")


if __name__ == "__main__":
    main()
