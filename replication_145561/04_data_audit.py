"""Phase 3: Data audit — coverage, distributions, logical consistency."""

import numpy as np
import pandas as pd

from utils import load_2x2_amp, load_3x3, load_mat


def audit_2x2():
    d = load_2x2_amp()
    n = len(d["Period"])
    subs = np.unique(d["Subject"])
    print(f"2x2 AMP: N={n}, subjects={len(subs)}, sessions={len(np.unique(d['session']))}")

    # Balance: each subject should play all 10 games × 8 periods = 80 obs
    counts = pd.Series(d["Subject"]).value_counts()
    print(f"  obs per subject: min={counts.min()}, max={counts.max()}")
    assert counts.min() == 80 and counts.max() == 80

    # Role split per subject: should be 50/50 row/col within each game block
    g = pd.DataFrame({"subj": d["Subject"], "game": d["game"], "row": d["row"]})
    role_mix = g.groupby(["subj", "game"])["row"].nunique().max()
    print(f"  max distinct roles per (subject, game): {role_mix} (expect 1)")
    assert role_mix == 1

    # Beliefs and choices in bounds
    assert d["choice"].min() >= 0 and d["choice"].max() <= 1
    assert d["belief"].min() >= 0 and d["belief"].max() <= 100
    print(f"  choice ∈ {{0,1}}: yes; belief ∈ [0, 100]: yes")

    # Game pair balance: 1&6, 2&7, ..., 5&10 should have similar sizes
    counts_g = pd.Series(d["game"]).value_counts().sort_index()
    print(f"  observations per game: {counts_g.to_dict()}")
    assert (counts_g == 512).all(), "Each game should have 64 subjects × 8 rounds = 512"


def audit_3x3():
    print()
    for name, n_expected in [("3x3_DS1", 480), ("3x3_DS2", 480), ("3x3_Nologit", 480), ("3x3_KM", 240)]:
        choices, beliefs, payoff = load_3x3(name)
        n = len(choices)
        print(f"{name}: N={n} (expected {n_expected})")
        assert n == n_expected

        # Exactly one action chosen per row
        row_sum_choice = choices.sum(axis=1)
        assert (row_sum_choice == 1).all(), f"Non-unit choice row in {name}"

        # Beliefs sum to 100 (integer percentages)
        row_sum_belief = beliefs.sum(axis=1)
        print(f"  belief row sums: min={row_sum_belief.min()}, max={row_sum_belief.max()}")
        assert row_sum_belief.min() == 100 and row_sum_belief.max() == 100

        # Action frequencies (aggregate)
        freqs = choices.mean(axis=0)
        print(f"  aggregate choice freq: A={freqs[0]:.3f}, B={freqs[1]:.3f}, C={freqs[2]:.3f}")

        # Average beliefs
        bel_means = beliefs.mean(axis=0) / 100
        print(f"  average beliefs:        A={bel_means[0]:.3f}, B={bel_means[1]:.3f}, C={bel_means[2]:.3f}")
        print(f"  payoff matrix:\n{payoff.astype(int)}")


def audit_ids():
    print()
    idd = load_mat("id_data")["ID"]
    print(f"id_data ID shape: {idd.shape}")
    for col, game_name in enumerate(["DS1", "DS2", "NL", "KM"]):
        uniq = np.unique(idd[:, col])
        # KM only has 240 real obs; last 240 slots are padded with zero
        if game_name == "KM":
            uniq_nz = uniq[uniq > 0]
            print(f"  {game_name}: {len(uniq_nz)} real IDs (+ zero padding for rows 241-480)")
        else:
            print(f"  {game_name}: {len(uniq)} unique IDs")


def duplicates_check():
    print()
    for name in ["3x3_DS1", "3x3_DS2", "3x3_Nologit", "3x3_KM"]:
        d = load_mat(name)
        # Check each (Subject/SubID, Period) combination unique
        sid_key = "SubID" if "SubID" in d else ("subID" if "subID" in d else "Subject")
        df = pd.DataFrame({"sub": d[sid_key], "period": d.get("Period", d.get("round1"))})
        dups = df.duplicated().sum()
        print(f"{name}: duplicate (subject, period) rows = {dups}")


if __name__ == "__main__":
    audit_2x2()
    audit_3x3()
    audit_ids()
    duplicates_check()
    print("\nAll audits passed.")
