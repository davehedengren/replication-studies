"""Phase 2: Load and sanity check experimental data."""

import numpy as np
from utils import load_2x2_amp, load_3x3, load_mat


def main():
    d = load_2x2_amp()
    print("=" * 60)
    print("2x2 AMP data")
    print("=" * 60)
    n_obs = len(d["Period"])
    print(f"Total observations: {n_obs}")
    print(f"Sessions: {n_obs // 1280}")
    print(f"Subjects per session: 16, total subjects: {16 * (n_obs // 1280)}")
    print(f"Periods per subject: {int(d['Period'].max())}")
    print(f"Games: {sorted(np.unique(d['game']).tolist())}")
    print(f"Choice values (after relabel): {sorted(np.unique(d['choice']).tolist())}")
    print(f"Belief range (after relabel): [{d['belief'].min()}, {d['belief'].max()}]")

    print()
    print("=" * 60)
    print("3x3 games")
    print("=" * 60)
    for name in ["3x3_DS1", "3x3_DS2", "3x3_Nologit", "3x3_KM"]:
        choices, beliefs, payoff = load_3x3(name)
        print(f"{name}: N={len(choices)}  choice_row_sums={np.unique(choices.sum(axis=1))}  "
              f"belief_row_sums≈[{beliefs.sum(axis=1).min()}, {beliefs.sum(axis=1).max()}]")

    idd = load_mat("id_data")
    print()
    print(f"id_data: ID shape {idd['ID'].shape}, unique IDs per column: "
          f"{[len(np.unique(idd['ID'][:, c])) for c in range(idd['ID'].shape[1])]}")


if __name__ == "__main__":
    main()
