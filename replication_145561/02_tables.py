"""Phase 2: Reproduce Table 2 (best response fractions) and Online-Appendix
Table 1 (Hotelling p-values comparing games)."""

import numpy as np
from utils import load_2x2_amp, best_response_2x2, hotelling_two_sample, hotelling_one_sample


def table_2():
    d = load_2x2_amp()
    subj = d["Subject"]
    sess = d["session"]
    rnd = d["round1"]
    game = d["game"]
    row = d["row"]
    choice = d["choice"]
    belief = d["belief"]
    Row = d["Row"]  # 2x2x10
    Col = d["Col"]

    GAMES = 10
    SUBS = 16
    SESSIONS = 4
    BLOCK = 8

    # Compute best response given stated beliefs (uses game 1 payoff because
    # of the relabeling — all games in a pair share the same BR structure)
    n = len(game)
    BR = np.zeros(n)
    pay_row = Row[:, :, 0].astype(float)
    pay_col = Col[:, :, 0].astype(float).T
    for i in range(n):
        b = belief[i] / 100.0
        if row[i] == 1:
            BR[i] = best_response_2x2(pay_row, b)
        else:
            BR[i] = best_response_2x2(pay_col, b)

    # Fraction of best responses: for each session x game, average per-subject
    Rbr = np.zeros((SUBS // 2, BLOCK, GAMES, SESSIONS))
    Cbr = np.zeros((SUBS // 2, BLOCK, GAMES, SESSIONS))

    for s in range(SESSIONS):
        subs_in_sess = np.arange(1, SUBS + 1) + s * SUBS
        for g in range(1, GAMES + 1):
            rind = 0
            cind = 0
            for sub in subs_in_sess:
                mask_one = (subj == sub) & (rnd == 1) & (game == g)
                if not mask_one.any():
                    continue
                is_row = row[mask_one][0] == 1
                for roun in range(1, BLOCK + 1):
                    m = (subj == sub) & (rnd == roun) & (game == g)
                    if not m.any():
                        continue
                    matched = BR[m][0] == choice[m][0]
                    if is_row:
                        Rbr[rind, roun - 1, g - 1, s] = matched
                    else:
                        Cbr[cind, roun - 1, g - 1, s] = matched
                if is_row:
                    rind += 1
                else:
                    cind += 1

    # Average over subjects within role, rounds, sessions -> per-game mean
    mmR = Rbr.mean(axis=(0, 1, 3))
    mmC = Cbr.mean(axis=(0, 1, 3))
    # Pair equivalent games (AMP1 = games 1&6, ..., AMP5 = 5&10)
    mmR_pair = 0.5 * (mmR[:5] + mmR[5:])
    mmC_pair = 0.5 * (mmC[:5] + mmC[5:])

    print("=" * 68)
    print("Table 2: Fraction of best responses (replication)")
    print("=" * 68)
    header = f"{'':<10}" + "".join(f"AMP{g:<6}" for g in range(1, 6)) + "avg"
    print(header)
    print(
        f"{'Row':<10}"
        + "".join(f"{v:.3f}  " for v in mmR_pair)
        + f"{mmR_pair.mean():.3f}"
    )
    print(
        f"{'Column':<10}"
        + "".join(f"{v:.3f}  " for v in mmC_pair)
        + f"{mmC_pair.mean():.3f}"
    )
    avg = 0.5 * (mmR_pair + mmC_pair)
    print(
        f"{'average':<10}"
        + "".join(f"{v:.3f}  " for v in avg)
        + f"{avg.mean():.3f}"
    )

    print()
    print("Published Table 2:")
    print("Row      .61   .56   .55   .60   .58   .58")
    print("Column   .75   .73   .66   .72   .75   .72")
    print("average  .68   .65   .61   .66   .67   .65")


def table_onlineapp_1():
    """Online appendix Table 1: Hotelling p-values for pairwise comparisons
    of choices and beliefs across games, and p-values comparing beliefs to
    mean opponent choice (support for Result 1(iii))."""
    d = load_2x2_amp()
    choice = d["choice"]
    belief = d["belief"]
    game = d["game"]
    row = d["row"]

    # Per game, per role: vectors of choices/beliefs
    ChRow, ChCol, BelRow, BelCol = {}, {}, {}, {}
    for g in range(1, 11):
        ChRow[g] = choice[(row == 1) & (game == g)]
        ChCol[g] = choice[(row == 0) & (game == g)]
        BelRow[g] = belief[(row == 1) & (game == g)]
        BelCol[g] = belief[(row == 0) & (game == g)]

    def stack(g, which):  # stack [row, col] columns for paired games
        a = np.column_stack([which[0][g], which[1][g]])
        b = np.column_stack([which[0][g + 5], which[1][g + 5]])
        return np.vstack([a, b])

    ch_pairs = (ChRow, ChCol)
    bel_pairs = (BelRow, BelCol)

    pvals_ch = np.full((4, 4), np.nan)
    pvals_bel = np.full((4, 4), np.nan)
    for i in range(1, 5):
        for j in range(i + 1, 6):
            xi = stack(i, ch_pairs)
            xj = stack(j, ch_pairs)
            pvals_ch[i - 1, j - 2] = hotelling_two_sample(xi, xj)
            yi = stack(i, bel_pairs)
            yj = stack(j, bel_pairs)
            pvals_bel[i - 1, j - 2] = hotelling_two_sample(yi, yj)

    # p-values comparing beliefs to mean opponent choice (Result 1(iii))
    pvals_belch = np.zeros(5)
    for i in range(1, 6):
        beliefs_stacked = stack(i, bel_pairs)
        # mean of opponent choices: [ChCol, ChRow] (opponent roles)
        a = np.column_stack([ChCol[i], ChRow[i]])
        b = np.column_stack([ChCol[i + 5], ChRow[i + 5]])
        mu = np.vstack([a, b]).mean(axis=0)
        pvals_belch[i - 1] = hotelling_one_sample(beliefs_stacked, mu)

    print()
    print("=" * 68)
    print("Online Appendix Table 1: Hotelling p-values across AMP games")
    print("=" * 68)
    labels = ["AMP1", "AMP2", "AMP3", "AMP4"]
    cols = ["AMP2", "AMP3", "AMP4", "AMP5"]
    print("Choices:")
    print(f"{'':<6}" + "".join(f"{c:>10}" for c in cols))
    for i, lab in enumerate(labels):
        vals = "".join(
            (f"{pvals_ch[i, j]:>10.4f}" if not np.isnan(pvals_ch[i, j]) else f"{'':>10}")
            for j in range(4)
        )
        print(f"{lab:<6}{vals}")
    print()
    print("Beliefs:")
    print(f"{'':<6}" + "".join(f"{c:>10}" for c in cols))
    for i, lab in enumerate(labels):
        vals = "".join(
            (f"{pvals_bel[i, j]:>10.4f}" if not np.isnan(pvals_bel[i, j]) else f"{'':>10}")
            for j in range(4)
        )
        print(f"{lab:<6}{vals}")

    print()
    print("One-sample Hotelling (beliefs vs mean opponent choice), by game:")
    for i, p in enumerate(pvals_belch, 1):
        print(f"  AMP{i}: p = {p:.6f}")


if __name__ == "__main__":
    table_2()
    table_onlineapp_1()
