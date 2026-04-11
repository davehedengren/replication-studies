"""Phase 1 / 2: inspect the raw simulation outputs and verify the design grid."""
from __future__ import annotations
import pandas as pd
from utils import (DATA, LAMBDA_FULL, K_VEC, N_VEC, load_balanced, load_k,
                   load_sigs, load_unbalanced)


def main() -> None:
    bal = load_balanced()
    print("Balanced CSV:", bal.shape)
    print("  N grid     :", sorted(bal["N"].unique().tolist()))
    print("  lambdaD    :", sorted(bal["lambdaD"].unique().tolist()))
    print("  lambdaH    :", sorted(bal["lambdaH"].unique().tolist()))
    assert sorted(bal["N"].unique().tolist()) == N_VEC
    assert sorted(bal["lambdaD"].unique().tolist()) == LAMBDA_FULL
    assert sorted(bal["lambdaH"].unique().tolist()) == LAMBDA_FULL
    assert len(bal) == len(N_VEC) * len(LAMBDA_FULL) ** 2

    ks = load_k()
    print("K CSV       :", ks.shape)
    assert sorted(ks["k"].unique().tolist()) == K_VEC
    assert len(ks) == len(K_VEC) * len(LAMBDA_FULL) ** 2

    ub = load_unbalanced()
    print("Unbal CSV   :", ub.shape, "nD:", sorted(ub["nD"].unique().tolist()))

    sigs = load_sigs()
    print("SIGS CSV    :", sigs.shape)

    print("All raw-data shape checks pass.")


if __name__ == "__main__":
    main()
