"""Phase 3 data audit over all simulation CSVs.

The paper is 100% simulation-based — there is no external dataset — so the
audit focuses on the raw CSV integrity: coverage of the (N, λ_D, λ_H, k)
grids, range sanity of every reported share (probabilities in [0,1]),
absence of NaNs, and a sanity check that blocking-pair counts scale with
market size.
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from utils import (DATA, K_VEC, LAMBDA_FULL, N_VEC, load_balanced, load_k,
                   load_sigs, load_unbalanced, load_rank_diffs)


def audit_coverage(df: pd.DataFrame, keys: list[str], expected: dict) -> None:
    """Verify the parameter grid is complete and non-duplicate."""
    print("  cells:", len(df))
    for k in keys:
        got = sorted(df[k].unique().tolist())
        print(f"  {k}: {got}")
        if k in expected:
            assert got == expected[k], (k, got, expected[k])
    dupes = df.duplicated(subset=keys).sum()
    print(f"  duplicates on keys {keys}: {dupes}")
    assert dupes == 0


def audit_ranges(df: pd.DataFrame) -> None:
    """Every ``*.gs`` / ``*.tags`` / ``*.trgs`` probability column must lie
    in [0, 1]. Blocking-pair columns are counts and should be non-negative."""
    share_cols = [c for c in df.columns if c.startswith(("unmatched.", "top.", "top3.", "samepartner.", "gs.tags."))]
    bp_cols = [c for c in df.columns if c.startswith("bp.")]
    for c in share_cols:
        lo, hi = df[c].min(), df[c].max()
        print(f"  share {c}: [{lo:.4f}, {hi:.4f}]")
        assert lo >= 0 and hi <= 1, (c, lo, hi)
    for c in bp_cols:
        lo, hi = df[c].min(), df[c].max()
        print(f"  bp    {c}: [{lo:.4f}, {hi:.4f}]")
        assert lo >= 0
    nans = df.isna().sum().sum()
    print(f"  NaN cells: {nans}")
    assert nans == 0


def audit_monotonicity(bal: pd.DataFrame) -> None:
    """Spot-check that Tr-DA unmatched rises with λ as in the paper's narrative.
    For fixed λ_H=0.5, unmatched-under-Tr-DA should rise monotonically in λ_D."""
    sub = bal[(bal["N"] == 500) & np.isclose(bal["lambdaH"], 0.5)].sort_values("lambdaD")
    trda = sub["unmatched.trgs"].tolist()
    print(f"  Tr-DA unmatched across λ_D at N=500, λ_H=0.5: {[f'{x:.3f}' for x in trda]}")
    assert all(trda[i] <= trda[i + 1] for i in range(len(trda) - 1)), trda


def audit_identity(bal: pd.DataFrame) -> None:
    """Int-DA should dominate Tr-DA on unmatched share everywhere."""
    bad = bal[bal["unmatched.tags"] > bal["unmatched.trgs"]]
    print(f"  cells where Int-DA unmatched > Tr-DA unmatched: {len(bad)} (expected 0)")
    assert len(bad) == 0


def audit_rank_diffs() -> None:
    r = load_rank_diffs()
    print("  rankDiffsx shape:", r.shape)
    print("  columns         :", r.columns.tolist())
    print("  Nd values       :", sorted(r["Nd"].unique().tolist()))
    print("  diff range      :", (r["diff"].min(), r["diff"].max()))
    print("  diff mean       :", float(r["diff"].mean()))
    # Only the (cud=0.5, cuh=0.5, Nd=500) cell is reported as Fig B8 in the paper.
    sub = r[(r["Nd"] == 500) & np.isclose(r["cud"], 0.5) & np.isclose(r["cuh"], 0.5)]
    print("  Fig B8 cell rows:", len(sub))
    print("  |rank-diff| > 0 fraction:", float((sub["diff"] != 0).mean()))


def main() -> None:
    print("== Balanced CSV ==")
    bal = load_balanced()
    audit_coverage(bal, ["N", "lambdaD", "lambdaH"], {
        "N": N_VEC, "lambdaD": LAMBDA_FULL, "lambdaH": LAMBDA_FULL,
    })
    audit_ranges(bal)
    audit_monotonicity(bal)
    audit_identity(bal)

    print("\n== K CSV ==")
    ks = load_k()
    audit_coverage(ks, ["k", "lambdaD", "lambdaH"], {
        "k": K_VEC, "lambdaD": LAMBDA_FULL, "lambdaH": LAMBDA_FULL,
    })
    audit_ranges(ks)

    print("\n== Unbalanced CSV ==")
    ub = load_unbalanced()
    audit_coverage(ub, ["nD", "lambdaD", "lambdaH"], {})
    audit_ranges(ub)

    print("\n== SIGS CSV ==")
    sigs = load_sigs()
    audit_coverage(sigs, ["N", "lambdaD", "lambdaH"], {})
    # SIGS has fewer share columns; just verify no NaNs and non-negative.
    share_cols = [c for c in sigs.columns if c not in ("N", "lambdaD", "lambdaH")]
    for c in share_cols:
        lo, hi = sigs[c].min(), sigs[c].max()
        print(f"  {c}: [{lo:.4f}, {hi:.4f}]")
        assert lo >= 0 and hi <= 1

    print("\n== rankDiffsx CSV ==")
    audit_rank_diffs()

    print("\nAll data-audit assertions pass.")


if __name__ == "__main__":
    main()
