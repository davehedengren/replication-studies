"""Phase 2: reproduce every appendix simulation table and compare to
published values in ./141541-V1/tables/*.tex.

The paper's appendix contains three table families, each emitted by the
Mathematica notebook:
  * balanced N tables (N in {50,100,200,500,1000,1700}, k=5)
  * interview-capacity tables (k in {2,5,10,20}, N=500)
  * an unbalanced table (nD=600, nH=500, k=5)

Every number in those tables comes from a single cell of one of the raw
simulation CSVs, so exact matches are expected (up to the published rounding).
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from utils import (LAMBDA_D_TABLE, LAMBDA_H_TABLE, N_VEC, K_VEC,
                   TABLES, load_balanced, load_k, load_unbalanced,
                   parse_tex_table, pct)


ROW_LABELS = ["DA", "Int-DA", "Tr-DA"]
# (panel tag, paper row label, variable stem)
PANEL_SPEC = [
    ("unmatched", "Int-DA", "UnmatchedIntDA"),
    ("unmatched", "Tr-DA", "UnmatchedTrDA"),
    ("top", "DA", "TopDA"),
    ("top", "Int-DA", "TopIntDA"),
    ("top", "Tr-DA", "TopTrDA"),
    ("top3", "DA", "Top3DA"),
    ("top3", "Int-DA", "Top3IntDA"),
    ("top3", "Tr-DA", "Top3TrDA"),
    ("samepartner", "DA", "SamePartnerDA"),
    ("samepartner", "Int-DA", "SamePartnerIntDA"),
    ("identical", "Int-DA", "IdenticalIntDA"),
    ("bp", "Matched", "BPMatched"),
    ("bp", "Unmatched", "BPUnmatched"),
]


def build_table_cells(df: pd.DataFrame, key_col: str, key_val) -> dict:
    """Compute our replication of a published table as {(panel,label,lamD,lamH) -> %}."""
    out: dict = {}
    for panel, label, stem in PANEL_SPEC:
        for lamD in LAMBDA_D_TABLE:
            for lamH in LAMBDA_H_TABLE:
                out[(panel, label, lamD, lamH)] = pct(
                    df, stem, lamD, lamH, key=key_col, n=key_val
                )
    return out


def compare(name: str, ours: dict, theirs: dict, tol: float = 0.1) -> tuple[int, int, list]:
    """Compare our computed values to the parsed tex table. Returns (matched,total,mismatches)."""
    matched = 0
    mismatches: list[str] = []
    total = 0
    for key, ours_val in ours.items():
        if key not in theirs:
            continue
        total += 1
        theirs_val = theirs[key]
        if abs(ours_val - theirs_val) <= tol:
            matched += 1
        else:
            mismatches.append(f"  {key}: ours={ours_val:.2f}% published={theirs_val:.2f}%")
    print(f"  {name}: {matched}/{total} cells within ±{tol} pp")
    for m in mismatches[:8]:
        print(m)
    if len(mismatches) > 8:
        print(f"  ... {len(mismatches) - 8} more")
    return matched, total, mismatches


def main() -> None:
    bal = load_balanced()
    ks = load_k()
    ub = load_unbalanced()

    all_matched = 0
    all_total = 0

    print("== Balanced-N appendix tables ==")
    for n in N_VEC:
        ours = build_table_cells(bal, "N", n)
        theirs = parse_tex_table(TABLES / f"app_tbl_balanced_{n}.tex")
        m, t, _ = compare(f"N={n}", ours, theirs)
        all_matched += m
        all_total += t

    print("\n== Interview-capacity appendix tables (N=500) ==")
    for k in K_VEC:
        ours = build_table_cells(ks, "k", k)
        theirs = parse_tex_table(TABLES / f"app_tbl_Interview_K{k}.tex")
        m, t, _ = compare(f"k={k}", ours, theirs)
        all_matched += m
        all_total += t

    print("\n== Unbalanced table (nD=600, nH=500) ==")
    ours = build_table_cells(ub, "nD", 600)
    theirs = parse_tex_table(TABLES / "app_tbl_Unbalanced600.tex")
    m, t, _ = compare("nD=600", ours, theirs)
    all_matched += m
    all_total += t

    print(f"\nTOTAL: {all_matched}/{all_total} appendix cells within ±0.1 pp")
    if all_matched != all_total:
        sys.exit(1)


if __name__ == "__main__":
    main()
