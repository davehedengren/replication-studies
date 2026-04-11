"""Reproduce Table 1: simulation outcomes rescaled to NRMP market size.

The Mathematica notebook fits a separate logit of the form
    logit(f(N)) = β0 + β1 * log(N)
to each panel-variable x (λD, λH) combination across the six simulated
market sizes N ∈ {50, 100, 200, 500, 1000, 1700}, then predicts f(n) at each
of the 30 NRMP sub-market sizes listed in the 2020 NRMP report (Table 13,
p.41), and takes the enrolment-weighted mean across sub-markets:

    f_NRMP(λD, λH) = sum_n ( n / Σn * f̂(n; λD, λH) )

If any simulated value hits 100% (impossible in logit space), the notebook
substitutes log(20000) for the logit transform so the regression still
runs; we do the same. The output percentages must match Table 1 of the
paper to within rounding.
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from utils import (LAMBDA_D_TABLE, LAMBDA_H_TABLE, MARKET_SIZES, MARKET_SUM,
                   N_VEC, TABLES, VAR_FNS, load_balanced, parse_tex_table)


# Same row ordering as the published Table 1.
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


def make_logit_model(bal: pd.DataFrame, stem: str, lamD: float, lamH: float):
    """Fit logit(f) = a + b*log(N) on the 6 balanced-N points and return a
    callable f̂(n) mapping a market size back to a percentage."""
    xs, ys = [], []
    for N in N_VEC:
        row = bal[(bal["N"] == N) & np.isclose(bal["lambdaD"], lamD) & np.isclose(bal["lambdaH"], lamH)].iloc[0]
        v = float(VAR_FNS[stem](row, N))
        if v >= 100.0:
            # Notebook substitutes Log[20000] when the raw value is at the
            # boundary (f==100%), so the linear fit is still well defined.
            lg = np.log(20000.0)
        else:
            lg = np.log(v / (100.0 - v))
        xs.append(np.log(N))
        ys.append(lg)
    xs = np.asarray(xs)
    ys = np.asarray(ys)
    b, a = np.polyfit(xs, ys, 1)  # ys = a + b*xs
    def f_hat(n: float) -> float:
        lg = a + b * np.log(n)
        e = np.exp(lg)
        return 100.0 * e / (1.0 + e)
    return f_hat


def nrmp_cell(bal: pd.DataFrame, stem: str, lamD: float, lamH: float) -> float:
    f = make_logit_model(bal, stem, lamD, lamH)
    weighted = 0.0
    for n in MARKET_SIZES:
        weighted += (n / MARKET_SUM) * f(n)
    return weighted


def main() -> None:
    bal = load_balanced()
    ours: dict = {}
    for panel, label, stem in PANEL_SPEC:
        for lamD in LAMBDA_D_TABLE:
            for lamH in LAMBDA_H_TABLE:
                ours[(panel, label, lamD, lamH)] = nrmp_cell(bal, stem, lamD, lamH)

    theirs = parse_tex_table(TABLES / "table1.tex")
    print("== Table 1 (NRMP-rescaled) reproduction ==")
    ok = 0
    tot = 0
    mismatches: list[str] = []
    for k, v in ours.items():
        if k not in theirs:
            continue
        tot += 1
        diff = abs(v - theirs[k])
        if diff <= 0.15:
            ok += 1
        else:
            mismatches.append(f"  {k}: ours={v:.2f}% published={theirs[k]:.2f}%")
    print(f"  {ok}/{tot} cells within ±0.15 pp")
    for m in mismatches:
        print(m)

    # Render the table in the paper's cell order so a human can eyeball it.
    print("\nSide-by-side:")
    header = f"{'panel':<13}{'row':<8}" + "".join(
        f"lD{lD}H{lH} ".ljust(13) for lH in LAMBDA_H_TABLE for lD in LAMBDA_D_TABLE
    )
    print(header)
    for panel, label, stem in PANEL_SPEC:
        cells = []
        for lH in LAMBDA_H_TABLE:
            for lD in LAMBDA_D_TABLE:
                o = ours[(panel, label, lD, lH)]
                t = theirs.get((panel, label, lD, lH), float("nan"))
                cells.append(f"{o:4.1f}/{t:4.1f}")
        print(f"{panel:<13}{label:<8}" + " ".join(c.ljust(12) for c in cells))


if __name__ == "__main__":
    main()
