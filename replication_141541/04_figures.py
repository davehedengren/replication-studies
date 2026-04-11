"""Reproduce the headline Figure 1 comparison in a simplified Python form.

The paper's Fig 1a/1b are Mathematica RegionPlots: they shade the range of
``First-ranked | Matched`` (and ``Top-three | Matched``) as λ_H sweeps from
0.05 to 0.95, with λ_D on the x-axis, for both DA and Int-DA. Here we
compute the NRMP-scaled value of ``TopDA`` / ``TopIntDA`` / ``Top3DA`` /
``Top3IntDA`` at the same (λ_D, λ_H) grid used in the notebook and plot
line-ranges that reproduce the qualitative picture in Figure 1. We also
dump the underlying values so they can be compared to the individual
``samepartner``-conditioned values that drive the published figures.
"""
from __future__ import annotations
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from utils import (LAMBDA_FULL, MARKET_SIZES, MARKET_SUM, OUT, VAR_FNS,
                   load_balanced)

# The notebook computes TopDA, etc. at NRMP size by first taking the raw
# simulation value at each N, dividing by the matched share (the Mathematica
# ``TopIntDAc`` function) to condition on matched, and then fitting the logit
# model over N. To keep this script self-contained we reimplement that
# pipeline directly here.


def conditional_on_matched(row, stem: str, N: int) -> float:
    """Top*/Matched -> rescale by 1 / (1 - unmatched share)."""
    v = float(VAR_FNS[stem](row, N))
    if "Tr" in stem:
        unmatched = row["unmatched.trgs"]
    elif "Int" in stem:
        unmatched = row["unmatched.tags"]
    else:
        unmatched = row["unmatched.gs"]
    return v / (1.0 - unmatched)


def nrmp_curve(bal: pd.DataFrame, stem: str, lamH_vals, lamD: float):
    """Return the NRMP-scaled conditional match value for each λ_H in lamH_vals."""
    out = []
    for lamH in lamH_vals:
        # Per Mathematica MakeModel, fit logit on log(N).
        xs, ys = [], []
        for N in [50, 100, 200, 500, 1000, 1700]:
            row = bal[(bal["N"] == N) & np.isclose(bal["lambdaD"], lamD) & np.isclose(bal["lambdaH"], lamH)].iloc[0]
            v = conditional_on_matched(row, stem, N)
            if v >= 100.0:
                lg = np.log(20000.0)
            else:
                lg = np.log(v / (100.0 - v))
            xs.append(np.log(N))
            ys.append(lg)
        b, a = np.polyfit(np.asarray(xs), np.asarray(ys), 1)
        # NRMP weighted prediction.
        weighted = 0.0
        for n in MARKET_SIZES:
            e = np.exp(a + b * np.log(n))
            weighted += (n / MARKET_SUM) * 100.0 * e / (1.0 + e)
        out.append(weighted)
    return np.asarray(out)


def main() -> None:
    bal = load_balanced()
    lamDs = np.asarray(LAMBDA_FULL)
    lamHs = np.asarray(LAMBDA_FULL)

    fig, axes = plt.subplots(1, 2, figsize=(11, 5))
    for ax, top_stems, title in (
        (axes[0], ("TopDA", "TopIntDA"), "First-ranked | matched"),
        (axes[1], ("Top3DA", "Top3IntDA"), "Top-three | matched"),
    ):
        da_low, da_high, int_low, int_high = [], [], [], []
        for lamD in lamDs:
            da_vals = nrmp_curve(bal, top_stems[0], lamHs, lamD)
            int_vals = nrmp_curve(bal, top_stems[1], lamHs, lamD)
            da_low.append(da_vals.min())
            da_high.append(da_vals.max())
            int_low.append(int_vals.min())
            int_high.append(int_vals.max())
        ax.fill_between(lamDs, da_low, da_high, color="C3", alpha=0.25, label="DA range (λ_H in [0.05, 0.95])")
        ax.fill_between(lamDs, int_low, int_high, color="C0", alpha=0.25, label="Int-DA range")
        ax.plot(lamDs, da_low, "--", color="C3"); ax.plot(lamDs, da_high, "--", color="C3")
        ax.plot(lamDs, int_low, "-", color="C0"); ax.plot(lamDs, int_high, "-", color="C0")
        ax.set_xlabel("Doctor common weight, λ_D")
        ax.set_ylabel("Share of matched doctors (%)")
        ax.set_title(title)
        ax.set_ylim(0, 100)
        ax.legend(loc="upper right", fontsize=8)
    fig.suptitle("Replicated Figure 1 — NRMP-rescaled match ranges (λ_H envelope)")
    fig.tight_layout()
    fig.savefig(OUT / "fig1_replicated.png", dpi=150)
    print(f"Saved {OUT / 'fig1_replicated.png'}")


if __name__ == "__main__":
    main()
