"""Python translation of compute_moments_quarterly.do (correlation moments).

Reproduces the RHO_D1 / RHO_D4 sheets in us_de_misc_moments_quarterly.xlsx
across all maturities, post-1986Q2 sample. These correlation moments are
empirical targets plotted against model-implied correlations in Appendix
figure C1 of the paper.

This script also documents a **mislabeled-column bug** in the Stata script:

  compute_moments_quarterly.do line 72:
      corr RHO_Dydiff RHO_DyF if `time_period'
      local idx_row = colnumb(RHO_D`dd'_res, "rho_Dydiff_De")

The column header claims to be the correlation of the yield differential
with the exchange rate change, but the Stata `corr` command on line 72
is computing corr(dy_diff, dyF) -- the correlation of the yield differential
with the **foreign yield change**. The value that is written to the xlsx
matches the dy_diff/dyF correlation, not dy_diff/de.

The mislabeled column is never consumed anywhere in the Julia estimation
or summary code (grep of the package shows `rho_Dydiff` appears only in
the Stata script), so the bug has **no effect on any published result**.
"""

import numpy as np
import pandas as pd

from utils import MATURITIES, OUT, in_sample, load_published_moments, load_quarterly


def forward_diff(s, k):
    return s.shift(-k) - s


def compute_rho(df, k):
    rows = []
    DrH = forward_diff(df["yH_12"], k)
    DrF = forward_diff(df["yF_12"], k)
    De = forward_diff(df["log_E"], k)

    for m in MATURITIES:
        DyH = forward_diff(df[f"yH_{m}"], k)
        DyF = forward_diff(df[f"yF_{m}"], k)
        Ddiff = DyH - DyF

        def c(a, b):
            d = pd.DataFrame({"a": a, "b": b}).dropna()
            if len(d) < 2:
                return np.nan
            return d["a"].corr(d["b"])

        rows.append({
            "tau": m // 12,
            "rho_DyH_DrH_py": c(DyH, DrH),
            "rho_DyF_DrF_py": c(DyF, DrF),
            "rho_DyH_De_py": c(DyH, De),
            "rho_DyF_De_py": c(DyF, De),
            # AS LABELED (correct interpretation): corr(Ddiff, De)
            "rho_Dydiff_De_CORRECT_py": c(Ddiff, De),
            # AS COMPUTED IN STATA: corr(Ddiff, DyF)
            "rho_Dydiff_DyF_STATA_py": c(Ddiff, DyF),
            "rho_DyH_DyF_py": c(DyH, DyF),
        })
    return pd.DataFrame(rows)


def compare(py, published, label):
    m = py.merge(published.rename(columns={"tau_arr": "tau"}), on="tau")
    cols = ["rho_DyH_DrH", "rho_DyF_DrF", "rho_DyH_De", "rho_DyF_De", "rho_DyH_DyF"]
    worst = 0.0
    print(f"\n== {label}: labelled moments ==")
    for c in cols:
        diff = (m[f"{c}_py"] - m[c]).abs()
        print(f"  {c}: max|py-st|={diff.max():.4g}")
        worst = max(worst, diff.max())

    # The bug: the xlsx column labelled 'rho_Dydiff_De' actually contains
    # corr(Ddiff, DyF). Our Python reproduces BOTH interpretations so we
    # can prove which one the Stata code produced.
    d_lab = (m["rho_Dydiff_De_CORRECT_py"] - m["rho_Dydiff_De"]).abs().max()
    d_bug = (m["rho_Dydiff_DyF_STATA_py"] - m["rho_Dydiff_De"]).abs().max()
    print(f"  rho_Dydiff_De  vs. correct corr(Ddiff,De) : max|diff|={d_lab:.4g}")
    print(f"  rho_Dydiff_De  vs. Stata's corr(Ddiff,DyF): max|diff|={d_bug:.4g}")
    if d_bug < 1e-6 < d_lab:
        print("  ==> CONFIRMED: xlsx column holds corr(Ddiff,DyF), not corr(Ddiff,De)")
    return m, worst


def main():
    df = in_sample(load_quarterly())
    pub = load_published_moments()

    rho1 = compute_rho(df, 1)
    rho4 = compute_rho(df, 4)

    compare(rho1, pub["RHO_D1"], "1Q changes (RHO_D1)")
    compare(rho4, pub["RHO_D4"], "4Q changes (RHO_D4)")

    rho1.to_csv(OUT / "rho_d1_py.csv", index=False)
    rho4.to_csv(OUT / "rho_d4_py.csv", index=False)


if __name__ == "__main__":
    main()
