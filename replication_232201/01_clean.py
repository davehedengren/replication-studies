"""Load the cached LaLonde and IRS datasets, replicate the paper's Table 1
summary-statistics panel (tables/stats.csv), and save cleaned pickles."""

from __future__ import annotations

import numpy as np
import pandas as pd

from utils import COVAR_LDW, OUT_DIR, load_irs, load_lalonde, load_trimmed


def main():
    lal = load_lalonde()
    trim = load_trimmed()
    irs = load_irs()

    print("nsw_tr:", lal["nsw_tr"].shape,
          "nsw_co:", lal["nsw_co"].shape,
          "ldw_tr:", lal["ldw_tr"].shape,
          "ldw_co:", lal["ldw_co"].shape,
          "cps1:", lal["cps1"].shape,
          "psid1:", lal["psid1"].shape)

    # Table 1: replicate paper/tables/stats.csv
    X0 = ["age", "education", "nodegree", "married", "black", "hispanic", "re75", "u75"]
    X = ["age", "education", "nodegree", "married", "black", "hispanic",
         "re75", "u75", "re74", "u74"]

    cols = {
        "nsw.tr": (lal["nsw_tr"], X0),
        "nsw.co": (lal["nsw_co"], X0),
        "cps":    (lal["cps1"],   X),
        "psid":   (lal["psid1"],  X),
        "ldw.tr": (lal["ldw_tr"], X),
        "ldw.co": (lal["ldw_co"], X),
    }
    order = ["nsw.tr", "nsw.co", "cps", "psid", "ldw.tr", "ldw.co"]
    rows = X  # 10 rows; NSW columns will have NaN for re74/u74
    means = pd.DataFrame(index=rows, columns=order, dtype=float)
    sds = pd.DataFrame(index=rows, columns=order, dtype=float)
    for c in order:
        df, cov = cols[c]
        for v in rows:
            if v in df.columns:
                means.at[v, c] = float(df[v].mean())
                sds.at[v, c] = float(df[v].std(ddof=1))
            else:
                means.at[v, c] = np.nan
                sds.at[v, c] = np.nan
    # scale earnings to thousands for presentation
    for v in ("re75", "re74"):
        means.loc[v] = means.loc[v] / 1000.0
        sds.loc[v] = sds.loc[v] / 1000.0

    print("\nTable 1 - Summary statistics (means; SDs in parens below):")
    for v in rows:
        vals = []
        for c in order:
            m, s = means.at[v, c], sds.at[v, c]
            if np.isnan(m):
                vals.append(" " * 14 + "NA")
            else:
                vals.append(f"{m:9.2f} ({s:6.2f})")
        print(f"  {v:9s} " + "  ".join(vals))

    means.to_csv(OUT_DIR / "out_stats_means.csv")
    sds.to_csv(OUT_DIR / "out_stats_sds.csv")

    # Dump trimmed samples summary
    print("\nCached trimmed samples (from authors' lalonde2_trim.R pipeline):")
    for k in sorted(trim.keys()):
        print(f"  {k:24s} n = {len(trim[k])}, treated = {int(trim[k]['treat'].sum())}")

    # IRS basic summary
    print(f"\nIRS lottery: n={len(irs)}, winners={int(irs['winner'].sum())}, "
          f"big={int(irs['tr1'].sum())}, small={int(irs['tr2'].sum())}, "
          f"non-winners={int(irs['co'].sum())}")


if __name__ == "__main__":
    main()
