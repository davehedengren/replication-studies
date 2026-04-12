"""Replicate IRS lottery results (paper tables/irs.csv).

Four columns:
  (1) Big winners vs non-winners, outcome = post-lottery earnings average.
  (2) Big winners vs non-winners, placebo = pre-lottery earnings average.
  (3) Small winners vs non-winners, post-lottery.
  (4) Small winners vs non-winners, placebo pre-lottery.
"""

from __future__ import annotations

import warnings

import pandas as pd

from utils import (COVAR_IRS, OUT_DIR, diff_est, dr_est, ipw_est, load_irs,
                   nn_match_att, om_reg_est, reg_est)

warnings.filterwarnings("ignore")


def run_panel(d, y, label):
    rows = []
    rows.append(("diff",   *diff_est(d, y, "tr")))
    rows.append(("reg",    *reg_est(d, y, "tr", COVAR_IRS)))
    rows.append(("om.reg", *om_reg_est(d, y, "tr", COVAR_IRS)))
    rows.append(("match",  *nn_match_att(d, y, "tr", COVAR_IRS, M=5)))
    rows.append(("ipw",    *ipw_est(d, y, "tr", COVAR_IRS)))
    rows.append(("dr",     *dr_est(d, y, "tr", COVAR_IRS)))
    out = pd.DataFrame(rows, columns=["method", "est", "se"])
    out["panel"] = label
    return out


def main():
    d = load_irs()
    s1 = d[(d["tr1"] == 1) | (d["co"] == 1)].copy()
    s1["tr"] = s1["tr1"]
    s2 = d[(d["tr2"] == 1) | (d["co"] == 1)].copy()
    s2["tr"] = s2["tr2"]

    print(f"Big winners sample: n={len(s1)}, treated={int(s1['tr'].sum())}")
    print(f"Small winners sample: n={len(s2)}, treated={int(s2['tr'].sum())}")

    panels = [
        (s1, "yearn.avg", "Big vs Non: post"),
        (s1, "xearn.avg", "Big vs Non: placebo"),
        (s2, "yearn.avg", "Small vs Non: post"),
        (s2, "xearn.avg", "Small vs Non: placebo"),
    ]
    all_out = []
    for d_, y, lab in panels:
        print(f"\n--- {lab} ---")
        out = run_panel(d_, y, lab)
        print(out.to_string(index=False,
              formatters={"est": lambda x: f"{x:7.2f}",
                          "se":  lambda x: f"({x:6.2f})"}))
        all_out.append(out)

    full = pd.concat(all_out, ignore_index=True)
    full.to_csv(OUT_DIR / "out_irs_estimates.csv", index=False)


if __name__ == "__main__":
    main()
