"""Replicate the LDW results table (paper Figure 2 / tables/ldw.csv).

For each of the four panels in the paper
    (LDW-CPS full, LDW-PSID full, LDW-CPS trimmed, LDW-PSID trimmed)
we compute the experimental benchmark plus six estimators. The three
deterministic estimators (diff, reg, om.reg) should match the paper
exactly. The three ML-flavored estimators (NN matching, IPW, AIPW) use
Python substitutes for the R GRF/DML/hbal methods and will differ at the
~10-20% level; the writeup records the gap.
"""

from __future__ import annotations

import pandas as pd

from utils import (COVAR_LDW, OUT_DIR, diff_est, dr_est, ipw_est,
                   load_lalonde, load_trimmed, nn_match_att, om_reg_est,
                   reg_est)

Y = "re78"
T = "treat"


def run_panel(data, label, covar=COVAR_LDW):
    rows = []
    rows.append(("diff",   *diff_est(data, Y, T)))
    rows.append(("reg",    *reg_est(data, Y, T, covar)))
    rows.append(("om.reg", *om_reg_est(data, Y, T, covar)))
    rows.append(("match",  *nn_match_att(data, Y, T, covar, M=5)))
    rows.append(("ipw",    *ipw_est(data, Y, T, covar)))
    rows.append(("dr",     *dr_est(data, Y, T, covar)))
    df = pd.DataFrame(rows, columns=["method", "est", "se"])
    df["panel"] = label
    return df


def main():
    lal = load_lalonde()
    trim = load_trimmed()

    # Experimental benchmarks
    bench_full, _ = diff_est(lal["ldw"], Y, T), None
    bench_cps_trim, _ = diff_est(trim["ldw_trim_cps"], Y, T), None
    bench_psid_trim, _ = diff_est(trim["ldw_trim_psid"], Y, T), None

    panels = {
        "LDW-CPS full":    lal["ldw_cps"],
        "LDW-PSID full":   lal["ldw_psid"],
        "LDW-CPS trimmed": trim["ldw_cps_trim"],
        "LDW-PSID trimmed": trim["ldw_psid_trim"],
    }

    all_out = []
    for lab, d in panels.items():
        print(f"\n--- {lab} (n={len(d)}, treated={int(d[T].sum())}) ---")
        out = run_panel(d, lab)
        print(out.to_string(index=False,
              formatters={"est": lambda x: f"{x:8.0f}",
                          "se":  lambda x: f"({x:6.0f})"}))
        all_out.append(out)

    print("\nExperimental benchmarks (Row 1 of paper table):")
    print(f"  LDW full            ATT = {diff_est(lal['ldw'], Y, T)[0]:8.1f}  "
          f"SE = ({diff_est(lal['ldw'], Y, T)[1]:6.1f})")
    print(f"  LDW-CPS trimmed exp ATT = {diff_est(trim['ldw_trim_cps'], Y, T)[0]:8.1f}  "
          f"SE = ({diff_est(trim['ldw_trim_cps'], Y, T)[1]:6.1f})")
    print(f"  LDW-PSID trim   exp ATT = {diff_est(trim['ldw_trim_psid'], Y, T)[0]:8.1f}  "
          f"SE = ({diff_est(trim['ldw_trim_psid'], Y, T)[1]:6.1f})")

    full = pd.concat(all_out, ignore_index=True)
    full.to_csv(OUT_DIR / "out_ldw_estimates.csv", index=False)
    print(f"\nWrote {OUT_DIR / 'out_ldw_estimates.csv'}")


if __name__ == "__main__":
    main()
