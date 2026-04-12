"""Replicate the parametric rows of Table 2 (ak_robustness_table.tex)."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import numpy as np
import pandas as pd

from utils import load_long, OUT
from importlib import import_module
ak = import_module("02_ak_descriptives")


def row(margin, label, X, sigma, cutoffs, symmetric, model):
    fit = ak.ak_fit(X, sigma, cutoffs=cutoffs, symmetric=symmetric, model=model)
    p = fit.x
    lam, tau = p[0], p[1]
    # For multi-cutoff cases, the tex table reports the LAST estimated beta
    # (the 0<=t<1.96 region for twocut), matching R's twocut_beta = two_val[5].
    if model == "t":
        beta = p[-1]
    else:
        beta = p[-1]
    return {
        "margin": margin, "diff_from_baseline": label,
        "epsilon": X.mean(), "beta_p": beta,
        "theta": lam, "tau": tau,
        "ci_lo": lam - 1.64 * tau, "ci_hi": lam + 1.64 * tau,
    }


def main():
    df = load_long()
    df_wo_hunt = load_long(drop_hunt=True)

    rows = []
    # ---- RR baseline and variants ----
    rr = df[df["pbd_vs_rr"] == "RR"]
    X, s = rr["elasticity"].values, rr["se"].values

    rows.append(row("RR", "",                  X, s, (1.96,),       False, "t"))
    rows.append(row("RR", "Normal Distribution", X, s, (1.96,),     False, "normal"))
    rows.append(row("RR", "Symmetric p(t)",    X, s, (1.96,),       True,  "t"))
    rows.append(row("RR", "Extra p(t) cutoff", X, s, (0.0, 1.96),   False, "t"))

    rr_h = df_wo_hunt[df_wo_hunt["pbd_vs_rr"] == "RR"]
    X_h, s_h = rr_h["elasticity"].values, rr_h["se"].values
    rows.append(row("RR", "Drop Hunt (1995)",  X_h, s_h, (1.96,),   False, "t"))

    # ---- PBD baseline + variants (for reference; paper's Table 2 only shows RR) ----
    pbd = df[df["pbd_vs_rr"] == "PBD"]
    X_p, s_p = pbd["elasticity"].values, pbd["se"].values
    rows.append(row("PBD", "",                 X_p, s_p, (1.96,),   False, "t"))
    rows.append(row("PBD", "Normal Distribution", X_p, s_p, (1.96,), False, "normal"))
    rows.append(row("PBD", "Symmetric p(t)",   X_p, s_p, (1.96,),   True,  "t"))
    rows.append(row("PBD", "Extra p(t) cutoff", X_p, s_p, (0.0, 1.96), False, "t"))

    out = pd.DataFrame(rows)
    print("AK robustness rows (parametric only):")
    print(out.to_string(index=False, float_format=lambda v: f"{v:.3f}"))

    # Published parametric rows (first five RR + first four PBD rows of tex)
    pub = pd.DataFrame([
        {"margin": "RR",  "diff_from_baseline": "",                    "epsilon_p": 0.43, "beta_p_p": 0.12, "theta_p": 0.21, "tau_p": 0.26},
        {"margin": "RR",  "diff_from_baseline": "Normal Distribution", "epsilon_p": 0.43, "beta_p_p": 0.08, "theta_p": 0.07, "tau_p": 0.50},
        {"margin": "RR",  "diff_from_baseline": "Symmetric p(t)",      "epsilon_p": 0.43, "beta_p_p": 0.19, "theta_p": 0.33, "tau_p": 0.23},
        {"margin": "RR",  "diff_from_baseline": "Extra p(t) cutoff",   "epsilon_p": 0.43, "beta_p_p": 0.13, "theta_p": 0.19, "tau_p": 0.28},
        {"margin": "RR",  "diff_from_baseline": "Drop Hunt (1995)",    "epsilon_p": 0.52, "beta_p_p": 0.11, "theta_p": 0.21, "tau_p": 0.26},
    ])
    print("\nPublished (Table 2 RR rows):")
    print(pub.to_string(index=False))

    out.to_csv(OUT / "ak_robustness.csv", index=False)
    print(f"\nSaved {OUT/'ak_robustness.csv'}")


if __name__ == "__main__":
    main()
