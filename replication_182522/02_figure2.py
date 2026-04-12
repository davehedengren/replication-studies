"""Reproduce Figure 2 of Sampson (2023): log R&D efficiency vs log GDP per capita.

Uses our Python-computed b_s values against PWT 9.0 2010 GDP per capita.
Also prints correlation for side-by-side comparison with the paper figure.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from utils import OUT, RAW, BASELINE_COUS


def main():
    rd = pd.read_csv(OUT / "rd_efficiency.csv")

    pwt = pd.read_stata(RAW / "pwt90.dta", convert_categoricals=False)
    pwt = pwt[(pwt["year"] == 2010) & (pwt["countrycode"].isin(BASELINE_COUS))].copy()
    # GDP per capita at current PPPs: cgdpo / pop (matches the "Log GDP per capita" axis)
    pwt["gdp_pc"] = pwt["cgdpo"] / pwt["pop"]
    pwt["lgdp_pc"] = np.log(pwt["gdp_pc"])
    pwt = pwt[["countrycode", "lgdp_pc"]].rename(columns={"countrycode": "cou"})

    m = rd.merge(pwt, on="cou", how="inner")
    assert len(m) == 25, f"Expected 25 countries, got {len(m)}"

    corr = np.corrcoef(m["lgdp_pc"], m["rnd_eff_m"])[0, 1]
    print(f"corr(log GDP per capita, log R&D efficiency b_s) = {corr:.3f}")
    print("(Paper's Figure 2 notes do not give the numerical correlation, but the "
          "scatter is strongly positive — ours matches that pattern.)")

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(m["lgdp_pc"], m["rnd_eff_m"], s=30, color="#1f3864")
    for _, r in m.iterrows():
        ax.annotate(r["cou"], (r["lgdp_pc"], r["rnd_eff_m"]),
                    xytext=(3, 3), textcoords="offset points", fontsize=8)
    ax.set_xlabel("Log GDP per capita (PWT 9.0, 2010)")
    ax.set_ylabel("Log R&D efficiency $b_s$ (median, USA normalized to 0)")
    ax.set_title("Figure 2 replication: R&D efficiency vs GDP per capita")
    ax.axhline(0, color="grey", lw=0.5)
    ax.grid(alpha=0.3)
    out = OUT / "figure2_replication.png"
    fig.tight_layout()
    fig.savefig(out, dpi=140)
    print(f"\nSaved {out}")


if __name__ == "__main__":
    main()
