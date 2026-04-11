"""Replicate Figures 11 and 12 from the computed measurement.

- Figure 11: histograms of annual growth rate of internal distortions
  (world, USA, Europe, JPN, CHN).
- Figure 12: manufacturing vs service TFP index over 1995-2011, plus
  ranked annual TFP growth rates by sector.

Uses only the computed (Python) intermediates — does NOT depend on MATLAB
counterfactual outputs.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from utils import OUT

OUT_I = OUT / "intermediate"
FIG = OUT / "figures"
FIG.mkdir(exist_ok=True)

EUR = {
    "AUT", "BEL", "BGR", "CYP", "CZE", "DEU", "DNK", "ESP", "EST", "FIN",
    "FRA", "GBR", "GRC", "HUN", "IRL", "ITA", "LTU", "LUX", "LVA", "MLT",
    "NLD", "POL", "PRT", "ROU", "ROM", "SVK", "SVN", "SWE",
}

# WIOD sector 1 = Agriculture, 2 = Mining, 3-16 = Manufacturing, 17 = Utilities
# 18 = Construction, 19-35 = Services (per Readme)
MANUF = set(range(3, 17))  # 3..16 manufacturing (food to manuf nec)
SERVICES = set(range(19, 35))  # 19..34 services (paper drops 35 = Private HH)


def annual_growth_from_taus(d: pd.DataFrame) -> pd.DataFrame:
    """Return dataframe of log-diff annual growth rates (percent)."""
    d = d.sort_values(["importer", "exporter", "sec_imp", "sec_exp", "year"]).copy()
    keys = ["importer", "exporter", "sec_imp", "sec_exp"]
    d["log_tau"] = np.log(d["int_taus_std"].replace(0, np.nan))
    d["dlog"] = d.groupby(keys)["log_tau"].diff() * 100.0  # approx percent change
    return d.dropna(subset=["dlog"])


def figure11() -> None:
    itau = pd.read_parquet(OUT_I / "int_taus_std_1995_2011.parquet")
    # drop identical-sector self-loops (tau==1 by construction)
    itau = itau[itau.sec_imp != itau.sec_exp]
    itau = itau.dropna(subset=["int_taus_std"])
    itau = itau[itau["int_taus_std"] > 0]
    growth = annual_growth_from_taus(itau)
    growth = growth[np.isfinite(growth["dlog"])]
    print(f"Figure 11: n(growth obs) = {len(growth):,}")

    def panel(ax, data, title):
        data = data[np.isfinite(data)]
        # trim extreme tails for plot readability, like the paper's fixed axes
        data = data[(data > np.percentile(data, 0.5)) & (data < np.percentile(data, 99.5))]
        ax.hist(data, bins=40, color="#BFB56B", edgecolor="none", density=True)
        ax.set_title(title, fontsize=9)
        ax.set_xlabel("Annual Growth")
        ax.set_ylabel("Density")

    fig, axes = plt.subplots(3, 2, figsize=(8, 10))
    axes = axes.ravel()
    panel(axes[0], growth["dlog"].values, "Annual Avg Growth Rate of Global Internal Distortions")
    axes[1].axis("off")
    panel(axes[2], growth[growth.importer == "USA"]["dlog"].values, "USA")
    panel(axes[3], growth[growth.importer.isin(EUR)]["dlog"].values, "Europe")
    panel(axes[4], growth[growth.importer == "JPN"]["dlog"].values, "JPN")
    panel(axes[5], growth[growth.importer == "CHN"]["dlog"].values, "CHN")
    fig.suptitle("Figure 11 (replicated): Distribution of changes in internal distortions", fontsize=11)
    fig.tight_layout()
    fig.savefig(FIG / "figure11_replicated.png", dpi=150)
    plt.close(fig)

    # Report summary stats
    rep = pd.DataFrame({
        "group": ["World", "USA", "Europe", "JPN", "CHN"],
        "n": [
            len(growth),
            (growth.importer == "USA").sum(),
            growth.importer.isin(EUR).sum(),
            (growth.importer == "JPN").sum(),
            (growth.importer == "CHN").sum(),
        ],
        "mean": [
            growth["dlog"].mean(),
            growth[growth.importer == "USA"]["dlog"].mean(),
            growth[growth.importer.isin(EUR)]["dlog"].mean(),
            growth[growth.importer == "JPN"]["dlog"].mean(),
            growth[growth.importer == "CHN"]["dlog"].mean(),
        ],
        "sd": [
            growth["dlog"].std(),
            growth[growth.importer == "USA"]["dlog"].std(),
            growth[growth.importer.isin(EUR)]["dlog"].std(),
            growth[growth.importer == "JPN"]["dlog"].std(),
            growth[growth.importer == "CHN"]["dlog"].std(),
        ],
    })
    print("\n--- Figure 11 summary (paper claims: CHN > Europe > USA ~ JPN dispersion) ---")
    print(rep.to_string(index=False, float_format=lambda x: f"{x:8.3f}"))
    rep.to_csv(OUT / "figure11_summary.csv", index=False)


def figure12() -> None:
    tfp = pd.read_parquet(OUT_I / "TFP_std_1995_2011.parquet")
    gross = pd.read_parquet(OUT_I / "gross_output.parquet")
    # Merge gross output weights
    df = tfp.merge(gross, on=["year", "importer", "sec_imp"], how="left")
    df = df[df.sec_imp != 1]  # drop agriculture (the numeraire)
    df = df[df.sec_imp != 35]  # drop Private Households sector
    df = df.dropna(subset=["TFP_std"])
    df = df[np.isfinite(df["TFP_std"]) & (df["TFP_std"] > 0)]

    # --- Upper panel: manufacturing vs services TFP index ---
    def wmean(x, w):
        m = np.isfinite(x) & np.isfinite(w) & (w > 0)
        if not m.any():
            return np.nan
        return np.average(x[m], weights=w[m])

    idx_rows = []
    for year in sorted(df.year.unique()):
        sub = df[df.year == year]
        m = sub[sub.sec_imp.isin(MANUF)]
        s = sub[sub.sec_imp.isin(SERVICES)]
        idx_rows.append({
            "year": year,
            "manuf_tfp": wmean(m["TFP_std"].values, m["gross_output"].values),
            "serv_tfp": wmean(s["TFP_std"].values, s["gross_output"].values),
        })
    idx = pd.DataFrame(idx_rows)
    # Normalize so 1996 == 1 (paper uses 1996=1 on y axis)
    for col in ["manuf_tfp", "serv_tfp"]:
        base = idx.loc[idx.year == 1996, col].values[0]
        idx[col] = idx[col] / base
    idx.to_csv(OUT / "figure12_upper.csv", index=False)

    fig, ax = plt.subplots(figsize=(6, 3.5))
    ax.plot(idx["year"], idx["manuf_tfp"], "-", label="Manufacturing", color="#333")
    ax.plot(idx["year"], idx["serv_tfp"], "--", label="Service", color="#333")
    ax.set_title("Figure 12 (upper): World TFP growth (1996=1)")
    ax.set_xlabel("Year"); ax.set_ylabel("TFP Growth")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIG / "figure12_upper.png", dpi=150)
    plt.close(fig)

    # --- Lower panels: annual growth rate by sector ---
    # Compute each country-sector's average annual log growth 1995-2011, then
    # weight across countries by gross output.
    gr = df.sort_values(["importer", "sec_imp", "year"]).copy()
    gr["log_tfp"] = np.log(gr["TFP_std"].replace(0, np.nan))
    gr["dlog"] = gr.groupby(["importer", "sec_imp"])["log_tfp"].diff() * 100.0
    gr = gr.dropna(subset=["dlog", "gross_output"])

    sector_gr = (
        gr.groupby("sec_imp")
          .apply(lambda g: np.average(g["dlog"], weights=g["gross_output"]))
          .rename("annual_pct").reset_index()
    )
    sector_gr.to_csv(OUT / "figure12_lower_sector_growth.csv", index=False)
    print("\n--- Figure 12 lower: annual TFP growth (%) by sector ---")
    manuf_sorted = sector_gr[sector_gr.sec_imp.isin(MANUF)].sort_values("annual_pct", ascending=False)
    serv_sorted = sector_gr[sector_gr.sec_imp.isin(SERVICES)].sort_values("annual_pct", ascending=False)
    print("Manufacturing:")
    print(manuf_sorted.to_string(index=False, float_format=lambda x: f"{x:7.3f}"))
    print("\nServices:")
    print(serv_sorted.to_string(index=False, float_format=lambda x: f"{x:7.3f}"))


def main() -> None:
    figure11()
    figure12()
    print(f"\nFigures saved to {FIG}")


if __name__ == "__main__":
    main()
