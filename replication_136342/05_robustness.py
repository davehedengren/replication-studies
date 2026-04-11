"""Robustness checks on the measurement step.

All checks operate on the recomputed measurement objects from 01_measure.py.
The paper's main empirical claims the checks stress-test:
  (a) "significant heterogeneity" in the growth rate of internal distortions
       across countries and sectors (Fig. 11)
  (b) manufacturing TFP grows slower than services (Fig. 12)
  (c) median internal distortion changes are small over 2007-2009 (GFC table)
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from utils import OUT, SIGMA, THETA

OUT_I = OUT / "intermediate"


def recompute_taus(sigma: float, theta: float, base_year: int = 1995) -> pd.DataFrame:
    """Recompute internal taus with alternative (sigma, theta, base_year)."""
    alphas = pd.read_parquet(OUT_I / "alphas.parquet")
    dom_gammas = pd.read_parquet(OUT_I / "dom_gammas.parquet")

    gammas_ikik = dom_gammas[dom_gammas.sec_exp == dom_gammas.sec_imp][
        ["year", "importer", "sec_exp", "dom_gammas"]
    ].rename(columns={"dom_gammas": "gammas_ikik"})
    gterm = dom_gammas.merge(gammas_ikik, on=["year", "sec_exp", "importer"], how="left")

    alphas_ij = alphas[["year", "importer", "sec_exp", "alphas"]].rename(
        columns={"sec_exp": "sec_imp", "alphas": "alphas_ij"}
    )
    alphas_ik = alphas[["year", "importer", "sec_exp", "alphas"]].rename(
        columns={"alphas": "alphas_ik"}
    )
    taus = gterm.merge(alphas_ij, on=["year", "importer", "sec_imp"], how="left")
    taus = taus.merge(alphas_ik, on=["year", "importer", "sec_exp"], how="left")
    taus["taus"] = (
        (taus["gammas_ikik"] / taus["dom_gammas"]) ** (1 / theta)
        * (taus["alphas_ij"] / taus["alphas_ik"]) ** (1 / (1 - sigma))
    )

    base = taus[taus.year == base_year][["importer", "sec_imp", "sec_exp", "taus"]].rename(
        columns={"taus": "taus_base"}
    )
    m = taus.merge(base, on=["importer", "sec_imp", "sec_exp"], how="left")
    m = m[m.importer != "RoW"]
    m["int_taus_std"] = m["taus"] / m["taus_base"]
    return m[["year", "importer", "sec_imp", "sec_exp", "int_taus_std"]]


def growth_stats(d: pd.DataFrame, name: str, tag: str) -> dict:
    d = d[d.sec_imp != d.sec_exp].dropna(subset=["int_taus_std"])
    d = d[d.int_taus_std > 0].sort_values(["importer", "sec_imp", "sec_exp", "year"]).copy()
    d["dlog"] = d.groupby(["importer", "sec_imp", "sec_exp"]).int_taus_std.transform(
        lambda x: np.log(x).diff() * 100.0
    )
    d = d.dropna(subset=["dlog"])
    d = d[np.isfinite(d.dlog)]
    return {
        "check": name,
        "tag": tag,
        "n": len(d),
        "mean": d.dlog.mean(),
        "sd": d.dlog.std(),
        "p05": d.dlog.quantile(0.05),
        "p95": d.dlog.quantile(0.95),
    }


def main() -> None:
    rows: list[dict] = []

    # --- Baseline ---
    base = pd.read_parquet(OUT_I / "int_taus_std_1995_2011.parquet")
    rows.append(growth_stats(base, "1. Baseline (sigma=4, theta=4)", "world"))
    rows.append(growth_stats(base[base.importer == "USA"], "2. Baseline - USA only", "usa"))
    rows.append(growth_stats(base[base.importer == "CHN"], "3. Baseline - CHN only", "chn"))

    # --- Alt sigma ---
    alt_s2 = recompute_taus(sigma=2.0, theta=THETA)
    rows.append(growth_stats(alt_s2, "4. Alt elasticity sigma=2", "world"))

    # --- Alt theta ---
    alt_t8 = recompute_taus(sigma=SIGMA, theta=8.0)
    rows.append(growth_stats(alt_t8, "5. Alt trade elasticity theta=8", "world"))

    # --- Alt base year (1996) ---
    alt_b = recompute_taus(sigma=SIGMA, theta=THETA, base_year=1996)
    rows.append(growth_stats(alt_b, "6. Normalize to 1996 (not 1995)", "world"))

    # --- Drop agriculture and private households (sectors 1 and 35) ---
    dropped = base[~base.sec_imp.isin([1, 35]) & ~base.sec_exp.isin([1, 35])]
    rows.append(growth_stats(dropped, "7. Drop sec 1 (Agri) & 35 (PrivHH)", "world"))

    # --- Drop RoW already done upstream; check BRIICS subsample ---
    briics = {"BRA", "RUS", "IND", "IDN", "CHN", "ZAF"}
    rows.append(growth_stats(base[base.importer.isin(briics)], "8. BRIICS subsample", "briics"))

    # --- Trim top/bottom 1% of growth rates ---
    bcopy = base[base.sec_imp != base.sec_exp].dropna(subset=["int_taus_std"]).copy()
    bcopy = bcopy[bcopy.int_taus_std > 0].sort_values(
        ["importer", "sec_imp", "sec_exp", "year"]
    )
    bcopy["dlog"] = bcopy.groupby(["importer", "sec_imp", "sec_exp"]).int_taus_std.transform(
        lambda x: np.log(x).diff() * 100.0
    )
    bcopy = bcopy.dropna(subset=["dlog"])
    bcopy = bcopy[np.isfinite(bcopy.dlog)]
    lo, hi = bcopy.dlog.quantile(0.01), bcopy.dlog.quantile(0.99)
    trimmed = bcopy[(bcopy.dlog >= lo) & (bcopy.dlog <= hi)]
    rows.append({
        "check": "9. Trim 1%/99% growth rates",
        "tag": "world",
        "n": len(trimmed),
        "mean": trimmed.dlog.mean(),
        "sd": trimmed.dlog.std(),
        "p05": trimmed.dlog.quantile(0.05),
        "p95": trimmed.dlog.quantile(0.95),
    })

    # --- Heterogeneity-by-country: sd should vary across countries ---
    het = (
        bcopy.groupby("importer")["dlog"]
             .std()
             .sort_values(ascending=False)
    )
    print("\nAcross-country SD of annual distortion growth (robustness check 10):")
    print(het.describe())
    print("Top 5 most volatile:", het.head().index.tolist())
    print("Bottom 5 least volatile:", het.tail().index.tolist())

    # --- TFP manufacturing vs services claim ---
    tfp = pd.read_parquet(OUT_I / "TFP_std_1995_2011.parquet")
    gross = pd.read_parquet(OUT_I / "gross_output.parquet")
    tfp = tfp.merge(gross, on=["year", "importer", "sec_imp"], how="left")
    tfp = tfp.dropna(subset=["TFP_std", "gross_output"])
    tfp = tfp[np.isfinite(tfp["TFP_std"]) & (tfp["gross_output"] > 0)]
    # Keep only country-sector-year cells where TFP is finite and positive
    tfp = tfp[(tfp.TFP_std > 0)]
    MANUF = set(range(3, 17))
    SERV = set(range(19, 35))  # paper drops sec 35 (Private Households)

    def annual_weighted_mean(d):
        rs = []
        for y, g in d.groupby("year"):
            if g.gross_output.sum() == 0:
                continue
            rs.append((y, np.average(g.TFP_std, weights=g.gross_output)))
        return pd.DataFrame(rs, columns=["year", "tfp"]).set_index("year")

    m_tfp = annual_weighted_mean(tfp[tfp.sec_imp.isin(MANUF)])
    s_tfp = annual_weighted_mean(tfp[tfp.sec_imp.isin(SERV)])
    m_norm = m_tfp / m_tfp.loc[1996].values
    s_norm = s_tfp / s_tfp.loc[1996].values
    print("\nCheck 11: Manufacturing vs Services TFP 2011/1996 index")
    print(f"  Manufacturing (1996=1): {m_norm.loc[2011, 'tfp']:.4f}")
    print(f"  Services     (1996=1): {s_norm.loc[2011, 'tfp']:.4f}")
    print("  Paper claim: Service > Manufacturing growth -> ",
          "CONFIRMED" if s_norm.loc[2011, "tfp"] > m_norm.loc[2011, "tfp"] else "REJECTED")

    # --- Paper claim ordering: dispersion(JPN) < USA < Europe < CHN ---
    EUR = {"AUT", "BEL", "BGR", "CYP", "CZE", "DEU", "DNK", "ESP", "EST", "FIN",
           "FRA", "GBR", "GRC", "HUN", "IRL", "ITA", "LTU", "LUX", "LVA", "MLT",
           "NLD", "POL", "PRT", "ROU", "SVK", "SVN", "SWE"}
    disp = {
        "JPN": bcopy[bcopy.importer == "JPN"].dlog.std(),
        "USA": bcopy[bcopy.importer == "USA"].dlog.std(),
        "Europe (pooled)": bcopy[bcopy.importer.isin(EUR)].dlog.std(),
        "CHN": bcopy[bcopy.importer == "CHN"].dlog.std(),
    }
    print("\nCheck 12: Dispersion ranking (paper: JPN ~ USA < EUR < CHN)")
    for k, v in disp.items():
        print(f"  {k:>15s}: SD={v:.3f}")

    out = pd.DataFrame(rows)
    print("\n=== Growth-rate robustness table ===")
    print(out.to_string(index=False, float_format=lambda x: f"{x:8.3f}"))
    out.to_csv(OUT / "robustness_summary.csv", index=False)


if __name__ == "__main__":
    main()
