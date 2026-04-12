"""Replicate Table 2 Panel B from Artiles (2025).

Columns 1-2: parish-level sanitation outcomes (same as Panel A).
Columns 3-5: ENAHO household survey — log household per-capita consumption
regressed on ethnic diversity, interaction, individual chars, parish and
year FE. Columns 3 is pooled, 4 is anio<2011, 5 is anio>=2011.

The original Stata uses vce(cluster ID) where ID is a household-level
cluster; this isn't present in data_enaho.dta, so we approximate with
par_id clustering. The published SE corresponds to a finer cluster; we
note this in the writeup.
"""
from pathlib import Path
import sys
import pandas as pd
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from utils import load_doctrinas, reghdfe, CONTROLS_ALL

PKG = Path(__file__).resolve().parent.parent / "235123-V1" / "Z1_Replication"
ENAHO_DTA = PKG / "02_output" / "02b_final" / "data" / "data_enaho.dta"

INDIVIDUAL_BASE = ["female", "age", "age2", "years_sch_1", "years_sch_2_4",
                   "years_sch_5", "years_sch_6"]
# p209_g* and p300a_g* are added dynamically after loading ENAHO.
PUB = {
    "C1": {"y": "CS17_sh_agua_red", "d_div": (-0.088, 0.086), "inter": (0.122, 0.124), "N": 336, "ymean": 0.763},
    "C2": {"y": "CS17_sh_hig_red", "d_div": (-0.096, 0.106), "inter": (0.134, 0.175), "N": 336, "ymean": 0.468},
    "C3": {"y": "ln_consumo_pc", "d_div": (-0.301, 0.161), "inter": (0.491, 0.228), "N": 53361, "ymean": 8.098},
    "C4": {"y": "ln_consumo_pc_lt2011", "d_div": (-0.386, 0.202), "inter": (0.601, 0.282), "N": 21258, "ymean": 7.918},
    "C5": {"y": "ln_consumo_pc_gte2011", "d_div": (-0.238, 0.168), "inter": (0.420, 0.235), "N": 32103, "ymean": 8.217},
}


def main():
    df = load_doctrinas()

    # C1, C2
    print("Table 2 Panel B: Ethnic Diversity x Crop Exchange (incl. household consumption)")
    print("=" * 90)
    xvars_base = ["d_div", "g_all_idi_grouplevel_4_n_w", "inter_d_div_all"] + CONTROLS_ALL
    for label, y in [("C1", "CS17_sh_agua_red"), ("C2", "CS17_sh_hig_red")]:
        out = reghdfe(df, y, xvars_base, absorb="par_id")
        pub = PUB[label]
        print(f"\n[{label}] {y}  N={out['N']} (pub {pub['N']}) ymean={out['ymean']:.3f}")
        print(f"  d_div: {out['params']['d_div']:+.3f} [{out['se']['d_div']:.3f}]"
              f"   (pub {pub['d_div'][0]:+.3f} [{pub['d_div'][1]:.3f}])")
        print(f"  inter: {out['params']['inter_d_div_all']:+.3f} [{out['se']['inter_d_div_all']:.3f}]"
              f"   (pub {pub['inter'][0]:+.3f} [{pub['inter'][1]:.3f}])")

    # Load ENAHO and merge (1:m on u_id)
    enaho = pd.read_stata(ENAHO_DTA, convert_categoricals=False)
    merged = enaho.merge(df, on="u_id", how="inner", suffixes=("", "_par"))
    print(f"\nENAHO merge: {merged.shape[0]} rows, "
          f"{merged['u_id'].nunique()} parishes, anio range "
          f"{merged['anio'].min():.0f}-{merged['anio'].max():.0f}")

    # Build log consumption
    # Drop non-positive consumption before log
    merged = merged[merged["gashog1d_not_pc_r"] > 0].copy()
    merged["lnc"] = np.log(merged["gashog1d_not_pc_r"])
    # Match Stata individual control set: p209_g*, p300a_g* included
    p209_g = sorted([c for c in merged.columns if c.startswith("p209_g")])
    p300a_g = sorted([c for c in merged.columns if c.startswith("p300a_g")])
    individual = INDIVIDUAL_BASE + p209_g + p300a_g
    xvars = xvars_base + individual
    sample = merged.dropna(subset=["lnc"] + xvars + ["anio", "par_id"]).copy()
    print(f"Sample after dropping NAs: {sample.shape[0]} (pub C3: 53361)")

    # Use statsmodels OLS with explicit parish + year dummies, cluster par_id.
    import statsmodels.api as sm

    def run_reg(d, label):
        par_dum = pd.get_dummies(d["par_id"], prefix="par", drop_first=True).astype(float)
        yr_dum = pd.get_dummies(d["anio"], prefix="yr", drop_first=True).astype(float)
        X = pd.concat(
            [d[xvars].reset_index(drop=True),
             par_dum.reset_index(drop=True),
             yr_dum.reset_index(drop=True)],
            axis=1,
        )
        X = sm.add_constant(X)
        y = d["lnc"].reset_index(drop=True)
        groups = d["par_id"].reset_index(drop=True).values
        res = sm.OLS(y, X).fit(cov_type="cluster", cov_kwds={"groups": groups})
        pub = PUB[label]
        bd = res.params["d_div"]; sdv = res.bse["d_div"]
        bi = res.params["inter_d_div_all"]; siv = res.bse["inter_d_div_all"]
        print(f"\n[{label}] N={int(res.nobs)} (pub {pub['N']}) ymean={y.mean():.3f} (pub {pub['ymean']:.3f})")
        print(f"  d_div: {bd:+.3f} [{sdv:.3f}]   (pub {pub['d_div'][0]:+.3f} [{pub['d_div'][1]:.3f}])")
        print(f"  inter: {bi:+.3f} [{siv:.3f}]   (pub {pub['inter'][0]:+.3f} [{pub['inter'][1]:.3f}])")

    run_reg(sample, "C3")
    run_reg(sample[sample["anio"] < 2011].copy(), "C4")
    run_reg(sample[sample["anio"] >= 2011].copy(), "C5")


if __name__ == "__main__":
    main()
