"""Replicate Table 2 Panel A from Artiles (2025).

Original Stata:
    reghdfe y d_div g_all_idi_grouplevel_4_n_w inter_d_div_all $controls_all,
        abs(par_id) vce(cluster par_id) keepsing

Outcomes (5 columns):
    1: l_nl_pc13_viirs       — log nightlights per capita 2013
    2: l_mean_nl_pc10_13     — log mean nightlights per capita 2010-13
    3: n_f_1013_p100         — firms per 100 pop 2010-13
    4: CN12_d_ua_venta       — 2012 ag census: sells production (dummy)
    5: CN12_sh_ua_venta      — 2012 ag census: share of parcels selling

We absorb par_id (44 provinces), cluster on par_id.
"""
from pathlib import Path
import sys
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from utils import load_doctrinas, reghdfe, CONTROLS_ALL, PANEL_A_OUTCOMES

PUB = {
    "l_nl_pc13_viirs":  {"d_div": (-0.036, 0.016, 0.023), "inter": (0.042, 0.021)},
    "l_mean_nl_pc10_13": {"d_div": (-0.107, 0.033, 0.056), "inter": (0.119, 0.042)},
    "n_f_1013_p100":    {"d_div": (-11.991, 7.041, 23.992), "inter": (19.147, 10.425)},
    "CN12_d_ua_venta":  {"d_div": (-0.416, 0.163, 0.500), "inter": (0.593, 0.233)},
    "CN12_sh_ua_venta": {"d_div": (-0.131, 0.069, 0.369), "inter": (0.151, 0.100)},
}


def main():
    df = load_doctrinas()
    print("Data shape:", df.shape)
    xvars = ["d_div", "g_all_idi_grouplevel_4_n_w", "inter_d_div_all"] + CONTROLS_ALL
    rows = []
    for y in PANEL_A_OUTCOMES:
        out = reghdfe(df, y, xvars, absorb="par_id")
        b_div = out["params"]["d_div"]
        se_div = out["se"]["d_div"]
        b_int = out["params"]["inter_d_div_all"]
        se_int = out["se"]["inter_d_div_all"]
        ymean = out["ymean"]
        pub_div_b, pub_div_se, pub_ymean = PUB[y]["d_div"]
        pub_int_b, pub_int_se = PUB[y]["inter"]
        rows.append({
            "outcome": y,
            "N": out["N"],
            "b_div": b_div, "se_div": se_div,
            "pub_b_div": pub_div_b, "pub_se_div": pub_div_se,
            "b_int": b_int, "se_int": se_int,
            "pub_b_int": pub_int_b, "pub_se_int": pub_int_se,
            "ymean": ymean, "pub_ymean": pub_ymean,
        })
    res = pd.DataFrame(rows)
    pd.set_option("display.width", 200)
    pd.set_option("display.max_columns", None)
    print()
    print("Table 2 Panel A: Ethnic Diversity x Crop Exchange")
    print("=" * 90)
    for _, r in res.iterrows():
        print()
        print(f"[{r.outcome}]  N={r.N}  ymean={r.ymean:.3f} (pub {r.pub_ymean:.3f})")
        print(f"  Ethnic diversity        : {r.b_div:+.3f} [{r.se_div:.3f}]"
              f"   (pub {r.pub_b_div:+.3f} [{r.pub_se_div:.3f}])")
        print(f"  Ethnic div x Exchange   : {r.b_int:+.3f} [{r.se_int:.3f}]"
              f"   (pub {r.pub_b_int:+.3f} [{r.pub_se_int:.3f}])")

    out_csv = Path(__file__).resolve().parent / "out_table2_panelA.csv"
    res.to_csv(out_csv, index=False)
    print(f"\nWrote {out_csv}")


if __name__ == "__main__":
    main()
